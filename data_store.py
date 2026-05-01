"""data_store.py — shared GitHub-backed JSON store for the MRB Capital apps.

Both the client portal and the advisor app read/write the same JSON files
through this module, so changes made in one app become visible to the other
within ~60 seconds. The backing store is a private GitHub repo named in
Streamlit secrets:

    [github]
    token = "github_pat_..."
    data_repo = "owner/reponame"

The token must have Contents: Read and write on that one repo.

Public API (designed to drop in for shared.load_json / shared.update_json):

    load_json(path, default=...)      -> dict / list — read JSON from store
    update_json(path, mutator)        -> None       — read, mutate, write back

Both accept either a bare filename ("client_profiles.json") or a full path
("/some/dir/client_profiles.json"); the basename is what gets used inside
the GitHub repo.

Behavior:

  - If [github] secrets are configured, all reads/writes go to GitHub.
  - If they are NOT configured (e.g. local dev with no .streamlit/secrets.toml),
    operations fall back to local-disk JSON in the original location. This
    keeps `streamlit run client_portal.py` on a laptop working without
    requiring a token.
  - Reads are cached for 60 seconds per file to avoid hammering the API.
    update_json() invalidates the cache for the file it touches.
  - Writes use the GitHub Contents API with the file's SHA, so concurrent
    writes get a 409 and we retry once. (Last-write-wins semantics in the
    rare case of a true conflict.)

This file is the ONLY place that talks to the GitHub API. If a third app
ever needs to share data, it imports this module and gets the same view.
"""
from __future__ import annotations

import base64
import json
import os
import time
from threading import Lock
from typing import Any, Callable, Optional

import requests
import streamlit as st


GITHUB_API     = "https://api.github.com"
_CACHE_TTL_SEC = 60.0


# ── Config ──────────────────────────────────────────────────────────────────
def _config() -> Optional[tuple[str, str]]:
    """Return (token, 'owner/repo') from Streamlit secrets, or None if the
    secrets aren't present. Never raises — callers fall back to local mode
    when this returns None."""
    try:
        gh = st.secrets["github"]
    except (KeyError, FileNotFoundError, AttributeError):
        return None
    token = gh.get("token") if hasattr(gh, "get") else gh["token"]
    repo  = gh.get("data_repo") if hasattr(gh, "get") else gh["data_repo"]
    if not token or not repo or "/" not in repo:
        return None
    return token, repo


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept":         "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def is_remote() -> bool:
    """True if we're talking to GitHub. False if falling back to local disk
    (e.g. on a developer laptop without secrets configured)."""
    return _config() is not None


# ── Cache ───────────────────────────────────────────────────────────────────
# Keyed by basename. Value is (expires_at, parsed_json, sha).
# A lock guards the dict because Streamlit can fire concurrent reruns.
_cache: dict[str, tuple[float, Any, Optional[str]]] = {}
_cache_lock = Lock()


def _cache_get(name: str) -> Optional[tuple[Any, Optional[str]]]:
    with _cache_lock:
        entry = _cache.get(name)
        if entry is None:
            return None
        expires, value, sha = entry
        if time.time() > expires:
            _cache.pop(name, None)
            return None
        return value, sha


def _cache_put(name: str, value: Any, sha: Optional[str]) -> None:
    with _cache_lock:
        _cache[name] = (time.time() + _CACHE_TTL_SEC, value, sha)


def _cache_invalidate(name: str) -> None:
    with _cache_lock:
        _cache.pop(name, None)


def clear_cache() -> None:
    """Drop all cached reads. Useful if you know remote state changed and
    don't want to wait out the TTL."""
    with _cache_lock:
        _cache.clear()


# ── GitHub I/O ──────────────────────────────────────────────────────────────
class _ConflictError(Exception):
    """Internal — caller (update_json) catches this and retries once."""


def _github_get(name: str) -> tuple[Any, Optional[str]]:
    """Fetch JSON file from the data repo. Returns (parsed_value, sha).
    Returns (None, None) if the file doesn't exist yet."""
    cfg = _config()
    if cfg is None:
        raise RuntimeError("data_store._github_get called without secrets")
    token, repo = cfg
    url = f"{GITHUB_API}/repos/{repo}/contents/{name}"
    r = requests.get(url, headers=_headers(token), timeout=10)
    if r.status_code == 404:
        return None, None
    if r.status_code != 200:
        raise RuntimeError(
            f"GitHub GET {name} failed: {r.status_code} {r.text[:300]}"
        )
    payload = r.json()
    raw = base64.b64decode(payload["content"]).decode("utf-8")
    parsed = json.loads(raw) if raw.strip() else None
    return parsed, payload.get("sha")


def _github_put(
    name: str,
    value: Any,
    prev_sha: Optional[str],
    message: str,
) -> str:
    """Write JSON to the data repo. Returns the new SHA. Raises on failure."""
    cfg = _config()
    if cfg is None:
        raise RuntimeError("data_store._github_put called without secrets")
    token, repo = cfg
    url = f"{GITHUB_API}/repos/{repo}/contents/{name}"
    body = json.dumps(value, indent=2, default=str).encode("utf-8")
    payload = {
        "message": message,
        "content": base64.b64encode(body).decode("ascii"),
        "branch":  "main",
    }
    if prev_sha:
        payload["sha"] = prev_sha
    r = requests.put(url, headers=_headers(token),
                     json=payload, timeout=15)
    if r.status_code in (200, 201):
        return r.json()["content"]["sha"]
    if r.status_code == 409 or (r.status_code == 422 and "sha" in r.text):
        raise _ConflictError(r.text)
    raise RuntimeError(
        f"GitHub PUT {name} failed: {r.status_code} {r.text[:300]}"
    )


# ── Public API ──────────────────────────────────────────────────────────────
def load_json(path: str, default: Any = None) -> Any:
    """Read a JSON file from the shared store (or local disk if no
    secrets are configured). Returns `default` if the file doesn't
    exist. The `path` argument may be a full filesystem path; only its
    basename is used to address the file in the GitHub repo."""
    name = os.path.basename(path)
    fallback = default if default is not None else {}

    cfg = _config()
    if cfg is None:
        # Local fallback — read from disk like the original shared.load_json.
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback

    cached = _cache_get(name)
    if cached is not None:
        value, _sha = cached
        return value if value is not None else fallback

    try:
        value, sha = _github_get(name)
    except RuntimeError:
        # Remote read failed — return default rather than crashing.
        # Don't cache the failure; we'll retry on the next call.
        return fallback

    _cache_put(name, value, sha)
    return value if value is not None else fallback


def save_json(path: str, value: Any) -> None:
    """Write `value` as the entire contents of the JSON file. This is a
    full-file overwrite — use update_json() instead if you only want to
    change part of the file and preserve concurrent edits.

    Same signature as shared.save_json — drop-in replacement."""
    name = os.path.basename(path)

    cfg = _config()
    if cfg is None:
        # Local fallback — atomic-ish write to disk.
        os.makedirs(
            os.path.dirname(os.path.abspath(path)) or ".",
            exist_ok=True,
        )
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, default=str)
        os.replace(tmp, path)
        return

    # Remote — fetch current SHA (if any) so the write succeeds even if
    # the file already exists. We don't pass the cached SHA because save
    # is intentionally a "blow away whatever's there" operation.
    try:
        _existing, sha = _github_get(name)
    except RuntimeError:
        sha = None

    new_sha = _github_put(
        name, value, sha,
        message=f"save {name}",
    )
    _cache_put(name, value, new_sha)


def update_json(path: str, mutator: Callable[[Any], None]) -> None:
    """Read the JSON file, apply `mutator(value)` in-place, write it back.
    Same signature as shared.update_json — drop-in replacement.

    Retries once on a SHA conflict (someone else wrote between our read
    and our write)."""
    name = os.path.basename(path)

    cfg = _config()
    if cfg is None:
        # Local fallback — atomic-ish read/mutate/write to disk.
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        mutator(value)
        os.makedirs(
            os.path.dirname(os.path.abspath(path)) or ".",
            exist_ok=True,
        )
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, default=str)
        os.replace(tmp, path)
        return

    for attempt in (1, 2):
        try:
            value, sha = _github_get(name)
        except RuntimeError as e:
            raise RuntimeError(f"data_store.update_json read failed: {e}")

        if value is None:
            value = {}
        if not isinstance(value, (dict, list)):
            value = {}

        mutator(value)

        try:
            new_sha = _github_put(
                name, value, sha,
                message=f"update {name}",
            )
        except _ConflictError:
            if attempt == 2:
                raise RuntimeError(
                    f"data_store.update_json: conflict writing {name} "
                    "after retry."
                )
            _cache_invalidate(name)
            time.sleep(0.5)
            continue

        _cache_put(name, value, new_sha)
        return


# ── Selftest (kept as a diagnostic) ─────────────────────────────────────────
def selftest() -> dict:
    """Verify GitHub credentials with a write/read/delete round-trip on a
    test file. Returns a dict; never raises."""
    out: dict = {"step": "start"}
    try:
        cfg = _config()
        if cfg is None:
            out["status"] = "error"
            out["mode"]   = "local"
            out["error"]  = "No [github] secrets configured."
            return out
        token, repo = cfg
        out["repo"] = repo
        out["mode"] = "remote"
        out["step"] = "config_ok"

        path = ".selftest/ping.txt"
        url  = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        body = "selftest from data_store.py\n"
        b64  = base64.b64encode(body.encode("utf-8")).decode("ascii")

        r_get = requests.get(url, headers=_headers(token), timeout=10)
        sha = r_get.json().get("sha") if r_get.status_code == 200 else None

        payload = {"message": "data_store selftest",
                   "content": b64, "branch": "main"}
        if sha:
            payload["sha"] = sha
        r_put = requests.put(url, headers=_headers(token),
                             json=payload, timeout=15)
        if r_put.status_code not in (200, 201):
            out["status"] = "error"
            out["step"]   = "write_failed"
            out["http"]   = r_put.status_code
            out["detail"] = r_put.json()
            return out

        r_get2 = requests.get(url, headers=_headers(token), timeout=10)
        if r_get2.status_code != 200:
            out["status"] = "error"
            out["step"]   = "read_failed"
            return out
        round_trip = base64.b64decode(
            r_get2.json()["content"]).decode("utf-8")
        out["round_trip_ok"] = (round_trip == body)

        requests.delete(
            url, headers=_headers(token),
            json={"message": "data_store selftest cleanup",
                  "sha": r_get2.json()["sha"], "branch": "main"},
            timeout=10,
        )
        out["status"] = "ok"
        out["step"]   = "cleanup_ok"
        return out

    except Exception as e:
        out["status"] = "error"
        out["error"]  = f"{type(e).__name__}: {e}"
        return out


def render_selftest_page():
    """Diagnostic page — visit /?selftest=1 to see results."""
    st.markdown("### data_store selftest")
    st.caption(
        "Mode: " +
        ("remote (GitHub)" if is_remote() else "local fallback (no token)")
    )
    with st.spinner("Pinging GitHub..."):
        result = selftest()
    if result.get("status") == "ok":
        st.success("All steps passed. The data layer is connected.")
    else:
        st.error("Selftest failed — see details below.")
    st.json(result)
