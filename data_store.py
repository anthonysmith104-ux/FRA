"""data_store.py — shared GitHub-backed data layer for the MRB Capital apps.

This module is the single point through which the client portal and advisor
app read/write shared state (client risk assessments, firm settings, etc.).
Both apps point at the same GitHub repo, set in Streamlit secrets:

    [github]
    token = "github_pat_..."
    data_repo = "owner/reponame"

The token must have Contents: Read and write on that repo. Nothing else.

This file is currently a stub — only `selftest()` is implemented. Once the
selftest passes in both apps, the real load_*/save_* helpers will be added
and the apps' existing JSON helpers will start delegating to them.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import requests
import streamlit as st


GITHUB_API = "https://api.github.com"


# ── Config ──────────────────────────────────────────────────────────────────
def _config() -> tuple[str, str]:
    """Return (token, 'owner/repo') from Streamlit secrets, or raise a clear
    error explaining what's missing."""
    try:
        gh = st.secrets["github"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "Missing [github] section in Streamlit secrets. "
            "Add it via the app's Settings → Secrets page."
        )
    token = gh.get("token")
    repo  = gh.get("data_repo")
    if not token:
        raise RuntimeError("github.token is missing from Streamlit secrets.")
    if not repo or "/" not in repo:
        raise RuntimeError(
            "github.data_repo must be in 'owner/reponame' format "
            "(got %r)." % repo
        )
    return token, repo


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept":         "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ── Selftest ────────────────────────────────────────────────────────────────
def selftest() -> dict:
    """Verify the GitHub credentials by writing, reading, and deleting a
    small test file at .selftest/ping.txt in the data repo. Returns a dict
    with status info; never raises (errors come back as status='error').
    """
    out: dict = {"step": "start"}
    try:
        token, repo = _config()
        out["repo"] = repo
        out["step"] = "config_ok"

        path  = ".selftest/ping.txt"
        url   = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        body  = f"selftest from data_store.py\n"
        b64   = base64.b64encode(body.encode("utf-8")).decode("ascii")

        # Check whether file already exists (we need its sha to overwrite).
        r_get = requests.get(url, headers=_headers(token), timeout=10)
        sha = r_get.json().get("sha") if r_get.status_code == 200 else None
        out["step"] = "get_ok"

        # Write (create or update).
        payload = {
            "message": "data_store selftest",
            "content": b64,
            "branch":  "main",
        }
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
        out["step"] = "write_ok"

        # Read it back.
        r_get2 = requests.get(url, headers=_headers(token), timeout=10)
        if r_get2.status_code != 200:
            out["status"] = "error"
            out["step"]   = "read_failed"
            out["http"]   = r_get2.status_code
            out["detail"] = r_get2.json()
            return out
        round_trip = base64.b64decode(
            r_get2.json()["content"]
        ).decode("utf-8")
        out["round_trip_ok"] = (round_trip == body)
        out["step"] = "read_ok"

        # Clean up.
        del_payload = {
            "message": "data_store selftest cleanup",
            "sha":     r_get2.json()["sha"],
            "branch":  "main",
        }
        requests.delete(url, headers=_headers(token),
                        json=del_payload, timeout=10)
        out["step"]   = "cleanup_ok"
        out["status"] = "ok"
        return out

    except Exception as e:
        out["status"] = "error"
        out["error"]  = f"{type(e).__name__}: {e}"
        return out


# ── Streamlit page (visit /?selftest=1 to see results) ─────────────────────
def render_selftest_page():
    """Render the selftest results inline. Call this from client_portal.py
    when ?selftest=1 is in the URL, so we can verify credentials without
    touching any real app flow yet."""
    st.markdown("### data_store selftest")
    with st.spinner("Pinging GitHub..."):
        result = selftest()
    if result.get("status") == "ok":
        st.success("All steps passed. The data layer is connected.")
    else:
        st.error("Selftest failed — see details below.")
    st.json(result)
