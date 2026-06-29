"""data_store.py — shared Firestore-backed JSON store for the MRB Capital apps.

Drop-in replacement for the previous GitHub-backed store. The public interface
is IDENTICAL — load_json / save_json / update_json / is_remote / clear_cache /
selftest — so neither client_portal.py nor app.py changes. Only the backend
moved: GitHub repo  →  Cloud Firestore. The point of the move is to get client
PII out of a git repo and into a private database.

Secrets (already present from the Firebase auth work):

    [firebase_service_account]
    type = "service_account"
    project_id = "risk-checkup"
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "...@risk-checkup.iam.gserviceaccount.com"
    ...

Public API (unchanged):

    load_json(path, default=...)   -> dict / list   — read JSON from store
    save_json(path, value)         -> None          — full-file overwrite
    update_json(path, mutator)     -> None          — read, mutate, write back

Both accept a bare filename ("ra_users.json") or a full path; the basename is
what addresses the data in Firestore.

Storage scheme
--------------
Most files are small config blobs → stored as ONE Firestore document:
    collection "_files", document <basename>, field "_value" = <the JSON>

The three big record files are KEY->RECORD maps that grow with the client base.
To stay clear of Firestore's 1 MB-per-document ceiling and to make concurrent
registrations safe, each is "sharded" — one document per client:
    collection "<name without .json>", document <client key>,
        fields { "_key": <original key>, "_value": <that client's record> }
load_json reassembles the full {key: record} dict; update_json writes back only
the documents that actually changed (so two clients registering at once touch
different documents and never clobber each other).

Behavior preserved from the GitHub version
------------------------------------------
  - If the Firebase service-account secret is NOT configured (local dev with no
    secrets), operations fall back to local-disk JSON in the original path, so
    `streamlit run client_portal.py` on a laptop keeps working.
  - Reads are cached 60 s per file; writes invalidate that file's cache.
  - update_json applies the mutator and writes back; on the rare concurrent
    edit of the same record, last-write-wins (same semantics as before).
"""
from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Any, Callable, Optional

import streamlit as st

import firebase_admin
from firebase_admin import credentials, firestore

_CACHE_TTL_SEC = 60.0

# The big key->record maps, stored one document per key. Everything else is a
# single document under the "_files" collection.
#
# client_proposals.json is a {client_key: {version_id: proposal}} map. It used
# to be a single document, but holding every client's every proposal version in
# one doc eventually crossed Firestore's 1 MiB-per-document ceiling. Sharding it
# per client (one document per client_key) keeps each document small. On the
# first read after this file joined the set, the legacy single document is
# migrated into the sharded collection automatically (see _fs_read).
_SHARDED = {"ra_users.json", "risk_profiles.json", "client_holdings.json",
            "client_proposals.json"}
_FILES_COLLECTION = "_files"

# Firestore commits at most 500 writes per batch; stay under it.
_BATCH_LIMIT = 450


# ── Config / connection ─────────────────────────────────────────────────────
def _has_firebase() -> bool:
    """True if a Firebase service account is configured in secrets."""
    try:
        sa = st.secrets["firebase_service_account"]
    except (KeyError, FileNotFoundError, AttributeError):
        return False
    return bool(sa)


def is_remote() -> bool:
    """True if we're talking to Firestore. False if falling back to local disk
    (e.g. a developer laptop without secrets)."""
    return _has_firebase()


_fs_client = None


def _db():
    """Return a cached Firestore client, initializing the Admin SDK once."""
    global _fs_client
    if _fs_client is not None:
        return _fs_client
    if not firebase_admin._apps:
        sa = dict(st.secrets["firebase_service_account"])
        if "private_key" in sa:
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")
        firebase_admin.initialize_app(credentials.Certificate(sa))
    _fs_client = firestore.client()
    return _fs_client


def _doc_id(key: str) -> str:
    """Make a safe Firestore document id from a record key (e.g. an email).
    Firestore ids can't contain '/', can't be '.' or '..', and can't match the
    reserved __.*__ pattern. The original key is always stored in the doc's
    _key field, so reassembly is exact regardless of this sanitization."""
    s = (str(key) or "").replace("/", "_").strip()
    if s in ("", ".", ".."):
        s = "k_" + s
    # Reserved: ids wrapped in double underscores (e.g. __name__).
    if s.startswith("__") and s.endswith("__"):
        s = "k_" + s.strip("_")
    return s[:1500] or "k_blank"


# ── Cache ───────────────────────────────────────────────────────────────────
# Keyed by basename. Value is (expires_at, parsed_json).
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = Lock()


def _cache_get(name: str) -> Optional[tuple]:
    with _cache_lock:
        entry = _cache.get(name)
        if entry is None:
            return None
        expires, value = entry
        if time.time() > expires:
            _cache.pop(name, None)
            return None
        return (value,)  # wrap so a cached None is distinguishable from a miss


def _cache_put(name: str, value: Any) -> None:
    with _cache_lock:
        _cache[name] = (time.time() + _CACHE_TTL_SEC, value)


def _cache_invalidate(name: str) -> None:
    with _cache_lock:
        _cache.pop(name, None)


def clear_cache() -> None:
    """Drop all cached reads."""
    with _cache_lock:
        _cache.clear()


# ── Firestore I/O ───────────────────────────────────────────────────────────
# Values are stored as a JSON string under "_json" rather than as a live nested
# map. This sidesteps every Firestore document restriction at once (reserved
# keys, nested arrays-of-arrays, null entities) and is invisible to the apps.
# Legacy docs written with a raw "_value" map are still read for safety.
def _encode(value: Any) -> dict:
    return {"_json": json.dumps(value, default=str)}


def _decode(doc: dict) -> Any:
    if doc is None:
        return None
    if "_json" in doc:
        try:
            return json.loads(doc["_json"])
        except Exception:
            return None
    return doc.get("_value")


def _migrate_legacy_single_to_sharded(name: str, coll: str) -> Optional[dict]:
    """One-time migration for a file that was newly added to _SHARDED.

    Its data still lives in the legacy single document under _files/<name>.
    Copy each top-level record into its own document in the sharded collection,
    then delete the legacy document so records deleted later can't be
    resurrected if the collection legitimately empties.

    Safety: records are written one at a time (not in a 450-write batch) so a
    single oversized record can't fail the whole migration. If ANY write fails,
    the partial migration is rolled back and the legacy document is left intact,
    so nothing is lost — the caller still receives the full legacy value for
    this run and the file stays in single-document mode until it can migrate
    cleanly. Returns the full {key: record} dict, or None if there's nothing to
    migrate."""
    db = _db()
    legacy_ref = db.collection(_FILES_COLLECTION).document(name)
    legacy = legacy_ref.get()
    if not legacy.exists:
        return None

    value = _decode(legacy.to_dict() or {})
    if not isinstance(value, dict) or not value:
        # Empty / non-map legacy doc — clear it so we don't re-check forever.
        try:
            legacy_ref.delete()
        except Exception:
            pass
        return value if isinstance(value, dict) else None

    written: list = []
    try:
        for k, v in value.items():
            db.collection(coll).document(_doc_id(k)).set({"_key": k, **_encode(v)})
            written.append(k)
    except Exception:
        # Roll back so the collection isn't left half-populated (which would
        # make subsequent reads return only the migrated subset). Keep the
        # legacy doc; the app still sees everything via the returned value.
        for k in written:
            try:
                db.collection(coll).document(_doc_id(k)).delete()
            except Exception:
                pass
        return value

    # Fully migrated — remove the legacy single document.
    try:
        legacy_ref.delete()
    except Exception:
        pass
    return value


def _fs_read(name: str) -> Any:
    """Read the value of a file from Firestore. Returns the parsed value, or
    None if it doesn't exist yet."""
    db = _db()
    if name in _SHARDED:
        coll = name[:-5] if name.endswith(".json") else name
        out: dict = {}
        for snap in db.collection(coll).stream():
            d = snap.to_dict() or {}
            key = d.get("_key", snap.id)
            out[key] = _decode(d)
        if out:
            return out
        # Sharded collection is empty — this file may have just joined _SHARDED
        # with its data still in the legacy single document. Migrate it once.
        return _migrate_legacy_single_to_sharded(name, coll)
    snap = db.collection(_FILES_COLLECTION).document(name).get()
    if not snap.exists:
        return None
    return _decode(snap.to_dict() or {})


def _fs_write_single(name: str, value: Any) -> None:
    _db().collection(_FILES_COLLECTION).document(name).set(_encode(value))


# Tracks sharded files whose stale legacy single-document copy has already been
# purged this process, so we attempt the (idempotent) delete only once per file.
_legacy_purged: set = set()


def _purge_legacy_single(name: str) -> None:
    """After a sharded file has been written, ensure no stale single-document
    copy survives under the _files collection. If one lingered, an empty
    sharded collection later (e.g. after every record is deleted) would trigger
    re-migration and resurrect those deleted records. Idempotent; runs at most
    once per file per process. Deleting a non-existent document is a no-op."""
    if name in _legacy_purged:
        return
    _legacy_purged.add(name)
    try:
        _db().collection(_FILES_COLLECTION).document(name).delete()
    except Exception:
        pass


def _chunked(items, n):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _fs_write_sharded(name: str, value: dict,
                      delete_missing: bool = True,
                      prev_keys: Optional[set] = None) -> None:
    """Write a {key: record} dict to a sharded collection. When delete_missing
    is True, documents whose key is absent from `value` are removed."""
    db = _db()
    coll = name[:-5] if name.endswith(".json") else name
    if not isinstance(value, dict):
        value = {}

    if delete_missing and prev_keys is None:
        prev_keys = {(s.to_dict() or {}).get("_key", s.id)
                     for s in db.collection(coll).stream()}

    for chunk in _chunked(list(value.items()), _BATCH_LIMIT):
        batch = db.batch()
        for k, v in chunk:
            batch.set(db.collection(coll).document(_doc_id(k)),
                      {"_key": k, **_encode(v)})
        batch.commit()

    if delete_missing and prev_keys:
        removed = [k for k in prev_keys if k not in value]
        for chunk in _chunked(removed, _BATCH_LIMIT):
            batch = db.batch()
            for k in chunk:
                batch.delete(db.collection(coll).document(_doc_id(k)))
            batch.commit()


def _fs_write_changed(name: str, new_value: dict, old_value: dict) -> None:
    """Write only the records that changed between old_value and new_value
    (and delete the ones that were removed). Keeps concurrent registrations of
    different clients from clobbering each other."""
    db = _db()
    coll = name[:-5] if name.endswith(".json") else name

    changed = [(k, v) for k, v in new_value.items()
               if k not in old_value or old_value[k] != v]
    removed = [k for k in old_value if k not in new_value]

    for chunk in _chunked(changed, _BATCH_LIMIT):
        batch = db.batch()
        for k, v in chunk:
            batch.set(db.collection(coll).document(_doc_id(k)),
                      {"_key": k, **_encode(v)})
        batch.commit()
    for chunk in _chunked(removed, _BATCH_LIMIT):
        batch = db.batch()
        for k in chunk:
            batch.delete(db.collection(coll).document(_doc_id(k)))
        batch.commit()


# ── Public API ──────────────────────────────────────────────────────────────
def load_json(path: str, default: Any = None) -> Any:
    """Read a JSON file from the shared store (or local disk if no secrets are
    configured). Returns `default` if the file doesn't exist. `path` may be a
    full filesystem path; only its basename addresses the data."""
    name = os.path.basename(path)
    fallback = default if default is not None else {}

    if not _has_firebase():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback

    cached = _cache_get(name)
    if cached is not None:
        value = cached[0]
        return value if value is not None else fallback

    try:
        value = _fs_read(name)
    except Exception:
        # Read failed — return default rather than crashing. Don't cache the
        # failure; retry on the next call.
        return fallback

    _cache_put(name, value)
    return value if value is not None else fallback


def save_json(path: str, value: Any) -> None:
    """Write `value` as the entire contents of the file (full overwrite).
    Use update_json() to change part of a file while preserving concurrent
    edits. Same signature as before."""
    name = os.path.basename(path)

    if not _has_firebase():
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, default=str)
        os.replace(tmp, path)
        return

    if name in _SHARDED:
        _fs_write_sharded(name, value if isinstance(value, dict) else {},
                          delete_missing=True)
        _purge_legacy_single(name)
    else:
        _fs_write_single(name, value)
    _cache_put(name, value)


def update_json(path: str, mutator: Callable[[Any], None]) -> None:
    """Read the JSON file, apply `mutator(value)` in-place, write it back.
    Same signature as before. On the rare concurrent edit of the same record,
    last-write-wins (matching the previous behavior)."""
    name = os.path.basename(path)

    if not _has_firebase():
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        mutator(value)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, default=str)
        os.replace(tmp, path)
        return

    # Read fresh (not from cache) so we mutate current state.
    try:
        value = _fs_read(name)
    except Exception as e:
        raise RuntimeError(f"data_store.update_json read failed: {e}")

    if value is None:
        value = {}
    if not isinstance(value, (dict, list)):
        value = {}

    if name in _SHARDED:
        if not isinstance(value, dict):
            value = {}
        # Snapshot before mutation so we can write back only what changed.
        old_snapshot = json.loads(json.dumps(value, default=str))
        mutator(value)
        _fs_write_changed(name, value, old_snapshot)
        _purge_legacy_single(name)
    else:
        mutator(value)
        _fs_write_single(name, value)

    _cache_put(name, value)


# ── Selftest (kept as a diagnostic) ─────────────────────────────────────────
def selftest() -> dict:
    """Verify Firestore credentials with a write/read/delete round-trip on a
    test document. Returns a dict; never raises."""
    out: dict = {"step": "start"}
    try:
        if not _has_firebase():
            out["status"] = "error"
            out["mode"] = "local"
            out["error"] = "No [firebase_service_account] secret configured."
            return out
        db = _db()
        out["project"] = st.secrets["firebase_service_account"].get("project_id", "?")
        out["mode"] = "remote"
        out["step"] = "config_ok"

        ref = db.collection(_FILES_COLLECTION).document("_selftest")
        token = f"selftest {time.time()}"
        ref.set(_encode(token))
        out["step"] = "write_ok"

        got = ref.get()
        out["round_trip_ok"] = bool(got.exists and
                                    _decode(got.to_dict() or {}) == token)
        ref.delete()
        out["status"] = "ok"
        out["step"] = "cleanup_ok"
        return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def render_selftest_page():
    """Diagnostic page — visit /?selftest=1 to see results."""
    st.markdown("### data_store selftest")
    st.caption("Mode: " +
               ("remote (Firestore)" if is_remote() else "local fallback (no secret)"))
    with st.spinner("Pinging Firestore..."):
        result = selftest()
    if result.get("status") == "ok":
        st.success("All steps passed. The Firestore data layer is connected.")
    else:
        st.error("Selftest failed — see details below.")
    st.json(result)
