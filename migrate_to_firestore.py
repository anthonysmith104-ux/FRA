"""migrate_to_firestore.py — one-time copy of the GitHub JSON store into Firestore.

Run this ONCE, before swapping data_store.py to the Firestore version. It reads
every .json file from your GitHub data repo and writes it into Firestore using
the exact same storage scheme data_store.py expects, so the moment you cut over,
the apps find their data already there.

It is deliberately self-contained — it does NOT import data_store — so it works
no matter which data_store.py is currently deployed. It writes in MERGE mode
(it never deletes), so running it can't clobber anything; your GitHub repo is
also left completely untouched as a backup.

How to run (easiest path):
  1. Commit this file to your repo.
  2. In Streamlit Cloud, create a TEMPORARY new app from the same repo with this
     file as the main file. It reuses the same secrets (it needs both the
     [github] and [firebase_service_account] blocks, which are already there).
  3. Open it, click "Dry run" to preview, then "Run migration."
  4. Verify the counts, then delete the temporary app.
  5. Swap data_store.py to the Firestore version and redeploy the real apps.

Both secret blocks must be present:
    [github] token = "..."   data_repo = "owner/repo"
    [firebase_service_account] ...
"""
import base64
import json

import requests
import streamlit as st

import firebase_admin
from firebase_admin import credentials, firestore

# ── MUST MATCH data_store.py ────────────────────────────────────────────────
_SHARDED = {"ra_users.json", "risk_profiles.json", "client_holdings.json"}
_FILES_COLLECTION = "_files"
_BATCH_LIMIT = 450

GITHUB_API = "https://api.github.com"


# ── GitHub side ─────────────────────────────────────────────────────────────
def _gh_cfg():
    gh = st.secrets["github"]
    token = gh.get("token") if hasattr(gh, "get") else gh["token"]
    repo = gh.get("data_repo") if hasattr(gh, "get") else gh["data_repo"]
    return token, repo


def _gh_headers(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def gh_list_json_files():
    token, repo = _gh_cfg()
    r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/",
                     headers=_gh_headers(token), timeout=15)
    r.raise_for_status()
    return [item["name"] for item in r.json()
            if item.get("type") == "file" and item["name"].endswith(".json")]


def gh_read_json(name):
    token, repo = _gh_cfg()
    r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{name}",
                     headers=_gh_headers(token), timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    raw = base64.b64decode(r.json()["content"]).decode("utf-8")
    if not raw.strip():
        return None
    return json.loads(raw)


# ── Firestore side (scheme identical to data_store.py) ──────────────────────
_fs = None


def _db():
    global _fs
    if _fs is not None:
        return _fs
    if not firebase_admin._apps:
        sa = dict(st.secrets["firebase_service_account"])
        if "private_key" in sa:
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")
        firebase_admin.initialize_app(credentials.Certificate(sa))
    _fs = firestore.client()
    return _fs


def _doc_id(key):
    s = (str(key) or "").replace("/", "_").strip()
    if s in ("", ".", ".."):
        s = "_" + s
    return s[:1500]


def _chunked(items, n):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fs_write_merge(name, value):
    """Write a file into Firestore in MERGE mode — sets documents, never
    deletes. Returns the number of records written."""
    db = _db()
    if name in _SHARDED:
        if not isinstance(value, dict):
            value = {}
        n = 0
        for chunk in _chunked(list(value.items()), _BATCH_LIMIT):
            batch = db.batch()
            coll = name[:-5]
            for k, v in chunk:
                batch.set(db.collection(coll).document(_doc_id(k)),
                          {"_key": k, "_value": v})
                n += 1
            batch.commit()
        return n
    db.collection(_FILES_COLLECTION).document(name).set({"_value": value})
    return 1


def fs_count(name):
    """Read back a count from Firestore for verification."""
    db = _db()
    if name in _SHARDED:
        coll = name[:-5]
        return sum(1 for _ in db.collection(coll).stream())
    snap = db.collection(_FILES_COLLECTION).document(name).get()
    return 1 if snap.exists else 0


def _record_count(value):
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 1


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Migrate to Firestore", page_icon="📦")
st.title("GitHub → Firestore migration")

# Preflight: confirm both secrets are present.
_have_gh = False
_have_fb = False
try:
    _gh_cfg()
    _have_gh = True
except Exception:
    pass
try:
    _have_fb = bool(st.secrets["firebase_service_account"])
except Exception:
    pass

c1, c2 = st.columns(2)
c1.metric("GitHub secret", "✓ present" if _have_gh else "✗ missing")
c2.metric("Firebase secret", "✓ present" if _have_fb else "✗ missing")

if not (_have_gh and _have_fb):
    st.error("Both [github] and [firebase_service_account] secrets must be "
             "configured in this app's Secrets before migrating.")
    st.stop()

st.caption("Reads every .json file from your GitHub data repo and copies it "
           "into Firestore. Merge mode — nothing is deleted, and your GitHub "
           "repo is left untouched as a backup.")

colA, colB = st.columns(2)

with colA:
    if st.button("Dry run (preview, no writes)", use_container_width=True):
        try:
            files = gh_list_json_files()
        except Exception as e:
            st.error(f"Couldn't list the GitHub repo: {e}")
            st.stop()
        rows = []
        for name in files:
            try:
                val = gh_read_json(name)
                rows.append({"file": name,
                             "records": _record_count(val) if val is not None else 0,
                             "sharded": name in _SHARDED})
            except Exception as e:
                rows.append({"file": name, "records": f"read error: {e}",
                             "sharded": name in _SHARDED})
        st.write("Files found in the data repo:")
        st.table(rows)

with colB:
    if st.button("▶ Run migration", type="primary", use_container_width=True):
        try:
            files = gh_list_json_files()
        except Exception as e:
            st.error(f"Couldn't list the GitHub repo: {e}")
            st.stop()
        results = []
        prog = st.progress(0.0)
        for i, name in enumerate(files, 1):
            try:
                val = gh_read_json(name)
                if val is None:
                    results.append({"file": name, "status": "skipped (empty)",
                                    "written": 0, "verified": fs_count(name)})
                else:
                    written = fs_write_merge(name, val)
                    results.append({"file": name, "status": "ok",
                                    "written": written, "verified": fs_count(name)})
            except Exception as e:
                results.append({"file": name, "status": f"ERROR: {e}",
                                "written": 0, "verified": 0})
            prog.progress(i / max(len(files), 1))
        st.success("Migration complete. Check the verified counts match.")
        st.table(results)
        st.caption("Next: confirm these counts look right (and spot-check the "
                   "Firestore console), then swap data_store.py to the Firestore "
                   "version and redeploy your apps. Delete this temporary app "
                   "when you're done.")
