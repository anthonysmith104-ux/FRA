"""migrate_to_firestore.py — one-time copy of the GitHub JSON store into Firestore.

Paste-the-JSON version. The Firebase service-account credentials are pasted
directly on the page (no Secrets-box TOML, no \\n escaping to get wrong). The
GitHub side reads from [github] secrets if present, otherwise you enter the
token and repo on the page.

Run it ONCE, verify the counts, then swap data_store.py to the Firestore
version. It writes in MERGE mode (never deletes) and never touches your GitHub
repo, so it's safe to run. Delete this temporary app when you're done — the
pasted key lives only in this app's memory while it runs.
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
DEFAULT_DATA_REPO = "anthonysmith104-ux/shared-data"


# ── Safe secrets access (never crashes if secrets are absent/partial) ───────
def _secret_github():
    try:
        gh = st.secrets["github"]
        token = gh.get("token") if hasattr(gh, "get") else gh["token"]
        repo = gh.get("data_repo") if hasattr(gh, "get") else gh["data_repo"]
        return token or "", repo or ""
    except Exception:
        return "", ""


# ── GitHub side ─────────────────────────────────────────────────────────────
def _gh_headers(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def gh_list_json_files(token, repo):
    r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/",
                     headers=_gh_headers(token), timeout=15)
    r.raise_for_status()
    return [item["name"] for item in r.json()
            if item.get("type") == "file" and item["name"].endswith(".json")]


def gh_read_json(token, repo, name):
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
def get_db(sa_dict):
    """Initialize (once) and return a Firestore client from a service-account
    dict pasted on the page."""
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(sa_dict))
    return firestore.client()


def _doc_id(key):
    s = (str(key) or "").replace("/", "_").strip()
    if s in ("", ".", ".."):
        s = "_" + s
    return s[:1500]


def _chunked(items, n):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fs_write_merge(db, name, value):
    """Write a file into Firestore in MERGE mode — sets documents, never
    deletes. Returns the number of records written."""
    if name in _SHARDED:
        if not isinstance(value, dict):
            value = {}
        n = 0
        coll = name[:-5]
        for chunk in _chunked(list(value.items()), _BATCH_LIMIT):
            batch = db.batch()
            for k, v in chunk:
                batch.set(db.collection(coll).document(_doc_id(k)),
                          {"_key": k, "_value": v})
                n += 1
            batch.commit()
        return n
    db.collection(_FILES_COLLECTION).document(name).set({"_value": value})
    return 1


def fs_count(db, name):
    if name in _SHARDED:
        return sum(1 for _ in db.collection(name[:-5]).stream())
    snap = db.collection(_FILES_COLLECTION).document(name).get()
    return 1 if snap.exists else 0


def _record_count(value):
    if isinstance(value, (dict, list)):
        return len(value)
    return 1


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Migrate to Firestore", page_icon="📦")
st.title("GitHub → Firestore migration")
st.caption("Copies every .json file from your GitHub data repo into Firestore. "
           "Merge mode — nothing is deleted, GitHub is left untouched as a backup.")

# --- GitHub config: secrets if present, otherwise on-page inputs ---
_gh_token, _gh_repo = _secret_github()
st.subheader("1 · GitHub source")
if _gh_token and _gh_repo:
    st.success(f"Using GitHub secret · repo `{_gh_repo}`")
else:
    st.info("No [github] secret found — enter it here.")
    _gh_repo = st.text_input("Data repo (owner/name)",
                             value=DEFAULT_DATA_REPO)
    _gh_token = st.text_input("GitHub token", type="password",
                              help="A token with read access to that repo.")

# --- Firebase: paste the service-account JSON ---
st.subheader("2 · Firebase service account")
st.caption("Open the service-account JSON file you downloaded, copy ALL of it, "
           "and paste it below. (This is your admin key — it stays in this "
           "temporary app only. Delete the app when you're done.)")
_sa_text = st.text_area("Service-account JSON", height=180,
                        placeholder='{ "type": "service_account", '
                                    '"project_id": "risk-checkup", ... }')

_sa_dict = None
_sa_ok = False
if _sa_text.strip():
    try:
        _sa_dict = json.loads(_sa_text)
        if _sa_dict.get("type") == "service_account" and _sa_dict.get("private_key"):
            _sa_ok = True
            st.success(f"Valid service account · project "
                       f"`{_sa_dict.get('project_id', '?')}`")
        else:
            st.error("That JSON parsed but doesn't look like a service account "
                     "(missing type/private_key).")
    except Exception as e:
        st.error(f"Couldn't parse that as JSON: {e}")

_ready = bool(_gh_token and _gh_repo and _sa_ok)

st.subheader("3 · Run")
if not _ready:
    st.caption("Fill in the GitHub source and paste a valid service-account "
               "JSON above to enable the buttons.")

colA, colB = st.columns(2)

with colA:
    if st.button("Dry run (preview, no writes)",
                 use_container_width=True, disabled=not _ready):
        try:
            files = gh_list_json_files(_gh_token, _gh_repo)
        except Exception as e:
            st.error(f"Couldn't list the GitHub repo: {e}")
            st.stop()
        rows = []
        for name in files:
            try:
                val = gh_read_json(_gh_token, _gh_repo, name)
                rows.append({"file": name,
                             "records": _record_count(val) if val is not None else 0,
                             "sharded": name in _SHARDED})
            except Exception as e:
                rows.append({"file": name, "records": f"read error: {e}",
                             "sharded": name in _SHARDED})
        st.write("Files found in the data repo:")
        st.table(rows)

with colB:
    if st.button("▶ Run migration", type="primary",
                 use_container_width=True, disabled=not _ready):
        try:
            db = get_db(_sa_dict)
        except Exception as e:
            st.error(f"Couldn't connect to Firestore: {e}")
            st.stop()
        try:
            files = gh_list_json_files(_gh_token, _gh_repo)
        except Exception as e:
            st.error(f"Couldn't list the GitHub repo: {e}")
            st.stop()
        results = []
        prog = st.progress(0.0)
        for i, name in enumerate(files, 1):
            try:
                val = gh_read_json(_gh_token, _gh_repo, name)
                if val is None:
                    results.append({"file": name, "status": "skipped (empty)",
                                    "written": 0, "verified": fs_count(db, name)})
                else:
                    written = fs_write_merge(db, name, val)
                    results.append({"file": name, "status": "ok",
                                    "written": written,
                                    "verified": fs_count(db, name)})
            except Exception as e:
                results.append({"file": name, "status": f"ERROR: {e}",
                                "written": 0, "verified": 0})
            prog.progress(i / max(len(files), 1))
        st.success("Migration complete. Check that written and verified match.")
        st.table(results)
        st.caption("Next: confirm the counts look right (spot-check the Firestore "
                   "console too), then swap data_store.py to the Firestore version "
                   "and redeploy your apps. Delete this temporary app when done.")
