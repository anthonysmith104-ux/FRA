"""
firebase_auth — drop-in Firebase login for a Streamlit app.

Provides Google, Facebook, and email/password sign-in via a small custom
component that runs the Firebase Web SDK in the browser, returns a Firebase
ID token to Python, and verifies that token server-side with firebase-admin.

Typical use in client_portal.py:

    import firebase_auth

    user = firebase_auth.login_gate()
    if not user:
        st.stop()

    # ...authenticated app below...
    # user == {"uid", "email", "name", "picture", "email_verified", "provider"}

Secrets required (see SETUP.md):
    [firebase_web]              # public web config, safe in the browser
    apiKey = "..."
    authDomain = "your-project.firebaseapp.com"
    projectId = "your-project"
    appId = "..."

    [firebase_service_account]  # private admin credentials, server-only
    type = "service_account"
    project_id = "..."
    private_key = "..."
    client_email = "..."
    ...
"""

import os

import streamlit as st
import streamlit.components.v1 as components

import firebase_admin
from firebase_admin import auth as admin_auth
from firebase_admin import credentials

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component = components.declare_component("firebase_login", path=_COMPONENT_DIR)

SESSION_KEY = "auth_user"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _web_config():
    """Public Firebase web config. These values are not secret — Firebase web
    apps expose them by design — but we keep them in secrets to avoid editing
    the HTML and to keep dev/prod values separate."""
    cfg = st.secrets["firebase_web"]
    return {
        "apiKey": cfg["apiKey"],
        "authDomain": cfg["authDomain"],
        "projectId": cfg["projectId"],
        "appId": cfg["appId"],
        "storageBucket": cfg.get("storageBucket", ""),
        "messagingSenderId": cfg.get("messagingSenderId", ""),
    }


def _init_admin():
    """Initialize the Firebase Admin SDK once per process."""
    if firebase_admin._apps:
        return
    sa = dict(st.secrets["firebase_service_account"])
    # Secrets escape newlines in the private key; restore them.
    if "private_key" in sa:
        sa["private_key"] = sa["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(sa)
    firebase_admin.initialize_app(cred)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_login(key="firebase_login"):
    """Render the login UI. Returns a Firebase ID token (str) once the user
    signs in with any method, otherwise None."""
    value = _component(config=_web_config(), mode="auth", key=key, default=None)
    if isinstance(value, str) and value and value != "signed_out":
        return value
    return None


def verify_token(id_token):
    """Verify a Firebase ID token. Returns decoded claims dict, or None."""
    _init_admin()
    try:
        return admin_auth.verify_id_token(id_token)
    except Exception:
        return None


def current_user():
    """Return the signed-in user dict for this session, or None."""
    return st.session_state.get(SESSION_KEY)


def login_gate():
    """Gate the app behind Firebase auth.

    Returns the user dict when authenticated, or None when not yet signed in
    (caller should st.stop() in that case)."""
    user = st.session_state.get(SESSION_KEY)
    if user:
        return user

    id_token = render_login()
    if not id_token:
        return None

    claims = verify_token(id_token)
    if not claims:
        st.error("We couldn't verify that sign-in. Please try again.")
        return None

    firebase_claims = claims.get("firebase", {}) or {}
    user = {
        "uid": claims.get("uid", ""),
        "email": (claims.get("email") or "").strip().lower(),
        "name": claims.get("name") or "",
        "picture": claims.get("picture") or "",
        "email_verified": bool(claims.get("email_verified", False)),
        "provider": firebase_claims.get("sign_in_provider", ""),
    }
    st.session_state[SESSION_KEY] = user
    st.rerun()


def logout(key="firebase_logout"):
    """Sign the user out of both the Streamlit session and the in-browser
    Firebase session. Call this, then st.rerun() in your app."""
    # Render the component once in signout mode so the iframe clears its
    # persisted Firebase session (otherwise a refresh would silently re-login).
    _component(config=_web_config(), mode="signout", key=key, default=None)
    st.session_state.pop(SESSION_KEY, None)
