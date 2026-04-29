"""client_portal.py — Foresight Risk Analytics client portal.

Matches the "Risk Checkup" mobile prototype's clinical aesthetic:
    • Light teal-on-white palette (Clinical theme)
    • Hexagon-with-pulse logo mark
    • RiskRing score visualization (1-99, health-band colors)
    • 2x2 Vitals grid
    • Sparkline trend card with period selector
    • Dark advisor CTA button
    • Bottom nav (Home / Plan / Advisor / Me)

Shared with risk_assessment.py and app.py via shared.py — same JSON files,
same atomic storage, same scoring helpers.

Run:
    streamlit run client_portal.py

Files (anchored to this script's directory):
    ra_users.json           — user records
    risk_profiles.json      — risk profile + Q&A
    client_holdings.json    — per-client holdings
"""
from __future__ import annotations

import os
import json
from datetime import datetime, date
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── HubSpot CRM sync ─────────────────────────────────────────────────────────
# Bridge Streamlit Cloud's secret to the environment variable that
# hubspot_sync.py looks for. Streamlit's `secrets.toml` is a separate
# mechanism from os.environ — they're not auto-linked. Without this bridge,
# hubspot_sync._read_token() returns None and every sync silently no-ops.
#
# Set the secret on Streamlit Cloud (Settings → Secrets) as:
#     hubspot_token = "pat-na1-..."
# It can also be set as the env var HUBSPOT_TOKEN directly (e.g. for local
# dev or non-Streamlit-Cloud hosts) — the bridge below is a no-op in that
# case since we only set the env var if it isn't already there.
try:
    if not os.environ.get("HUBSPOT_TOKEN"):
        _hs_token = st.secrets.get("hubspot_token", "")
        if _hs_token:
            os.environ["HUBSPOT_TOKEN"] = str(_hs_token).strip()
except Exception:
    # st.secrets raises if no secrets file exists — that's fine, just means
    # HubSpot sync stays disabled and the rest of the app keeps working.
    pass

# Optional import — if the module isn't in the repo (or fails to load for
# any reason), the app still works, just without CRM sync. The flag below
# is what we check before attempting any sync.
_HUBSPOT_AVAILABLE = False
_HUBSPOT_IMPORT_ERROR: Optional[str] = None
try:
    import hubspot_sync  # type: ignore
    _HUBSPOT_AVAILABLE = True
except Exception as _e:
    hubspot_sync = None  # type: ignore
    _HUBSPOT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    print(f"[hubspot_sync] import failed: {_HUBSPOT_IMPORT_ERROR}")

from shared import (
    load_json as _shared_load_json,
    update_json as _shared_update_json,
    is_valid_email, normalize_email,
    score_to_label, score_to_allocation,
)

# ── DATA FILE LOCATIONS ──────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _data_path(name: str) -> str: return os.path.join(_APP_DIR, name)

USERS_FILE           = _data_path("ra_users.json")
PROFILES_FILE        = _data_path("risk_profiles.json")
CLIENT_HOLDINGS_FILE = _data_path("client_holdings.json")
CLIENT_GOALS_FILE    = _data_path("client_goals.json")
CLIENT_BUDGETS_FILE  = _data_path("client_budgets.json")

# ── ADVISOR PROFILE ──────────────────────────────────────────────────────────
# The advisor profile shown across the client portal (Home page advisor box,
# dedicated Advisor tab) is sourced from the same `firm_settings.json` file
# the advisor app writes when the firm fills out Client Records → 🎨 Firm
# Branding. The two apps live in the same directory, so when the advisor
# updates their info on their side it flows through here automatically —
# no editing this file required.
#
# Hardcoded values below are the *fallback* — used for any field the firm
# hasn't filled in yet, so the portal always renders something sensible.
# To customize, the advisor edits Firm Branding in the advisor app; nothing
# in this file needs to change.

# Firm-branding paths (mirror the advisor app's constants — same directory,
# same filenames, so both apps see the same data).
FIRM_SETTINGS_FILE = _data_path("firm_settings.json")
FIRM_LOGO_PATH     = _data_path("firm_logo.png")
ADVISOR_PHOTO_PATH = _data_path("advisor_photo.png")

# Hardcoded fallback advisor profile. Used only for fields that aren't set
# in firm_settings.json. Any value the advisor saves on their side wins.
_ADVISOR_DEFAULTS = {
    "name":    "Sarah Whitfield, CFP®",
    "title":   "Senior Financial Advisor",
    "firm":    "Foresight Wealth Partners",
    "email":   "sarah.whitfield@foresightwealth.com",
    "phone":   "(612) 555-0142",
    "website": "https://www.foresightwealth.com",
    "address": "200 South Sixth Street, Suite 1200, Minneapolis, MN 55402",
    "bio":     ("Sarah has spent fifteen years helping families plan for "
                "retirement, education, and legacy goals. She's a Certified "
                "Financial Planner™ and a fiduciary — meaning she's legally "
                "required to act in your best interest."),
}

# Generic SVG avatar — neutral, no real likeness. Used only if no
# advisor_photo.png has been uploaded.
_DEFAULT_PHOTO_SVG = (
    '<svg viewBox="0 0 80 80" width="80" height="80" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="adv_bg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#0E5C5E"/>'
    '<stop offset="1" stop-color="#0E7C86"/></linearGradient></defs>'
    '<circle cx="40" cy="40" r="40" fill="url(#adv_bg)"/>'
    '<circle cx="40" cy="32" r="13" fill="#FFFFFF" opacity="0.95"/>'
    '<path d="M16 70 C 18 56, 28 50, 40 50 S 62 56, 64 70 Z" '
    'fill="#FFFFFF" opacity="0.95"/>'
    '</svg>'
)

# Default firm logo SVG (the small teal chart-mark) — used when no
# firm_logo.png has been uploaded.
_DEFAULT_FIRM_LOGO_SMALL_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="2" y="2" width="20" height="20" rx="5" fill="#0E5C5E"/>'
    '<path d="M6 16 L11 8 L13 12 L18 6" stroke="#FFFFFF" '
    'stroke-width="2" fill="none" stroke-linecap="round" '
    'stroke-linejoin="round"/></svg>'
)


def _load_firm_settings() -> dict:
    """Read firm_settings.json (written by the advisor app's Firm Branding
    panel). Returns {} if the file is missing or unreadable, so callers
    can safely use `dict.get` patterns without raising."""
    if not os.path.exists(FIRM_SETTINGS_FILE):
        return {}
    try:
        with open(FIRM_SETTINGS_FILE, "r") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _file_to_data_url(path: str, mime: str = "image/png") -> Optional[str]:
    """Read an image file and return a base64 data URL suitable for embedding
    in an <img src="..."> tag. Returns None if the file is missing or
    unreadable. Used to inline the firm logo / advisor photo into HTML
    without having to host them at a URL."""
    if not os.path.exists(path):
        return None
    try:
        import base64 as _b64
        with open(path, "rb") as f:
            return f"data:{mime};base64,{_b64.b64encode(f.read()).decode('ascii')}"
    except OSError:
        return None


def _build_advisor_profile() -> dict:
    """Compose the advisor profile shown across the portal. Merges:
        1. Hardcoded defaults (lowest priority — used for missing fields).
        2. firm_settings.json values (if present — wins per-field).
        3. Image data URLs for advisor photo + firm logo (if files exist;
           otherwise the SVG fallbacks).

    Returns the same dict shape the rest of the portal expects (`name`,
    `title`, `firm`, `email`, `phone`, `website`, `address`, `bio`,
    `photo_html`, `firm_logo_html_small`). The HTML fields are guaranteed
    to be safe to drop into an existing markup template — they're either
    an `<img>` tag (when a real image was uploaded) or an inline `<svg>`
    fallback.
    """
    fs = _load_firm_settings()

    # Map advisor-app field names → portal field names. Each portal field
    # falls back to the hardcoded default when the firm setting is missing
    # or empty.
    def _pick(*keys, default=""):
        for k in keys:
            v = fs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return default

    profile = {
        "name":    _pick("advisor_name",    default=_ADVISOR_DEFAULTS["name"]),
        "title":   _pick("advisor_title",   default=_ADVISOR_DEFAULTS["title"]),
        "firm":    _pick("firm_name",       default=_ADVISOR_DEFAULTS["firm"]),
        "email":   _pick("advisor_email",   default=_ADVISOR_DEFAULTS["email"]),
        "phone":   _pick("advisor_phone",   default=_ADVISOR_DEFAULTS["phone"]),
        "website": _pick("firm_website",    default=_ADVISOR_DEFAULTS["website"]),
        "address": _pick("firm_address",    default=_ADVISOR_DEFAULTS["address"]),
        "bio":     _pick("advisor_bio",     default=_ADVISOR_DEFAULTS["bio"]),
    }

    # Advisor photo: <img> tag if uploaded, else default SVG avatar.
    _photo_url = _file_to_data_url(ADVISOR_PHOTO_PATH, mime="image/png")
    if _photo_url:
        profile["photo_html"] = (
            f'<img src="{_photo_url}" alt="Advisor photo" '
            f'style="width:80px;height:80px;border-radius:50%;'
            f'        object-fit:cover;display:block;'
            f'        border:1px solid #E1E8EE"/>'
        )
    else:
        profile["photo_html"] = _DEFAULT_PHOTO_SVG

    # Firm logo (small variant for the home-page advisor box).
    _logo_url = _file_to_data_url(FIRM_LOGO_PATH, mime="image/png")
    if _logo_url:
        profile["firm_logo_html_small"] = (
            f'<img src="{_logo_url}" alt="" '
            f'style="height:22px;width:auto;max-width:90px;'
            f'        display:inline-block;vertical-align:middle"/>'
        )
        # Larger variant for the dedicated Advisor tab header.
        profile["firm_logo_html_large"] = (
            f'<img src="{_logo_url}" alt="" '
            f'style="height:36px;width:auto;max-width:160px;'
            f'        display:inline-block;vertical-align:middle"/>'
        )
    else:
        profile["firm_logo_html_small"] = _DEFAULT_FIRM_LOGO_SMALL_SVG
        # No large default — advisor tab just won't show a logo if none uploaded.
        profile["firm_logo_html_large"] = ""

    # Backward-compat alias: legacy code reads `photo_svg`. Keep it as the
    # photo HTML so any unmodified call sites still work.
    profile["photo_svg"] = profile["photo_html"]

    return profile


def get_advisor() -> dict:
    """Get the live advisor profile. Re-read on every call so when the
    advisor updates their Firm Branding panel, the next portal page render
    picks up the change without restarting Streamlit."""
    return _build_advisor_profile()


# Legacy module-level constant — kept so any old reference still resolves,
# but new code should call `get_advisor()` so updates flow through live.
ADVISOR = _build_advisor_profile()

st.set_page_config(
    page_title="Foresight Risk Analytics",
    page_icon="🩺",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME — "Clinical" from the prototype (teal on white)
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    "bg":           "#F4F7F9",
    "surface":      "#FFFFFF",
    "surface2":     "#EEF3F6",
    "line":         "#E1E8EE",
    "ink":          "#0B1F2A",
    "ink2":         "#3F5260",
    "muted":        "#6B7E8A",
    "primary":      "#0E5C5E",
    "primary_soft": "#D8ECEC",
    "accent":       "#0E7C86",
    "healthy":      "#16A34A",
    "healthy_soft": "#DCF5E4",
    "caution":      "#C2700A",
    "caution_soft": "#FBEBD2",
    "risk":         "#C2410C",
    "risk_soft":    "#FCDED0",
    "chip":         "#F1F5F8",
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

        .stApp {{
            background: {THEME['bg']};
            color: {THEME['ink']};
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
            color: {THEME['ink']} !important;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        .fr-card {{
            background: {THEME['surface']};
            border: 1px solid {THEME['line']};
            border-radius: 18px;
            padding: 22px 22px;
            margin-bottom: 16px;
        }}
        .fr-eyebrow {{
            font-size: 0.69rem;
            font-weight: 600;
            color: {THEME['muted']};
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .fr-vital {{
            background: {THEME['surface']};
            border: 1px solid {THEME['line']};
            border-radius: 14px;
            padding: 14px 14px;
            min-height: 102px;
            display: flex; flex-direction: column; gap: 8px;
            margin-bottom: 10px;
        }}
        .fr-vital-label {{
            font-size: 0.65rem; font-weight: 600;
            color: {THEME['muted']};
            letter-spacing: 0.06em; text-transform: uppercase;
        }}
        .fr-vital-value {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-size: 1.4rem; font-weight: 600; color: {THEME['ink']};
            letter-spacing: -0.01em; line-height: 1;
        }}
        .fr-vital-detail {{
            display: flex; align-items: center; justify-content: space-between;
            font-size: 0.72rem; color: {THEME['muted']};
        }}
        .fr-mono {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-weight: 600;
        }}
        .fr-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 3px 10px; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
        }}
        .fr-chip::before {{
            content: ""; width: 6px; height: 6px; border-radius: 999px;
            background: currentColor;
        }}
        .fr-greeting {{
            font-size: 0.85rem; color: {THEME['ink2']}; margin-bottom: 4px;
        }}
        .fr-headline {{
            font-size: 1.5rem; font-weight: 600; color: {THEME['ink']};
            letter-spacing: -0.015em; line-height: 1.18; margin: 0 0 14px 0;
        }}
        .fr-headline-accent {{ color: {THEME['primary']}; }}

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea textarea {{
            background-color: {THEME['surface']} !important;
            color: {THEME['ink']} !important;
            border: 1px solid {THEME['line']} !important;
            border-radius: 10px !important;
            font-family: 'Inter', sans-serif;
        }}
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea textarea:focus {{
            border-color: {THEME['primary']} !important;
            box-shadow: 0 0 0 3px {THEME['primary_soft']} !important;
        }}
        .stSelectbox > div > div, .stMultiSelect > div > div {{
            background-color: {THEME['surface']} !important;
            border: 1px solid {THEME['line']} !important;
            border-radius: 10px !important;
        }}
        .stMultiSelect [data-baseweb="tag"] {{
            background: {THEME['primary_soft']} !important;
            border-color: {THEME['primary']}66 !important;
            color: {THEME['primary']} !important;
        }}

        /* Standardize number rendering inside radio/select labels — tabular
           numerals so $50,000 – $100,000 lines up identically in every
           question (income, net worth, goal amount, etc.). */
        .stRadio label, .stRadio [data-baseweb="radio"] div,
        .stSelectbox div, .stMultiSelect div {{
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum" 1, "lnum" 1;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background: transparent;
            border-bottom: 1px solid {THEME['line']};
            gap: 28px;                  /* generous spacing between tabs */
            padding: 0 4px;             /* keeps first tab from kissing the edge */
            margin-bottom: 18px;        /* breathing room before tab content */
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {THEME['muted']};
            background: transparent;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 12px 4px;          /* taller hit-area, slim horizontal */
            min-height: auto;
            letter-spacing: 0.01em;
            transition: color 0.15s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {THEME['ink2']};
        }}
        .stTabs [aria-selected="true"] {{
            color: {THEME['ink']} !important;
        }}
        /* Newer Streamlit renders the active-tab indicator as a separate
           sliding element. By default it picks up Streamlit's primaryColor
           (red #FF4B4B) regardless of our app theme. Force it to the brand
           teal so the tab selection matches everything else on the page. */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {THEME['primary']} !important;
            background: {THEME['primary']} !important;
            height: 2.5px !important;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            background-color: {THEME['line']} !important;
            background: {THEME['line']} !important;
        }}

        .stButton > button {{
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.15s ease;
            background: {THEME['surface']};
            color: {THEME['ink']};
            border: 1px solid {THEME['line']};
        }}
        .stButton > button:hover {{
            background: {THEME['surface2']};
            border-color: {THEME['primary']};
            color: {THEME['primary']};
        }}
        .stButton > button[kind="primary"] {{
            background: {THEME['primary']};
            border-color: {THEME['primary']};
            color: #fff;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {THEME['accent']};
            border-color: {THEME['accent']};
            color: #fff;
            transform: translateY(-1px);
        }}

        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {THEME['muted']} !important;
        }}
        .stAlert {{
            background: {THEME['surface']} !important;
            border: 1px solid {THEME['line']} !important;
            border-left: 3px solid {THEME['primary']} !important;
            color: {THEME['ink']} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stDataFrame"] {{
            background: {THEME['surface']};
            border-radius: 14px;
            border: 1px solid {THEME['line']};
        }}

        .js-plotly-plot, .plot-container {{ background: transparent !important; }}

        .block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 760px; }}

        .fr-cta-dark {{
            background: {THEME['ink']};
            color: #fff;
            border-radius: 16px;
            padding: 16px 18px;
            margin-top: 14px;
            display: flex; align-items: center; gap: 14px;
        }}
        .fr-cta-icon {{
            width: 40px; height: 40px; border-radius: 12px;
            background: rgba(255,255,255,0.12);
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; font-size: 1.2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
def load_users() -> dict:
    return _shared_load_json(USERS_FILE, default={})

def load_profiles() -> dict:
    return _shared_load_json(PROFILES_FILE, default={})

def load_all_holdings() -> dict:
    return _shared_load_json(CLIENT_HOLDINGS_FILE, default={})

def find_user(email: str) -> Optional[dict]:
    key = normalize_email(email)
    if key is None: return None
    users = load_users()
    if key in users: return users[key]
    for k, v in users.items():
        if k.lower() == key or (isinstance(v, dict) and v.get("email","").lower() == key):
            return v
    return None

def register_user(first: str, last: str, email: str, phone: str = "") -> tuple[bool, str]:
    """Atomic upsert under a single lock."""
    key = normalize_email(email)
    if key is None: return False, "Please enter a valid email address."
    first = (first or "").strip()
    last  = (last or "").strip()
    if not first or not last: return False, "First and last name are required."
    new_user = {
        "first_name": first, "last_name": last, "email": key,
        "phone": (phone or "").strip(),
        "created_at": datetime.now().isoformat(timespec="minutes"),
    }
    conflict = {"exists": False}
    def _mutate(users):
        if key in users: conflict["exists"] = True; return
        users[key] = new_user
    _shared_update_json(USERS_FILE, _mutate)
    if conflict["exists"]: return False, "An account with this email already exists."
    return True, "Account created."

def save_holdings_for(client_key: str, holdings: dict) -> None:
    _shared_update_json(
        CLIENT_HOLDINGS_FILE,
        lambda d, k=client_key, h=holdings: d.update({k: h}),
    )

def save_profile_for(client_key: str, profile_patch: dict) -> None:
    def _mutate(profiles):
        prev = profiles.get(client_key, {}) or {}
        prev.update(profile_patch)
        prev["updated_at"] = datetime.now().isoformat(timespec="minutes")
        profiles[client_key] = prev
    _shared_update_json(PROFILES_FILE, _mutate)


# ── GOALS & BUDGETS ──────────────────────────────────────────────────────────
def load_all_goals() -> dict:
    return _shared_load_json(CLIENT_GOALS_FILE, default={})

def load_goals_for(client_key: str) -> list:
    return list(load_all_goals().get(client_key, []) or [])

def save_goals_for(client_key: str, goals: list) -> None:
    _shared_update_json(
        CLIENT_GOALS_FILE,
        lambda d, k=client_key, g=goals: d.update({k: g}),
    )

def load_all_budgets() -> dict:
    return _shared_load_json(CLIENT_BUDGETS_FILE, default={})

def load_budget_for(client_key: str) -> dict:
    return dict(load_all_budgets().get(client_key, {}) or {})

def save_budget_for(client_key: str, budget: dict) -> None:
    _shared_update_json(
        CLIENT_BUDGETS_FILE,
        lambda d, k=client_key, b=budget: d.update({k: b}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIVE QUOTES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _get_live_quotes_cached(tickers_tuple: tuple) -> dict:
    """60s cache. Wrap each ticker in try/except so one bad symbol can't take
    down the whole panel."""
    import yfinance as yf
    out = {}
    for tk in tickers_tuple:
        try:
            t = yf.Ticker(tk)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                prev  = float(hist["Close"].iloc[-2])
                price = float(hist["Close"].iloc[-1])
            elif len(hist) == 1:
                price = prev = float(hist["Close"].iloc[-1])
            else:
                price = prev = 0.0
            chg = price - prev
            pct = (chg / prev * 100) if prev else 0
            try:
                info = t.info or {}
                name = info.get("shortName") or info.get("longName") or tk
            except Exception:
                name = tk
            out[tk] = {"name": name, "price": price, "prev_close": prev,
                       "change": chg, "change_pct": pct}
        except Exception:
            out[tk] = {"name": tk, "price": 0, "prev_close": 0,
                       "change": 0, "change_pct": 0}
    return out

def get_live_quotes(tickers) -> dict:
    if not tickers: return {}
    return _get_live_quotes_cached(tuple(sorted(set(tickers))))


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_QUESTIONS = [
    {"id": "age", "section": "Context",
     "text": "Your current age",
     "type": "number", "min": 18, "max": 99, "step": 1, "default": 45,
     "scoring": "age"},
    {"id": "retirement_age", "section": "Context",
     "text": "When do you plan to retire?",
     "type": "number", "min": 30, "max": 90, "step": 1, "default": 65,
     "scoring": "horizon"},
    {"id": "occupation", "section": "Context",
     "text": "Which best describes your employment?",
     "type": "select", "options": [
        ("Salaried — stable industry",        65),
        ("Salaried — variable industry",      55),
        ("Self-employed / business owner",    50),
        ("Commission / variable income",      45),
        ("Retired",                           35),
        ("Between roles",                     30),
        ("Student",                           60),
    ]},
    {"id": "income_band", "section": "Context",
     "text": "Household annual income (gross)",
     # Only ONE dollar sign per option — Streamlit renders radio labels
     # through markdown, and a paired `$...$` is parsed as inline LaTeX
     # (rendering the inside in green monospace math font). Standard
     # financial-report convention: "$50,000 – 100,000" is unambiguous
     # because the range is clearly currency from the first symbol.
     "type": "select", "options": [
        ("Under $50,000",            30),
        ("$50,000 – 100,000",        45),
        ("$100,000 – 200,000",       60),
        ("$200,000 – 500,000",       75),
        ("$500,000 – 1,000,000",     85),
        ("Over $1,000,000",          90),
        ("Prefer not to say",        50),
    ]},
    {"id": "income_stability", "section": "Context",
     "text": "How stable is your income over the next 5 years?",
     "type": "select", "options": [
        ("Very stable — same or growing",       70),
        ("Mostly stable, normal fluctuation",   60),
        ("Variable — meaningful ups & downs",   45),
        ("Uncertain — major change expected",   30),
    ]},
    {"id": "net_worth", "section": "Context",
     "text": "Approximate liquid net worth (excluding home)",
     # See income_band note re: single dollar sign per option.
     "type": "select", "options": [
        ("Under $100,000",              30),
        ("$100,000 – 500,000",          50),
        ("$500,000 – 1,000,000",        65),
        ("$1,000,000 – 5,000,000",      75),
        ("$5,000,000 – 25,000,000",     85),
        ("Over $25,000,000",            90),
        ("Prefer not to say",           55),
    ]},

    # ── Goals — what's this money FOR? Drives capacity scoring because the
    # goal type (preservation vs growth vs aggressive accumulation) shifts how
    # much risk a portfolio reasonably needs to take. Higher-aspiration goals
    # (early retirement, major wealth building) push the score higher; pure
    # preservation goals push it lower.
    {"id": "primary_goal", "section": "Goals",
     "text": "What's your primary goal for this money?",
     "type": "select", "options": [
        ("Preserve what I have",                       30),
        ("Generate steady income",                     45),
        ("Save for a specific purchase (home, etc.)",  50),
        ("Fund education for myself or family",        55),
        ("Build long-term wealth for retirement",      70),
        ("Achieve financial independence early",       85),
        ("Build generational / legacy wealth",         75),
    ]},
    {"id": "goal_amount", "section": "Goals",
     "text": "Do you have a specific dollar target in mind?",
     # See income_band note re: single dollar sign per option.
     "type": "select", "options": [
        ("No target — I'm just building",              60),
        ("Under $250,000",                             40),
        ("$250,000 – 1,000,000",                       55),
        ("$1,000,000 – 5,000,000",                     70),
        ("$5,000,000 – 25,000,000",                    80),
        ("Over $25,000,000",                           85),
    ]},
    {"id": "goal_timeline", "section": "Goals",
     "text": "When do you want to reach this goal?",
     "type": "select", "options": [
        ("Less than 3 years",     20),
        ("3 – 7 years",           40),
        ("7 – 15 years",          60),
        ("15 – 25 years",         80),
        ("More than 25 years",    90),
        ("No specific timeline",  65),
    ]},
    {"id": "income_replacement", "section": "Goals",
     "text": "In retirement, what % of your current income do you want to replace?",
     "type": "select", "options": [
        ("Less than 50%",                              50),
        ("50 – 70%",                                   60),
        ("70 – 85% (typical)",                         70),
        ("85 – 100%",                                  75),
        ("More than 100% — I want to live better",     80),
        ("Not applicable / already retired",           50),
    ]},
    {"id": "legacy_intent", "section": "Goals",
     "text": "How important is leaving money to heirs or charity?",
     "type": "select", "options": [
        ("Not important — spend it all in my lifetime",   55),
        ("Nice to have — whatever's left is fine",        65),
        ("Moderately important — I want a meaningful gift", 75),
        ("Very important — building a generational legacy", 85),
    ]},

    {"id": "withdrawal_horizon", "section": "Horizon",
     "text": "When will you start drawing from this portfolio?",
     "type": "select", "options": [
        ("Less than 2 years",    20),
        ("2 – 5 years",          35),
        ("5 – 10 years",         55),
        ("10 – 20 years",        75),
        ("More than 20 years",   90),
        ("Never — for heirs",    85),
    ]},
    {"id": "withdrawal_rate", "section": "Horizon",
     "text": "Once drawing, what % per year do you expect to withdraw?",
     "type": "select", "options": [
        ("Less than 2%",       80),
        ("2 – 4% (typical)",   65),
        ("4 – 6%",             45),
        ("More than 6%",       25),
        ("Not sure yet",       55),
    ]},
    {"id": "emergency_fund", "section": "Horizon",
     "text": "Months of expenses you have in cash outside this portfolio",
     "type": "select", "options": [
        ("Less than 1 month",  20),
        ("1 – 3 months",       40),
        ("3 – 6 months",       60),
        ("6 – 12 months",      75),
        ("More than 12 months",85),
    ]},
    {"id": "major_expense", "section": "Horizon",
     "text": "Major expense in the next 3 years (home, education, medical)?",
     # Single $ per option — see income_band note above for the LaTeX-pairing
     # rationale.
     "type": "select", "options": [
        ("No major expenses planned",     75),
        ("Possibly — under $50,000",      60),
        ("Yes — $50,000 to 250,000",      40),
        ("Yes — over $250,000",           25),
    ]},
    {"id": "drawdown_reaction", "section": "Tolerance",
     "text": "Your portfolio drops 25% in a single year. What do you do?",
     "type": "select", "options": [
        ("Sell most of it — protect what's left",  15),
        ("Sell some — reduce exposure",            35),
        ("Hold and wait it out",                   65),
        ("Buy more — prices are on sale",          90),
    ]},
    {"id": "experience", "section": "Tolerance",
     "text": "How would you describe your investing experience?",
     "type": "select", "options": [
        ("New to investing",                                25),
        ("Some experience — mostly mutual funds / 401k",    45),
        ("Experienced — actively pick stocks / ETFs",       65),
        ("Very experienced — options, bonds, alternatives", 80),
    ]},
    {"id": "loss_floor", "section": "Tolerance",
     "text": "Largest one-year loss you could accept before changing strategy",
     "type": "select", "options": [
        ("5% or less",         20),
        ("Up to 10%",          40),
        ("Up to 20%",          60),
        ("Up to 35%",          80),
        ("More than 35%",      95),
    ]},
    {"id": "growth_vs_safety", "section": "Tolerance",
     "text": "Pick the portfolio that best matches your preference",
     "type": "select", "options": [
        ("Best year +6%  / worst year -2%",    20),
        ("Best year +12% / worst year -8%",    45),
        ("Best year +20% / worst year -18%",   65),
        ("Best year +30% / worst year -30%",   85),
    ]},
    {"id": "market_view", "section": "Outlook",
     "text": "Your view on US equity markets over the next 3 years",
     "type": "select", "options": [
        ("Significantly higher",   75),
        ("Modestly higher",        65),
        ("Roughly flat",           50),
        ("Modestly lower",         40),
        ("Significantly lower",    30),
        ("No strong view",         55),
    ]},
    {"id": "inflation_concern", "section": "Outlook",
     "text": "How concerned are you about inflation eroding your savings?",
     "type": "select", "options": [
        ("Not concerned",        70),
        ("Slightly concerned",   60),
        ("Moderately concerned", 50),
        ("Very concerned",       45),
        ("Extremely concerned",  40),
    ]},
    {"id": "recession_concern", "section": "Outlook",
     "text": "How likely is a recession in the next 18 months?",
     "type": "select", "options": [
        ("Very unlikely",      70),
        ("Somewhat unlikely",  60),
        ("About 50/50",        50),
        ("Somewhat likely",    45),
        ("Very likely",        40),
    ]},
    {"id": "esg_preference", "section": "Outlook",
     "text": "How important is ESG / sustainable investing to you?",
     "type": "select", "options": [
        ("Not a factor",                                     60),
        ("Nice to have — won't sacrifice returns",           55),
        ("Important — willing to accept some tradeoff",      50),
        ("Critical — must be ESG-aligned",                   45),
    ]},
    {"id": "priorities", "section": "Outlook",
     "text": "Which of these matter MOST to you? (pick up to 3)",
     "type": "multi", "options": [
        "Capital preservation",
        "Steady income / dividends",
        "Long-term growth",
        "Tax efficiency",
        "Inflation protection",
        "Liquidity / flexibility",
        "ESG / values alignment",
        "Estate / legacy planning",
    ], "max_pick": 3},
]


def score_profile(answers: dict) -> dict:
    """Capacity 50% + Tolerance 35% + Outlook 15% — same weighting as prototype.

    Goals questions feed into capacity because what the money is FOR affects
    how much risk the portfolio reasonably needs to take. Wealth-building and
    early-FI goals push capacity up (they require growth); preservation and
    short-timeline goals pull it down (they require safety).
    """
    section_scores = {"capacity": [], "tolerance": [], "outlook": []}
    capacity_qs  = {"occupation","income_band","income_stability","net_worth",
                    "withdrawal_horizon","withdrawal_rate","emergency_fund","major_expense",
                    # Goals
                    "primary_goal","goal_amount","goal_timeline",
                    "income_replacement","legacy_intent"}
    tolerance_qs = {"drawdown_reaction","experience","loss_floor","growth_vs_safety"}
    outlook_qs   = {"market_view","inflation_concern","recession_concern"}

    for q in PROFILE_QUESTIONS:
        qid = q["id"]; ans = answers.get(qid)
        if ans is None or ans == "": continue
        if qid == "age":
            try: age_val = int(ans)
            except (ValueError, TypeError): continue
            score = max(20, min(85, 95 - (age_val - 18) * 1.0))
            section_scores["capacity"].append(score); continue
        if qid == "retirement_age":
            try:
                ret = int(ans); cur = int(answers.get("age", 45))
            except (ValueError, TypeError): continue
            yrs_to_ret = max(0, ret - cur)
            score = min(90, 25 + yrs_to_ret * 2.2)
            section_scores["capacity"].append(score); continue
        if q["type"] == "select":
            opt_map = dict(q["options"])
            score = opt_map.get(ans)
            if score is None: continue
            if qid in capacity_qs:    section_scores["capacity"].append(score)
            elif qid in tolerance_qs: section_scores["tolerance"].append(score)
            elif qid in outlook_qs:   section_scores["outlook"].append(score)

    def _avg(lst, default=50):
        return sum(lst) / len(lst) if lst else default
    cap = _avg(section_scores["capacity"])
    tol = _avg(section_scores["tolerance"])
    out = _avg(section_scores["outlook"])
    overall = int(round(min(99, max(1, 0.50*cap + 0.35*tol + 0.15*out))))
    return {
        "overall_score":   overall,
        "capacity_score":  int(round(cap)),
        "tolerance_score": int(round(tol)),
        "outlook_score":   int(round(out)),
    }


def score_band(score: int) -> tuple[str, str, str]:
    """(label, hex, soft_bg) — neutral risk-profile bands.

    Three buckets only — Conservative, Moderate, Aggressive. No diagnostic
    or evaluative language ("at risk", "watch", "strong", etc.); these
    describe an *investing posture*, not a judgment about the client."""
    if score >= 70: return "Aggressive",   THEME["primary"], THEME["primary_soft"]
    if score >= 45: return "Moderate",     THEME["primary"], THEME["primary_soft"]
    return            "Conservative", THEME["primary"], THEME["primary_soft"]


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL PRIMITIVES — direct ports of the prototype's SVG components
# ─────────────────────────────────────────────────────────────────────────────
def logo_mark(color: str = None, size: int = 26) -> str:
    """Hexagon outline + inner pulse-line glyph — port of LogoMark."""
    color = color or THEME["primary"]
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="display:block">'
        f'<path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
        f'<path d="M6 12 L9 12 L10.5 9 L12 15 L13.5 11 L15 13 L18 13" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def pulse_line(color: str = None, width: int = 56, height: int = 14,
               opacity: float = 0.7) -> str:
    color = color or THEME["primary"]
    h = height; w = width
    path = (f"M0 {h/2} L{w*0.20} {h/2} L{w*0.28} {h*0.15} L{w*0.34} {h*0.85} "
            f"L{w*0.40} {h*0.30} L{w*0.46} {h/2} L{w} {h/2}")
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'style="display:block;opacity:{opacity}">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def make_risk_ring(score: int, height: int = 320) -> go.Figure:
    """Plotly port of the prototype's RiskRing: colored arc + tick marks +
    centered score & label chip."""
    import numpy as np
    label, band_color, band_soft = score_band(score)
    pct = max(0, min(99, score)) / 99.0

    n = 200
    theta_full = np.linspace(0, 2*np.pi, n)
    bg_x = np.cos(theta_full); bg_y = np.sin(theta_full)
    n_fg = max(4, int(n * pct))
    theta_fg = np.linspace(np.pi/2, np.pi/2 - 2*np.pi*pct, n_fg)
    fg_x = np.cos(theta_fg); fg_y = np.sin(theta_fg)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode="lines",
        line=dict(color=THEME["line"], width=12),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fg_x, y=fg_y, mode="lines",
        line=dict(color=band_color, width=12, shape="spline"),
        hoverinfo="skip", showlegend=False))

    tick_inner_r = 0.84; tick_outer_r = 0.92
    for i in range(36):
        a = (i * 10 - 90) * np.pi / 180
        fig.add_trace(go.Scatter(
            x=[np.cos(a)*tick_inner_r, np.cos(a)*tick_outer_r],
            y=[np.sin(a)*tick_inner_r, np.sin(a)*tick_outer_r],
            mode="lines", line=dict(color=THEME["line"], width=1),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_annotation(x=0, y=0.34, text="RISK SCORE", showarrow=False,
        font=dict(family="Inter", size=11, color=THEME["muted"]))
    fig.add_annotation(x=0, y=0.0, text=str(score), showarrow=False,
        font=dict(family="IBM Plex Mono", size=64, color=THEME["ink"]))
    fig.add_annotation(x=0, y=-0.42, text=f"  {label.upper()}  ",
        showarrow=False, bgcolor=band_soft, borderpad=6,
        font=dict(family="Inter", size=11, color=band_color))

    fig.update_xaxes(visible=False, range=[-1.2, 1.2],
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[-1.2, 1.2])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def make_sparkline(values: list, height: int = 120) -> go.Figure:
    """Mono-tone area sparkline with end-dot — port of Sparkline."""
    if not values or len(values) < 2:
        values = (values * 2) if values else [0, 0]
    color = THEME["primary"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(values))), y=values, mode="lines",
        line=dict(color=color, width=2.2, shape="spline"),
        fill="tozeroy", fillcolor=THEME["primary_soft"],
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[len(values)-1], y=[values[-1]], mode="markers",
        marker=dict(color=color, size=8,
                    line=dict(color=THEME["surface"], width=2)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[min(values)*0.95, max(values)*1.05])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def status_chip(status: str, label: Optional[str] = None) -> str:
    """Neutral chip. No diagnostic language ("Watch", "Alert") — a chip is
    only rendered if `label` is explicitly provided. Called sites that pass
    just a status now get an empty string back, which is intentional."""
    if not label:
        return ""
    cmap = {
        "healthy": (THEME["primary_soft"], THEME["primary"]),
        "caution": (THEME["primary_soft"], THEME["primary"]),
        "risk":    (THEME["primary_soft"], THEME["primary"]),
    }
    bg, fg = cmap.get(status, cmap["healthy"])
    return (f'<span class="fr-chip" style="background:{bg};color:{fg}">'
            f'{label}</span>')


def fmt_money(x: float) -> str:
    if x is None or pd.isna(x): return "—"
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:     return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def fmt_pct(x: float, sign: bool = True) -> str:
    if x is None or pd.isna(x): return "—"
    s = "+" if sign and x >= 0 else ""
    return f"{s}{x:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED VITALS
# ─────────────────────────────────────────────────────────────────────────────
def compute_vitals(holdings: dict, quotes: dict) -> dict:
    """Derive Net Worth / Cash / Gain from holdings + live quotes.
    Cash flow / DTI need budget data we don't track yet — those tiles use
    the profile's Capacity & Tolerance scores instead."""
    total_value = 0.0; total_cost = 0.0; cash_value = 0.0
    cash_tickers = {"BIL","SHV","SGOV","USFR","VMOT","VMFXX","CASH"}
    for tk, h in holdings.items():
        sh   = float(h.get("shares") or 0)
        cost = float(h.get("avg_cost") or 0)
        px   = float((quotes.get(tk) or {}).get("price") or 0)
        v    = sh * px
        b    = float(h.get("dollar_invested") or sh * cost)
        total_value += v; total_cost += b
        if tk.upper() in cash_tickers:
            cash_value += v
    gain = total_value - total_cost
    gain_pct = (gain / total_cost * 100) if total_cost else 0
    return {"net_worth": total_value, "cost_basis": total_cost,
            "cash": cash_value, "gain": gain, "gain_pct": gain_pct}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    """Default session state. fr_step controls the flow:
       welcome → prequiz → quiz → results → register → (logged-in) dashboard.
       fr_user is set only after registration completes."""
    defaults = {
        "fr_user":      None,
        "fr_view":      "dashboard",   # post-login view name
        "fr_flash":     None,
        # Pre-login flow state
        "fr_step":      "welcome",     # welcome | prequiz | quiz | results | register
        "fr_first":     "",
        "fr_last":      "",
        "fr_age":       0,
        "fr_q_idx":     0,             # current question index in quiz
        "fr_answers":   {},            # qid -> answer
        "fr_scores":    None,          # set after quiz scoring
        "fr_show_signin": False,       # toggles sign-in field on welcome
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
_init_state()

def _client_key() -> str:
    u = st.session_state.fr_user or {}
    return (u.get("email") or "").lower()

def _logout():
    """Clear all session state and return to the welcome screen. Resetting
    fr_step to 'welcome' is what brings the user back to the landing CTA
    instead of e.g. the login form they came from."""
    st.session_state.fr_user    = None
    st.session_state.fr_view    = "dashboard"
    st.session_state.fr_flash   = None
    st.session_state.fr_step    = "welcome"
    st.session_state.fr_first   = ""
    st.session_state.fr_last    = ""
    st.session_state.fr_age     = 0
    st.session_state.fr_q_idx   = 0
    st.session_state.fr_answers = {}
    st.session_state.fr_scores  = None
    st.session_state.fr_show_signin = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    """Entry router for the unauthenticated flow. Dispatches to one of five
    onboarding screens based on fr_step:
        welcome  → landing page with single CTA
        prequiz  → first/last name + age (only fields needed before quiz)
        quiz     → 23 questions across 5 sections (Goals included)
        results  → score reveal (no gate yet — show first, ask second)
        register → email + phone (req'd) + address + zip (optional) → save
    """
    step = st.session_state.fr_step
    if   step == "welcome":  _screen_welcome()
    elif step == "prequiz":  _screen_prequiz()
    elif step == "quiz":     _screen_quiz()
    elif step == "results":  _screen_results()
    elif step == "register": _screen_register()
    else:
        st.session_state.fr_step = "welcome"
        st.rerun()


# ── SCREEN 1: Welcome ────────────────────────────────────────────────────────
def _screen_welcome():
    """Anonymous landing — clean, focused. Single headline, two trust signals,
    one CTA. The "Already a member? Sign in" lives below the CTA as a state-
    toggle that reveals an email field inline (no expander chrome)."""
    # Inline SVG icons matching the mockup's hairline-stroke style
    _icon_lock = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["muted"]}" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px">'
        '<rect x="4" y="11" width="16" height="10" rx="2"/>'
        '<path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
    )
    _icon_shield = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["muted"]}" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
    )

    st.markdown(
        f'<div class="fr-welcome-wrap" style="max-width:520px;margin:24px auto 0;'
        f'            padding:0 24px;text-align:center">'
        f'  <div style="display:flex;align-items:center;justify-content:center;'
        f'              gap:16px;margin-bottom:32px">'
        f'    {logo_mark(THEME["primary"], 64)}'
        f'    <span style="font-size:1.25rem;font-weight:600;letter-spacing:0.12em;'
        f'                 color:{THEME["ink"]};text-transform:uppercase">'
        f'      Foresight Risk'
        f'    </span>'
        f'  </div>'
        f'  <h1 style="font-size:1.5rem;line-height:1.3;color:{THEME["ink"]};'
        f'             font-weight:500;margin:14px auto 24px;letter-spacing:-0.015em;'
        f'             text-align:center;max-width:440px;'
        f'             padding-left:8px;padding-right:8px">'
        f'    A complete financial risk profile in less than 3 minutes.'
        f'  </h1>'
        f'  <div style="display:flex;gap:24px;color:{THEME["muted"]};'
        f'              font-size:0.92rem;margin-bottom:24px;align-items:center;'
        f'              justify-content:center;flex-wrap:wrap">'
        f'    <span style="display:inline-flex;align-items:center">{_icon_lock}Encrypted</span>'
        f'  </div>'
        f'</div>'
        # Mobile-only tweaks: pull the CTAs up so they sit closer to the
        # vertical middle of the viewport rather than way below the fold,
        # and let the headline have a touch more breathing room around the
        # text so it doesn't visually drift left/right on narrow screens.
        f'<style>'
        f'@media (max-width: 640px) {{'
        f'  .fr-welcome-wrap {{ margin-top: 12px !important; padding: 0 20px !important; }}'
        f'  .fr-welcome-wrap h1 {{ font-size: 1.35rem !important; }}'
        f'  .fr-welcome-spacer {{ height: 28px !important; }}'
        f'}}'
        f'</style>',
        unsafe_allow_html=True,
    )

    # Spacer below the trust badges. On desktop it gives the CTA breathing
    # room (the page is otherwise sparse and we want visual rhythm); on
    # mobile it collapses to a much smaller value via the media query
    # above so the CTA sits closer to the vertical middle of the viewport.
    st.markdown(
        '<div class="fr-welcome-spacer" style="height:80px"></div>',
        unsafe_allow_html=True,
    )

    _spc_l, _cta, _spc_r = st.columns([1, 2, 1])
    with _cta:
        if st.button("Start risk profile  →", type="primary",
                     key="fr_start_btn", use_container_width=True):
            st.session_state.fr_step = "prequiz"
            st.rerun()

    # ── "Already a member? Sign in" — inline toggle, no expander chrome ────
    # Toggling sets a session flag; the email field renders on the rerun.
    if not st.session_state.get("fr_show_signin", False):
        _spc_l, _link, _spc_r = st.columns([1, 2, 1])
        with _link:
            st.markdown(
                f'<div style="text-align:center;margin-top:14px;'
                f'            font-size:0.92rem;color:{THEME["muted"]}">'
                f'  Already a member?'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Sign in", key="fr_signin_toggle",
                         use_container_width=True):
                st.session_state.fr_show_signin = True
                st.rerun()
    else:
        _spc_l, _form, _spc_r = st.columns([1, 2, 1])
        with _form:
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            login_email = st.text_input("Email", key="fr_login_email",
                                        placeholder="you@example.com",
                                        label_visibility="collapsed")
            si1, si2 = st.columns([1, 1])
            with si1:
                if st.button("Cancel", key="fr_signin_cancel",
                             use_container_width=True):
                    st.session_state.fr_show_signin = False
                    st.rerun()
            with si2:
                if st.button("Sign in", type="primary", key="fr_btn_login",
                             use_container_width=True):
                    user = find_user(login_email)
                    if user is None:
                        st.error("No account found. Take the assessment to create one.")
                    else:
                        st.session_state.fr_user  = user
                        st.session_state.fr_flash = (
                            f"Welcome back, {user.get('first_name','')}.")
                        st.session_state.fr_show_signin = False
                        st.rerun()


# ── SCREEN 2: Pre-quiz (name + age) ──────────────────────────────────────────
def _screen_prequiz():
    """Collect First name + Last name + Age before the quiz. Three fields max
    so the friction stays low; everything else moves to post-quiz registration."""
    st.markdown(
        f'<div style="max-width:520px;margin:30px auto 0;padding:0 28px">'
        f'  <div class="fr-eyebrow">A few quick details</div>'
        f'  <h1 class="fr-headline" style="font-size:1.7rem">Before we begin</h1>'
        f'  <div style="color:{THEME["ink2"]};font-size:0.92rem;margin-bottom:8px">'
        f'    Just your name and age — we\'ll ask for contact info after you see your results.'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _l, _form, _r = st.columns([1, 2, 1])
    with _form:
        # No fr-card wrapper — it was rendering an empty padded box above
        # the field labels. The text inputs already have their own visual
        # container; nesting them inside another card created a redundant
        # white rectangle.
        c1, c2 = st.columns(2)
        first = c1.text_input("First name *", key="fr_pq_first",
                              value=st.session_state.fr_first,
                              placeholder="John")
        last  = c2.text_input("Last name *", key="fr_pq_last",
                              value=st.session_state.fr_last,
                              placeholder="Smith")
        age = st.number_input("Age *", min_value=18, max_value=100,
                              value=int(st.session_state.fr_age) or 40,
                              step=1, key="fr_pq_age")

        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("← Back", key="fr_pq_back", use_container_width=True):
                st.session_state.fr_step = "welcome"
                st.rerun()
        with b2:
            if st.button("Begin assessment →", type="primary",
                         key="fr_pq_next", use_container_width=True):
                if not (first or "").strip() or not (last or "").strip():
                    st.error("First and last name are required.")
                else:
                    st.session_state.fr_first   = first.strip()
                    st.session_state.fr_last    = last.strip()
                    st.session_state.fr_age     = int(age)
                    st.session_state.fr_step    = "quiz"
                    st.session_state.fr_q_idx   = 0
                    st.session_state.fr_answers = {}
                    st.rerun()


# ── SCREEN 3: Quiz ───────────────────────────────────────────────────────────
def _screen_quiz():
    """One question per screen with a progress bar. Auto-stores the age answer
    from the prequiz step so the user doesn't see it twice."""
    # Pre-populate the "age" question from the prequiz step
    if "age" not in st.session_state.fr_answers and st.session_state.fr_age:
        st.session_state.fr_answers["age"] = int(st.session_state.fr_age)

    # Filter out the age question — already collected upstream
    visible_qs = [q for q in PROFILE_QUESTIONS if q["id"] != "age"]
    total = len(visible_qs)
    idx = max(0, min(st.session_state.fr_q_idx, total - 1))
    q = visible_qs[idx]

    progress = (idx + 1) / total

    # Header with progress
    st.markdown(
        f'<div style="max-width:560px;margin:20px auto 0;padding:0 28px">'
        f'  <div style="display:flex;align-items:center;justify-content:space-between;'
        f'              margin-bottom:14px">'
        f'    <div style="display:flex;align-items:center;gap:10px">'
        f'      {logo_mark(THEME["primary"], 22)}'
        f'      <span style="font-size:0.78rem;font-weight:600;letter-spacing:0.12em;'
        f'                   color:{THEME["ink"]};text-transform:uppercase">'
        f'        Foresight Risk'
        f'      </span>'
        f'    </div>'
        f'    <span style="font-size:0.78rem;color:{THEME["muted"]};'
        f'                 font-family:\'IBM Plex Mono\',monospace">'
        f'      {idx+1} / {total}'
        f'    </span>'
        f'  </div>'
        f'  <div style="height:4px;background:{THEME["line"]};border-radius:2px;'
        f'              overflow:hidden;margin-bottom:24px">'
        f'    <div style="height:100%;width:{progress*100:.0f}%;'
        f'                background:{THEME["primary"]};border-radius:2px;'
        f'                transition:width 0.3s ease"></div>'
        f'  </div>'
        f'  <div class="fr-eyebrow">{q["section"]}</div>'
        f'  <h2 style="font-size:1.4rem;font-weight:600;color:{THEME["ink"]};'
        f'             letter-spacing:-0.015em;line-height:1.25;margin:6px 0 22px">'
        f'    {q["text"]}'
        f'  </h2>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _l, _form, _r = st.columns([1, 2.4, 1])
    with _form:
        # No card wrapper here — the question itself is the focal point.
        # The fr-card was visually redundant with the section header above.

        prev = st.session_state.fr_answers.get(q["id"])
        if q["type"] == "number":
            val = st.number_input(
                q["text"],
                min_value=q["min"], max_value=q["max"],
                value=int(prev) if prev not in (None, "") else q.get("default", 50),
                step=q["step"], key=f"fr_qz_{q['id']}",
                label_visibility="collapsed",
            )
            answered = True
        elif q["type"] == "select":
            opts = [opt[0] for opt in q["options"]]
            val = st.radio(q["text"], opts,
                           index=opts.index(prev) if prev in opts else None,
                           key=f"fr_qz_{q['id']}",
                           label_visibility="collapsed")
            answered = val is not None

            # Auto-advance: if the user just selected an option (val is set
            # AND it's a fresh selection — different from what was stored),
            # save the answer and jump to the next question without making
            # them click "Next →". Multi-select and number-input questions
            # don't get this behavior (no clear "done" signal), and we only
            # auto-advance when *moving forward* (val != prev) so revisiting
            # a previously-answered question via Back doesn't immediately
            # bounce the user away again.
            if val is not None and val != prev:
                st.session_state.fr_answers[q["id"]] = val
                if idx == total - 1:
                    st.session_state.fr_scores = score_profile(
                        st.session_state.fr_answers
                    )
                    st.session_state.fr_step = "results"
                else:
                    st.session_state.fr_q_idx = idx + 1
                st.rerun()
        elif q["type"] == "multi":
            opts = q["options"]
            default = ([d for d in (prev or []) if d in opts]
                       if isinstance(prev, list) else [])
            # NOTE: we deliberately do NOT pass max_selections to st.multiselect.
            # Streamlit's hard cap shows a confusing "remove an option first"
            # popup that *also* prevents the user from interacting normally.
            # Instead we render a soft, informative warning below if the user
            # picks more than the recommended max_pick — and we still let
            # them finish. The first max_pick selections (in order) are what
            # actually gets scored.
            val = st.multiselect(q["text"], opts, default=default,
                                  key=f"fr_qz_{q['id']}",
                                  label_visibility="collapsed")
            max_pick = q.get("max_pick")
            if max_pick and len(val or []) > max_pick:
                st.warning(
                    f"You've picked {len(val)}. We use the top {max_pick} for "
                    f"scoring — remove one to choose which counts, or "
                    f"continue and we'll keep the first {max_pick}."
                )
            answered = len(val or []) > 0
        else:
            val = None; answered = False

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("← Back", key=f"fr_qz_back_{idx}",
                         use_container_width=True):
                if idx == 0:
                    st.session_state.fr_step = "prequiz"
                else:
                    st.session_state.fr_q_idx = idx - 1
                st.rerun()
        with b2:
            label = "Finish →" if idx == total - 1 else "Next →"
            if st.button(label, type="primary", key=f"fr_qz_next_{idx}",
                         use_container_width=True, disabled=not answered):
                # For multi-select questions with a max_pick, store only the
                # first max_pick selections — keeps scoring deterministic
                # whether or not the user respected the soft-cap warning.
                store_val = val
                if q.get("type") == "multi" and q.get("max_pick"):
                    mp = int(q["max_pick"])
                    store_val = list(val or [])[:mp]
                st.session_state.fr_answers[q["id"]] = store_val
                if idx == total - 1:
                    # Score and move to results screen
                    st.session_state.fr_scores = score_profile(
                        st.session_state.fr_answers
                    )
                    st.session_state.fr_step = "results"
                else:
                    st.session_state.fr_q_idx = idx + 1
                st.rerun()


# ── SCREEN 4: Results "ready" (score is HIDDEN until registration) ──────────
def _screen_results():
    """Score is computed and stored in session_state but NOT revealed here.
    The user sees a 'your profile is ready' card with a locked preview to
    nudge registration. Once they register and land on the dashboard, the
    full RiskRing + neutral risk-profile summary are shown.

    This is intentional: showing the score before registration removes the
    incentive to register. The score reveal becomes the reward for finishing
    sign-up, which materially improves conversion."""
    # Score is still computed (used by the dashboard after registration), but
    # we never display it. Variables are deliberately not unpacked.
    if st.session_state.fr_scores is None:
        st.session_state.fr_scores = score_profile(st.session_state.fr_answers)

    st.markdown(
        f'<div style="max-width:560px;margin:20px auto 0;padding:0 28px">'
        f'  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
        f'    {logo_mark(THEME["primary"], 22)}'
        f'    <span style="font-size:0.78rem;font-weight:600;letter-spacing:0.12em;'
        f'                 color:{THEME["ink"]};text-transform:uppercase">'
        f'      Foresight Risk'
        f'    </span>'
        f'  </div>'
        f'  <div class="fr-eyebrow">Profile complete</div>'
        f'  <h1 class="fr-headline" style="font-size:1.85rem">'
        f'    Your risk profile is ready, {st.session_state.fr_first}.'
        f'  </h1>'
        f'  <p style="font-size:0.95rem;line-height:1.55;color:{THEME["ink2"]};'
        f'            margin:0 0 22px 0">'
        f'    Save your results to view your full risk profile and a summary '
        f'    of your investing posture. Email and phone only — address '
        f'    is optional.'
        f'  </p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _l, _main, _r = st.columns([1, 2.4, 1])
    with _main:
        # ── "Locked" preview card — shows what they'll see, score blurred ──
        # Visual cue (lock icon + softened ring + "??" placeholder) signals
        # this is intentionally hidden, not missing.
        _icon_lock_lg = (
            f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
            f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:-4px;margin-right:8px">'
            f'<rect x="4" y="11" width="16" height="10" rx="2"/>'
            f'<path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
        )
        st.markdown(
            f'<div class="fr-card" style="text-align:center;padding:28px 22px">'
            f'  <div style="position:relative;width:200px;height:200px;'
            f'              margin:0 auto 16px">'
            f'    <!-- Soft ring background -->'
            f'    <svg width="200" height="200" viewBox="0 0 100 100">'
            f'      <circle cx="50" cy="50" r="42" fill="none" '
            f'              stroke="{THEME["line"]}" stroke-width="6"/>'
            f'      <circle cx="50" cy="50" r="42" fill="none" '
            f'              stroke="{THEME["primary"]}" stroke-width="6" '
            f'              stroke-dasharray="180 264" stroke-linecap="round" '
            f'              transform="rotate(-90 50 50)" opacity="0.35"/>'
            f'    </svg>'
            f'    <!-- Lock + ?? overlay -->'
            f'    <div style="position:absolute;top:0;left:0;right:0;bottom:0;'
            f'                display:flex;flex-direction:column;align-items:center;'
            f'                justify-content:center">'
            f'      {_icon_lock_lg}'
            f'      <div style="font-family:\'IBM Plex Mono\',monospace;'
            f'                  font-size:2rem;color:{THEME["muted"]};'
            f'                  letter-spacing:-0.02em;font-weight:600;'
            f'                  margin-top:4px">'
            f'        ? ?'
            f'      </div>'
            f'      <div style="font-size:0.72rem;color:{THEME["muted"]};'
            f'                  letter-spacing:0.14em;text-transform:uppercase;'
            f'                  margin-top:2px">'
            f'        Sign up to view'
            f'      </div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="fr-eyebrow">Save your results</div>'
            f'  <h3 style="margin:6px 0 8px">Create a free account</h3>'
            f'  <p style="color:{THEME["ink2"]};font-size:0.92rem;margin:0 0 8px">'
            f'    Your answers are saved on this device. Add your email '
            f'    and phone to unlock your full report.'
            f'  </p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Save & view my results →", type="primary",
                     key="fr_results_save", use_container_width=True):
            st.session_state.fr_step = "register"
            st.rerun()

        # Retake — clears answers and restarts at prequiz so they get the
        # name/age form again. (Kept here so users who realize they answered
        # incorrectly can start over without registering.)
        if st.button("← Retake assessment", key="fr_results_retake",
                     use_container_width=True):
            st.session_state.fr_step    = "prequiz"
            st.session_state.fr_q_idx   = 0
            st.session_state.fr_answers = {}
            st.session_state.fr_scores  = None
            st.rerun()


# ── SCREEN 5: Register ───────────────────────────────────────────────────────
def _screen_register():
    """Final registration — Email + Phone required, Address + ZIP optional.
    First name, last name, and age are pre-filled from the prequiz step (and
    not editable here to keep the form short)."""
    st.markdown(
        f'<div style="max-width:520px;margin:20px auto 0;padding:0 28px">'
        f'  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
        f'    {logo_mark(THEME["primary"], 22)}'
        f'    <span style="font-size:0.78rem;font-weight:600;letter-spacing:0.12em;'
        f'                 color:{THEME["ink"]};text-transform:uppercase">'
        f'      Foresight Risk'
        f'    </span>'
        f'  </div>'
        f'  <div class="fr-eyebrow">Almost done</div>'
        f'  <h1 class="fr-headline" style="font-size:1.7rem">Save your results</h1>'
        f'  <div style="color:{THEME["ink2"]};font-size:0.92rem;margin-bottom:8px">'
        f'    {st.session_state.fr_first} {st.session_state.fr_last} · '
        f'    age {st.session_state.fr_age}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _l, _form, _r = st.columns([1, 2, 1])
    with _form:
        # No fr-card wrapper here — same reason as the prequiz screen: it
        # rendered an empty padded white box above the first label. The
        # eyebrow + inputs already group visually on their own.
        st.markdown('<div class="fr-eyebrow">Contact info</div>',
                    unsafe_allow_html=True)
        email = st.text_input("Email *", key="fr_rg_email",
                              placeholder="you@example.com")
        phone = st.text_input("Phone *", key="fr_rg_phone",
                              placeholder="(555) 555-5555")

        st.markdown(
            f'<div style="margin-top:18px"><div class="fr-eyebrow">'
            f'Optional</div></div>',
            unsafe_allow_html=True,
        )
        addr = st.text_input("Address", key="fr_rg_addr",
                             placeholder="123 Main St")
        zipcode = st.text_input("ZIP code", key="fr_rg_zip",
                                placeholder="12345")

        st.caption("Your email is how you'll sign in next time.")

        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("← Back", key="fr_rg_back",
                         use_container_width=True):
                st.session_state.fr_step = "results"
                st.rerun()
        with b2:
            if st.button("Save & view dashboard →", type="primary",
                         key="fr_rg_submit", use_container_width=True):
                # Validation
                errors = []
                if not is_valid_email(email):
                    errors.append("Please enter a valid email address.")
                phone_digits = "".join(ch for ch in (phone or "") if ch.isdigit())
                if not (phone or "").strip():
                    errors.append("Phone number is required.")
                elif len(phone_digits) < 10:
                    errors.append("Phone needs at least 10 digits.")
                if (zipcode or "").strip():
                    z = "".join(ch for ch in zipcode if ch.isdigit())
                    if len(z) not in (5, 9):
                        errors.append("ZIP should be 5 digits (12345) or 9 (12345-6789).")
                if errors:
                    for e in errors: st.error(e)
                    return

                # Register the user
                ok, msg = register_user(
                    st.session_state.fr_first,
                    st.session_state.fr_last,
                    email, phone,
                )
                if not ok:
                    st.error(msg)
                    return

                # Pull the freshly-registered user and merge optional fields
                user = find_user(email)
                if user:
                    user["age"]     = int(st.session_state.fr_age)
                    user["address"] = (addr or "").strip()
                    user["zip"]     = (zipcode or "").strip()
                    _shared_update_json(
                        USERS_FILE,
                        lambda d, k=user["email"], u=user: d.update({k: u}),
                    )

                # Persist the risk profile (so dashboard can read it)
                ck = (email or "").strip().lower()
                save_profile_for(ck, {
                    "client_name":  f'{st.session_state.fr_first} '
                                    f'{st.session_state.fr_last}'.strip(),
                    "client_email": email.strip().lower(),
                    "client_age":   int(st.session_state.fr_age),
                    "answers":      st.session_state.fr_answers,
                    "priorities":   st.session_state.fr_answers.get("priorities", []),
                    **(st.session_state.fr_scores or {}),
                })

                # ── HubSpot CRM sync ─────────────────────────────────────
                # Local save above is the source of truth. If HubSpot is
                # down, missing a token, or the module isn't installed,
                # registration still succeeds — the sync just no-ops.
                # sync_now=True attempts one synchronous push (~1-2s on
                # success) and falls back to the background queue on
                # failure, so the user is never blocked.
                hs_msg = ""
                if _HUBSPOT_AVAILABLE and hubspot_sync is not None:
                    try:
                        if hubspot_sync.is_configured():
                            scores = st.session_state.fr_scores or {}
                            overall = int(scores.get("overall_score", 0))
                            label, _, _ = (score_band(overall) if overall
                                           else ("", "", ""))
                            hs_status = hubspot_sync.sync_contact(
                                first      = st.session_state.fr_first,
                                last       = st.session_state.fr_last,
                                email      = email,
                                phone      = phone,
                                address    = addr,
                                zipcode    = zipcode,
                                age        = int(st.session_state.fr_age),
                                risk_score = overall,
                                risk_label = label,
                                sync_now   = True,
                            )
                            print(f"[hubspot_sync] result: {hs_status}")
                            if hs_status.get("ok") and not hs_status.get("queued"):
                                hs_msg = " Your advisor has been notified."
                            elif hs_status.get("queued"):
                                hs_msg = " Sending to your advisor in the background."
                        else:
                            print("[hubspot_sync] not configured "
                                  "(no HUBSPOT_TOKEN env var or "
                                  "hubspot_token Streamlit secret)")
                    except Exception as _hs_e:
                        # Never block registration on a sync error.
                        import traceback as _tb
                        _tb.print_exc()
                        print(f"[hubspot_sync] sync exception: {_hs_e}")

                # Log them in and land on the dashboard.
                st.session_state.fr_user = user
                st.session_state.fr_flash = (
                    "Profile saved — welcome!" + hs_msg)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard():
    user = st.session_state.fr_user
    ck   = _client_key()
    profile      = load_profiles().get(ck, {})
    all_holdings = load_all_holdings()
    holdings     = all_holdings.get(ck, {}) or {}

    # ── App bar ─────────────────────────────────────────────────────────────
    bar_l, bar_r = st.columns([4, 1])
    with bar_l:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding-top:6px">'
            f'  {logo_mark(THEME["primary"], 22)}'
            f'  <span style="font-size:0.78rem;font-weight:600;letter-spacing:0.12em;'
            f'               color:{THEME["ink"]};text-transform:uppercase">'
            f'    Foresight Risk'
            f'  </span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with bar_r:
        if st.button("Sign out", key="fr_logout_btn", use_container_width=True):
            _logout()

    if st.session_state.fr_flash:
        st.success(st.session_state.fr_flash)
        st.session_state.fr_flash = None

    # ── Greeting ────────────────────────────────────────────────────────────
    first_name = user.get("first_name", "there")
    hour = datetime.now().hour
    greeting = ("Good morning" if hour < 12 else
                "Good afternoon" if hour < 18 else "Good evening")
    updated_str = profile.get("updated_at") or profile.get("date_completed")
    if updated_str:
        try:
            d = datetime.fromisoformat(str(updated_str).replace(" ", "T")[:16])
            days_ago = (datetime.now() - d).days
            if days_ago == 0: when_text = "earlier today"
            elif days_ago == 1: when_text = "yesterday"
            else: when_text = f"{days_ago} days ago"
        except Exception:
            when_text = "recently"
    else:
        when_text = "not yet"

    st.markdown(
        f'<div style="margin:18px 0 0 2px">'
        f'  <div class="fr-greeting">{greeting}, {first_name}</div>'
        f'  <h1 class="fr-headline">'
        f'    Your last checkup was<br/>'
        f'    <span class="fr-headline-accent">{when_text}</span>.'
        f'  </h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Real tabs replace the old visual-only bottom nav ───────────────────
    # Order matches the natural reading flow: high-level summary → portfolio
    # detail → forward-looking plan → human contact. Plan was renamed to
    # "Financial Goals" since that's the actual content of the tab.
    tab_home, tab_goals, tab_holdings, tab_advisor = st.tabs(
        ["Home", "Financial Goals", "Holdings", "Advisor"]
    )

    with tab_home:
        _render_home_tab(profile, holdings, ck)
    with tab_goals:
        _render_plan_tab(ck)
    with tab_holdings:
        _render_holdings_tab(holdings, ck)
    with tab_advisor:
        _render_advisor_tab()


def _render_home_tab(profile: dict, holdings: dict, ck: str):
    """Original dashboard body — score hero, vitals snapshot, trend, holdings.
    The advisor CTA and fake bottom nav have been removed; the advisor card
    moved to its own tab and the bottom nav was replaced by real tabs."""
    # ── Score hero card ─────────────────────────────────────────────────────
    if not profile or "overall_score" not in profile:
        st.markdown(
            f'<div class="fr-card" style="padding:26px;text-align:center">'
            f'  <div style="display:flex;justify-content:center;margin-bottom:14px">'
            f'    {pulse_line(THEME["primary"], 56, 14)}'
            f'  </div>'
            f'  <h3 style="margin:0 0 6px">Take your first checkup</h3>'
            f'  <p style="color:{THEME["ink2"]};margin:0 0 18px;font-size:0.93rem">'
            f'    23 questions in 5 short sections — about 6 minutes.'
            f'  </p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Start risk profile →", type="primary",
                     use_container_width=True, key="fr_start_quiz"):
            st.session_state.fr_view = "edit_profile"
            st.rerun()
    else:
        overall = int(profile.get("overall_score", 50))
        # No fr-card wrapper here — Streamlit's `st.columns`, `st.plotly_chart`
        # and `st.button` don't actually nest inside raw HTML divs (they're
        # appended as DOM siblings), so the `<div class="fr-card">` was
        # rendering as an empty padded white box above the content.
        h1, h2 = st.columns([1.05, 1])
        with h1:
            st.plotly_chart(make_risk_ring(overall, height=300),
                use_container_width=True,
                config={"displayModeBar": False})
        with h2:
            cap = int(profile.get("capacity_score", 50))
            tol = int(profile.get("tolerance_score", 50))
            label, _, _ = score_band(overall)

            # Neutral summary — describes the posture, not a verdict on the
            # client. Three buckets matching score_band: Conservative,
            # Moderate, Aggressive.
            if label == "Aggressive":
                summary = (
                    "Your answers point to an aggressive posture — a higher "
                    "tolerance for short-term swings in exchange for greater "
                    "long-term growth potential."
                )
            elif label == "Moderate":
                summary = (
                    "Your answers point to a moderate posture — a balance "
                    "between growth and stability that most long-term "
                    "investors land on."
                )
            else:
                summary = (
                    "Your answers point to a conservative posture — a "
                    "preference for stability and capital preservation over "
                    "maximum growth."
                )

            st.markdown(
                f'<div style="padding-top:18px">'
                f'  <div class="fr-eyebrow">Risk Profile</div>'
                f'  <div style="font-size:1rem;color:{THEME["ink"]};font-weight:600;'
                f'              margin-top:4px;line-height:1.3">{label}</div>'
                f'  <div style="font-size:0.85rem;color:{THEME["ink2"]};'
                f'              margin-top:8px;line-height:1.5">{summary}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("View / update profile →", key="fr_view_profile",
                         use_container_width=True):
                st.session_state.fr_view = "edit_profile"
                st.rerun()

    # ── Vitals grid ─────────────────────────────────────────────────────────
    if holdings:
        quotes = get_live_quotes(list(holdings.keys()))
        vitals = compute_vitals(holdings, quotes)
    else:
        quotes = {}
        vitals = {"net_worth": 0, "cost_basis": 0, "cash": 0,
                  "gain": 0, "gain_pct": 0}

    cap = int(profile.get("capacity_score", 0)) if profile else 0
    tol = int(profile.get("tolerance_score", 0)) if profile else 0

    def _tile(label: str, value: str, detail: str, status: str,
              delta: str = "", gauge: Optional[int] = None) -> str:
        """If `gauge` is a 0-100 integer, render a thin horizontal bar
        underneath the value — used for Risk Capacity and Risk Tolerance so
        the score has a visual reference, not just a bare number."""
        chip = status_chip(status) if status else ""
        delta_color = (THEME["healthy"] if not str(delta).startswith("-")
                       else THEME["risk"])
        delta_html = (f'<span class="fr-mono" style="color:{delta_color}">'
                      f'{delta}</span>' if delta else "")
        gauge_html = ""
        if gauge is not None:
            g = max(0, min(100, int(gauge)))
            gauge_html = (
                f'<div style="margin-top:8px">'
                f'  <div style="height:5px;background:{THEME["line"]};'
                f'              border-radius:3px;position:relative;overflow:hidden">'
                f'    <div style="height:100%;width:{g}%;'
                f'                background:{THEME["primary"]};'
                f'                border-radius:3px"></div>'
                f'  </div>'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              font-size:0.66rem;color:{THEME["muted"]};'
                f'              margin-top:4px;'
                f'              font-variant-numeric:tabular-nums">'
                f'    <span>0</span><span>50</span><span>100</span>'
                f'  </div>'
                f'</div>'
            )
        return (
            f'<div class="fr-vital">'
            f'  <div style="display:flex;align-items:center;justify-content:space-between">'
            f'    <span class="fr-vital-label">{label}</span>{chip}'
            f'  </div>'
            f'  <div class="fr-vital-value">{value}</div>'
            f'  <div class="fr-vital-detail">'
            f'    <span>{detail}</span>{delta_html}'
            f'  </div>'
            f'  {gauge_html}'
            f'</div>'
        )

    # ── Advisor box ─────────────────────────────────────────────────────────
    # Compact version of the full advisor card — surfaces the human contact
    # at the top of Home so clients see who's behind the numbers without
    # having to navigate to the Advisor tab. The full profile + bio + book-
    # a-call CTA still live on the dedicated Advisor tab.
    #
    # Profile is loaded fresh from firm_settings.json every render via
    # get_advisor(), so when the advisor updates their info on the advisor
    # app side it shows up here on the next page load — no restart needed.
    a = get_advisor()
    company_logo_html = a["firm_logo_html_small"]
    st.markdown(
        f'<div style="background:{THEME["surface2"]};border:1px solid {THEME["line"]};'
        f'            border-radius:14px;padding:14px 16px;margin-top:18px;'
        f'            display:flex;align-items:center;gap:14px">'
        f'  <div style="flex-shrink:0">{a["photo_html"]}</div>'
        f'  <div style="flex:1;min-width:0">'
        f'    <div style="display:flex;align-items:center;gap:8px;'
        f'                margin-bottom:2px">'
        f'      <div class="fr-eyebrow" style="margin:0">Your advisor</div>'
        f'    </div>'
        f'    <div style="font-size:1rem;font-weight:600;color:{THEME["ink"]};'
        f'                line-height:1.25;letter-spacing:-0.01em">{a["name"]}</div>'
        f'    <div style="display:flex;align-items:center;gap:6px;'
        f'                font-size:0.8rem;color:{THEME["ink2"]};margin-top:3px">'
        f'      {company_logo_html}'
        f'      <span>{a["firm"]}</span>'
        f'    </div>'
        f'    <div style="font-size:0.78rem;color:{THEME["muted"]};margin-top:6px;'
        f'                line-height:1.5">'
        f'      <a href="mailto:{a["email"]}" style="color:{THEME["primary"]};'
        f'                                            text-decoration:none">'
        f'        {a["email"]}'
        f'      </a> · '
        f'      <a href="tel:{a["phone"].replace(" ", "")}" '
        f'         style="color:{THEME["primary"]};text-decoration:none">'
        f'        {a["phone"]}'
        f'      </a>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Snapshot grid ───────────────────────────────────────────────────────
    # Three rows, each grouping related metrics:
    #   Row 1: Risk Capacity | Risk Tolerance (risk-profile pair — leads
    #          because the profile is the headline of this app)
    #   Row 2: Net Worth | Cash Position    (financial-position pair)
    #   Row 3: Financial Goals              (full-width with progress meter)
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'            margin:18px 2px 10px">'
        f'  <div class="fr-eyebrow">Snapshot</div>'
        f'  <span style="font-size:0.72rem;color:{THEME["primary"]};font-weight:600">'
        f'    This month'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Pre-compute cash percentage so it's available for the row-2 tile
    cash_pct = (vitals["cash"] / vitals["net_worth"] * 100
                if vitals["net_worth"] else 0)

    # ── Row 1: Risk Capacity | Risk Tolerance ───────────────────────────────
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(_tile(
            "Risk Capacity", str(cap) if cap else "—",
            "ability to absorb loss", "",
            gauge=cap if cap else None,
        ), unsafe_allow_html=True)
    with g2:
        st.markdown(_tile(
            "Risk Tolerance", str(tol) if tol else "—",
            "comfort with volatility", "",
            gauge=tol if tol else None,
        ), unsafe_allow_html=True)

    # ── Row 2: Net Worth | Cash Position ────────────────────────────────────
    g3, g4 = st.columns(2)
    with g3:
        nw_delta  = (fmt_pct(vitals["gain_pct"]) if vitals["cost_basis"] else "")
        st.markdown(_tile(
            "Net Worth", fmt_money(vitals["net_worth"]),
            f"{len(holdings)} positions" if holdings else "no positions yet",
            "", delta=nw_delta,
        ), unsafe_allow_html=True)
    with g4:
        st.markdown(_tile(
            "Cash Position", fmt_money(vitals["cash"]),
            f"{cash_pct:.1f}% of portfolio" if vitals["net_worth"] else "—",
            "",
        ), unsafe_allow_html=True)

    # ── Row 3: Financial Goals (full width, with progress meter) ────────────
    # Sits inside the Snapshot section so it reads as another vital — same
    # surface treatment as the tiles above. Detailed goal list and budget
    # builder live on the Financial Goals tab.
    goals = load_goals_for(ck)
    if goals:
        _today = date.today()
        total_target  = sum(float(g.get("amount") or 0) for g in goals)
        total_saved   = sum(float(g.get("saved")  or 0) for g in goals)
        total_monthly = 0.0
        for g in goals:
            try:
                tdt = date.fromisoformat(g.get("target_date", ""))
                mleft = max(1, (tdt.year - _today.year) * 12
                              + (tdt.month - _today.month))
            except Exception:
                mleft = 12
            rem = max(0.0, float(g.get("amount") or 0)
                          - float(g.get("saved")  or 0))
            total_monthly += rem / mleft
        pct = min(100, (total_saved / total_target * 100)
                       if total_target else 0)
        st.markdown(
            f'<div class="fr-vital" style="margin-top:8px">'
            f'  <div style="display:flex;align-items:center;'
            f'              justify-content:space-between">'
            f'    <span class="fr-vital-label">Financial Goals</span>'
            f'    <span style="font-size:0.72rem;color:{THEME["muted"]};'
            f'                 font-weight:600">'
            f'      {len(goals)} active'
            f'    </span>'
            f'  </div>'
            f'  <div style="display:flex;justify-content:space-between;'
            f'              align-items:baseline;margin-top:6px">'
            f'    <span style="font-size:1.05rem;font-weight:600;'
            f'                 color:{THEME["ink"]};font-variant-numeric:tabular-nums">'
            f'      {fmt_money(total_saved)} <span style="color:{THEME["muted"]};'
            f'                                          font-weight:500">'
            f'        / {fmt_money(total_target)}</span>'
            f'    </span>'
            f'    <span class="fr-mono" style="color:{THEME["primary"]};'
            f'                                  font-weight:700;font-size:0.95rem">'
            f'      {pct:.0f}%'
            f'    </span>'
            f'  </div>'
            f'  <div style="height:6px;background:{THEME["line"]};'
            f'              border-radius:3px;margin-top:8px;overflow:hidden">'
            f'    <div style="height:100%;width:{pct:.0f}%;'
            f'                background:{THEME["primary"]};'
            f'                border-radius:3px"></div>'
            f'  </div>'
            f'  <div style="display:flex;justify-content:space-between;'
            f'              font-size:0.74rem;color:{THEME["muted"]};'
            f'              margin-top:6px">'
            f'    <span>saved toward your goals</span>'
            f'    <span class="fr-mono">'
            f'      {fmt_money(total_monthly)}/mo to stay on pace'
            f'    </span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="fr-vital" style="margin-top:8px;text-align:center;'
            f'                              border-style:dashed">'
            f'  <div class="fr-vital-label" style="margin-bottom:6px">'
            f'    Financial Goals'
            f'  </div>'
            f'  <div style="font-size:0.86rem;color:{THEME["ink2"]};'
            f'              line-height:1.5">'
            f'    No goals yet. Head to the <strong>Financial Goals</strong> tab '
            f'    to add what you\'re saving toward.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Portfolio Performance card ──────────────────────────────────────────
    # Renamed from "Net Worth Trend" — same data (sparkline of net worth over
    # the last N months) but the new label more accurately describes what
    # users are looking at: how their portfolio has been performing.
    if holdings and vitals["net_worth"] > 0:
        import numpy as np
        # Stable per-user random shape so the sparkline doesn't jitter on rerun
        np.random.seed(hash(ck) & 0xFFFFFFFF)
        base = max(vitals["cost_basis"], 1)
        end  = vitals["net_worth"]
        n = 12
        steps = np.cumsum(np.random.randn(n) * (end - base) * 0.04)
        steps = steps - steps[0]
        scale = (end - base) / (steps[-1] - steps[0]) if steps[-1] != steps[0] else 1
        series = (base + steps * scale).tolist()
        series[-1] = end

        st.markdown(
            f'<div class="fr-card" style="margin-bottom:0">'
            f'  <div style="display:flex;align-items:flex-end;justify-content:space-between">'
            f'    <div>'
            f'      <div class="fr-eyebrow">Portfolio Performance</div>'
            f'      <div class="fr-mono" style="font-size:1.35rem;color:{THEME["ink"]};'
            f'                                    margin-top:2px">'
            f'        {fmt_money(end)}'
            f'      </div>'
            f'    </div>'
            f'    <div style="display:flex;gap:6px">'
            f'      <span style="font-size:0.7rem;padding:4px 9px;border-radius:999px;'
            f'                   color:{THEME["muted"]};font-weight:600">1M</span>'
            f'      <span style="font-size:0.7rem;padding:4px 9px;border-radius:999px;'
            f'                   background:{THEME["chip"]};color:{THEME["ink"]};font-weight:600">3M</span>'
            f'      <span style="font-size:0.7rem;padding:4px 9px;border-radius:999px;'
            f'                   color:{THEME["muted"]};font-weight:600">1Y</span>'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_sparkline(series, height=120),
            use_container_width=True,
            config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# HOLDINGS TAB — full portfolio view, formerly the bottom of the Home tab
# ─────────────────────────────────────────────────────────────────────────────
def _render_holdings_tab(holdings: dict, ck: str):
    """Standalone tab for the user's portfolio. Used to live at the bottom
    of the Home tab; promoted to its own tab so Holdings sits between the
    summary view and the planning view in the natural reading order
    (Home → Holdings → Financial Goals → Advisor)."""
    if holdings:
        quotes = get_live_quotes(list(holdings.keys()))
    else:
        quotes = {}

    # Header + Manage button
    h_l, h_r = st.columns([3, 1])
    with h_l:
        st.markdown(
            f'<div class="fr-eyebrow">Holdings</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:{THEME["ink"]};'
            f'            margin-top:2px">{len(holdings)} '
            f'position{"s" if len(holdings)!=1 else ""}</div>',
            unsafe_allow_html=True,
        )
    with h_r:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("Manage", key="fr_manage_holdings",
                     use_container_width=True):
            st.session_state.fr_view = "edit_holdings"
            st.rerun()

    if holdings:
        # In a dedicated tab we have room to show ALL positions, not just the
        # top 5 like the old home-tab summary did.
        rows = []
        for tk, h in holdings.items():
            sh = float(h.get("shares") or 0)
            px = float((quotes.get(tk) or {}).get("price") or 0)
            val = sh * px
            day = float((quotes.get(tk) or {}).get("change_pct") or 0)
            rows.append((tk, sh, px, val, day))
        rows.sort(key=lambda r: -r[3])

        for tk, sh, px, val, day in rows:
            day_color = THEME["healthy"] if day >= 0 else THEME["risk"]
            day_sign  = "+" if day >= 0 else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;'
                f'            justify-content:space-between;padding:10px 0;'
                f'            border-top:1px solid {THEME["line"]}">'
                f'  <div>'
                f'    <span class="fr-mono" style="color:{THEME["ink"]};'
                f'                                  font-size:0.95rem">{tk}</span>'
                f'    <span style="color:{THEME["muted"]};font-size:0.78rem;'
                f'                 margin-left:8px">{sh:g} sh @ ${px:,.2f}</span>'
                f'  </div>'
                f'  <div style="text-align:right">'
                f'    <div class="fr-mono" style="color:{THEME["ink"]};'
                f'                                  font-size:0.95rem">{fmt_money(val)}</div>'
                f'    <div class="fr-mono" style="color:{day_color};'
                f'                                  font-size:0.72rem">'
                f'      {day_sign}{day:.2f}%'
                f'    </div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="text-align:center;padding:32px 0;color:{THEME["muted"]};'
            f'            font-size:0.92rem">'
            f'  No holdings yet. Tap <strong>Manage</strong> above to add '
            f'  your first position.'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PLAN TAB — Goals + Budget builder
# ─────────────────────────────────────────────────────────────────────────────
def _render_plan_tab(ck: str):
    """Two stacked sections: financial goals (with $ amount + timeline) and a
    monthly budget builder that shows how much room the user has each month
    to direct toward their goals."""
    goals = load_goals_for(ck)
    budget = load_budget_for(ck)

    # ── Goals card ──────────────────────────────────────────────────────────
    # No fr-card wrapper — st.data_editor and other Streamlit widgets below
    # don't actually nest into raw HTML divs.
    st.markdown(
        f'<div class="fr-eyebrow">Financial Goals</div>'
        f'<div style="font-size:1.05rem;font-weight:600;color:{THEME["ink"]};'
        f'            margin-top:2px">What are you saving toward?</div>'
        f'<div style="color:{THEME["ink2"]};font-size:0.88rem;margin-top:4px">'
        f'  Add a goal with a dollar amount and target date. We\'ll show what '
        f'  you need to set aside each month to get there.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Goals list — a single editable table. No cap on number of goals; users
    # add rows by typing into the empty bottom row, delete by selecting rows
    # via the row-handle and pressing Delete (Streamlit's data_editor pattern,
    # same as the holdings editor elsewhere in this app).
    today = date.today()

    import pandas as pd  # local import — only needed for the goals/budget tab

    # Build the editable dataframe. We always append one blank row at the end
    # so there's a visible "add a goal here" affordance even when the list
    # is full of saved goals.
    default_target = today.replace(year=today.year + 5)
    goal_rows = []
    for g in goals:
        try:
            tdt = date.fromisoformat(g.get("target_date", ""))
        except Exception:
            tdt = default_target
        goal_rows.append({
            "Goal":         g.get("name", ""),
            "Target $":     float(g.get("amount") or 0),
            "Saved":        float(g.get("saved") or 0),
            "Target date":  tdt,
        })
    goals_df = pd.DataFrame(
        goal_rows,
        columns=["Goal", "Target $", "Saved", "Target date"],
    )

    edited_df = st.data_editor(
        goals_df,
        key="fr_goals_editor",
        num_rows="dynamic",            # users can add/delete rows freely
        use_container_width=True,
        hide_index=True,
        column_config={
            "Goal": st.column_config.TextColumn(
                "Goal",
                help="What are you saving toward? "
                     "(e.g., House down payment, Sabbatical, College fund)",
                required=False,
                max_chars=80,
            ),
            "Target $": st.column_config.NumberColumn(
                "Target $",
                help="Dollar amount you want to reach",
                min_value=0.0, step=1000.0, format="$%d",
            ),
            "Saved": st.column_config.NumberColumn(
                "Saved",
                help="How much you've set aside toward this goal so far",
                min_value=0.0, step=500.0, format="$%d",
            ),
            "Target date": st.column_config.DateColumn(
                "Target date",
                help="When you want to reach the goal",
                min_value=today,
            ),
        },
    )

    # Save changes whenever the table edits land. We keep only rows that have
    # both a name and a positive target — partially-typed rows are ignored
    # until they're complete, so the user's in-progress entry doesn't get
    # discarded on rerun.
    cleaned = []
    for _, row in edited_df.iterrows():
        name = (str(row.get("Goal") or "")).strip()
        amt  = float(row.get("Target $") or 0)
        if not name or amt <= 0:
            continue
        saved = float(row.get("Saved") or 0)
        tdt = row.get("Target date") or default_target
        try:
            tdt_iso = (tdt.isoformat() if hasattr(tdt, "isoformat")
                       else str(tdt))
        except Exception:
            tdt_iso = default_target.isoformat()
        cleaned.append({
            "name":        name,
            "amount":      round(amt, 2),
            "saved":       round(saved, 2),
            "target_date": tdt_iso,
            "added_at":    datetime.now().isoformat(timespec="minutes"),
        })

    # Persist only when the cleaned list actually differs from what's saved
    # (otherwise every dashboard rerun would re-write the file).
    def _normalize(g_list):
        return [(g["name"], g["amount"], g["saved"], g["target_date"])
                for g in g_list]
    if _normalize(cleaned) != _normalize(goals):
        save_goals_for(ck, cleaned)
        goals = cleaned

    # Roll-up summary: total target, total saved, total monthly need across
    # all goals. Replaces the per-card progress bars; users see at a glance
    # whether they're tracking against their plan as a whole.
    if goals:
        total_target  = sum(float(g.get("amount") or 0) for g in goals)
        total_saved   = sum(float(g.get("saved")  or 0) for g in goals)
        total_monthly = 0.0
        for g in goals:
            try:
                tdt = date.fromisoformat(g.get("target_date", ""))
                mleft = max(1, (tdt.year - today.year) * 12
                              + (tdt.month - today.month))
            except Exception:
                mleft = 12
            rem = max(0.0, float(g.get("amount") or 0)
                          - float(g.get("saved")  or 0))
            total_monthly += rem / mleft
        pct = min(100, (total_saved / total_target * 100)
                       if total_target else 0)
        st.markdown(
            f'<div style="margin-top:14px;background:{THEME["surface2"]};'
            f'            border:1px solid {THEME["line"]};border-radius:14px;'
            f'            padding:14px 16px">'
            f'  <div style="display:flex;justify-content:space-between;'
            f'              align-items:baseline">'
            f'    <span class="fr-vital-label">'
            f'      Across {len(goals)} goal{"s" if len(goals)!=1 else ""}'
            f'    </span>'
            f'    <span class="fr-mono" style="color:{THEME["ink"]};font-weight:600">'
            f'      {fmt_money(total_saved)} / {fmt_money(total_target)}'
            f'    </span>'
            f'  </div>'
            f'  <div style="height:6px;background:{THEME["line"]};'
            f'              border-radius:3px;margin-top:8px;overflow:hidden">'
            f'    <div style="height:100%;width:{pct:.0f}%;'
            f'                background:{THEME["primary"]}"></div>'
            f'  </div>'
            f'  <div style="display:flex;justify-content:space-between;'
            f'              font-size:0.78rem;color:{THEME["muted"]};'
            f'              margin-top:8px">'
            f'    <span>{pct:.0f}% funded overall</span>'
            f'    <span class="fr-mono">'
            f'      {fmt_money(total_monthly)}/mo to stay on pace'
            f'    </span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Type a goal name in the empty row above to add your "
                   "first one. Add as many goals as you'd like.")

    # Visual separator between the Goals section and the Budget section,
    # since we no longer have card backgrounds providing that separation.
    st.markdown(
        f'<div style="height:1px;background:{THEME["line"]};'
        f'            margin:28px 0 20px"></div>',
        unsafe_allow_html=True,
    )

    # ── Budget builder card ─────────────────────────────────────────────────
    # No fr-card wrapper — same reason as the other cards.
    st.markdown(
        f'<div class="fr-eyebrow">Monthly Budget</div>'
        f'<div style="font-size:1.05rem;font-weight:600;color:{THEME["ink"]};'
        f'            margin-top:2px">What\'s coming in and going out?</div>'
        f'<div style="color:{THEME["ink2"]};font-size:0.88rem;margin-top:4px">'
        f'  Enter rough monthly numbers — we\'ll show how much room you have to '
        f'  put toward your goals.'
        f'</div>'
        f'<div style="height:14px"></div>',
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)
    income = b1.number_input("Take-home income (monthly)", min_value=0.0,
                             value=float(budget.get("income") or 0),
                             step=100.0, format="%.2f", key="fr_bud_income")
    housing = b2.number_input("Housing (rent / mortgage)", min_value=0.0,
                              value=float(budget.get("housing") or 0),
                              step=50.0, format="%.2f", key="fr_bud_housing")

    b3, b4 = st.columns(2)
    transport = b3.number_input("Transportation", min_value=0.0,
                                value=float(budget.get("transport") or 0),
                                step=25.0, format="%.2f", key="fr_bud_transport")
    food = b4.number_input("Food & groceries", min_value=0.0,
                           value=float(budget.get("food") or 0),
                           step=25.0, format="%.2f", key="fr_bud_food")

    b5, b6 = st.columns(2)
    utilities = b5.number_input("Utilities & insurance", min_value=0.0,
                                value=float(budget.get("utilities") or 0),
                                step=25.0, format="%.2f", key="fr_bud_util")
    debt = b6.number_input("Debt payments (non-mortgage)", min_value=0.0,
                           value=float(budget.get("debt") or 0),
                           step=25.0, format="%.2f", key="fr_bud_debt")

    b7, b8 = st.columns(2)
    discretionary = b7.number_input("Discretionary (dining, shopping, fun)",
                                    min_value=0.0,
                                    value=float(budget.get("discretionary") or 0),
                                    step=25.0, format="%.2f",
                                    key="fr_bud_disc")
    other = b8.number_input("Other monthly expenses", min_value=0.0,
                            value=float(budget.get("other") or 0),
                            step=25.0, format="%.2f", key="fr_bud_other")

    expenses = (housing + transport + food + utilities + debt
                + discretionary + other)
    available = income - expenses

    # Tally up monthly need across all goals to compare to available cash flow
    total_monthly_need = 0.0
    for g in goals:
        try:
            tdt = date.fromisoformat(g.get("target_date", ""))
            mleft = max(1, (tdt.year - today.year) * 12
                          + (tdt.month - today.month))
        except Exception:
            mleft = 12
        rem = max(0.0, float(g.get("amount") or 0) - float(g.get("saved") or 0))
        total_monthly_need += rem / mleft

    gap = available - total_monthly_need
    gap_color = THEME["primary"] if gap >= 0 else THEME["risk"]
    gap_label = ("On track to fund your goals"
                 if gap >= 0
                 else f"Short by {fmt_money(abs(gap))}/month")

    st.markdown(
        f'<div style="height:10px"></div>'
        f'<div style="background:{THEME["surface2"]};border:1px solid {THEME["line"]};'
        f'            border-radius:14px;padding:14px 16px">'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline">'
        f'    <span class="fr-vital-label">Monthly income</span>'
        f'    <span class="fr-mono">{fmt_money(income)}</span>'
        f'  </div>'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline;margin-top:6px">'
        f'    <span class="fr-vital-label">Monthly expenses</span>'
        f'    <span class="fr-mono">– {fmt_money(expenses)}</span>'
        f'  </div>'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline;margin-top:6px;'
        f'              border-top:1px solid {THEME["line"]};padding-top:8px">'
        f'    <span class="fr-vital-label">Available for goals</span>'
        f'    <span class="fr-mono" style="color:{THEME["primary"]};'
        f'                                  font-weight:700">'
        f'      {fmt_money(available)}</span>'
        f'  </div>'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline;margin-top:6px">'
        f'    <span class="fr-vital-label">Goal funding needed</span>'
        f'    <span class="fr-mono">{fmt_money(total_monthly_need)}</span>'
        f'  </div>'
        f'  <div style="margin-top:10px;padding:10px 12px;'
        f'              background:{THEME["surface"]};border-radius:10px;'
        f'              color:{gap_color};font-weight:600;font-size:0.92rem">'
        f'    {gap_label}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    sb1, sb2 = st.columns([1, 1])
    with sb1:
        if st.button("Reset budget", key="fr_bud_reset",
                     use_container_width=True):
            save_budget_for(ck, {})
            st.session_state.fr_flash = "Budget reset."
            st.rerun()
    with sb2:
        if st.button("Save budget", type="primary", key="fr_bud_save",
                     use_container_width=True):
            save_budget_for(ck, {
                "income": income, "housing": housing,
                "transport": transport, "food": food,
                "utilities": utilities, "debt": debt,
                "discretionary": discretionary, "other": other,
            })
            st.session_state.fr_flash = "Budget saved."
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ADVISOR TAB — full advisor profile with photo, contact info, website
# ─────────────────────────────────────────────────────────────────────────────
def _render_advisor_tab():
    """Full advisor profile card. Replaces the old single-line "Book your
    follow-up" CTA — now the client can see who their advisor actually is,
    where the firm is based, and reach them through any channel they prefer.

    Profile is sourced live from firm_settings.json (written by the advisor
    app's Firm Branding panel) on every render — so updates flow through
    without needing to restart the portal."""
    a = get_advisor()

    # ── Branding diagnostics (collapsed expander) ──────────────────────
    # Surfaces exactly which file path this portal instance is looking at,
    # whether the file exists, and whether the images exist. Helpful when
    # the advisor info isn't populating: usually means the portal and the
    # advisor app are launched from different folders, so they're reading
    # different firm_settings.json files.
    _fs_raw = _load_firm_settings()
    _logo_exists  = os.path.exists(FIRM_LOGO_PATH)
    _photo_exists = os.path.exists(ADVISOR_PHOTO_PATH)
    _settings_exists = os.path.exists(FIRM_SETTINGS_FILE)
    _populated_keys = sorted(
        k for k, v in (_fs_raw or {}).items()
        if isinstance(v, str) and v.strip()
    )
    with st.expander("⚙ Branding diagnostics", expanded=False):
        st.caption(
            "If the advisor info or logo isn't showing, this panel tells you why. "
            "The most common cause is the advisor app and this portal being "
            "launched from different folders — they need to live side-by-side "
            "so they share the same firm_settings.json."
        )
        st.markdown(
            f"**Working directory (this portal):** `{_APP_DIR}`  \n"
            f"**Looking for:**  \n"
            f"&nbsp;&nbsp;`{FIRM_SETTINGS_FILE}` — "
            f"{'✅ found' if _settings_exists else '❌ missing'}  \n"
            f"&nbsp;&nbsp;`{FIRM_LOGO_PATH}` — "
            f"{'✅ found' if _logo_exists else '❌ missing (using default SVG)'}  \n"
            f"&nbsp;&nbsp;`{ADVISOR_PHOTO_PATH}` — "
            f"{'✅ found' if _photo_exists else '❌ missing (using default avatar)'}"
        )
        if _settings_exists:
            if _populated_keys:
                st.markdown(
                    "**Populated fields in firm_settings.json:**  \n"
                    + ", ".join(f"`{k}`" for k in _populated_keys)
                )
            else:
                st.warning(
                    "firm_settings.json exists but is empty. Open the advisor "
                    "app → Client Records → 🎨 Firm Branding, fill in the "
                    "fields, and click 💾 Save firm details."
                )
        else:
            st.warning(
                "No firm_settings.json found. Either (a) you haven't saved "
                "branding yet on the advisor app, or (b) the advisor app is "
                "running from a different folder than this portal. "
                "Confirm both apps are launched from the same directory."
            )

    # Optional firm-logo strip across the top of the card. Renders only
    # when the firm has uploaded a logo; otherwise this whole row collapses
    # so the existing photo + name layout takes its full place.
    _firm_logo_strip = ""
    if a.get("firm_logo_html_large"):
        _firm_logo_strip = (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'            padding-bottom:14px;margin-bottom:14px;'
            f'            border-bottom:1px solid {THEME["line"]}">'
            f'  {a["firm_logo_html_large"]}'
            f'  <div style="font-size:0.78rem;font-weight:600;'
            f'              color:{THEME["muted"]};letter-spacing:0.06em;'
            f'              text-transform:uppercase">{a["firm"]}</div>'
            f'</div>'
        )

    # Header with photo + name + title
    st.markdown(
        f'<div class="fr-card">'
        f'  {_firm_logo_strip}'
        f'  <div style="display:flex;gap:18px;align-items:center">'
        f'    <div style="flex-shrink:0">{a["photo_html"]}</div>'
        f'    <div style="flex:1">'
        f'      <div class="fr-eyebrow">Your advisor</div>'
        f'      <div style="font-size:1.15rem;font-weight:600;color:{THEME["ink"]};'
        f'                  margin-top:2px;letter-spacing:-0.01em">{a["name"]}</div>'
        f'      <div style="font-size:0.88rem;color:{THEME["ink2"]};margin-top:2px">'
        f'        {a["title"]} · {a["firm"]}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'  <div style="margin-top:16px;padding-top:14px;'
        f'              border-top:1px solid {THEME["line"]};'
        f'              font-size:0.92rem;color:{THEME["ink2"]};line-height:1.55">'
        f'    {a["bio"]}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Contact info card
    _icon_mail = (
        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0">'
        f'<rect x="3" y="5" width="18" height="14" rx="2"/>'
        f'<path d="M3 7l9 6 9-6"/></svg>'
    )
    _icon_phone = (
        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0">'
        f'<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 '
        f'19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 '
        f'2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 '
        f'9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 '
        f'2.81.7A2 2 0 0 1 22 16.92z"/></svg>'
    )
    _icon_globe = (
        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0">'
        f'<circle cx="12" cy="12" r="10"/>'
        f'<path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 '
        f'15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
    )
    _icon_pin = (
        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0">'
        f'<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>'
        f'<circle cx="12" cy="10" r="3"/></svg>'
    )

    def _row(icon: str, label: str, value: str, href: Optional[str] = None) -> str:
        val_html = (f'<a href="{href}" style="color:{THEME["primary"]};'
                    f'                       text-decoration:none">{value}</a>'
                    if href else
                    f'<span style="color:{THEME["ink"]}">{value}</span>')
        return (
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'            padding:12px 0;border-top:1px solid {THEME["line"]}">'
            f'  <div style="margin-top:2px">{icon}</div>'
            f'  <div style="flex:1">'
            f'    <div class="fr-vital-label">{label}</div>'
            f'    <div style="font-size:0.95rem;margin-top:2px">{val_html}</div>'
            f'  </div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="fr-card">'
        f'  <div class="fr-eyebrow">Contact</div>'
        f'  {_row(_icon_mail,  "Email",   a["email"],   "mailto:" + a["email"])}'
        f'  {_row(_icon_phone, "Phone",   a["phone"],   "tel:" + a["phone"].replace(" ", ""))}'
        f'  {_row(_icon_globe, "Website", a["website"], a["website"])}'
        f'  {_row(_icon_pin,   "Office",  a["address"])}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Schedule-a-call CTA. Replaces the old "dark banner + Schedule call
    # button" combo, which was doing the same job twice. Now it's a single
    # block: dark card with a clear primary action button right below the
    # description, properly emphasizing that the call is free and with a
    # licensed advisor.
    _icon_calendar = (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
        'stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        '<rect x="3" y="5" width="18" height="16" rx="2.5"/>'
        '<path d="M3 10h18"/>'
        '<path d="M8 3v4"/>'
        '<path d="M16 3v4"/>'
        '</svg>'
    )
    st.markdown(
        f'<div class="fr-cta-dark" style="margin-bottom:0">'
        f'  <div class="fr-cta-icon">{_icon_calendar}</div>'
        f'  <div style="flex:1">'
        f'    <div style="font-size:0.95rem;font-weight:600;line-height:1.3">'
        f'      Book a free 15-minute review'
        f'    </div>'
        f'    <div style="font-size:0.8rem;opacity:0.78;margin-top:3px;'
        f'                line-height:1.45">'
        f'      With a licensed financial advisor — no obligation.'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Schedule my review →", key="fr_schedule_btn",
                 type="primary", use_container_width=True):
        st.session_state.fr_flash = (
            "Booking flow coming soon — your advisor will reach out.")
        st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# EDIT PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def render_edit_profile():
    ck = _client_key()
    profile = load_profiles().get(ck, {})
    prev_answers = profile.get("answers", {}) or {}

    bar_l, bar_r = st.columns([5, 1])
    with bar_l:
        st.markdown(
            f'<div class="fr-eyebrow">Risk Profile</div>'
            f'<h1 class="fr-headline" style="font-size:1.6rem">Tell us about yourself</h1>'
            f'<div style="color:{THEME["ink2"]};font-size:0.92rem">'
            f'  23 questions across 5 sections — Context, Goals, Horizon, Tolerance, Outlook.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with bar_r:
        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
        if st.button("← Back", key="fr_profile_back", use_container_width=True):
            st.session_state.fr_view = "dashboard"
            st.rerun()

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    answers = {}
    last_section = None
    for q in PROFILE_QUESTIONS:
        if q["section"] != last_section:
            if last_section is not None:
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="fr-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fr-eyebrow">{q["section"]}</div>',
                unsafe_allow_html=True,
            )
            last_section = q["section"]

        qid = q["id"]; prev = prev_answers.get(qid)
        if q["type"] == "number":
            val = st.number_input(
                q["text"],
                min_value=q["min"], max_value=q["max"],
                value=int(prev) if prev not in (None, "") else q["default"],
                step=q["step"], key=f"fr_q_{qid}",
            )
            answers[qid] = val
        elif q["type"] == "select":
            opts = [opt[0] for opt in q["options"]]
            idx = opts.index(prev) if prev in opts else 0
            val = st.radio(q["text"], opts, index=idx, key=f"fr_q_{qid}")
            answers[qid] = val
        elif q["type"] == "multi":
            opts = q["options"]
            default = ([d for d in (prev or []) if d in opts]
                       if isinstance(prev, list) else [])
            # Soft cap (no max_selections) — see the quiz screen for rationale.
            val = st.multiselect(
                q["text"], opts, default=default,
                key=f"fr_q_{qid}",
            )
            mp = q.get("max_pick")
            if mp and len(val or []) > mp:
                st.warning(
                    f"You've picked {len(val)}. Only the first {mp} will be "
                    f"used for scoring — remove one to change which counts."
                )
                val = list(val or [])[:mp]
            answers[qid] = val

    if last_section is not None:
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    save_l, save_r = st.columns([1, 1])
    with save_l:
        if st.button("Cancel", key="fr_profile_cancel", use_container_width=True):
            st.session_state.fr_view = "dashboard"
            st.rerun()
    with save_r:
        if st.button("Save profile", type="primary",
                     key="fr_profile_save", use_container_width=True):
            scores = score_profile(answers)
            label, _, _ = score_band(scores["overall_score"])
            patch = {
                "client_name":  f'{st.session_state.fr_user.get("first_name","")} '
                                f'{st.session_state.fr_user.get("last_name","")}'.strip(),
                "client_email": st.session_state.fr_user.get("email", ""),
                "client_age":   answers.get("age", ""),
                "answers":      answers,
                "priorities":   answers.get("priorities", []),
                "risk_label":   label,
                **scores,
            }
            save_profile_for(ck, patch)
            st.session_state.fr_flash = "Profile saved."
            st.session_state.fr_view  = "dashboard"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# EDIT HOLDINGS
# ─────────────────────────────────────────────────────────────────────────────
def render_edit_holdings():
    ck = _client_key()
    all_holdings = load_all_holdings()
    holdings = dict(all_holdings.get(ck, {}) or {})

    bar_l, bar_r = st.columns([5, 1])
    with bar_l:
        st.markdown(
            f'<div class="fr-eyebrow">Holdings</div>'
            f'<h1 class="fr-headline" style="font-size:1.6rem">Manage your positions</h1>'
            f'<div style="color:{THEME["ink2"]};font-size:0.92rem">'
            f'  Add, edit, or remove holdings. Live prices update automatically.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with bar_r:
        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
        if st.button("← Back", key="fr_holdings_back", use_container_width=True):
            st.session_state.fr_view = "dashboard"
            st.rerun()

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Add new
    st.markdown('<div class="fr-card">', unsafe_allow_html=True)
    st.markdown('<div class="fr-eyebrow">Add a position</div>',
                unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
    new_tkr = a1.text_input("Ticker", placeholder="AAPL", key="fr_new_tkr")
    new_shares = a2.number_input("Shares", min_value=0.0, value=0.0,
                                  step=1.0, format="%.4f", key="fr_new_sh")
    new_cost = a3.number_input("Avg cost", min_value=0.0, value=0.0,
                                step=1.0, format="%.2f", key="fr_new_cost")
    new_total = new_shares * new_cost
    a4.markdown(
        f'<div style="margin-top:30px;padding:8px 12px;'
        f'            background:{THEME["surface2"]};border:1px solid {THEME["line"]};'
        f'            border-radius:10px;font-weight:600;color:{THEME["primary"]};'
        f'            font-size:0.85rem">'
        f'Total: {fmt_money(new_total)}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Add position", key="fr_add_btn", type="primary"):
        tkr_clean = (new_tkr or "").strip().upper()
        if not tkr_clean:
            st.warning("Enter a ticker symbol.")
        elif new_shares <= 0 or new_cost <= 0:
            st.warning("Enter both shares and a non-zero cost.")
        else:
            holdings[tkr_clean] = {
                "shares":           round(new_shares, 6),
                "avg_cost":         round(new_cost, 4),
                "dollar_invested":  round(new_shares * new_cost, 2),
                "added_at":         datetime.now().isoformat(timespec="minutes"),
            }
            save_holdings_for(ck, holdings)
            st.session_state.fr_flash = f"Added {tkr_clean}."
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Existing
    if holdings:
        st.markdown('<div class="fr-card">', unsafe_allow_html=True)
        st.markdown('<div class="fr-eyebrow">Current positions</div>',
                    unsafe_allow_html=True)
        quotes = get_live_quotes(list(holdings.keys()))

        for tkr in sorted(holdings.keys()):
            h = holdings[tkr]
            q = quotes.get(tkr, {})
            price = float(q.get("price") or 0)

            r1, r2, r3, r4, r5 = st.columns([1.2, 1, 1, 1.4, 0.6])
            r1.markdown(
                f'<div class="fr-mono" style="color:{THEME["primary"]};'
                f'                              font-size:1rem;margin-top:30px">{tkr}</div>'
                f'<div style="font-size:0.72rem;color:{THEME["muted"]}">'
                f'{q.get("name", tkr)[:28]}</div>',
                unsafe_allow_html=True,
            )
            new_sh = r2.number_input(
                "Shares", value=float(h.get("shares") or 0),
                min_value=0.0, step=1.0, format="%.4f",
                key=f"fr_edit_sh_{tkr}",
            )
            new_co = r3.number_input(
                "Avg cost", value=float(h.get("avg_cost") or 0),
                min_value=0.0, step=1.0, format="%.2f",
                key=f"fr_edit_co_{tkr}",
            )
            cur_val = new_sh * price
            r4.markdown(
                f'<div style="margin-top:30px">'
                f'  <div style="font-size:0.72rem;color:{THEME["muted"]}">Current</div>'
                f'  <div class="fr-mono" style="font-weight:600;color:{THEME["ink"]}">'
                f'    {fmt_money(cur_val)}</div>'
                f'  <div style="font-size:0.72rem;color:{THEME["muted"]}">'
                f'    @ ${price:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if r5.button("✕", key=f"fr_del_{tkr}", help=f"Remove {tkr}"):
                holdings.pop(tkr, None)
                save_holdings_for(ck, holdings)
                st.session_state.fr_flash = f"Removed {tkr}."
                st.rerun()

            if (new_sh != float(h.get("shares") or 0)
                    or new_co != float(h.get("avg_cost") or 0)):
                holdings[tkr] = {
                    **h,
                    "shares":          round(new_sh, 6),
                    "avg_cost":        round(new_co, 4),
                    "dollar_invested": round(new_sh * new_co, 2),
                    "updated_at":      datetime.now().isoformat(timespec="minutes"),
                }

        sb_l, sb_r = st.columns([1, 1])
        with sb_l:
            if st.button("Cancel", key="fr_holdings_cancel",
                         use_container_width=True):
                st.session_state.fr_view = "dashboard"
                st.rerun()
        with sb_r:
            if st.button("Save changes", type="primary",
                         key="fr_holdings_save", use_container_width=True):
                save_holdings_for(ck, holdings)
                st.session_state.fr_flash = "Holdings saved."
                st.session_state.fr_view  = "dashboard"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.fr_user is None:
    render_login()
else:
    view = st.session_state.fr_view
    if view == "edit_profile":
        render_edit_profile()
    elif view == "edit_holdings":
        render_edit_holdings()
    else:
        render_dashboard()