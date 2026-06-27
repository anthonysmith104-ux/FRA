"""client_portal.py — Foresight Risk Analytics client portal.

White-labelable wealth-firm client portal. Foresight Risk Analytics is the
product (this code); the firm using it (currently MRB Capital Group) is
defined by firm_settings.json — colors, copy, logos, advisor identity all
come from there via mrb_design.load_settings(). To onboard a new firm,
swap firm_settings.json for theirs and the portal repaints automatically.

Layout:
    • Brand crest + firm-name header bar
    • Navy + gold palette (from firm_settings.brand)
    • Crest-badge risk score visualization (1-99, navy/gold motif)
    • 2x2 Vitals grid (Risk Capacity, Risk Tolerance, Net Worth, Cash)
    • Financial Goals progress card
    • Tabs: Home / Financial Goals / Holdings / Advisor / My Info

Shared with the advisor app (app.py) via:
    • shared.py — scoring helpers, normalization
    • data_store.py — atomic JSON I/O backed by GitHub
    • mrb_design.py — design tokens, SVG generators, color/copy lookups
    • firm_settings.json — single source of truth for firm branding

Run:
    streamlit run client_portal.py

Files (anchored to this script's directory):
    ra_users.json           — user records
    risk_profiles.json      — risk profile + Q&A
    client_holdings.json    — per-client holdings
    firm_settings.json      — firm branding + design tokens (v2.4+)
"""
from __future__ import annotations

import os
import json
import math
import secrets
from datetime import datetime, date
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

# ── Firebase sign-in (Google / Facebook / email) ─────────────────────────────
# Verified sign-in for returning clients, replacing the old email-only lookup.
# Guarded so a missing/broken module never takes down the open quiz funnel —
# only the returning-client sign-in path degrades.
try:
    import firebase_auth  # type: ignore
    _FIREBASE_AVAILABLE = True
except Exception as _fb_err:  # pragma: no cover
    firebase_auth = None  # type: ignore
    _FIREBASE_AVAILABLE = False
    print(f"[firebase_auth] import failed: {_fb_err}")

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
    is_valid_email, normalize_email,
    score_to_label, score_to_allocation,
)

# Design system — colors, fonts, copy, and SVG helpers. All visual tokens
# come from firm_settings.json via load_settings(); the app stays brand-
# agnostic and can be repointed at a different firm by swapping the
# settings file. See mrb_design.py docstring for full details.
from mrb_design import (
    load_settings,
    resolve_color_key,
    pick_alignment_tier,
)

# All shared JSON I/O now goes through data_store, which transparently
# reads/writes to a GitHub-backed shared repo (configured in Streamlit
# secrets) so the client portal and advisor app see the same data.
# When secrets aren't configured (local dev), data_store falls back to
# local-disk JSON in the same paths shared.load_json used. Drop-in
# replacement — same signatures.
from data_store import (
    load_json   as _shared_load_json,
    update_json as _shared_update_json,
)

# ── DATA FILE LOCATIONS ──────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _data_path(name: str) -> str: return os.path.join(_APP_DIR, name)

USERS_FILE           = _data_path("ra_users.json")
PROFILES_FILE        = _data_path("risk_profiles.json")
CLIENT_HOLDINGS_FILE = _data_path("client_holdings.json")
CLIENT_GOALS_FILE    = _data_path("client_goals.json")
CLIENT_BUDGETS_FILE  = _data_path("client_budgets.json")

# ── ADVISOR & FIRM PROFILE ───────────────────────────────────────────────────
# All firm and advisor identity comes from firm_settings.json via mrb_design.
# The dict below is just the shape the rendering code expects — every value
# gets overwritten on module import by _load_firm_settings_into_advisor()
# reading the v2.4 nested schema. The fallback strings are intentionally
# obvious placeholders so a misconfigured deploy is visible at a glance
# rather than silently shipping someone else's brand.
ADVISOR = {
    "name":    "(advisor name unset)",
    "title":   "(title unset)",
    "firm":    "(firm name unset)",
    "email":   "",
    "phone":   "",
    "website": "",
    "address": "",
    "bio":     "",
    # Default photo — neutral SVG silhouette. Replaced at module-import
    # time if assets/advisor_photo.png is present. Colors below are
    # rebuilt from settings AFTER load_settings runs (see below), so
    # this initial value is harmless — it's replaced before render.
    "photo_svg": "",
}

# Load the design system. All subsequent reads of colors, fonts, and copy
# go through SETTINGS or the resolve_color_key / format_proposal_copy
# helpers. Reading this once at module scope means a settings change
# requires a container restart, which is the right granularity for
# brand changes.
SETTINGS = load_settings()


def _load_firm_settings_into_advisor():
    """Read firm + advisor identity from settings v2.4 nested schema.

    Maps settings["firm"] and settings["advisor"] into the ADVISOR dict
    that the rendering code consumes. Clean break from the legacy flat
    schema (advisor_name, firm_name, ...) — v2.4 uses firm.name,
    advisor.name, etc.
    """
    firm = SETTINGS.get("firm", {}) or {}
    advisor = SETTINGS.get("advisor", {}) or {}

    # Firm identity (rendered as the brand line in headers)
    if firm.get("name"):     ADVISOR["firm"]    = firm["name"].strip()
    if firm.get("website"):  ADVISOR["website"] = firm["website"].strip()

    # Advisor identity
    if advisor.get("name"):   ADVISOR["name"]  = advisor["name"].strip()
    if advisor.get("title"):  ADVISOR["title"] = advisor["title"].strip()
    if advisor.get("email"):  ADVISOR["email"] = advisor["email"].strip()
    if advisor.get("phone"):  ADVISOR["phone"] = advisor["phone"].strip()

    # Optional fields — not all firms will populate these
    if firm.get("address"):    ADVISOR["address"] = firm["address"].strip()
    if advisor.get("bio"):     ADVISOR["bio"]     = advisor["bio"].strip()


def _overlay_shared_identity():
    """Overlay the EDITABLE firm/advisor identity from the shared GitHub store
    on top of the committed defaults.

    load_settings() (mrb_design) only reads the local committed
    firm_settings.json — it never touches data_store — so edits the advisor
    makes in the advisor app's Firm Branding panel (which writes to the shared
    store) never reached the portal. This pulls those fields from the same
    shared store and applies them last, so the panel actually drives what the
    client sees. Design tokens (brand/typography/copy) still come from the
    committed file via load_settings(); only identity fields are overlaid.

    Handles both the v2.4 nested schema (firm.* / advisor.*) written by the
    patched advisor app and the legacy flat schema (advisor_name, advisor_bio,
    …) in case an older write is still in the store. Falls back silently to
    the committed values if the shared store is unavailable."""
    try:
        shared = _shared_load_json("firm_settings.json", default={}) or {}
    except Exception:
        return
    if not isinstance(shared, dict):
        return

    s_firm = shared.get("firm", {}) or {}
    s_adv  = shared.get("advisor", {}) or {}

    def _set(key, val):
        if val and str(val).strip():
            ADVISOR[key] = str(val).strip()

    # v2.4 nested schema (what the patched advisor app writes)
    _set("firm",    s_firm.get("name"))
    _set("website", s_firm.get("website"))
    _set("address", s_firm.get("address"))
    _set("name",    s_adv.get("name"))
    _set("title",   s_adv.get("title"))
    _set("email",   s_adv.get("email"))
    _set("phone",   s_adv.get("phone"))
    _set("bio",     s_adv.get("bio"))

    # Legacy flat schema (pre-patch advisor app), applied last as a fallback.
    _set("firm",    shared.get("firm_name"))
    _set("website", shared.get("firm_website"))
    _set("address", shared.get("firm_address"))
    _set("name",    shared.get("advisor_name"))
    _set("title",   shared.get("advisor_title"))
    _set("email",   shared.get("advisor_email"))
    _set("phone",   shared.get("advisor_phone"))
    _set("bio",     shared.get("advisor_bio"))


_load_firm_settings_into_advisor()
_overlay_shared_identity()


# ── Logo / photo asset loading ────────────────────────────────────────────────
# Load firm_logo.png and advisor_photo.png from disk if present, encode as
# base64 data URIs so they can be embedded directly in HTML/SVG markup
# without separate image requests. If a file is missing or can't be read,
# the helper returns None and the calling code falls back to the SVG
# default (hexagon mark for the logo, generic silhouette for the photo).
import base64 as _b64

def _load_image_as_data_uri(filename: str, mime: str = "image/png") -> Optional[str]:
    """Read a PNG file from the script's directory and return a data URI,
    or None if the file is missing or unreadable. Used for firm_logo.png
    and advisor_photo.png — both are seed assets committed to the repo.
    """
    p = _data_path(filename)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            raw = f.read()
        return f"data:{mime};base64,{_b64.b64encode(raw).decode('ascii')}"
    except OSError:
        return None


# These run once at module import. Globals stay None if the files aren't
# in the repo, and downstream code uses the SVG fallback.
FIRM_LOGO_DATA_URI    = _load_image_as_data_uri("firm_logo.png")
ADVISOR_PHOTO_DATA_URI = _load_image_as_data_uri("advisor_photo.png")

# Overlay images uploaded in the advisor app's Firm Branding panel. That
# panel base64-encodes the logo/photo into firm_settings.json (the same
# shared store that already carries the text identity), so an uploaded image
# flows through to the portal exactly like the name/bio do. The committed
# firm_logo.png / advisor_photo.png remain as fallback seeds when nothing
# has been uploaded.
try:
    _brand_fs   = _shared_load_json("firm_settings.json", default={}) or {}
    _brand_firm = _brand_fs.get("firm", {}) or {}
    _brand_adv  = _brand_fs.get("advisor", {}) or {}
    if _brand_firm.get("logo_data_uri"):
        FIRM_LOGO_DATA_URI = _brand_firm["logo_data_uri"]
        # Drop it to disk too, so the favicon (set_page_config below needs a
        # file path) reflects the uploaded logo rather than the seed.
        try:
            with open(_data_path("firm_logo.png"), "wb") as _lf:
                _lf.write(_b64.b64decode(FIRM_LOGO_DATA_URI.split(",", 1)[1]))
        except Exception:
            pass
    if _brand_adv.get("photo_data_uri"):
        ADVISOR_PHOTO_DATA_URI = _brand_adv["photo_data_uri"]
except Exception:
    pass

# Build the default advisor photo SVG using brand colors. This used to
# hardcode the teal Clinical palette (#0E5C5E → #0E7C86); now it pulls
# navy + cream from settings so the placeholder doesn't fight the brand
# layer when the real photo isn't uploaded yet.
_NAVY = resolve_color_key("brand.primary.navy", SETTINGS)
_GOLD = resolve_color_key("brand.accent.gold", SETTINGS)
def advisor_photo_svg(uid: str = "main") -> str:
    """Return the advisor photo (or silhouette fallback) as an 80x80 CIRCULAR
    SVG. `uid` makes the internal element ids unique per call site. Streamlit
    renders every tab into the DOM simultaneously, so reusing one id (the
    clipPath) across the home card and the Advisor-tab card made the second
    instance's circular clip silently fail and render as a square. Unique ids
    per usage keep every copy circular."""
    if ADVISOR_PHOTO_DATA_URI:
        cid = f"adv_photo_clip_{uid}"
        return (
            '<svg viewBox="0 0 80 80" width="80" height="80" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<defs><clipPath id="{cid}">'
            '<circle cx="40" cy="40" r="40"/></clipPath></defs>'
            f'<image href="{ADVISOR_PHOTO_DATA_URI}" '
            'x="0" y="0" width="80" height="80" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>'
            '</svg>'
        )
    gid = f"adv_bg_{uid}"
    return (
        '<svg viewBox="0 0 80 80" width="80" height="80" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{_NAVY}"/>'
        f'<stop offset="1" stop-color="{_GOLD}"/></linearGradient></defs>'
        f'<circle cx="40" cy="40" r="40" fill="url(#{gid})"/>'
        '<circle cx="40" cy="32" r="13" fill="#FFFFFF" opacity="0.95"/>'
        '<path d="M16 70 C 18 56, 28 50, 40 50 S 62 56, 64 70 Z" '
        'fill="#FFFFFF" opacity="0.95"/>'
        '</svg>'
    )

# Default (home/compact card) variant. The Advisor tab calls
# advisor_photo_svg("advisor") directly so its ids don't collide with this.
ADVISOR["photo_svg"] = advisor_photo_svg("main")

# Browser-tab favicon. Streamlit's page_icon accepts an emoji, a URL, a
# PIL Image, or a local file path. Prefer the firm's own logo at
# firm_logo.png (same asset already used inline via FIRM_LOGO_DATA_URI)
# so the tab favicon matches the in-app brand. Falls back to the
# stethoscope emoji only if the logo file is missing — keeps the app
# bootable in environments where branding assets aren't deployed yet.
_FAVICON_PATH = _data_path("firm_logo.png")
st.set_page_config(
    page_title=f'{SETTINGS["firm"]["name"]} · Risk Profile',
    page_icon=_FAVICON_PATH if os.path.exists(_FAVICON_PATH) else "🩺",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME — derived from firm_settings.json via mrb_design
# ─────────────────────────────────────────────────────────────────────────────
# Same dict shape as the legacy "Clinical" theme so all 170+ downstream
# THEME["..."] references continue to work without changes — only the
# values are different. When firm_settings.json changes (e.g. a different
# firm's brand swaps in), this dict rebuilds on next container restart
# and every styled element repaints automatically.
#
# Mapping rationale:
#   bg            → cream surface (was light gray)
#   surface       → white (cards)
#   surface2      → cream_warm (featured surfaces, advisor box)
#   line          → light border
#   ink/ink2/muted → navy-tinted text scale
#   primary       → NAVY (was teal) — brand structural color
#   primary_soft  → cream_warm (was light teal background)
#   accent        → GOLD (was deeper teal) — brand accent
#   healthy/caution/risk → semantic green/amber/red for alignment tiers
#   chip          → cream — neutral chip surface
_B = SETTINGS["brand"]
_S = SETTINGS["brand"]["semantic"]
THEME = {
    "bg":           _B["surface"]["cream"],
    "surface":      _B["surface"]["white"],
    "surface2":     _B["surface"]["cream_warm"],
    "line":         _B["border"]["light"],
    "ink":          _B["text"]["primary"],
    "ink2":         _B["text"]["secondary"],
    "muted":        _B["text"]["muted"],
    "primary":      _B["primary"]["navy"],
    "primary_soft": _B["surface"]["cream_warm"],
    "accent":       _B["accent"]["gold"],
    "healthy":      _S["success"],
    "healthy_soft": _S["success_bg"],
    "caution":      _S["warning"],
    "caution_soft": _S["warning_bg"],
    "risk":         _S["danger"],
    "risk_soft":    _S["danger_bg"],
    "chip":         _B["surface"]["cream"],
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+Pro:wght@400;500;600;700&display=swap');

        #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

        /* Hide Streamlit's auto-generated header-anchor link icons. They
           appear as a small chain icon next to every <h1>/<h2>/<h3> in
           markdown blocks and look like a broken/dead link to users. */
        .stApp h1 a, .stApp h2 a, .stApp h3 a,
        .stApp h4 a, .stApp h5 a, .stApp h6 a,
        [data-testid="stHeaderActionElements"],
        [data-testid="stMarkdownHeadingActionElements"] {{
            display: none !important;
        }}

        .stApp {{
            background: {THEME['bg']};
            color: {THEME['ink']};
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
            color: {THEME['ink']} !important;
            font-family: 'Source Serif Pro', Georgia, serif;
            font-weight: 500;
            letter-spacing: -0.01em;
        }}

        .fr-card {{
            background: {THEME['surface2']};
            border: 1.5px solid {THEME['primary']};
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
            background: {THEME['surface2']};
            border: 1.5px solid {THEME['primary']};
            border-radius: 14px;
            padding: 14px 14px;
            min-height: 102px;
            display: flex; flex-direction: column; gap: 8px;
            margin-bottom: 10px;
        }}
        /* Tiles tagged .fr-tablink navigate to another tab on click — see the
           _render_tab_link_bridge() JS helper. Give them a pointer cursor and
           a subtle hover lift so they read as tappable. */
        .fr-tablink {{ cursor: pointer; transition: box-shadow .15s ease; }}
        .fr-tablink:hover {{ box-shadow: 0 0 0 3px {THEME['primary_soft']}; }}
        /* Hidden helper buttons behind the Portfolio Performance pills — the
           pills are styled HTML spans; these do the real period switch when
           the JS bridge clicks them. */
        .st-key-fr_perf_btns {{ display: none !important; }}
        /* Advisor bio — real spacing between paragraphs (the source text's
           line breaks are otherwise collapsed by HTML). */
        .fr-bio p {{ margin: 0 0 0.7em; }}
        .fr-bio p:last-child {{ margin-bottom: 0; }}
        .fr-vital-label {{
            font-size: 0.65rem; font-weight: 600;
            color: {THEME['muted']};
            letter-spacing: 0.06em; text-transform: uppercase;
        }}
        .fr-vital-value {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 1.4rem; font-weight: 600; color: {THEME['ink']};
            letter-spacing: -0.01em; line-height: 1;
            font-variant-numeric: tabular-nums;
        }}
        .fr-vital-detail {{
            display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 2px 8px;
            font-size: 0.72rem; color: {THEME['muted']};
        }}
        .fr-mono {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
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
            font-size: 1.275rem; color: {THEME['ink2']}; margin-bottom: 4px;
        }}
        .fr-headline {{
            font-family: 'Source Serif Pro', Georgia, serif;
            font-size: 1.5rem; font-weight: 500; color: {THEME['ink']};
            letter-spacing: -0.015em; line-height: 1.18; margin: 0 0 14px 0;
        }}
        .fr-headline-accent {{ color: {THEME['primary']}; }}
        /* Questionnaire question text — sized via a class with !important so
           it reliably overrides Streamlit's default <h2> sizing. Inline
           font-size on injected markdown headings was being ignored, which is
           why earlier inline bumps had no visible effect. */
        .fr-question {{
            font-size: 2.1rem !important;
            font-weight: 600;
            color: {THEME['ink']};
            letter-spacing: -0.015em;
            line-height: 1.3;
            margin: 6px 0 22px;
            overflow-wrap: break-word;
            word-break: normal;
            hyphens: none;
        }}

        /* Inputs — blue outline on the whole box. Text and date fields are a
           single box, so we border that. */
        .stTextInput > div > div,
        .stDateInput > div > div {{
            background-color: {THEME['surface']} !important;
            border: 1.5px solid {THEME['primary']} !important;
            border-radius: 10px !important;
        }}
        /* Number inputs render the value field and the +/- steppers as TWO
           sibling boxes. Border the wrapper that holds both, and strip the
           inner borders/backgrounds so they read as one unified control
           (value on the left, steppers on the right). */
        .stNumberInput > div {{
            background-color: {THEME['surface']} !important;
            border: 1.5px solid {THEME['primary']} !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            gap: 0 !important;
        }}
        .stNumberInput > div > div {{
            border: none !important;
            background-color: transparent !important;
        }}
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stDateInput input {{
            background-color: transparent !important;
            color: {THEME['ink']} !important;
            border: none !important;
            font-family: 'Inter', sans-serif;
        }}
        .stTextArea textarea {{
            background-color: {THEME['surface']} !important;
            color: {THEME['ink']} !important;
            border: 1.5px solid {THEME['primary']} !important;
            border-radius: 10px !important;
            font-family: 'Inter', sans-serif;
        }}
        .stTextInput > div > div:focus-within,
        .stNumberInput > div:focus-within,
        .stDateInput > div > div:focus-within {{
            box-shadow: 0 0 0 3px {THEME['primary_soft']} !important;
        }}
        .stSelectbox > div > div, .stMultiSelect > div > div {{
            background-color: {THEME['surface']} !important;
            border: 1.5px solid {THEME['primary']} !important;
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

        /* Larger answer-option text — easier to read and tap on mobile. */
        .stRadio [data-baseweb="radio"] div {{
            font-size: 1.25rem !important;
            line-height: 1.45 !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background: transparent;
            border-bottom: 1px solid {THEME['line']};
            gap: 14px;                  /* tighter spacing — fits all 5 tabs on mobile */
            padding: 0 2px;             /* minimal edge padding */
            margin-bottom: 18px;        /* breathing room before tab content */
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {THEME['muted']};
            background: transparent;
            font-weight: 600;
            font-size: 0.88rem;         /* slightly smaller — more tabs per row */
            padding: 12px 2px;          /* taller hit-area, slim horizontal */
            min-height: auto;
            letter-spacing: 0.005em;    /* tightened from 0.01em for narrow widths */
            transition: color 0.15s ease;
            white-space: nowrap;        /* keep "Financial Goals" on one line */
        }}
        /* Desktop gets a bit more room — re-expand spacing above ~520px wide. */
        @media (min-width: 520px) {{
            .stTabs [data-baseweb="tab-list"] {{ gap: 22px; }}
            .stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; padding: 12px 4px; }}
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
            border: 1.5px solid {THEME['primary']};
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
            border: 1.5px solid {THEME['primary']} !important;
            border-left: 3px solid {THEME['primary']} !important;
            color: {THEME['ink']} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stDataFrame"] {{
            background: {THEME['surface']};
            border-radius: 14px;
            border: 1.5px solid {THEME['primary']};
            overflow: hidden;
        }}

        .js-plotly-plot, .plot-container {{ background: transparent !important; }}

        .block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 760px; }}

        /* ── Mobile text-clipping hardening ──────────────────────────────
           On narrow screens, side-by-side label/value rows
           (justify-content:space-between) and long unbroken strings (big
           dollar figures, emails) could run off the right edge and get
           clipped instead of wrapping. These make all in-card content wrap
           to fit; they engage only when content would otherwise overflow,
           so the desktop layout is unchanged. */
        [data-testid="stMarkdownContainer"] {{ overflow-wrap: break-word; }}
        [data-testid="stMarkdownContainer"] div[style*="space-between"] {{
            flex-wrap: wrap;
            gap: 2px 12px;
        }}

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

def find_user_by_uid(uid: str) -> Optional[dict]:
    """Find an account previously linked to this Firebase UID. Lets a returning
    client match even if their provider email differs from the email their
    record is keyed under (once the link has been established)."""
    if not uid:
        return None
    users = load_users()
    for v in users.values():
        if isinstance(v, dict) and v.get("firebase_uid") == uid:
            return v
    return None

def register_user(first: str, last: str, email: str, phone: str = "") -> tuple[bool, str]:
    """Atomic upsert under a single lock.

    Idempotency note (bug fix for "email already exists on first try"):
    data_store.update_json uses optimistic locking with retry. If the
    very first PUT succeeds at the GitHub end but the success response
    is lost to the client (network blip, proxy timeout, etc.), the
    retry path re-reads the just-committed state — which now contains
    the user we *just* wrote — and the second call to _mutate would
    see the email as "already taken" and report a spurious conflict.
    Users hit this and worked around it by registering with a second
    email; the original email was left as a ghost record they could
    still sign in with.

    Mitigation: _mutate compares the existing record's identifying
    fields against the new_user we're attempting to write. If they
    match exactly, the "conflict" is our own retry and we return
    success. A genuine conflict — same email key registered by a
    different person — still surfaces correctly because at least one
    identifying field (typically first_name, last_name, or phone)
    will differ.
    """
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
        existing = users.get(key)
        if existing is not None:
            # Compare every identifying field. ALL must match for this
            # to be safely treated as our own retry. Using a tuple of
            # the four fields gives a single comparison that's robust
            # to dict-ordering differences and to extra fields that
            # downstream code (update_user, HubSpot sync, etc.) may
            # have added on top of the original record.
            _our_fingerprint = (
                new_user["first_name"],
                new_user["last_name"],
                new_user["phone"],
                new_user["created_at"],
            )
            _their_fingerprint = (
                existing.get("first_name", ""),
                existing.get("last_name", ""),
                existing.get("phone", ""),
                existing.get("created_at", ""),
            )
            if _our_fingerprint == _their_fingerprint:
                # Our own write coming back through the retry path —
                # silently succeed (the write already landed).
                return
            conflict["exists"] = True
            return
        users[key] = new_user
    _shared_update_json(USERS_FILE, _mutate)
    if conflict["exists"]: return False, "An account with this email already exists."
    return True, "Account created."

def update_user(email: str, patch: dict) -> tuple[bool, str]:
    """In-place update of an existing user record, atomic under USERS_FILE lock.

    `email` is the canonical key (lowercased, stripped) — this is the field
    we never change, since it's also how the client signs in. Anything else
    on the user dict can be patched: first_name, last_name, phone, address,
    zip, age.

    Returns (ok, message). Message is a short user-facing explanation when
    ok is False; empty when ok is True.
    """
    key = normalize_email(email)
    if key is None:
        return False, "Invalid email — cannot identify the user record."
    found = {"yes": False}
    def _mutate(users, k=key, p=dict(patch)):
        if k not in users:
            return
        found["yes"] = True
        # Drop email from the patch if present so a malformed patch
        # can't accidentally re-key the entry. Trim whitespace on
        # string fields to keep the data clean.
        p.pop("email", None)
        for fk, fv in p.items():
            if isinstance(fv, str):
                fv = fv.strip()
            users[k][fk] = fv
        users[k]["updated_at"] = datetime.now().isoformat(timespec="minutes")
    _shared_update_json(USERS_FILE, _mutate)
    if not found["yes"]:
        return False, "User not found."
    return True, ""


# ── REFERRALS ────────────────────────────────────────────────────────────────
# Each client gets a short, opaque referral_code (NOT their email — the email
# is PII and must never travel in a shareable URL). The invite link carries
# ?ref=<code>; at signup we resolve the code back to the referring client and
# record the relationship on BOTH records, in a single atomic write to
# ra_users.json (so it's consistent and visible to the advisor app, which
# reads the same shared file):
#   • on the NEW client:      referred_by (referrer's email key), referred_by_name, referred_at
#   • on the REFERRING client: referrals (list of {email,name,at}) + referrals_sent (count)
# referrals_sent is denormalized for easy advisor-side display; it can drift if
# a referred client is later deleted, in which case it can be recomputed by
# scanning ra_users.json for referred_by == that client.

def get_or_create_referral_code() -> str:
    """Return the current client's referral code, generating + persisting one
    on first use. Reads from the session record when present to avoid a store
    round-trip on every render; only hits the store when a code must be made
    (or backfilled for clients created before this feature existed)."""
    u = st.session_state.fr_user or {}
    code = (u.get("referral_code") or "").strip()
    if code:
        return code
    key = (u.get("email") or "").lower()
    if not key:
        return ""
    made = {"code": ""}
    def _mutate(users):
        rec = users.get(key)
        if not rec:
            return
        existing = (rec.get("referral_code") or "").strip()
        if existing:
            made["code"] = existing
            return
        used = {(v.get("referral_code") or "") for v in users.values()
                if isinstance(v, dict)}
        c = secrets.token_hex(4)        # 8 hex chars, URL-safe, non-PII
        while c in used:
            c = secrets.token_hex(4)
        rec["referral_code"] = c
        made["code"] = c
    _shared_update_json(USERS_FILE, _mutate)
    if made["code"] and st.session_state.fr_user is not None:
        st.session_state.fr_user["referral_code"] = made["code"]
    return made["code"]


def record_referral(new_user_key: str, ref_code: str) -> None:
    """At signup: attribute the new client to whoever owns ref_code. Single
    atomic mutate over ra_users.json updating both records. Idempotent (won't
    double-count), and ignores unknown codes and self-referrals."""
    ref_code = (ref_code or "").strip()
    new_user_key = (new_user_key or "").lower()
    if not ref_code or not new_user_key:
        return

    def _mutate(users):
        referrer_key = None
        for k, v in users.items():
            if isinstance(v, dict) and (v.get("referral_code") or "") == ref_code:
                referrer_key = k
                break
        if not referrer_key or referrer_key == new_user_key:
            return  # unknown code, or someone using their own link
        nu = users.get(new_user_key)
        if not nu or nu.get("referred_by"):
            return  # missing, or already attributed
        ref = users.get(referrer_key) or {}
        ref_name = f'{ref.get("first_name","")} {ref.get("last_name","")}'.strip()
        nu_name  = f'{nu.get("first_name","")} {nu.get("last_name","")}'.strip()
        now = datetime.now().isoformat(timespec="minutes")
        nu["referred_by"]      = referrer_key
        nu["referred_by_name"] = ref_name
        nu["referred_at"]      = now
        lst = ref.get("referrals") or []
        if not any(isinstance(x, dict) and x.get("email") == new_user_key for x in lst):
            lst.append({"email": new_user_key, "name": nu_name, "at": now})
        ref["referrals"]      = lst
        ref["referrals_sent"] = len(lst)
        users[referrer_key]   = ref
    _shared_update_json(USERS_FILE, _mutate)


def referrals_sent_count(client_key: str) -> int:
    """How many people this client has referred who went on to sign up.
    Reads the denormalized count, falling back to the length of the list."""
    key = (client_key or "").lower()
    if not key:
        return 0
    rec = load_users().get(key) or {}
    n = rec.get("referrals_sent")
    if isinstance(n, int):
        return n
    lst = rec.get("referrals")
    return len(lst) if isinstance(lst, list) else 0


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
    down the whole panel.

    yfinance occasionally hands back the most-recent daily bar with a NaN
    Close — an incomplete/holiday bar, or thin-volume ETFs like SGOV/AVUV —
    which used to surface in the UI as "$nan" and "nan%". We now:
      • drop NaN closes before indexing, so price = last *valid* close;
      • if history yields nothing usable, fall back to previousClose via
        fast_info / info;
      • guard the final values so NaN can never reach the panel.
    Net effect: a holding without a fresh live print defaults to its previous
    close instead of showing nothing.
    """
    import math as _math
    import yfinance as yf

    def _clean(x: float) -> float:
        try:
            x = float(x)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if _math.isnan(x) else x

    out = {}
    for tk in tickers_tuple:
        try:
            t = yf.Ticker(tk)
            hist = t.history(period="5d")
            closes = (
                [c for c in hist["Close"].dropna().tolist()]
                if hist is not None and not hist.empty and "Close" in hist
                else []
            )

            price = prev = 0.0
            if len(closes) >= 2:
                prev, price = _clean(closes[-2]), _clean(closes[-1])
            elif len(closes) == 1:
                price = prev = _clean(closes[-1])

            # Fallback: no usable history → pull previous close from
            # fast_info, then info. Covers tickers whose 5d history comes
            # back empty or all-NaN.
            if not price:
                pc = 0.0
                try:
                    fi = getattr(t, "fast_info", None)
                    if fi is not None:
                        pc = _clean(fi.get("last_price")
                                    or fi.get("previous_close") or 0)
                except Exception:
                    pc = 0.0
                if not pc:
                    try:
                        info = t.info or {}
                        pc = _clean(info.get("regularMarketPrice")
                                    or info.get("previousClose") or 0)
                    except Exception:
                        pc = 0.0
                price = prev = pc

            # Final guard: never let a zero/NaN price ride alongside a good
            # prev (or vice-versa); a missing price defaults to prev close.
            if not price:
                price = prev
            if not prev:
                prev = price

            chg = price - prev
            pct = (chg / prev * 100) if prev else 0.0

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
# RISK-FREE RATE  (default assumed return for goal projections)
# ─────────────────────────────────────────────────────────────────────────────
# The Financial Goals tab can grow the already-saved balance and future
# monthly contributions at an assumed rate of return when working out how much
# a client needs to set aside each month. That rate defaults to the risk-free
# rate: ^IRX, the 13-week US T-bill yield, the conventional risk-free proxy.
# It's quoted in percent, so divide by 100. RISK_FREE_FALLBACK covers cold
# starts / fetch failures (Streamlit Cloud spins containers down) — keep it
# roughly current.
RISK_FREE_FALLBACK = 0.043  # ~4.3%

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_risk_free_rate(ticker: str = "^IRX") -> float:
    """Current risk-free rate as a decimal (0.043 == 4.3%). For long-horizon
    goals you may prefer to horizon-match with ^FVX (5y) or ^TNX (10y)."""
    import math as _math
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        closes = (hist["Close"].dropna().tolist()
                  if hist is not None and not hist.empty and "Close" in hist
                  else [])
        if closes:
            rate = float(closes[-1]) / 100.0
            if not _math.isnan(rate) and 0 < rate < 0.25:  # sanity guard
                return rate
    except Exception:
        pass
    return RISK_FREE_FALLBACK

def risk_free_pct() -> float:
    """Risk-free rate as a percent (e.g. 4.3), for prefilling rate inputs."""
    return round(get_risk_free_rate() * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# GOAL PROJECTION MATH
# ─────────────────────────────────────────────────────────────────────────────
# Monthly compounding, monthly contributions, end-of-period (ordinary annuity).
# annual_rate is a decimal (0.043 == 4.3%); 0.0 reproduces the old straight-
# line "remaining / months" behaviour.
def goal_future_value(saved: float, monthly: float, months: int,
                      annual_rate: float) -> float:
    """Projected value of the current balance + future monthly contributions
    at the target date, both grown at annual_rate."""
    n = max(0, int(months))
    r = (annual_rate or 0.0) / 12.0
    if n == 0:
        return float(saved)
    if r == 0:
        return float(saved) + float(monthly) * n
    growth = (1 + r) ** n
    return float(saved) * growth + float(monthly) * ((growth - 1) / r)

def goal_required_monthly(target: float, saved: float, months: int,
                          annual_rate: float) -> float:
    """Monthly contribution needed to reach `target`, growing both the
    already-saved balance and future contributions at annual_rate. Returns 0
    when the balance alone already grows past the target."""
    n = max(1, int(months))
    r = (annual_rate or 0.0) / 12.0
    if r == 0:
        return max(0.0, (float(target) - float(saved)) / n)
    growth = (1 + r) ** n
    annuity_factor = (growth - 1) / r
    pmt = (float(target) - float(saved) * growth) / annuity_factor
    return max(0.0, pmt)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_QUESTIONS = [
    # ─────────────────────────────────────────────────────────────────────
    # All question `text` fields are phrased as proper questions ending
    # in "?" — clients found a few of the legacy fragments ("Your view
    # on US equity markets...") read awkwardly because the answer
    # context didn't make clear they were prompts. Every entry is now
    # a complete interrogative.
    # ─────────────────────────────────────────────────────────────────────
    {"id": "age", "section": "Context",
     "text": "What is your current age?",
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
     "text": "What is your household annual income (gross)?",
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
     "text": "What is your liquid cash (excluding retirement)?",
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

    # ── Goals: kept as a smaller block (specific goal & timeline questions
    # were removed in the 2026-04-30 questionnaire trim — feedback was that
    # they duplicated context already captured in withdrawal_horizon and the
    # advisor's own kickoff conversation). What remains targets dimensions
    # the questionnaire is uniquely positioned to capture: income replacement
    # ratio in retirement, and the strength of legacy intent.
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
     "text": "When will you need access to funds within your investment / retirement portfolio?",
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
     "text": "How many months of expenses do you have in cash outside your investment / retirement portfolio?",
     "type": "select", "options": [
        ("Less than 1 month",  20),
        ("1 – 3 months",       40),
        ("3 – 6 months",       60),
        ("6 – 12 months",      75),
        ("More than 12 months",85),
    ]},
    # major_expense was removed in the 2026-04-30 questionnaire trim — the
    # specific 3-year-horizon expense question added marginal scoring signal
    # over what the withdrawal_horizon and emergency_fund questions already
    # capture, and clients found it confusing when the answer didn't match
    # their actual mental model of upcoming spending.

    # drawdown_reaction and experience were removed in the 2026-04-30
    # questionnaire trim. drawdown_reaction was a hypothetical that
    # over-weighted self-perceived behavior; experience adds subjective
    # signal we don't act on. The remaining tolerance questions
    # (loss_floor, growth_vs_safety) measure tolerance through concrete
    # numeric tradeoffs which clients answer more consistently.
    {"id": "loss_floor", "section": "Tolerance",
     "text": "What is the largest one-year loss you could accept before changing strategy?",
     "type": "select", "options": [
        ("5% or less",         20),
        ("Up to 10%",          40),
        ("Up to 20%",          60),
        ("Up to 35%",          80),
        ("More than 35%",      95),
    ]},
    {"id": "growth_vs_safety", "section": "Tolerance",
     "text": "Of these portfolios, which best matches your preference?",
     "type": "select", "options": [
        ("Avg +4% · best year +6%  / worst year -2%",    20),
        ("Avg +6% · best year +12% / worst year -8%",    45),
        ("Avg +8% · best year +20% / worst year -18%",   65),
        ("Avg +9% · best year +30% / worst year -30%",   85),
    ]},
    # drawdown_history (added 2026): an actual-past-behavior question.
    # Revealed behavior predicts future behavior better than the
    # hypothetical drawdown_reaction removed in the 2026-04-30 trim. The
    # "haven't been through one" option maps to None, so the scoring loop
    # skips it (no penalty/reward for inexperience) and tolerance falls
    # back to loss_floor + growth_vs_safety for that client.
    {"id": "drawdown_history", "section": "Tolerance",
     "text": "In a past market downturn (2022, 2020, or 2008), what did you actually do?",
     "type": "select", "options": [
        ("Sold most or all of my investments",        20),
        ("Sold some or shifted to safer holdings",     40),
        ("Did nothing — stayed the course",            70),
        ("Bought more while prices were down",         90),
        ("I haven't invested through a downturn yet", None),
    ]},
    # market_view, inflation_concern, and recession_concern were removed
    # in the 2026-04-30 follow-up trim. These were "outlook" questions
    # asking the client to forecast macro conditions — useful in theory
    # but in practice they introduced more noise than signal: clients
    # tended to answer based on whatever they read in the news that
    # week, and the scoring weight (15% of overall) was big enough to
    # shift their risk profile based on transient mood. The remaining
    # outlook questions (esg_preference, priorities) capture preferences
    # that are stable over time.
    #
    # 2026-04-30 follow-up: esg_preference removed too. Advisor feedback
    # was that the standalone ESG question over-weighted what's a niche
    # preference for most clients, while ESG / values alignment remains
    # available as one of the eight options in the priorities multi-pick
    # — so a client who genuinely cares about it can still flag it there.
    {"id": "priorities", "section": "Outlook",
     "text": "Which of these matter MOST to you? (pick exactly 3)",
     "type": "multi", "options": [
        "Capital preservation",
        "Steady income / dividends",
        "Long-term growth",
        "Tax efficiency",
        "Inflation protection",
        "Liquidity / flexibility",
        "ESG / values alignment",
        "Estate / legacy planning",
    ], "min_pick": 3, "max_pick": 3},
]


def score_profile(answers: dict) -> dict:
    """Capacity 60% + Tolerance 40% — outlook dropped in 2026-04-30 trim.

    Goals questions feed into capacity because what the money is FOR affects
    how much risk the portfolio reasonably needs to take. Wealth-building and
    early-FI goals push capacity up (they require growth); preservation and
    short-timeline goals pull it down (they require safety).
    """
    section_scores = {"capacity": [], "tolerance": [], "outlook": []}
    # Capacity = the financial cushion / horizon that lets the portfolio
    # take risk. After 2026-04-30 trim: dropped major_expense,
    # primary_goal, goal_amount, goal_timeline.
    capacity_qs  = {"occupation","income_band","income_stability","net_worth",
                    "withdrawal_horizon","withdrawal_rate","emergency_fund",
                    # Goals
                    "income_replacement","legacy_intent"}
    # Tolerance = the client's emotional/behavioral capacity for
    # volatility. After 2026-04-30 trim: dropped drawdown_reaction
    # (hypothetical-behavior bias) and experience (subjective signal).
    tolerance_qs = {"loss_floor","growth_vs_safety","drawdown_history"}
    # Outlook = client's macro view. After 2026-04-30 follow-up trim:
    # dropped market_view, inflation_concern, recession_concern. The
    # set is empty for now — esg_preference and priorities don't fit
    # the "outlook" forecasting bucket; they're more about preferences
    # and feed scoring elsewhere. With nothing in the outlook bucket,
    # the overall score weighting collapses to capacity 50% +
    # tolerance 35% (rebalanced below).
    outlook_qs   = set()

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
    # Weighting: with the 2026-04-30 follow-up trim outlook has no
    # questions feeding it, so it's dropped from the overall formula.
    # Capacity and tolerance rebalance from 50/35 (out of 85) to a
    # clean 60/40 split. If outlook questions are added back later,
    # restore the third term — old formula was 0.50*cap + 0.35*tol +
    # 0.15*out. The outlook_score field is still returned in the
    # result dict (defaults to 50) so any UI reading it still works.
    # Capacity is a hard ceiling: the 60/40 blend governs while tolerance
    # is at or below capacity, but tolerance can never lift the score above
    # what the client can financially afford. (cap < tol -> pinned at cap;
    # tol < cap -> blend, which already sits below cap.)
    blended = 0.60*cap + 0.40*tol
    overall = int(round(max(1, min(99, min(blended, cap)))))
    cap_i, tol_i = int(round(cap)), int(round(tol))
    return {
        "overall_score":   overall,
        "capacity_score":  cap_i,
        "tolerance_score": tol_i,
        "outlook_score":   int(round(out)),
        "band":            score_band(cap_i, tol_i)[0],
    }


def score_band(capacity: int, tolerance: int) -> tuple[str, str, str]:
    """(label, hex, soft_bg) — risk profile from a capacity × tolerance matrix.

    Five Schwab-aligned tiers. Capacity is the ceiling: tolerance positions
    the client within what their capacity can support but can never lift the
    band above it. When tolerance sits at or below capacity the two tiers are
    blended (half-up); when tolerance exceeds capacity the band is pinned to
    the capacity tier. Banding on the two axes jointly (rather than on the
    single blended score) avoids the central-clustering that a 14-item
    average produces, so clients actually separate. No diagnostic or
    evaluative language; these describe an *investing posture*, not a
    judgment about the client."""
    bands = ("Conservative", "Moderately Conservative", "Moderate",
             "Moderately Aggressive", "Aggressive")
    def _tier(s: int) -> int:
        return 0 if s < 38 else 1 if s < 50 else 2 if s < 62 else 3 if s < 74 else 4
    ct, tt = _tier(capacity), _tier(tolerance)
    idx = ct if ct < tt else min(ct, (ct + tt + 1) // 2)
    return bands[idx], THEME["primary"], THEME["primary_soft"]


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL PRIMITIVES — direct ports of the prototype's SVG components
# ─────────────────────────────────────────────────────────────────────────────
def logo_mark(color: str = None, size: int = 26) -> str:
    """Returns either:
    - An <img> tag wrapping the firm's PNG logo (if firm_logo.png is in the
      repo and loaded successfully into FIRM_LOGO_DATA_URI), OR
    - The default hexagon-with-pulse SVG glyph (the original prototype mark).

    Either output renders inline at `size` x `size` pixels with display:block,
    so the call sites' surrounding flex layout stays unchanged.
    """
    if FIRM_LOGO_DATA_URI:
        # The firm_logo.png artwork sits off-center inside its square canvas
        # (more whitespace on the left than the right), so the rendered image
        # appears shifted right. Nudge it left by ~7.5% of its width to
        # optically center the artwork. Scales with `size` so it works at
        # both the 160px welcome mark and the 26px header mark.
        nudge = round(size * 0.075, 1)
        return (
            f'<img src="{FIRM_LOGO_DATA_URI}" '
            f'width="{size}" height="{size}" '
            f'alt="Firm logo" '
            f'style="display:block;object-fit:contain;'
            f'        margin-left:-{nudge}px"/>'
        )
    color = color or THEME["primary"]
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="display:block">'
        f'<path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
        f'<path d="M6 12 L9 12 L10.5 9 L12 15 L13.5 11 L15 13 L18 13" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# Public website the header brandmark links to. Prefer the website set in the
# advisor app's Firm Branding panel (synced via shared identity); fall back to
# the configured default if none is set.
FIRM_WEBSITE_URL = "https://www.mrb-capital-group.com/"

# Advisor scheduling page (HubSpot Meetings, synced to Google Calendar). The
# "Schedule my review" button opens this with the client's name/email
# pre-filled; HubSpot logs the booking against the contact and writes the
# event to the connected Google Calendar. Change the slug here if it moves.
SCHEDULE_URL = "https://meetings-na2.hubspot.com/anthony-smith"

# Advisor's dot.cards digital business card. The "Save my contact" button in
# the Advisor tab opens this; the client lands on the dot.cards profile and
# can tap to save the contact (no app needed). Change the slug if it moves.
DOT_CARD_URL = "https://dot.cards/anthony_smith_795"

# Public URL of THIS portal — the page someone lands on to register and take
# the assessment. Used by the "Share / Invite" card so a client can text or
# email a link to someone else. Leave as "" to auto-detect the live URL in the
# browser (origin + path); set it to the deployed Streamlit Cloud URL for a
# stable, canonical invite link that won't carry along any session query string.
PORTAL_URL = ""

# Client agreement shown at registration as an in-app popup. Replace the
# placeholder body text below with your CCO/counsel-approved language before
# launch — this code provides the popup and records acceptance (version +
# timestamp); it does not constitute the agreement and is not legal advice.
# Bump TOS_VERSION whenever the text changes so the audit trail stays accurate.
TOS_VERSION = "2026-06-01"

TERMS_TEXT = """
**Placeholder — replace with your approved Terms & Conditions. Not legal advice.**

**1. Acceptance.** By creating an account and checking the agreement box, you agree to these Terms and to the Privacy Policy.

**2. Nature of the service.** This portal provides an educational risk-profile assessment and related information. It is not investment, legal, or tax advice and is not an offer to buy or sell any security. Advisory services are provided only under a separate written agreement with MRB Capital Group.

**3. No guarantees.** All investing involves risk, including possible loss of principal. Risk-profile results are estimates based on the information you provide and do not guarantee any outcome.

**4. Your information.** You agree to provide accurate information and to keep your contact details current. Your email is used to sign in.

**5. Electronic communications.** You consent to receive communications and disclosures electronically.

**6. Privacy.** Your information is handled as described in the Privacy Policy.

**7. Limitation of liability.** To the fullest extent permitted by law, MRB Capital Group is not liable for indirect or consequential damages arising from use of this portal. _[Confirm with counsel.]_

**8. Governing law.** These Terms are governed by the laws of [STATE]. _[Confirm with counsel.]_

**9. Changes.** These Terms may be updated; the version shown reflects the current terms.

**10. Contact.** Questions? Reach your advisor through the Advisor tab.
"""

PRIVACY_TEXT = """
**Placeholder — replace with your approved Privacy Policy. Not legal advice.**

**What we collect.** Name, email, phone, optional address/ZIP, age, and your risk-questionnaire answers.

**How we use it.** To generate your risk profile, let you sign in, and allow your advisor to follow up with you.

**Sharing.** Information may be shared with the service providers that operate this platform (for example, the firm's CRM) and is not sold. _[Confirm provider list with counsel.]_

**Security.** Reasonable safeguards are used to protect your information.

**Your choices.** You can request to update or delete your information by contacting your advisor.

**Contact.** Reach your advisor through the Advisor tab.
"""

# Overlay editable agreement text from the advisor app (legal_content.json in
# the shared store) on top of the placeholder defaults above, so edits made in
# the advisor portal's PDF Content tab flow through here. Falls back to the
# placeholders if nothing is saved or the store is unavailable.
try:
    _legal_fs = _shared_load_json("legal_content.json", default={}) or {}
    if (_legal_fs.get("terms")   or "").strip(): TERMS_TEXT   = _legal_fs["terms"]
    if (_legal_fs.get("privacy") or "").strip(): PRIVACY_TEXT = _legal_fs["privacy"]
    if (_legal_fs.get("version") or "").strip(): TOS_VERSION  = _legal_fs["version"]
except Exception:
    pass


def _agreement_popup(title: str, body_md: str, close_key: str, version: str = None):
    """Show an agreement document as a modal popup. Uses st.dialog when the
    Streamlit version supports it (1.37+), falling back to an inline expander
    on older versions so the app never crashes."""
    def _render():
        if version:
            st.caption(f"Version {version}")
        st.markdown(body_md)
        if st.button("Close", key=close_key, use_container_width=True):
            st.rerun()
    if hasattr(st, "dialog"):
        st.dialog(title)(_render)()
    else:
        with st.expander(title, expanded=True):
            _render()

def _firm_url() -> str:
    w = (ADVISOR.get("website") or "").strip()
    if w:
        return w if w.startswith(("http://", "https://")) else "https://" + w
    return FIRM_WEBSITE_URL

def firm_brandmark(size: int = 88, extra: str = "") -> str:
    """Firm logo + wordmark wrapped as ONE live link to the firm website, so
    clicking either the logo or the lettering opens the site in a new tab.
    `extra` appends to the anchor's style (e.g. 'margin-bottom:14px')."""
    return (
        f'<a href="{_firm_url()}" target="_blank" rel="noopener" '
        f'   style="display:flex;align-items:center;gap:10px;text-decoration:none;{extra}">'
        f'{logo_mark(THEME["primary"], size)}'
        f'<span style="font-size:1.5rem;font-weight:700;letter-spacing:0.06em;'
        f'             color:{THEME["ink"]};text-transform:uppercase">'
        f'{ADVISOR["firm"]}'
        f'</span>'
        f'</a>'
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


def make_risk_ring(score: int, band: str, height: int = 320) -> str:
    """Profile-style risk badge — circular, EXACT port of the proportions
    used by app.py's risk_badge() function (the same function that
    generates the PROFILE badge on page 1 of the advisor PDF).

    Every dimension below is taken line-for-line from risk_badge() in
    app.py (around line 5427), expressed in units of a 200-unit virtual
    badge size mapped onto a 200×200 SVG viewBox. Visual parity with
    the PDF is the goal — clients see the same anchor visual whether
    they're in the portal or reading the PDF proposal, with no
    perceptible difference in proportions.

    Source proportions (size = badge diameter, here 200 units):
      • outer navy stroke      = max(0.8, size × 0.022)  → 4.4
      • inner gold radius      = size × 0.42             → 84
      • gauge arc span         = 90°, from −45° to +45°  (0° = top)
      • gauge stroke           = max(1.5, size × 0.035)  → 7
      • tick radial half-len   = max(2.5, size × 0.04)   → 8 per side
      • PROFILE eyebrow size   = max(5.5, size × 0.075)  → 15
      • PROFILE eyebrow y      = cy − size × 0.22        → 56 (SVG)
      • numeral font size      = size × 0.36             → 72
      • numeral baseline y     = cy + num_pt × 0.32      → 123 (SVG)
      • "/99" font size        = max(5.5, size × 0.085)  → 17
      • "/99" baseline y       = cy + size × 0.28        → 156 (SVG)

    Coordinate conversion notes:
      • ReportLab uses y-up; SVG uses y-down. Anywhere the source code
        writes "cy + N" (meaning N above center in ReportLab), the
        SVG equivalent is "cy − N". Anywhere it writes "cy − N"
        (below center in ReportLab), SVG uses "cy + N".
      • Both systems put a text element's y-coordinate at the
        baseline, so font-size and text positioning translate
        directly without further adjustment.
      • The score-to-angle mapping (score 0 → −45°, score 99 → +45°,
        x = cx + r·sin(θ), y = cy ± r·cos(θ)) is identical in both;
        only the y-sign flips for SVG.
    """
    label = band
    navy  = resolve_color_key("brand.primary.navy",  SETTINGS)
    gold  = resolve_color_key("brand.accent.gold",   SETTINGS)
    cream = resolve_color_key("brand.surface.cream", SETTINGS)
    muted = resolve_color_key("brand.text.muted",    SETTINGS)

    px = int(height * 0.625)

    # Tick position on the 90° gauge arc. Score 0 → −45°, score 99 → +45°,
    # measured clockwise from straight up. The tick is a short radial
    # line straddling the gauge stroke by ±8 (= max(2.5, size×0.04) for
    # size=200), matching risk_badge() in app.py exactly.
    s = max(1, min(99, int(score)))
    angle_rad = math.radians(-45.0 + (s / 99.0) * 90.0)
    _sin = math.sin(angle_rad)
    _cos = math.cos(angle_rad)
    _cx, _cy = 100.0, 100.0
    _tick_in_r  = 76.0   # inner gold radius (84) − 8
    _tick_out_r = 92.0   # inner gold radius (84) + 8
    _tx_in  = _cx + _tick_in_r  * _sin
    _ty_in  = _cy - _tick_in_r  * _cos
    _tx_out = _cx + _tick_out_r * _sin
    _ty_out = _cy - _tick_out_r * _cos

    # Unique gradient id per-render so multiple badges on one page
    # don't collide on the same <defs>.
    grad_id = f"profile_gauge_{s}"

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'            justify-content:center;padding:8px 0">'
        f'  <svg viewBox="0 0 200 200" width="{px}" height="{px}" '
        f'       xmlns="http://www.w3.org/2000/svg" '
        f'       role="img" aria-label="Risk profile {s} of 99">'
        f'    <defs>'
        f'      <linearGradient id="{grad_id}" x1="0" y1="0" x2="1" y2="0">'
        f'        <stop offset="0"   stop-color="#97C459"/>'
        f'        <stop offset="0.5" stop-color="#FAC775"/>'
        f'        <stop offset="1"   stop-color="#F09595"/>'
        f'      </linearGradient>'
        f'    </defs>'
        # Outer navy ring + cream interior. r=97.8 = (200/2 − 4.4/2);
        # stroke=4.4 = size × 0.022. Exact line-for-line port.
        f'    <circle cx="100" cy="100" r="97.8" fill="{cream}" '
        f'            stroke="{navy}" stroke-width="4.4"/>'
        # Inner gold ring at r=84 (size × 0.42), 0.8pt stroke.
        f'    <circle cx="100" cy="100" r="84" fill="none" '
        f'            stroke="{gold}" stroke-width="0.8"/>'
        # Gauge arc — 90° span on the inner ring (r=84), spanning
        # from −45° to +45° measured from straight up. Endpoints are
        # (cx ∓ 84·sin45°, cy − 84·cos45°) = (40.6, 40.6) and (159.4,
        # 40.6). Stroke=7 = max(1.5, size × 0.035). The linearGradient
        # paints green→amber→red along the path's bounding-box x-axis,
        # which lines up with the arc's natural left-to-right flow.
        f'    <path d="M 40.6 40.6 A 84 84 0 0 1 159.4 40.6" '
        f'          fill="none" stroke="url(#{grad_id})" '
        f'          stroke-width="7" stroke-linecap="round"/>'
        # Navy tick at score position — short radial line, 1.5pt
        # stroke, round caps. Coordinates computed above.
        f'    <line x1="{_tx_in:.2f}" y1="{_ty_in:.2f}" '
        f'          x2="{_tx_out:.2f}" y2="{_ty_out:.2f}" '
        f'          stroke="{navy}" stroke-width="1.5" '
        f'          stroke-linecap="round"/>'
        # PROFILE eyebrow at y=56 (= 100 − 44, where 44 = size × 0.22),
        # font 15pt = size × 0.075, Helvetica-Bold navy, letter-spaced.
        f'    <text x="100" y="56" text-anchor="middle" '
        f'          font-family="Helvetica, Arial, sans-serif" '
        f'          font-size="15" font-weight="bold" '
        f'          letter-spacing="1.5" fill="{navy}">PROFILE</text>'
        # Score numeral at baseline y=123 (= 100 + 72 × 0.32, the
        # vertical-center adjustment from risk_badge), font 72pt =
        # size × 0.36, Times-Roman / Source Serif Pro navy.
        f'    <text x="100" y="123" text-anchor="middle" '
        f'          font-family="\'Source Serif Pro\', Georgia, serif" '
        f'          font-size="72" fill="{navy}">{s}</text>'
        # "/99" at y=156 (= 100 + 56, where 56 = size × 0.28),
        # font 17pt = size × 0.085, Helvetica gray.
        f'    <text x="100" y="156" text-anchor="middle" '
        f'          font-family="Helvetica, Arial, sans-serif" '
        f'          font-size="17" fill="{muted}">/ 99</text>'
        f'  </svg>'
        # Band label below the badge (e.g. "Moderate") — preserved
        # from the previous implementation, in serif navy.
        f'  <div style="font-family:\'Source Serif Pro\',Georgia,serif;'
        f'              font-size:1.05rem;color:{navy};font-weight:500;'
        f'              letter-spacing:0.02em;margin-top:14px">'
        f'    {label}'
        f'  </div>'
        f'</div>'
    )


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
        "fr_perf_period": "3M",        # Portfolio Performance window: 1M | 3M | 1Y
        # Pre-login flow state
        "fr_step":      "welcome",     # welcome | prequiz | quiz | results | register
        "fr_first":     "",
        "fr_last":      "",
        "fr_age":       0,
        "fr_q_idx":     0,             # current question index in quiz
        "fr_q_max":     0,             # furthest question reached (frontier)
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
    st.session_state.fr_q_max   = 0
    st.session_state.fr_answers = {}
    st.session_state.fr_scores  = None
    st.session_state.fr_show_signin = False
    st.session_state.fr_fb_signout = True
    st.session_state.pop("fr_prefill_email", None)
    st.session_state.pop("fr_reg_linked", None)
    st.rerun()


def _render_sign_out(suffix: str):
    """Sign-out control. Shown only on the Home and My Info tabs (not every
    tab). `suffix` keeps the Streamlit button key unique per call site, since
    both tabs render into the DOM at once."""
    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
    _so_l, _so_c, _so_r = st.columns([1, 2, 1])
    with _so_c:
        if st.button("Sign out", key=f"fr_logout_btn_{suffix}",
                     use_container_width=True):
            _logout()
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    """Entry router for the unauthenticated flow. Dispatches to one of five
    onboarding screens based on fr_step:
        welcome  → landing page with single CTA
        prequiz  → first/last name + age (only fields needed before quiz)
        quiz     → 14 questions across 5 sections (Goals included)
        results  → score reveal (no gate yet — show first, ask second)
        register → email + phone (req'd) + address + zip (optional) → save
    """
    # Capture a referral code from the invite link (?ref=<code>) once, into the
    # session, so it survives the multi-step onboarding flow (the query param
    # can get dropped on later reruns). Resolved to the referrer at signup.
    _ref = st.query_params.get("ref")
    if _ref and not st.session_state.get("fr_ref_code"):
        st.session_state.fr_ref_code = str(_ref).strip()[:32]

    # After a sign-out, clear the in-browser Firebase session once so the
    # iframe doesn't silently re-authenticate the user on the next sign-in.
    if st.session_state.pop("fr_fb_signout", False) and _FIREBASE_AVAILABLE \
            and firebase_auth is not None:
        try:
            firebase_auth.logout(key="fr_fb_signout_run")
        except Exception:
            pass

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
    # Build the welcome markup as a single unindented HTML string. Streamlit
    # runs markdown before applying unsafe_allow_html, and lines indented 4+
    # spaces get parsed as code blocks — which can leak phantom whitespace
    # nodes into the rendered DOM (visible as ghost selection regions to
    # the right of the headline). Keeping the HTML on one logical string
    # without leading indentation avoids that.
    welcome_html = (
        f'<div style="max-width:520px;margin:30px auto 0;padding:0 28px;text-align:center">'
        f'<div style="display:flex;align-items:center;justify-content:center;margin-bottom:36px">'
        f'<a href="{_firm_url()}" target="_blank" rel="noopener" style="text-decoration:none;display:inline-flex">{logo_mark(THEME["primary"], 160)}</a>'
        f'</div>'
        f'<h1 style="font-size:1.5rem;line-height:1.3;color:{THEME["ink"]};'
        f'font-weight:500;margin:14px auto 28px;letter-spacing:-0.015em;text-align:center">'
        f'Get your free financial risk profile<br/>in less than 3 minutes!'
        f'</h1>'
        f'<div style="display:flex;gap:24px;color:{THEME["muted"]};'
        f'font-size:0.92rem;margin-bottom:24px;align-items:center;justify-content:center">'
        f'<span>{_icon_lock}Encrypted</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(welcome_html, unsafe_allow_html=True)

    # Tight spacer — buttons sit close under the "Encrypted" trust signal
    # rather than being pushed to the bottom of the viewport. (Previously
    # 120px to match the mockup's vertical rhythm; the long gap left an
    # uncomfortable empty band on mobile, so it's collapsed to a small
    # breathing-room margin.)
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    _spc_l, _cta, _spc_r = st.columns([1, 2, 1])
    with _cta:
        if st.button("Start risk profile  →", type="primary",
                     key="fr_start_btn", use_container_width=True):
            st.session_state.pop("fr_prefill_email", None)
            st.session_state.pop("fr_reg_linked", None)
            st.session_state.fr_step = "prequiz"
            st.rerun()

    # ── "Already a member? Sign in" — inline toggle, no expander chrome ────
    # Toggling sets a session flag; the email field renders on the rerun.
    if not st.session_state.get("fr_show_signin", False):
        _spc_l, _link, _spc_r = st.columns([1, 2, 1])
        with _link:
            st.markdown(
                f'<div style="text-align:center;margin-top:6px;'
                f'            font-size:0.92rem;color:{THEME["muted"]}">'
                f'  Already registered?'
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

            if not _FIREBASE_AVAILABLE or firebase_auth is None:
                st.error("Sign-in is temporarily unavailable. Please try again shortly.")
                if st.button("Cancel", key="fr_signin_cancel",
                             use_container_width=True):
                    st.session_state.fr_show_signin = False
                    st.rerun()
            else:
                # Verified sign-in via Google, Facebook, or email. Replaces the
                # old email-only lookup (which let anyone who knew a client's
                # address into that client's data). Firebase returns a signed
                # ID token; we verify it server-side, then map the verified
                # email onto the existing account.
                _token = firebase_auth.render_login(key="fr_fb_login")

                if st.button("Cancel", key="fr_signin_cancel",
                             use_container_width=True):
                    st.session_state.fr_show_signin = False
                    st.rerun()

                if _token:
                    _claims = firebase_auth.verify_token(_token)
                    if not _claims:
                        st.error("We couldn't verify that sign-in. Please try again.")
                    else:
                        _email = (_claims.get("email") or "").strip().lower()
                        _uid = _claims.get("uid") or _claims.get("user_id") or ""
                        # 1) Match by Firebase UID first — handles a returning
                        #    client whose record was already linked, even if the
                        #    email on the record differs from the provider email.
                        _user = find_user_by_uid(_uid)
                        # 2) Otherwise match by verified email, and link the UID
                        #    onto that record the first time. Safe because
                        #    Firebase has proven they own this email and the
                        #    account is keyed under it.
                        if _user is None:
                            _user = find_user(_email)
                            if (_user is not None and _uid
                                    and not _user.get("firebase_uid")):
                                update_user(_user.get("email", _email),
                                            {"firebase_uid": _uid})
                                _user["firebase_uid"] = _uid
                        if _user is not None:
                            # Existing account → sign in.
                            st.session_state.fr_user = _user
                            st.session_state.fr_show_signin = False
                            st.rerun()
                        else:
                            # Verified identity, but no account on file yet.
                            # Send them straight into the assessment like any
                            # other visitor — no "set up an account" popup,
                            # since results are shown before the info is asked
                            # for. Name pre-fills from their Google/Facebook
                            # profile; the verified email is pinned for the
                            # registration step at the end, and the UID is
                            # carried so the new account is linked on creation.
                            _name = (_claims.get("name") or "").strip()
                            _first, _, _last = _name.partition(" ")
                            st.session_state.fr_first = _first
                            st.session_state.fr_last  = _last
                            st.session_state.fr_prefill_email = _email
                            st.session_state.fr_link_uid = _uid
                            st.session_state.fr_show_signin = False
                            st.session_state.fr_step = "prequiz"
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
                    st.session_state.fr_q_max   = 0
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

    # Frontier = furthest question the user has reached. The forward button is
    # hidden while progressing into new territory (select questions auto-
    # advance on answer, so no button is needed) and reappears only when the
    # user has gone BACK to an already-answered question — letting them move
    # forward again without re-picking. Number/multi questions never auto-
    # advance, so they always keep a forward button.
    st.session_state.fr_q_max = max(st.session_state.get("fr_q_max", 0), idx)
    frontier = st.session_state.fr_q_max

    progress = (idx + 1) / total

    # Header with progress
    st.markdown(
        f'<div style="max-width:560px;margin:20px auto 0;padding:0 28px">'
        f'  <div style="display:flex;align-items:center;justify-content:space-between;'
        f'              margin-bottom:14px">'
        f'    {firm_brandmark(88)}'
        f'    <span style="font-size:0.78rem;color:{THEME["muted"]};'
        f'                 font-variant-numeric:tabular-nums">'
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
        f'  <div class="fr-question" style="font-size:2.1rem;font-weight:600;'
        f'       line-height:1.3;letter-spacing:-0.015em;color:{THEME["ink"]};'
        f'       margin:6px 0 22px">'
        f'    {q["text"]}'
        f'  </div>'
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
            min_pick = q.get("min_pick")
            n_picked = len(val or [])
            if max_pick and n_picked > max_pick:
                st.warning(
                    f"You've picked {n_picked}. We use the top {max_pick} for "
                    f"scoring — remove one to choose which counts, or "
                    f"continue and we'll keep the first {max_pick}."
                )
            elif min_pick and n_picked < min_pick:
                st.info(
                    f"Pick {min_pick - n_picked} more to continue "
                    f"({n_picked} of {min_pick} selected)."
                )
            # Question is "answered" only when the floor is met. If
            # min_pick is set, the Next/Finish button stays disabled
            # until the user reaches the required count. If only
            # max_pick is set, any non-zero pick counts as answered
            # (legacy behavior).
            if min_pick:
                answered = n_picked >= min_pick
            else:
                answered = n_picked > 0
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
            # Auto-advancing select questions at the frontier need no button;
            # number/multi (no auto-advance) and any revisited question (behind
            # the frontier) still show it.
            auto_advances = (q["type"] == "select")
            show_forward = (not auto_advances) or (idx < frontier)
            label = "Finish →" if idx == total - 1 else "Next →"
            if show_forward and st.button(
                    label, type="primary", key=f"fr_qz_next_{idx}",
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
        f'  {firm_brandmark(88, "margin-bottom:14px")}'
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
            f'      <div style="font-family:\'Inter\',-apple-system,sans-serif;'
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
            st.session_state.fr_q_max   = 0
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
        f'  {firm_brandmark(88, "margin-bottom:14px")}'
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
        # ── Fast path: register with Google / Facebook ──────────────────
        # If they sign up with a provider here, the account is created under
        # their verified identity (email + Firebase UID), so a later "Sign in
        # with Google" matches every time — no looping back to the assessment.
        # Capturing their identity locks the email field and pre-fills the name
        # below; they only need to add a phone and accept the terms.
        if _FIREBASE_AVAILABLE and firebase_auth is not None \
                and not st.session_state.get("fr_prefill_email"):
            st.markdown('<div class="fr-eyebrow">Create your account</div>',
                        unsafe_allow_html=True)
            _reg_token = firebase_auth.render_login(
                key="fr_reg_fb",
                title="Create your account",
                subtitle="Sign up with Google, Facebook, or email.")
            if _reg_token and not st.session_state.get("fr_reg_linked"):
                _rc = firebase_auth.verify_token(_reg_token)
                if _rc:
                    _ve = (_rc.get("email") or "").strip().lower()
                    if _ve:
                        st.session_state.fr_prefill_email = _ve
                        st.session_state.fr_link_uid = (
                            _rc.get("uid") or _rc.get("user_id") or "")
                        _nm = (_rc.get("name") or "").strip()
                        if _nm and not st.session_state.get("fr_first"):
                            _f, _, _l2 = _nm.partition(" ")
                            st.session_state.fr_first = _f
                            st.session_state.fr_last = _l2
                        st.session_state.fr_reg_linked = True
                        # No st.rerun() here. The component emitting its token
                        # already triggers a Streamlit rerun, and the locked
                        # email field renders just below in this same pass. An
                        # explicit rerun yanked the whole block off-screen the
                        # instant a Firebase session already existed (the
                        # "flash then vanish").
            if st.session_state.get("fr_prefill_email"):
                st.success(f"Creating your account as: "
                           f"{st.session_state.get('fr_prefill_email')}")
            st.markdown('<div style="text-align:center;color:#6b6f78;'
                        'font-size:13px;margin:10px 0;">or enter your details'
                        '</div>', unsafe_allow_html=True)

        # No fr-card wrapper here — same reason as the prequiz screen: it
        # rendered an empty padded white box above the first label. The
        # eyebrow + inputs already group visually on their own.
        st.markdown('<div class="fr-eyebrow">Contact info</div>',
                    unsafe_allow_html=True)
        # If the user arrived here from a verified sign-in (Google / Facebook /
        # email) with no existing account, lock the email field to the verified
        # address. This guarantees the account is created under the exact email
        # their sign-in returns next time, so find_user() matches and they land
        # on their dashboard instead of being looped back into the assessment.
        # Anonymous funnel users (no verified email) still type theirs freely.
        _verified_email = (st.session_state.get("fr_prefill_email") or "").strip().lower()
        if _verified_email:
            st.text_input("Email *", value=_verified_email,
                          key="fr_rg_email_locked", disabled=True,
                          help="Verified from your sign-in — your account is "
                               "created under this address.")
            email = _verified_email
        else:
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

        # ── Required agreement ──────────────────────────────────────────
        # Two buttons open the documents directly through st.dialog. The
        # checkbox label previously held [Terms](?agree=terms)-style markdown
        # links; a Streamlit update changed link handling so clicking one
        # reloaded the whole app *inside* the dialog. Buttons calling
        # _agreement_popup don't depend on link or query-param behavior, so
        # the consent gate survives Streamlit upgrades. The submit button
        # stays disabled until the box is checked; acceptance (version +
        # timestamp) is recorded on the user record below.
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            if st.button("Terms & Conditions", key="fr_view_terms",
                         use_container_width=True):
                _agreement_popup("Terms & Conditions", TERMS_TEXT,
                                 "fr_terms_close", version=TOS_VERSION)
        with _tc2:
            if st.button("Privacy Policy", key="fr_view_privacy",
                         use_container_width=True):
                _agreement_popup("Privacy Policy", PRIVACY_TEXT,
                                 "fr_privacy_close")

        agreed = st.checkbox(
            "I agree to the Terms & Conditions and Privacy Policy",
            key="fr_rg_consent",
        )

        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("← Back", key="fr_rg_back",
                         use_container_width=True):
                st.session_state.fr_step = "results"
                st.rerun()
        with b2:
            # Button stays enabled even when the agreement box is unchecked,
            # so clicking it surfaces a clear reason rather than silently
            # doing nothing (a disabled button gave no feedback about WHY
            # sign-up wouldn't proceed). The agreement is validated below.
            if st.button("Save & view dashboard →", type="primary",
                         key="fr_rg_submit", use_container_width=True):
                # Validation
                errors = []
                if not agreed:
                    errors.append(
                        "Please check the box agreeing to the Terms & "
                        "Conditions and Privacy Policy to continue."
                    )
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
                    from datetime import datetime as _dt, timezone as _tz
                    user["age"]     = int(st.session_state.fr_age)
                    user["address"] = (addr or "").strip()
                    user["zip"]     = (zipcode or "").strip()
                    # If they arrived from a Google/Facebook/email sign-in,
                    # link the Firebase UID now so future sign-ins match by UID
                    # regardless of any later email change.
                    _luid = (st.session_state.get("fr_link_uid") or "").strip()
                    if _luid:
                        user["firebase_uid"] = _luid
                    # Audit trail: which agreement version the client accepted
                    # and when (UTC). Reaching here requires the consent box,
                    # since it gates the submit button above.
                    user["consent_tos_version"] = TOS_VERSION
                    user["consent_tos_at"] = _dt.now(_tz.utc).isoformat(timespec="seconds")
                    _shared_update_json(
                        USERS_FILE,
                        lambda d, k=user["email"], u=user: d.update({k: u}),
                    )

                    # Referral attribution — if they arrived via an invite link
                    # (?ref=<code>), credit the referring client. No-op for
                    # unknown codes or self-referrals; safe to call always.
                    _ref_code = (st.session_state.get("fr_ref_code") or "").strip()
                    if _ref_code:
                        record_referral(user["email"], _ref_code)

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
                            label, _, _ = (score_band(scores.get("capacity_score", 0),
                                                       scores.get("tolerance_score", 0))
                                           if overall else ("", "", ""))
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
                # No flash banner here — the dashboard's greeting already
                # acknowledges the user, and stacking a green
                # "Profile saved — welcome!" banner on top added visual
                # noise without information. (HubSpot sync messages, if
                # any, are still printed to logs for debugging — just not
                # shown as a UI banner.)
                st.session_state.fr_user = user
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
# ── RMD calculator (client tool) ────────────────────────────────────────────
# IRS Uniform Lifetime Table (Publication 590-B, Table III): distribution
# period by age. RMD = prior year-end balance ÷ factor.
_RMD_UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
    86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
    93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}


def _rmd_start_age(birth_year: int) -> int:
    """SECURE 2.0 required-beginning age by birth year: 72 for owners already
    in RMDs (born ≤1950), 73 for 1951–1959, 75 for 1960 or later."""
    if birth_year <= 1950:
        return 72
    if birth_year <= 1959:
        return 73
    return 75


def _rmd_schedule_link(label: str) -> str:
    """HubSpot Meetings link pre-filled with the signed-in client, styled to
    match the Advisor tab's booking button."""
    from urllib.parse import urlencode as _urlencode
    _su = st.session_state.get("fr_user") or {}
    _sp = {}
    if (_su.get("first_name") or "").strip(): _sp["firstName"] = _su["first_name"].strip()
    if (_su.get("last_name")  or "").strip(): _sp["lastName"]  = _su["last_name"].strip()
    if (_su.get("email")      or "").strip(): _sp["email"]     = _su["email"].strip()
    url = SCHEDULE_URL + (("?" + _urlencode(_sp)) if _sp else "")
    return (
        f'<a href="{url}" target="_blank" rel="noopener" '
        f'style="display:flex;align-items:center;justify-content:center;'
        f'width:100%;box-sizing:border-box;text-align:center;'
        f'background:{THEME["primary"]};color:#fff;padding:14px 16px;'
        f'border-radius:10px;text-decoration:none;font-weight:600;'
        f'font-size:0.95rem;margin:4px 0 14px">{label}</a>'
    )


def _render_rmd_tab():
    """Client-facing RMD estimator. Uses the IRS Uniform Lifetime Table and
    SECURE 2.0 starting ages; routes the spouse-more-than-10-years-younger
    case to the advisor (Joint Life table) and notes Roth/inherited
    exceptions in the disclaimer."""
    _serif = "font-family:'Source Serif Pro',Georgia,serif"
    navy = THEME["primary"]; ink = THEME["ink"]; ink2 = THEME["ink2"]
    line = THEME["line"]; gold = THEME["accent"]; muted = THEME["muted"]

    st.markdown('<div class="fr-eyebrow">Retirement planning</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:1.15rem;font-weight:600;color:{ink};'
        f'letter-spacing:-0.01em;margin-bottom:4px">'
        f'Required Minimum Distribution</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.9rem;color:{ink2};line-height:1.55;'
        f'margin-bottom:18px">Estimate the minimum you must withdraw from a '
        f'traditional IRA or workplace retirement plan this year under '
        f'current IRS rules.</div>',
        unsafe_allow_html=True,
    )

    bal = st.number_input(
        "Account balance — as of December 31 last year",
        min_value=0.0, step=1000.0, value=0.0, format="%.0f", key="rmd_bal",
    )
    c1, c2 = st.columns(2)
    with c1:
        birth = st.number_input("Birth year", min_value=1900, max_value=2010,
                                value=1953, step=1, key="rmd_birth")
    with c2:
        year = st.number_input("Distribution year", min_value=2023,
                               max_value=2100, value=datetime.now().year,
                               step=1, key="rmd_year")
    spouse = st.checkbox(
        "My spouse is my sole beneficiary and more than 10 years younger "
        "than me", key="rmd_spouse",
    )

    if not bal or bal <= 0:
        st.caption("Enter your account balance above to see your estimate.")
    else:
        age   = int(year) - int(birth)
        start = _rmd_start_age(int(birth))
        usd0  = lambda n: "${:,.0f}".format(n)

        def _stat(v, k):
            return (
                f'<div style="flex:1;text-align:center;padding:12px 10px">'
                f'<div style="{_serif};font-size:1.15rem;font-weight:600;'
                f'color:{navy}">{v}</div>'
                f'<div style="font-size:0.62rem;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{muted};margin-top:3px">{k}'
                f'</div></div>'
            )

        def _card(amount, sub, factor, pct, note, warn=False):
            border = THEME["risk"] if warn else gold
            nbg = THEME["risk_soft"] if warn else THEME["surface2"]
            return (
                f'<div class="fr-card">'
                f'<div class="fr-eyebrow">Estimated RMD for {int(year)}</div>'
                f'<div style="{_serif};font-size:2.4rem;font-weight:600;'
                f'color:{navy};line-height:1.05;letter-spacing:-0.02em">'
                f'{amount}</div>'
                f'<div style="font-size:0.85rem;color:{ink2};margin-top:4px">'
                f'{sub}</div>'
                f'<div style="display:flex;border:1px solid {line};'
                f'border-radius:10px;overflow:hidden;margin-top:16px;'
                f'background:{THEME["surface"]}">'
                f'{_stat(factor, "Factor")}'
                f'<div style="width:1px;background:{line}"></div>'
                f'{_stat(pct, "Of balance")}'
                f'<div style="width:1px;background:{line}"></div>'
                f'{_stat(age, "Age this year")}'
                f'</div>'
                f'<div style="margin-top:14px;font-size:0.85rem;color:{ink};'
                f'background:{nbg};border-left:3px solid {border};'
                f'padding:11px 14px;border-radius:0 8px 8px 0;line-height:1.5">'
                f'{note}</div>'
                f'</div>'
            )

        if age < start:
            html = _card(
                "$0", f"No RMD is required for {int(year)}.", "—", "—",
                f"Your first RMD year is <b>{int(birth) + start}</b>, the year "
                f"you turn <b>{start}</b>. That first withdrawal can be delayed "
                f"until April 1 of the following year.",
            )
        elif spouse:
            html = _card(
                "See your advisor", "Your case uses the IRS Joint Life table.",
                "Joint", "Lower",
                "When a spouse more than 10 years younger is your sole "
                "beneficiary, your RMD is figured from the IRS Joint Life &amp; "
                "Last Survivor table, which produces a <b>smaller</b> required "
                "amount. We'll calculate the exact figure with you.",
                warn=True,
            )
        else:
            look   = min(max(age, 72), 120)
            factor = _RMD_UNIFORM_LIFETIME[look]
            rmd    = bal / factor
            pct    = (1.0 / factor) * 100.0
            note = (f"Withdraw at least <b>{usd0(rmd)}</b> by "
                    f"<b>December 31, {int(year)}</b>.")
            if age == start:
                note += (f" Since this is your first RMD, you may delay it "
                         f"until <b>April 1, {int(year) + 1}</b> — but that "
                         f"means two taxable RMDs in {int(year) + 1}.")
            html = _card(
                usd0(rmd),
                f"{usd0(bal)} ÷ {factor:.1f} (factor for age {look})",
                f"{factor:.1f}", f"{pct:.2f}%", note,
            )

        st.markdown(html, unsafe_allow_html=True)
        st.markdown(_rmd_schedule_link("Talk through your RMD strategy →"),
                    unsafe_allow_html=True)

    st.caption(
        "Estimate only — not tax advice. Figures use the IRS Uniform Lifetime "
        "Table (Publication 590-B) and SECURE 2.0 starting ages (73 if born "
        "1951–1959, 75 if born 1960 or later). Roth IRAs have no lifetime RMD "
        "for the original owner; inherited IRAs and the spouse-more-than-10-"
        "years-younger case are handled separately. Confirm your figures with "
        "your tax professional or your advisor."
    )


def render_dashboard():
    user = st.session_state.fr_user
    ck   = _client_key()
    profile      = load_profiles().get(ck, {})
    all_holdings = load_all_holdings()
    holdings     = all_holdings.get(ck, {}) or {}

    # ── App bar ─────────────────────────────────────────────────────────────
    # Sign-out button moved to the bottom of the page (after the tabs) so
    # the top of the screen is reserved for branding and the user's data.
    st.markdown(
        f'{firm_brandmark(88, "padding-top:6px")}',
        unsafe_allow_html=True,
    )

    if st.session_state.fr_flash:
        st.toast(st.session_state.fr_flash, icon="✅")
        st.session_state.fr_flash = None

    # ── Greeting ────────────────────────────────────────────────────────────
    first_name = user.get("first_name", "there")
    hour = datetime.now().hour
    greeting = ("Good morning" if hour < 12 else
                "Good afternoon" if hour < 18 else "Good evening")
    # The "Last checkup: ..." annotation moved to _render_home_tab and is
    # computed there from the profile dict — keeping it close to where it
    # renders.

    st.markdown(
        f'<div style="margin:18px 0 0 2px">'
        f'  <div class="fr-greeting">{greeting}, {first_name}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Order matches the natural reading flow: high-level summary → portfolio
    # detail → forward-looking plan → personal info → human contact. "My Info"
    # was added in the 2026-04-30 update so clients can edit their own
    # contact details without having to message the advisor; placed at the
    # far right (rightmost = "settings-like" convention; Advisor sits to
    # its left as "their info to my info").
    (tab_home, tab_goals, tab_holdings, tab_rmd,
     tab_advisor, tab_my_info) = st.tabs(
        ["Home", "Financial Goals", "Holdings", "RMD Calculator",
         "Advisor", "My Info"]
    )

    with tab_home:
        _render_home_tab(profile, holdings, ck)
        _render_sign_out("home")
    with tab_goals:
        _render_plan_tab(ck)
    with tab_holdings:
        _render_holdings_tab(holdings, ck)
    with tab_rmd:
        _render_rmd_tab()
    with tab_advisor:
        _render_advisor_tab()
    with tab_my_info:
        _render_my_info_tab()
        _render_sign_out("myinfo")

    # Wire the snapshot tiles (.fr-go-holdings / .fr-go-goals) to their tabs.
    _render_tab_link_bridge()


def _render_tab_link_bridge():
    """Make snapshot tiles tagged .fr-go-* act as links to the matching tab.

    Streamlit's st.tabs can't be switched from Python, so a tiny same-origin
    component iframe reaches into the parent document and clicks the real tab
    button when a tagged tile is clicked. st.tabs renders every tab's content
    up front (just hidden), so the Home tiles are always in the DOM regardless
    of which tab is active. If the selectors ever drift in a future Streamlit
    release this degrades gracefully — the tiles simply stop navigating, the
    app is otherwise unaffected."""
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          const MAP = {"fr-go-holdings": "Holdings", "fr-go-goals": "Financial Goals", "fr-go-advisor": "Advisor"};
          const PERF = {"fr-perf-1m": "fr_perf_1M", "fr-perf-3m": "fr_perf_3M", "fr-perf-1y": "fr_perf_1Y"};
          function clickTab(name){
            const tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
            for (const t of tabs){
              if ((t.innerText || "").trim() === name){
                t.click();
                // Land at the top of the freshly shown tab — clicking a tab
                // doesn't reset scroll, so without this you stay at whatever
                // offset you were at on the Home page (mid-tab).
                const toTop = function(){
                  try { window.parent.scrollTo(0, 0); } catch(e){}
                  const sels = ['section.main', '[data-testid="stMain"]',
                                '[data-testid="stAppViewContainer"] section', '.main'];
                  for (const s of sels){
                    const el = doc.querySelector(s);
                    if (el){ try { el.scrollTo(0, 0); } catch(e){ el.scrollTop = 0; } }
                  }
                };
                toTop();
                setTimeout(toTop, 60);
                return;
              }
            }
          }
          function clickPerf(keyCls){
            const b = doc.querySelector('.st-key-' + keyCls + ' button');
            if (b){ b.click(); }
          }
          function wire(){
            doc.querySelectorAll('.fr-tablink').forEach(function(el){
              if (el.dataset.frWired) return;
              for (const cls in MAP){
                if (el.classList.contains(cls)){
                  el.dataset.frWired = "1";
                  el.addEventListener('click', function(ev){
                    if (ev.target.closest('a')) return;  // let email/phone links work
                    clickTab(MAP[cls]);
                  });
                  break;
                }
              }
            });
            doc.querySelectorAll('.fr-perf-chip').forEach(function(el){
              if (el.dataset.frWired) return;
              for (const cls in PERF){
                if (el.classList.contains(cls)){
                  el.dataset.frWired = "1";
                  el.addEventListener('click', function(){ clickPerf(PERF[cls]); });
                  break;
                }
              }
            });
          }
          wire();
          new MutationObserver(wire).observe(doc.body, {childList:true, subtree:true});
        })();
        </script>
        """,
        height=0,
    )


def _render_invite_button():
    """"Invite someone" action. Rendered as a real st.button so it matches the
    "View / update profile" button exactly — same component, same CSS, same
    Inter font. Getting an iframe-rendered button to match the native one
    proved unreliable (font load timing + separate document), so instead a
    hidden bridge (components.html, height 0) reaches into the parent document
    the same way _render_tab_link_bridge does: it appends the share-icon SVG
    after the label and wires the click to the native share sheet in the PARENT
    window — where the Web Share API is permitted, unlike the sandboxed
    component iframe.

    Shares a link to the assessment (PORTAL_URL -> live parent URL -> firm
    site). Where Web Share is unavailable (most desktop browsers) it copies the
    link instead. The click handler runs in the capture phase and stops
    propagation so Streamlit's own click handler never fires — the button is a
    pure share trigger with no server rerun. The Python return is ignored.
    """
    st.button("Invite someone", key="fr_invite_btn", use_container_width=True)

    firm = (ADVISOR.get("firm") or "our firm").strip()
    cfg_json = json.dumps({
        "firm":        firm,
        "portalUrl":   PORTAL_URL,
        "fallbackUrl": FIRM_WEBSITE_URL,
        "refCode":     get_or_create_referral_code(),
    })

    components.html(
        f"""
<script>
(function() {{
  const doc = window.parent.document;
  const CFG = {cfg_json};
  const SHARE_SVG =
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    + 'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    + 'stroke-linejoin="round"><circle cx="18" cy="5" r="3"/>'
    + '<circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>'
    + '<path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4"/></svg>';

  function inviteLink() {{
    let base;
    if (CFG.portalUrl) base = CFG.portalUrl;
    else {{
      try {{ const p = window.parent.location; base = p.origin + p.pathname; }}
      catch (e) {{ base = CFG.fallbackUrl || ""; }}
    }}
    if (CFG.refCode) {{
      base += (base.indexOf("?") >= 0 ? "&" : "?") + "ref=" + encodeURIComponent(CFG.refCode);
    }}
    return base;
  }}
  function findBtn() {{
    const btns = doc.querySelectorAll('.stButton button');
    for (const b of btns) {{
      if ((b.innerText || "").trim() === "Invite someone") return b;
    }}
    return null;
  }}
  function findViewBtn() {{
    const btns = doc.querySelectorAll('.stButton button');
    for (const b of btns) {{
      if ((b.innerText || "").trim().indexOf("View / update profile") === 0) return b;
    }}
    return null;
  }}
  function labelEl(btn) {{
    return btn.querySelector('[data-testid="stMarkdownContainer"] p')
        || btn.querySelector('p')
        || btn.querySelector('[data-testid="stMarkdownContainer"]')
        || btn;
  }}
  function bumpText(btn) {{
    // Label 20% larger, button the same size: lock the current height first,
    // then scale the font so the flex-centered label grows within the box.
    // Guard with a class on the label element (not a one-time flag): a
    // Streamlit rerun can rebuild the button and wipe our inline style, and a
    // class marker resets with it, so this re-applies from the base size
    // instead of getting stuck or compounding.
    const p = labelEl(btn);
    if (!p || p.classList.contains('inv-fs')) return;
    p.classList.add('inv-fs');
    const h = Math.round(btn.getBoundingClientRect().height);
    if (h > 0) btn.style.height = h + "px";
    const cur = parseFloat(window.parent.getComputedStyle(p).fontSize) || 16;
    p.style.fontSize = (cur * 1.2).toFixed(1) + "px";
  }}
  function sizeArrow(btn) {{
    // Wrap the trailing arrow so it can be a bit larger than the text.
    const p = labelEl(btn);
    if (!p || p.querySelector('.inv-arrow')) return;
    p.innerHTML = p.innerHTML.replace(
      /(\\u2192)\\s*$/,
      '<span class="inv-arrow" style="font-size:1.3em;line-height:1">$1</span>');
  }}
  function copyLink() {{
    try {{
      const nav = window.parent.navigator;
      if (nav.clipboard && nav.clipboard.writeText) nav.clipboard.writeText(inviteLink());
    }} catch (e) {{}}
  }}
  function wire() {{
    const btn = findBtn();
    if (!btn) return false;
    // Share icon inside the label so it flows right after the text (appending
    // to the button itself lands it at the far-right edge, since Streamlit's
    // label fills the button width). Re-added if a rerun wiped it.
    if (!btn.querySelector('.inv-share-ic')) {{
      const ic = doc.createElement('span');
      ic.className = 'inv-share-ic';
      ic.style.display = 'inline-flex';
      ic.style.alignItems = 'center';
      ic.style.verticalAlign = 'middle';
      ic.style.marginLeft = '8px';
      ic.innerHTML = SHARE_SVG;
      labelEl(btn).appendChild(ic);
    }}
    bumpText(btn);
    // Attach the share handler once per button node. The flag is a JS expando
    // (not an attribute) so React won't strip it on reconciliation; a brand-new
    // node from a rerun won't have it, so it re-wires without double-binding.
    if (!btn.__invWired) {{
      btn.__invWired = true;
      btn.addEventListener('click', function(ev) {{
        ev.preventDefault();
        ev.stopImmediatePropagation();   // block Streamlit's rerun handler
        const link  = inviteLink();
        const text  = "Take this quick risk-profile checkup from " + CFG.firm
                    + " \\u2014 it's free and takes about 4 minutes:";
        const title = CFG.firm + " \\u2014 free risk-profile checkup";
        const nav   = window.parent.navigator;
        if (nav.share) {{
          try {{ nav.share({{ title: title, text: text, url: link }}); return; }}
          catch (e) {{ if (e && e.name === 'AbortError') return; }}
        }}
        copyLink();
      }}, true);  // capture phase
    }}
    // Same 20% text bump on "View / update profile", plus a slightly larger
    // trailing arrow so the two buttons read in balance.
    const vbtn = findViewBtn();
    if (vbtn) {{ bumpText(vbtn); sizeArrow(vbtn); }}
    return true;
  }}
  if (!wire()) {{
    let n = 0;
    const t = setInterval(function() {{ if (wire() || ++n > 20) clearInterval(t); }}, 100);
  }}
}})();
</script>
""",
        height=0,
    )


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
            f'    14 questions in 5 short sections — about 4 minutes.'
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
            # Crest medallion badge — replaces the legacy Plotly RiskRing.
            # Renders as inline SVG/HTML via st.markdown rather than
            # st.plotly_chart, so it scales cleanly on mobile and doesn't
            # need staticPlot config to disable accidental gesture capture.
            st.markdown(make_risk_ring(overall, profile.get("band", "Moderate"), height=300),
                        unsafe_allow_html=True)
        with h2:
            cap = int(profile.get("capacity_score", 50))
            tol = int(profile.get("tolerance_score", 50))
            label, _, _ = score_band(cap, tol)

            # Neutral summary — describes the posture, not a verdict on the
            # client. Three buckets matching score_band: Conservative,
            # Moderate, Aggressive.
            summaries = {
                "Aggressive": (
                    "Your answers point to an aggressive posture — a higher "
                    "tolerance for short-term swings in exchange for greater "
                    "long-term growth potential."
                ),
                "Moderately Aggressive": (
                    "Your answers point to a moderately aggressive posture — "
                    "a tilt toward long-term growth, with some ballast to "
                    "soften the sharpest swings."
                ),
                "Moderate": (
                    "Your answers point to a moderate posture — a balance "
                    "between growth and stability that most long-term "
                    "investors land on."
                ),
                "Moderately Conservative": (
                    "Your answers point to a moderately conservative posture "
                    "— a lean toward stability, with a modest allocation aimed "
                    "at long-term growth."
                ),
                "Conservative": (
                    "Your answers point to a conservative posture — a "
                    "preference for stability and capital preservation over "
                    "maximum growth."
                ),
            }
            summary = summaries.get(label, summaries["Moderate"])

            st.markdown(
                f'<div style="padding-top:18px">'
                f'  <div class="fr-eyebrow" '
                f'       style="font-size:0.85rem;letter-spacing:0.12em">'
                f'    Risk Profile</div>'
                f'  <div style="font-size:1.5rem;color:{THEME["ink"]};'
                f'              font-weight:700;margin-top:6px;line-height:1.2;'
                f'              letter-spacing:-0.015em">{label}</div>'
                f'  <div style="font-size:0.85rem;color:{THEME["ink2"]};'
                f'              margin-top:8px;line-height:1.5">{summary}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("View / update profile →", key="fr_view_profile",
                         use_container_width=True):
                st.session_state.fr_view = "edit_profile"
                st.rerun()
            # Invite button — same width as the profile button, nested right
            # below it. A share-icon button that fires the OS share sheet
            # directly (no popup / channel picker); shares a link to the
            # assessment. See _render_invite_button.
            _render_invite_button()
            # Client-facing referral count — count only, no names (the people
            # they referred are other clients; the advisor sees the detail).
            # Shown only once they've referred someone.
            _ref_n = referrals_sent_count(ck)
            if _ref_n > 0:
                st.markdown(
                    f'<div style="text-align:center;color:{THEME["ink2"]};'
                    f'            font-size:0.8rem;margin-top:6px">'
                    f'  You\'ve referred {_ref_n} '
                    f'{"person" if _ref_n == 1 else "people"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            # Last checkup indicator — placed under the button so the
            # primary action stays visually dominant. Color matches the
            # secondary "ink2" theme tone (used by .fr-greeting and other
            # supporting text) so it reads as a soft annotation rather
            # than competing with the button.
            #
            # when_text is computed locally here from the `profile` param
            # rather than threaded in from render_dashboard() — this
            # function has its own scope and can't see the parent's
            # locals. Profile is passed in, so it's all we need.
            _updated_str = (profile.get("updated_at")
                            or profile.get("date_completed"))
            if _updated_str:
                try:
                    _d = datetime.fromisoformat(
                        str(_updated_str).replace(" ", "T")[:16]
                    )
                    _days_ago = (datetime.now() - _d).days
                    if _days_ago == 0:
                        when_text = "earlier today"
                    elif _days_ago == 1:
                        when_text = "yesterday"
                    else:
                        when_text = f"{_days_ago} days ago"
                except Exception:
                    when_text = "recently"
            else:
                when_text = "not yet"
            st.markdown(
                f'<div style="text-align:center;margin-top:14px;'
                f'            font-size:1.5rem;font-weight:700;'
                f'            color:{THEME["ink2"]};'
                f'            letter-spacing:-0.015em;line-height:1.3">'
                f'  Last checkup: <span style="color:{THEME["primary"]}">'
                f'    {when_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

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
              delta: str = "", gauge: Optional[int] = None,
              link: Optional[str] = None) -> str:
        """If `gauge` is a 0-100 integer, render a thin horizontal bar
        underneath the value — used for Risk Capacity and Risk Tolerance so
        the score has a visual reference, not just a bare number.

        If `link` is set (e.g. "holdings" or "goals"), tag the tile so the
        tab-link bridge wires a click-through to the matching tab."""
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
        tile_cls = "fr-vital"
        if link:
            tile_cls += f" fr-tablink fr-go-{link}"
        return (
            f'<div class="{tile_cls}">'
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
    a = ADVISOR
    # Use logo_mark() instead of a hand-drawn generic icon — when
    # firm_logo.png exists it returns an <img> tag with the real logo,
    # otherwise falls back to the hexagon SVG. Same visual size (22px)
    # as the previous teal-square icon.
    company_logo_svg = logo_mark(THEME["primary"], 22)
    st.markdown(
        f'<div class="fr-tablink fr-go-advisor" '
        f'     style="background:{THEME["surface2"]};'
        f'            border:1.5px solid {THEME["primary"]};'
        f'            border-radius:10px;padding:14px 16px;margin-top:18px;'
        f'            display:flex;align-items:center;gap:14px">'
        f'  <div style="flex-shrink:0">{a["photo_svg"]}</div>'
        f'  <div style="flex:1;min-width:0">'
        f'    <div style="display:flex;align-items:center;gap:8px;'
        f'                margin-bottom:2px">'
        f'      <div class="fr-eyebrow" style="margin:0">Your advisor</div>'
        f'    </div>'
        f'    <div style="font-size:1rem;font-weight:600;color:{THEME["ink"]};'
        f'                line-height:1.25;letter-spacing:-0.01em">{a["name"]}</div>'
        f'    <div style="display:flex;align-items:center;gap:6px;'
        f'                font-size:0.8rem;color:{THEME["ink2"]};margin-top:3px">'
        f'      {company_logo_svg}'
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
    # When the client hasn't entered any holdings yet, show "—" rather
    # than "$0" so the tile reads as "no data yet" instead of "your net
    # worth is literally zero" — accurate AND avoids accidental alarm
    # before onboarding is complete.
    g3, g4 = st.columns(2)
    with g3:
        nw_delta  = (fmt_pct(vitals["gain_pct"]) if vitals["cost_basis"] else "")
        st.markdown(_tile(
            "Net Worth",
            fmt_money(vitals["net_worth"]) if holdings else "—",
            f"{len(holdings)} positions" if holdings else "no positions yet",
            "", delta=nw_delta, link="holdings",
        ), unsafe_allow_html=True)
    with g4:
        st.markdown(_tile(
            "Cash Position",
            fmt_money(vitals["cash"]) if holdings else "—",
            f"{cash_pct:.1f}% of portfolio" if vitals["net_worth"]
                else "no positions yet",
            "", link="holdings",
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
        _rf_pct = risk_free_pct()
        for g in goals:
            try:
                tdt = date.fromisoformat(g.get("target_date", ""))
                mleft = max(1, (tdt.year - _today.year) * 12
                              + (tdt.month - _today.month))
            except Exception:
                mleft = 12
            g_rate = ((float(g.get("rate_pct", _rf_pct) or 0.0) / 100.0)
                      if bool(g.get("use_growth", True)) else 0.0)
            total_monthly += goal_required_monthly(
                float(g.get("amount") or 0),
                float(g.get("saved")  or 0),
                mleft, g_rate)
        pct = min(100, (total_saved / total_target * 100)
                       if total_target else 0)
        st.markdown(
            f'<div class="fr-vital fr-tablink fr-go-goals" style="margin-top:8px">'
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
            f'<div class="fr-vital fr-tablink fr-go-goals" style="margin-top:8px;text-align:center;'
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
        base = max(vitals["cost_basis"], 1)
        end  = vitals["net_worth"]

        # Window selection. The sparkline is an illustrative per-user shape
        # (we don't store real historical net worth), so each window just
        # shows a different slice: 1M = a small recent move, 1Y = the full
        # run from cost basis. (points, fraction-of-total-move, wiggle).
        period = st.session_state.fr_perf_period
        period_cfg = {
            "1M": (22, 0.16, 0.05),
            "3M": (13, 0.45, 0.07),
            "1Y": (12, 1.00, 0.09),
        }
        n, frac, wig_scale = period_cfg.get(period, period_cfg["3M"])
        start = end - (end - base) * frac
        # Seed includes the period so each window has its own stable shape.
        np.random.seed((hash(ck) ^ (hash(period) << 1)) & 0xFFFFFFFF)
        ramp = np.linspace(start, end, n)
        wig  = np.cumsum(np.random.randn(n))
        wig  = wig - np.linspace(wig[0], wig[-1], n)   # de-trend: endpoints unmoved
        series = (ramp + wig * (end - start) * wig_scale).tolist()
        series[0], series[-1] = start, end

        # Period pills. These are styled HTML spans (so they keep the exact
        # original look inside the card); the click-through is handled by the
        # hidden buttons rendered just below + the JS bridge, which clicks the
        # matching hidden button when a pill is tapped. The active pill gets
        # the chip background; the rest are muted.
        def _perf_chip(lab: str) -> str:
            on = (period == lab)
            tone = (f'background:{THEME["chip"]};color:{THEME["ink"]}' if on
                    else f'color:{THEME["muted"]}')
            return (
                f'<span class="fr-perf-chip fr-perf-{lab.lower()}" '
                f'      style="font-size:0.7rem;padding:4px 9px;border-radius:999px;'
                f'             font-weight:600;cursor:pointer;{tone}">{lab}</span>'
            )

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
            f'      {_perf_chip("1M")}{_perf_chip("3M")}{_perf_chip("1Y")}'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Hidden buttons that do the actual period switch. The container is
        # display:none (see .st-key-fr_perf_btns in the CSS), but a programmatic
        # .click() from the JS bridge still fires the Streamlit handler.
        with st.container(key="fr_perf_btns"):
            for lab in ("1M", "3M", "1Y"):
                if st.button(lab, key=f"fr_perf_{lab}"):
                    st.session_state.fr_perf_period = lab
                    st.rerun()

        # See risk ring above — staticPlot stops the chart from eating
        # touch scroll events.
        st.plotly_chart(make_sparkline(series, height=120),
            use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True})


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

    today = date.today()
    default_target = today.replace(year=today.year + 5)

    # Default assumed rate of return for goal projections = the risk-free rate
    # (13-week T-bill). Computed once per render; cached for 6h underneath.
    _rf_pct = risk_free_pct()

    # Mobile-first goals UI: each goal is a stacked card (no wide scrolling
    # grid), with an inline Edit/Remove panel. New goals are added through the
    # form at the bottom. This replaced st.data_editor, whose 4-column grid
    # forced horizontal scrolling that was unusable on a phone.
    if goals:
        for _i, _g in enumerate(goals):
            _name  = (str(_g.get("name") or "Goal")).strip() or "Goal"
            _amt   = float(_g.get("amount") or 0)
            _saved = float(_g.get("saved") or 0)
            try:
                _tdt   = date.fromisoformat(_g.get("target_date", ""))
                _mleft = max(1, (_tdt.year - today.year) * 12
                                + (_tdt.month - today.month))
                _tdt_str = _tdt.strftime("%b %Y")
            except Exception:
                _tdt, _mleft, _tdt_str = default_target, 12, "\u2014"
            _rem   = max(0.0, _amt - _saved)
            # Rate of return on the saved balance + future contributions.
            # Defaults ON at the risk-free rate; per-goal toggle/override is
            # in the Edit panel below. Backward-compatible with older goals
            # that predate these keys.
            _use_growth = bool(_g.get("use_growth", True))
            _rate_pct   = float(_g.get("rate_pct", _rf_pct) or 0.0)
            _rate       = (_rate_pct / 100.0) if _use_growth else 0.0
            _permo = goal_required_monthly(_amt, _saved, _mleft, _rate)
            _pct   = min(100, (_saved / _amt * 100) if _amt else 0)
            _rate_note = (f" \u00b7 {_rate_pct:.1f}% return"
                          if _use_growth and _rate_pct else "")

            st.markdown(
                f'<div style="margin-top:12px;background:{THEME["surface2"]};'
                f'            border:1.5px solid {THEME["primary"]};'
                f'            border-radius:14px;padding:14px 16px">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              align-items:baseline;gap:10px">'
                f'    <span style="font-weight:600;color:{THEME["ink"]};'
                f'                 font-size:0.98rem">{_name}</span>'
                f'    <span class="fr-mono" style="color:{THEME["ink"]};'
                f'                 font-weight:600;white-space:nowrap">'
                f'      {fmt_money(_saved)} / {fmt_money(_amt)}</span>'
                f'  </div>'
                f'  <div style="height:6px;background:{THEME["line"]};'
                f'              border-radius:3px;margin-top:8px;overflow:hidden">'
                f'    <div style="height:100%;width:{_pct:.0f}%;'
                f'                background:{THEME["primary"]}"></div>'
                f'  </div>'
                f'  <div style="display:flex;justify-content:space-between;'
                f'              font-size:0.78rem;color:{THEME["muted"]};'
                f'              margin-top:8px">'
                f'    <span>Target {_tdt_str}{_rate_note}</span>'
                f'    <span class="fr-mono">{fmt_money(_permo)}/mo</span>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Edit / remove"):
                with st.form(f"fr_goal_edit_{_i}", clear_on_submit=False):
                    _e_amt = st.number_input(
                        "Target amount ($)", min_value=0.0, step=1000.0,
                        value=float(_amt), format="%.0f",
                        key=f"fr_goal_amt_{_i}")
                    _e_saved = st.number_input(
                        "Already saved ($)", min_value=0.0, step=500.0,
                        value=float(_saved), format="%.0f",
                        key=f"fr_goal_saved_{_i}")
                    _e_date = st.date_input(
                        "Target date",
                        value=(_tdt if _tdt >= today else today),
                        min_value=today, key=f"fr_goal_date_{_i}")
                    _e_growth = st.checkbox(
                        "Grow savings at an assumed rate of return",
                        value=_use_growth, key=f"fr_goal_growth_{_i}")
                    _e_rate = st.number_input(
                        "Assumed annual return (%)",
                        min_value=0.0, max_value=25.0,
                        value=float(_rate_pct), step=0.25, format="%.2f",
                        key=f"fr_goal_rate_{_i}",
                        help="Applied to both your current balance and your "
                             "future monthly contributions. Defaults to the "
                             f"risk-free rate (currently ~{_rf_pct:.2f}%).")
                    _ce1, _ce2 = st.columns(2)
                    _save_clicked = _ce1.form_submit_button(
                        "Save", type="primary", use_container_width=True)
                    _del_clicked = _ce2.form_submit_button(
                        "Remove", use_container_width=True)
                if _save_clicked:
                    goals[_i] = {
                        "name":        _name,
                        "amount":      round(float(_e_amt), 2),
                        "saved":       round(float(_e_saved), 2),
                        "target_date": _e_date.isoformat(),
                        "use_growth":  bool(_e_growth),
                        "rate_pct":    round(float(_e_rate), 2),
                        "added_at":    _g.get("added_at")
                                       or datetime.now().isoformat(timespec="minutes"),
                    }
                    save_goals_for(ck, goals)
                    st.rerun()
                if _del_clicked:
                    goals.pop(_i)
                    save_goals_for(ck, goals)
                    st.rerun()

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
            g_rate = ((float(g.get("rate_pct", _rf_pct) or 0.0) / 100.0)
                      if bool(g.get("use_growth", True)) else 0.0)
            total_monthly += goal_required_monthly(
                float(g.get("amount") or 0),
                float(g.get("saved")  or 0),
                mleft, g_rate)
        pct = min(100, (total_saved / total_target * 100)
                       if total_target else 0)
        st.markdown(
            f'<div style="margin-top:14px;background:{THEME["surface2"]};'
            f'            border:1.5px solid {THEME["primary"]};border-radius:14px;'
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
        st.caption("No goals yet \u2014 use the \u201cAdd a goal\u201d "
                   "form below to create your first one.")

    # ── Add a goal (stacked form; mobile-friendly) ───────────────────────
    with st.expander("Add a goal", expanded=not goals):
        with st.form("fr_goal_add", clear_on_submit=True):
            _a_name = st.text_input(
                "Goal", placeholder="e.g., House down payment, College fund",
                max_chars=80, key="fr_goal_add_name")
            _a_amt = st.number_input(
                "Target amount ($)", min_value=0.0, step=1000.0, value=0.0,
                format="%.0f", key="fr_goal_add_amt")
            _a_saved = st.number_input(
                "Already saved ($)", min_value=0.0, step=500.0, value=0.0,
                format="%.0f", key="fr_goal_add_saved")
            _a_date = st.date_input(
                "Target date", value=default_target, min_value=today,
                key="fr_goal_add_date")
            _a_growth = st.checkbox(
                "Grow savings at an assumed rate of return",
                value=True, key="fr_goal_add_growth")
            _a_rate = st.number_input(
                "Assumed annual return (%)",
                min_value=0.0, max_value=25.0,
                value=float(_rf_pct), step=0.25, format="%.2f",
                key="fr_goal_add_rate",
                help="Applied to both your current balance and your future "
                     "monthly contributions. Defaults to the risk-free rate "
                     f"(currently ~{_rf_pct:.2f}%).")
            _a_submit = st.form_submit_button(
                "Add goal", type="primary", use_container_width=True)
        if _a_submit:
            _nm = (_a_name or "").strip()
            if not _nm:
                st.warning("Give your goal a name.")
            elif _a_amt <= 0:
                st.warning("Enter a target amount greater than $0.")
            else:
                goals.append({
                    "name":        _nm,
                    "amount":      round(float(_a_amt), 2),
                    "saved":       round(float(_a_saved), 2),
                    "target_date": _a_date.isoformat(),
                    "use_growth":  bool(_a_growth),
                    "rate_pct":    round(float(_a_rate), 2),
                    "added_at":    datetime.now().isoformat(timespec="minutes"),
                })
                save_goals_for(ck, goals)
                st.rerun()

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
        g_rate = ((float(g.get("rate_pct", _rf_pct) or 0.0) / 100.0)
                  if bool(g.get("use_growth", True)) else 0.0)
        total_monthly_need += goal_required_monthly(
            float(g.get("amount") or 0),
            float(g.get("saved")  or 0),
            mleft, g_rate)

    gap = available - total_monthly_need
    gap_color = THEME["primary"] if gap >= 0 else THEME["risk"]
    gap_label = ("On track to fund your goals"
                 if gap >= 0
                 else f"Short by {fmt_money(abs(gap))}/month")

    st.markdown(
        f'<div style="height:10px"></div>'
        f'<div style="background:{THEME["surface2"]};border:1.5px solid {THEME["primary"]};'
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

    # Even the vertical rhythm: give the Reset/Save row the same breathing
    # room above it (vs. the summary card) as there is between the two
    # buttons. Tunable — bump this height if the gap should be larger.
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
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
def _render_my_info_tab():
    """Personal contact info — editable by the client.

    All fields except email are editable: first/last name, phone, address,
    ZIP, age. Email is the database key and is shown read-only with a
    short note explaining why it can't be changed here. Saves write back
    to USERS_FILE atomically via update_user(), and on success we update
    the in-memory session_state so subsequent tabs see the new values
    without requiring a full reload.

    Edit mode is gated by a "Edit info" button so the default view is a
    clean read-only summary — the same shape as the Advisor tab, just
    populated with the client's own data.
    """
    user = st.session_state.fr_user or {}

    # Toggle between view-only and edit modes via session state. Default
    # is view; clicking "Edit" flips to the edit form.
    edit_key = "fr_my_info_editing"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = False
    is_editing = st.session_state[edit_key]

    # ── Header card (matches Advisor tab visual style) ─────────────────
    full_name = (
        f"{(user.get('first_name') or '').strip()} "
        f"{(user.get('last_name') or '').strip()}"
    ).strip() or "—"
    age_val = user.get("age")
    age_str = f"Age {age_val}" if age_val else "Age not set"

    st.markdown(
        f'<div class="fr-card">'
        f'  <div style="display:flex;gap:18px;align-items:center;'
        f'              justify-content:space-between">'
        f'    <div style="flex:1">'
        f'      <div class="fr-eyebrow">Your information</div>'
        f'      <div style="font-size:1.15rem;font-weight:600;'
        f'                  color:{THEME["ink"]};margin-top:2px;'
        f'                  letter-spacing:-0.01em">{full_name}</div>'
        f'      <div style="font-size:0.88rem;color:{THEME["ink2"]};margin-top:2px">'
        f'        {age_str}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Contact info card — read-only or editable form ─────────────────
    if not is_editing:
        # ── READ-ONLY VIEW ──
        # Use the same icon vocabulary as the Advisor tab so the two
        # surfaces feel paired (mail = email, phone = phone, etc.).
        _icon_mail_mi = (
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            f'stroke="{THEME["primary"]}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex-shrink:0">'
            f'<rect x="3" y="5" width="18" height="14" rx="2"/>'
            f'<path d="M3 7l9 6 9-6"/></svg>'
        )
        _icon_phone_mi = (
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            f'stroke="{THEME["primary"]}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex-shrink:0">'
            f'<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 '
            f'19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 '
            f'2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 '
            f'9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 '
            f'2.81.7A2 2 0 0 1 22 16.92z"/></svg>'
        )
        _icon_pin_mi = (
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
            f'stroke="{THEME["primary"]}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex-shrink:0">'
            f'<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>'
            f'<circle cx="12" cy="10" r="3"/></svg>'
        )

        def _row(icon_svg, label, value):
            shown = (value or "").strip() or "—"
            return (
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'            padding:10px 0;border-bottom:1px solid '
                f'            {THEME["line"]}">'
                f'  {icon_svg}'
                f'  <div style="flex:1">'
                f'    <div style="font-size:0.7rem;text-transform:uppercase;'
                f'                letter-spacing:0.08em;color:{THEME["muted"]};'
                f'                font-weight:600">{label}</div>'
                f'    <div style="font-size:0.92rem;color:{THEME["ink"]};'
                f'                margin-top:1px">{shown}</div>'
                f'  </div>'
                f'</div>'
            )

        addr_lines = []
        if (user.get("address") or "").strip():
            addr_lines.append(user["address"].strip())
        if (user.get("zip") or "").strip():
            addr_lines.append(user["zip"].strip())
        addr_combined = ", ".join(addr_lines) if addr_lines else ""

        st.markdown(
            f'<div class="fr-card">'
            f'  <div class="fr-eyebrow" style="margin-bottom:6px">Contact</div>'
            f'  {_row(_icon_mail_mi,  "Email",   user.get("email", ""))}'
            f'  {_row(_icon_phone_mi, "Phone",   user.get("phone", ""))}'
            f'  {_row(_icon_pin_mi,   "Address", addr_combined)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Edit info →", key="fr_my_info_edit_btn",
                     use_container_width=True):
            st.session_state[edit_key] = True
            st.rerun()

    else:
        # ── EDIT FORM ──
        st.markdown(
            f'<div class="fr-card">'
            f'  <div class="fr-eyebrow" style="margin-bottom:10px">'
            f'    Edit your information</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Two-column layout for first/last name, then full-width for the rest.
        n1, n2 = st.columns(2)
        with n1:
            new_first = st.text_input(
                "First name",
                value=(user.get("first_name") or ""),
                key="fr_mi_first",
            )
        with n2:
            new_last = st.text_input(
                "Last name",
                value=(user.get("last_name") or ""),
                key="fr_mi_last",
            )
        # Email is read-only — it's the database key.
        st.text_input(
            "Email (cannot be changed here)",
            value=(user.get("email") or ""),
            disabled=True,
            help="Your email is how you sign in. Contact your advisor "
                 "if you need to change it.",
            key="fr_mi_email_readonly",
        )
        new_phone = st.text_input(
            "Phone",
            value=(user.get("phone") or ""),
            placeholder="(555) 555-5555",
            key="fr_mi_phone",
        )
        new_addr = st.text_input(
            "Address",
            value=(user.get("address") or ""),
            placeholder="123 Main St",
            key="fr_mi_addr",
        )
        a1, a2 = st.columns([1, 1])
        with a1:
            new_zip = st.text_input(
                "ZIP code",
                value=(user.get("zip") or ""),
                placeholder="12345",
                key="fr_mi_zip",
            )
        with a2:
            new_age = st.number_input(
                "Age",
                min_value=18, max_value=99, step=1,
                value=int(user.get("age") or 45),
                key="fr_mi_age",
            )

        # Action buttons — Save / Cancel
        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("Cancel", key="fr_mi_cancel",
                         use_container_width=True):
                st.session_state[edit_key] = False
                st.rerun()
        with b2:
            if st.button("Save changes", type="primary",
                         key="fr_mi_save", use_container_width=True):
                # ── Validation ──
                errors = []
                if not (new_first or "").strip():
                    errors.append("First name is required.")
                if not (new_last or "").strip():
                    errors.append("Last name is required.")
                phone_digits = "".join(
                    ch for ch in (new_phone or "") if ch.isdigit()
                )
                if (new_phone or "").strip() and len(phone_digits) < 10:
                    errors.append("Phone needs at least 10 digits.")
                if (new_zip or "").strip():
                    z = "".join(ch for ch in new_zip if ch.isdigit())
                    if len(z) not in (5, 9):
                        errors.append(
                            "ZIP should be 5 digits (12345) or 9 (12345-6789)."
                        )
                if errors:
                    for e in errors:
                        st.error(e)
                    return

                # Persist
                ok, msg = update_user(user.get("email", ""), {
                    "first_name": new_first,
                    "last_name":  new_last,
                    "phone":      new_phone,
                    "address":    new_addr,
                    "zip":        new_zip,
                    "age":        int(new_age),
                })
                if not ok:
                    st.error(msg or "Could not save changes.")
                    return

                # Update the in-memory user object so the rest of the
                # session sees the new values immediately. Re-fetching
                # from disk also works but is one extra I/O.
                refreshed = find_user(user.get("email", ""))
                if refreshed:
                    st.session_state.fr_user = refreshed

                st.session_state[edit_key] = False
                st.session_state.fr_flash = "Your information was updated."
                st.rerun()


def _bio_html(text: str) -> str:
    """Render an advisor bio as separate <p> paragraphs so the line breaks
    from the source (pasted-in site copy) show as real spacing. Raw HTML
    collapses newlines, so we split on them and wrap each paragraph; the
    .fr-bio container supplies the inter-paragraph margin (see CSS)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paras)


def _render_advisor_tab():
    """Full advisor profile card. Replaces the old single-line "Book your
    follow-up" CTA — now the client can see who their advisor actually is,
    where the firm is based, and reach them through any channel they prefer."""
    a = ADVISOR

    # Header with photo + name + title
    st.markdown(
        f'<div class="fr-card">'
        f'  <div style="display:flex;gap:18px;align-items:center">'
        f'    <div style="flex-shrink:0">{advisor_photo_svg("advisor")}</div>'
        f'    <div style="flex:1">'
        f'      <div class="fr-eyebrow">Your advisor</div>'
        f'      <div style="font-size:1.15rem;font-weight:600;color:{THEME["ink"]};'
        f'                  margin-top:2px;letter-spacing:-0.01em">{a["name"]}</div>'
        f'      <div style="font-size:0.88rem;color:{THEME["ink2"]};margin-top:2px">'
        f'        {a["title"]} · {a["firm"]}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="fr-bio" style="margin-top:16px;padding-top:14px;'
        f'              border-top:1px solid {THEME["line"]};'
        f'              font-size:0.92rem;color:{THEME["ink2"]};line-height:1.55">'
        f'    {_bio_html(a["bio"])}'
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

    # Digital business card (dot.cards). Outlined so it reads as a secondary
    # action next to the filled "Book a free 15-minute review" CTA below. Opens
    # the advisor's dot.cards profile in a new tab, where the client can tap once
    # to save the contact to their phone — no app required.
    _icon_card = (
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        f'stroke="{THEME["primary"]}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0;vertical-align:-4px;'
        f'margin-right:8px">'
        f'<rect x="2" y="4" width="20" height="16" rx="2.5"/>'
        f'<circle cx="8" cy="11" r="2.2"/>'
        f'<path d="M4.5 17c.6-1.8 2-2.6 3.5-2.6s2.9.8 3.5 2.6"/>'
        f'<path d="M15 9h4M15 12.5h4M15 16h2.5"/></svg>'
    )
    st.markdown(
        f'<a href="{DOT_CARD_URL}" target="_blank" rel="noopener" '
        f'   style="display:flex;align-items:center;justify-content:center;'
        f'          width:100%;box-sizing:border-box;text-align:center;'
        f'          background:{THEME["surface"]};color:{THEME["primary"]};'
        f'          border:1.5px solid {THEME["primary"]};padding:11px 16px;'
        f'          border-radius:10px;text-decoration:none;font-weight:600;'
        f'          font-size:0.95rem;margin:10px 0 18px">'
        f'  {_icon_card}Save my contact card'
        f'</a>',
        unsafe_allow_html=True,
    )

    # Schedule-a-call CTA — a single clickable button that opens the advisor's
    # HubSpot meetings page in a new tab, pre-filled with the client's
    # name/email from their session. HubSpot writes the booking to the CRM
    # (associated to the contact the portal already created) and to the
    # connected Google Calendar. (Previously this was a dark info card stacked
    # on top of a separate "Schedule my review" button — collapsed to one
    # button per design so there's a single, unambiguous action.)
    _icon_calendar = (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true" '
        'style="flex-shrink:0;margin-right:10px">'
        '<rect x="3" y="5" width="18" height="16" rx="2.5"/>'
        '<path d="M3 10h18"/>'
        '<path d="M8 3v4"/>'
        '<path d="M16 3v4"/>'
        '</svg>'
    )
    from urllib.parse import urlencode as _urlencode
    _su = st.session_state.get("fr_user") or {}
    _sp = {}
    if (_su.get("first_name") or "").strip(): _sp["firstName"] = _su["first_name"].strip()
    if (_su.get("last_name")  or "").strip(): _sp["lastName"]  = _su["last_name"].strip()
    if (_su.get("email")      or "").strip(): _sp["email"]     = _su["email"].strip()
    _schedule_url = SCHEDULE_URL + (("?" + _urlencode(_sp)) if _sp else "")
    st.markdown(
        f'<a href="{_schedule_url}" target="_blank" rel="noopener" '
        f'   style="display:flex;align-items:center;justify-content:center;'
        f'          width:100%;box-sizing:border-box;text-align:center;'
        f'          background:{THEME["primary"]};color:#fff;padding:14px 16px;'
        f'          border-radius:10px;text-decoration:none;font-weight:600;'
        f'          font-size:0.95rem;margin:0 0 18px">'
        f'  {_icon_calendar}Book a free 15-minute review →'
        f'</a>',
        unsafe_allow_html=True,
    )



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
            f'  14 questions across 5 sections — Context, Goals, Horizon, Tolerance, Outlook.'
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
            # Section label only — no fr-card wrapper. The card divs rendered
            # as empty divider boxes because Streamlit widgets attach as DOM
            # siblings rather than nesting inside raw HTML. A spaced-out
            # eyebrow label separates the sections without an empty box.
            st.markdown(
                f'<div class="fr-eyebrow" style="margin-top:18px">{q["section"]}</div>',
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
            label, _, _ = score_band(scores["capacity_score"], scores["tolerance_score"])
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
        f'            background:{THEME["surface2"]};border:1.5px solid {THEME["primary"]};'
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