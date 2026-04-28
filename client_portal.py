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
from datetime import datetime, date
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import (
    load_json as _shared_load_json,
    update_json as _shared_update_json,
    is_valid_email, normalize_email,
    score_to_label, score_to_allocation,
)

# ── Optional HubSpot CRM sync ────────────────────────────────────────────────
# Non-blocking: if the module isn't installed or the token isn't configured,
# the portal works exactly as before. Registrations always save locally first;
# HubSpot sync runs in a background thread with retry-and-backoff.
_HUBSPOT_IMPORT_ERROR: Optional[str] = None
try:
    import hubspot_sync  # type: ignore
    _HUBSPOT_AVAILABLE = True
except Exception as _e:
    hubspot_sync = None
    _HUBSPOT_AVAILABLE = False
    _HUBSPOT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    # Also log it so it shows up in the Streamlit terminal output, not just
    # the in-app diagnostic.
    import traceback as _tb
    print(f"[hubspot_sync] import failed: {_HUBSPOT_IMPORT_ERROR}")
    _tb.print_exc()

# ── DATA FILE LOCATIONS ──────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _data_path(name: str) -> str: return os.path.join(_APP_DIR, name)

USERS_FILE           = _data_path("ra_users.json")
PROFILES_FILE        = _data_path("risk_profiles.json")
CLIENT_HOLDINGS_FILE = _data_path("client_holdings.json")

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

        .stTabs [data-baseweb="tab-list"] {{
            background: transparent;
            border-bottom: 1px solid {THEME['line']};
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {THEME['muted']};
            background: transparent;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            color: {THEME['ink']} !important;
            border-bottom: 2px solid {THEME['primary']} !important;
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
     "type": "select", "options": [
        ("Under $50,000",            30),
        ("$50,000 – $100,000",       45),
        ("$100,000 – $200,000",      60),
        ("$200,000 – $500,000",      75),
        ("$500,000 – $1,000,000",    85),
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
     "type": "select", "options": [
        ("Under $100,000",              30),
        ("$100,000 – $500,000",         50),
        ("$500,000 – $1,000,000",       65),
        ("$1,000,000 – $5,000,000",     75),
        ("$5,000,000 – $25,000,000",    85),
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
     "type": "select", "options": [
        ("No target — I'm just building",              60),
        ("Under $250,000",                             40),
        ("$250,000 – $1,000,000",                      55),
        ("$1,000,000 – $5,000,000",                    70),
        ("$5,000,000 – $25,000,000",                   80),
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
     "type": "select", "options": [
        ("No major expenses planned",   75),
        ("Possibly — under $50,000",    60),
        ("Yes — $50,000 to $250,000",   40),
        ("Yes — over $250,000",         25),
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
    """(label, hex, soft_bg) — prototype's Risk Checkup framing.
       75+ Strong, 65+ Stable, 40+ Watch, <40 At risk."""
    if score >= 75: return "Strong",  THEME["healthy"], THEME["healthy_soft"]
    if score >= 65: return "Stable",  THEME["healthy"], THEME["healthy_soft"]
    if score >= 40: return "Watch",   THEME["caution"], THEME["caution_soft"]
    return            "At risk", THEME["risk"],    THEME["risk_soft"]


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
    cmap = {
        "healthy": (THEME["healthy_soft"], THEME["healthy"], "Healthy"),
        "caution": (THEME["caution_soft"], THEME["caution"], "Watch"),
        "risk":    (THEME["risk_soft"],    THEME["risk"],    "Alert"),
    }
    bg, fg, default_text = cmap.get(status, cmap["healthy"])
    text = label or default_text
    return (f'<span class="fr-chip" style="background:{bg};color:{fg}">'
            f'{text}</span>')


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
        f'<div style="max-width:520px;margin:30px auto 0;padding:0 28px">'
        f'  <div style="display:flex;align-items:center;gap:10px;margin-bottom:36px">'
        f'    {logo_mark(THEME["primary"], 26)}'
        f'    <span style="font-size:0.82rem;font-weight:600;letter-spacing:0.12em;'
        f'                 color:{THEME["ink"]};text-transform:uppercase">'
        f'      Foresight Risk'
        f'    </span>'
        f'  </div>'
        f'  <h1 style="font-size:2rem;line-height:1.18;color:{THEME["ink"]};'
        f'             font-weight:500;margin:14px 0 28px;letter-spacing:-0.015em">'
        f'    A complete financial risk profile in less than 2 minutes.'
        f'  </h1>'
        f'  <div style="display:flex;gap:24px;color:{THEME["muted"]};'
        f'              font-size:0.92rem;margin-bottom:24px;align-items:center">'
        f'    <span>{_icon_lock}Encrypted</span>'
        f'    <span>{_icon_shield}Fiduciary</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Spacer pushes the CTA toward the bottom of the visible area, matching
    # the mockup's vertical rhythm
    st.markdown('<div style="height:120px"></div>', unsafe_allow_html=True)

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
        st.markdown('<div class="fr-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="fr-card">', unsafe_allow_html=True)

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
        elif q["type"] == "multi":
            opts = q["options"]
            default = ([d for d in (prev or []) if d in opts]
                       if isinstance(prev, list) else [])
            val = st.multiselect(q["text"], opts, default=default,
                                  key=f"fr_qz_{q['id']}",
                                  max_selections=q.get("max_pick"),
                                  label_visibility="collapsed")
            answered = len(val or []) > 0
        else:
            val = None; answered = False

        st.markdown('</div>', unsafe_allow_html=True)

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
                st.session_state.fr_answers[q["id"]] = val
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
    full RiskRing + diagnosis are shown.

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
        f'    Save your results to view your full risk score, diagnosis, and '
        f'    personalized recommendations. Email and phone only — address '
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
        st.markdown('<div class="fr-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

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
                # Local save above is the source of truth — if HubSpot is down,
                # missing a token, or even uninstalled, registration still
                # succeeds. sync_now=True tries one synchronous attempt
                # (~1-2s on success); failures auto-fall-back to the
                # background queue so nothing is lost.
                hs_status = None
                hs_skip_reason = None  # set when sync is skipped, for diagnostics

                if not _HUBSPOT_AVAILABLE:
                    hs_skip_reason = (
                        f"hubspot_sync module not importable "
                        f"({_HUBSPOT_IMPORT_ERROR or 'unknown error'})"
                    )
                elif not hubspot_sync.is_configured():
                    hs_skip_reason = (
                        "hubspot_sync.is_configured() returned False "
                        "(token missing or invalid?)"
                    )
                else:
                    try:
                        scores = st.session_state.fr_scores or {}
                        overall = int(scores.get("overall_score", 0))
                        label, _, _ = (score_band(overall) if overall
                                       else ("", "", ""))
                        hs_status = hubspot_sync.sync_contact(
                            first   = st.session_state.fr_first,
                            last    = st.session_state.fr_last,
                            email   = email,
                            phone   = phone,
                            address = addr,
                            zipcode = zipcode,
                            age     = int(st.session_state.fr_age),
                            risk_score = overall,
                            risk_label = label,
                            sync_now   = True,
                        )
                        # Log whatever came back so it's visible in the
                        # Streamlit terminal, not just the UI.
                        print(f"[hubspot_sync] sync_contact returned: {hs_status}")
                    except Exception as e:
                        import traceback as _tb
                        _tb.print_exc()
                        hs_status = {"ok": False, "error": str(e),
                                     "exception_type": type(e).__name__}

                # Log them in and land on the dashboard.  If HubSpot returned
                # a status, surface it on the dashboard via the flash banner so
                # we're not silent about failures the user might want to know
                # about (advisor team won't see this contact yet, etc.).
                st.session_state.fr_user = user
                if hs_skip_reason is not None:
                    # Sync was skipped before we ever reached HubSpot. Stash
                    # the reason where the dashboard can pick it up; don't
                    # alarm the end user with a stack trace.
                    st.session_state.fr_hubspot_debug = hs_skip_reason
                    st.session_state.fr_flash = "Profile saved — welcome!"
                elif hs_status is None:
                    st.session_state.fr_flash = "Profile saved — welcome!"
                elif hs_status.get("ok") and not hs_status.get("queued"):
                    st.session_state.fr_flash = (
                        "Profile saved — welcome! "
                        "Your advisor has been notified.")
                elif hs_status.get("queued"):
                    st.session_state.fr_flash = (
                        "Profile saved — welcome! "
                        "Sending a copy to your advisor in the background.")
                else:
                    # Sync ran but failed.  Stash the failure for the
                    # diagnostics panel; user-facing message stays calm.
                    st.session_state.fr_hubspot_debug = (
                        f"sync_contact failed: {hs_status}"
                    )
                    st.session_state.fr_flash = "Profile saved — welcome!"
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

    # ── HubSpot diagnostic panel ────────────────────────────────────────────
    # Set to False once the integration is verified to be working in production.
    # Leaves the panel hidden unless there's a debug payload to surface.
    _ADMIN_DIAGNOSTICS_ENABLED = True

    _hs_debug = st.session_state.get("fr_hubspot_debug")
    _hs_show_status = (
        _ADMIN_DIAGNOSTICS_ENABLED
        or st.session_state.get("fr_hubspot_show_status", False)
    )
    if _hs_debug or _hs_show_status:
        with st.expander("🔧 HubSpot sync status (admin)", expanded=False):
            st.write(f"**Module available:** `{_HUBSPOT_AVAILABLE}`")
            if _HUBSPOT_IMPORT_ERROR:
                st.error(f"Import error: {_HUBSPOT_IMPORT_ERROR}")
            if _HUBSPOT_AVAILABLE and hubspot_sync is not None:
                try:
                    st.write(f"**is_configured():** `{hubspot_sync.is_configured()}`")
                except Exception as e:
                    st.error(f"is_configured() raised: {type(e).__name__}: {e}")
                # Queue depth and dead-letter — these reveal whether past
                # syncs have been failing silently.
                try:
                    st.write(f"**Pending in queue:** `{hubspot_sync.pending_count()}`")
                except Exception as e:
                    st.write(f"pending_count() raised: {e}")
                try:
                    dead = hubspot_sync.get_deadletter()
                    st.write(f"**Dead-lettered (failed permanently):** `{len(dead)}`")
                    if dead:
                        st.write("Most recent failure:")
                        st.json(dead[-1])
                        if st.button("Clear dead-letter file",
                                     key="fr_clear_hs_dead"):
                            hubspot_sync.clear_deadletter()
                            st.rerun()
                except Exception as e:
                    st.write(f"get_deadletter() raised: {e}")
            if _hs_debug:
                st.write("**Last sync attempt:**")
                st.code(str(_hs_debug))
                if st.button("Clear debug", key="fr_clear_hs_debug"):
                    st.session_state.fr_hubspot_debug = None
                    st.rerun()

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
        st.markdown('<div class="fr-card">', unsafe_allow_html=True)
        h1, h2 = st.columns([1.05, 1])
        with h1:
            st.plotly_chart(make_risk_ring(overall, height=300),
                use_container_width=True,
                config={"displayModeBar": False})
        with h2:
            cap = int(profile.get("capacity_score", 50))
            tol = int(profile.get("tolerance_score", 50))
            if overall >= 75:
                diag = "On a strong trajectory."
                detail = "Capacity and tolerance both align — keep building toward your horizon."
            elif overall >= 60:
                diag = "On track, with two areas to watch."
                detail = "Strong signal across most dimensions. Review liquidity and outlook tilts."
            elif overall >= 40:
                diag = "Some areas need attention."
                detail = ("Lower-than-average " +
                          ("capacity" if cap < tol else "tolerance") +
                          " is pulling your score down.")
            else:
                diag = "Several areas at risk."
                detail = "Significant rebalancing recommended — talk to your advisor."

            st.markdown(
                f'<div style="padding-top:18px">'
                f'  <div class="fr-eyebrow">Diagnosis</div>'
                f'  <div style="font-size:1rem;color:{THEME["ink"]};font-weight:600;'
                f'              margin-top:4px;line-height:1.3">{diag}</div>'
                f'  <div style="font-size:0.85rem;color:{THEME["ink2"]};'
                f'              margin-top:8px;line-height:1.5">{detail}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("View full profile →", key="fr_view_profile",
                         use_container_width=True):
                st.session_state.fr_view = "edit_profile"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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
              delta: str = "") -> str:
        chip = status_chip(status) if status else ""
        delta_color = (THEME["healthy"] if not str(delta).startswith("-")
                       else THEME["risk"])
        delta_html = (f'<span class="fr-mono" style="color:{delta_color}">'
                      f'{delta}</span>' if delta else "")
        return (
            f'<div class="fr-vital">'
            f'  <div style="display:flex;align-items:center;justify-content:space-between">'
            f'    <span class="fr-vital-label">{label}</span>{chip}'
            f'  </div>'
            f'  <div class="fr-vital-value">{value}</div>'
            f'  <div class="fr-vital-detail">'
            f'    <span>{detail}</span>{delta_html}'
            f'  </div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'            margin:18px 2px 10px">'
        f'  <div class="fr-eyebrow">Financial Vitals</div>'
        f'  <span style="font-size:0.72rem;color:{THEME["primary"]};font-weight:600">'
        f'    This month'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        nw_status = "healthy" if vitals["net_worth"] > 0 else "caution"
        nw_delta  = (fmt_pct(vitals["gain_pct"]) if vitals["cost_basis"] else "")
        st.markdown(_tile(
            "Net Worth", fmt_money(vitals["net_worth"]),
            f"{len(holdings)} positions" if holdings else "no positions yet",
            nw_status, delta=nw_delta,
        ), unsafe_allow_html=True)

    with g2:
        cap_status = ("healthy" if cap >= 60 else
                      "caution" if cap >= 40 else "risk")
        st.markdown(_tile(
            "Risk Capacity", str(cap) if cap else "—",
            "ability to absorb loss", cap_status,
        ), unsafe_allow_html=True)

    with g3:
        cash_pct = (vitals["cash"] / vitals["net_worth"] * 100
                    if vitals["net_worth"] else 0)
        cash_status = "healthy" if 5 <= cash_pct <= 20 else "caution"
        st.markdown(_tile(
            "Cash Position", fmt_money(vitals["cash"]),
            f"{cash_pct:.1f}% of portfolio" if vitals["net_worth"] else "—",
            cash_status,
        ), unsafe_allow_html=True)

    with g4:
        tol_status = ("healthy" if tol >= 60 else
                      "caution" if tol >= 40 else "risk")
        st.markdown(_tile(
            "Risk Tolerance", str(tol) if tol else "—",
            "comfort with volatility", tol_status,
        ), unsafe_allow_html=True)

    # ── Trend card ──────────────────────────────────────────────────────────
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
            f'      <div class="fr-eyebrow">Net Worth Trend</div>'
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

    # ── Holdings card ───────────────────────────────────────────────────────
    st.markdown('<div class="fr-card">', unsafe_allow_html=True)
    h_l, h_r = st.columns([3, 1])
    with h_l:
        st.markdown(
            f'<div class="fr-eyebrow">Holdings</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:{THEME["ink"]};'
            f'            margin-top:2px">{len(holdings)} positions</div>',
            unsafe_allow_html=True,
        )
    with h_r:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("Manage", key="fr_manage_holdings",
                     use_container_width=True):
            st.session_state.fr_view = "edit_holdings"
            st.rerun()

    if holdings:
        rows = []
        for tk, h in holdings.items():
            sh = float(h.get("shares") or 0)
            px = float((quotes.get(tk) or {}).get("price") or 0)
            val = sh * px
            day = float((quotes.get(tk) or {}).get("change_pct") or 0)
            rows.append((tk, sh, px, val, day))
        rows.sort(key=lambda r: -r[3])
        rows = rows[:5]
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
        if len(holdings) > 5:
            st.markdown(
                f'<div style="text-align:center;padding-top:10px;'
                f'            font-size:0.78rem;color:{THEME["muted"]}">'
                f'+ {len(holdings) - 5} more — tap Manage to see all</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="text-align:center;padding:24px 0;color:{THEME["muted"]};'
            f'            font-size:0.9rem">'
            f'  No holdings yet. Add your first position to see your portfolio here.'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Advisor CTA ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="fr-cta-dark">'
        f'  <div class="fr-cta-icon">📅</div>'
        f'  <div style="flex:1">'
        f'    <div style="font-size:0.92rem;font-weight:600">Book your follow-up</div>'
        f'    <div style="font-size:0.78rem;opacity:0.7;margin-top:2px">'
        f'      15-min call with a fiduciary advisor'
        f'    </div>'
        f'  </div>'
        f'  <div style="font-size:1.2rem">→</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Schedule call", key="fr_schedule_btn",
                 use_container_width=True):
        st.session_state.fr_flash = "Booking flow coming soon — your advisor will reach out."
        st.rerun()

    # ── Bottom nav (visual indicator) ───────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;justify-content:space-around;padding:14px 0 6px;'
        f'            margin-top:18px;border-top:1px solid {THEME["line"]}">'
        f'  <div style="text-align:center;color:{THEME["primary"]};'
        f'              font-size:0.7rem;font-weight:600">'
        f'    <div style="font-size:1.05rem">🏠</div>Home'
        f'  </div>'
        f'  <div style="text-align:center;color:{THEME["muted"]};'
        f'              font-size:0.7rem;font-weight:600">'
        f'    <div style="font-size:1.05rem">⭐</div>Plan'
        f'  </div>'
        f'  <div style="text-align:center;color:{THEME["muted"]};'
        f'              font-size:0.7rem;font-weight:600">'
        f'    <div style="font-size:1.05rem">📅</div>Advisor'
        f'  </div>'
        f'  <div style="text-align:center;color:{THEME["muted"]};'
        f'              font-size:0.7rem;font-weight:600">'
        f'    <div style="font-size:1.05rem">👤</div>Me'
        f'  </div>'
        f'</div>',
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
            val = st.multiselect(
                q["text"], opts, default=default,
                key=f"fr_q_{qid}",
                max_selections=q.get("max_pick"),
            )
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