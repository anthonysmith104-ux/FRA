import streamlit as st
# ── Standard imports moved to top of file ─────────────────────────────────────
# These were previously imported around line 585, after several functions that
# used them. Worked because Streamlit defers function calls until after the
# module finishes loading, but fragile and confusing. Moved to top.
import os
import json
import functools
import hashlib
import secrets as _secrets
import warnings as _warnings
from io import BytesIO
from datetime import datetime, date

import pandas as pd
import numpy as np
import requests as _requests
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta

# Shared module — single source of truth for storage, scoring, validation
from shared import (
    is_valid_email,
    normalize_email,
    compute_risk_score as _shared_compute_risk_score,
    score_to_label as _shared_score_to_label,
    score_to_allocation as _shared_score_to_allocation,
    sharpe_ratio as _shared_sharpe,
    make_secure_token,
    DEFAULT_RISK_FREE_RATE,
)

# All shared JSON I/O now goes through data_store, which transparently
# reads/writes to a GitHub-backed shared repo (configured in Streamlit
# secrets) so the advisor app and client portal see the same data.
# When secrets aren't configured (local dev), data_store falls back to
# local-disk JSON. Drop-in replacement — same signatures as shared.*.
from data_store import (
    load_json   as _shared_load_json,
    save_json   as _shared_save_json,
    update_json as _shared_update_json,
)

# Schwab model portfolios (loaded from schwab_portfolios.json next to app.py).
# Powers the Broad-ETF Alternate tier in the proposal flow. Imported with a
# soft fallback so the app still runs if the file is missing — in that case
# the alternate tier falls back to the legacy build_tier_proposal output.
try:
    import portfolios as _schwab_portfolios
    _SCHWAB_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _schwab_portfolios = None
    _SCHWAB_AVAILABLE = False
    _warnings.warn(f"schwab_portfolios.json not loaded: {_e}", RuntimeWarning)

# Build the Schwab expense-ratio lookup once at module load. Used by
# _expense_ratio_for_ticker() as the highest-priority source so that
# Schwab-published ERs always win over Alpha Vantage / FMP / the
# hardcoded fallback table.
#
# Why this matters: the JSON carries authoritative per-fund ERs that
# Schwab publishes alongside their model portfolios. If we don't seed
# them, a portfolio containing PRFDX (T. Rowe Price Equity Income, ER
# 0.69%) gets resolved through the API path — and if the API doesn't
# have data for that mutual fund (Alpha Vantage's ETF_PROFILE has gaps
# for active MFs), the resolver silently returns 0.0%, understating
# the portfolio's true cost.
#
# The lookup is decimal-keyed (0.0069 = 0.69%) to match the format
# _expense_ratio_for_ticker returns. {} when Schwab module unavailable.
try:
    _SCHWAB_ER_LOOKUP = (_schwab_portfolios.get_all_expense_ratios()
                         if _SCHWAB_AVAILABLE else {})
except Exception as _e:
    _SCHWAB_ER_LOOKUP = {}
    _warnings.warn(f"Schwab ER lookup failed: {_e}", RuntimeWarning)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  OPENBB INTEGRATION                                                      │
# │  Install: pip install openbb                                             │
# │  OpenBB provides reliable financial data from multiple sources.          │
# │  Falls back to yfinance automatically if OpenBB is not installed.        │
# └─────────────────────────────────────────────────────────────────────────┘
import yfinance as yf

# ── OpenBB Integration ────────────────────────────────────────────────────────
# OpenBB Platform provides reliable, multi-source financial data
# Install: pip install openbb
# Falls back to yfinance if OpenBB not available
_OBB_AVAILABLE = False
try:
    from openbb import obb
    # Test that it works
    _OBB_AVAILABLE = True
except ImportError:
    pass

def _obb_get_prices(tickers, start_date, end_date):
    """Fetch prices via OpenBB Platform with automatic source fallback.
    Returns a DataFrame with ticker columns and date index, auto-adjusted (splits+dividends).
    """
    if not _OBB_AVAILABLE:
        return None
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        dfs = []
        for tkr in tickers:
            try:
                # Try ETF first, then equity
                try:
                    raw = obb.etf.historical(
                        tkr,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        adjustment="splits_and_dividends",
                        provider="yfinance"
                    ).to_dataframe()
                except Exception:
                    raw = obb.equity.price.historical(
                        tkr,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        adjustment="splits_and_dividends",
                        provider="yfinance"
                    ).to_dataframe()
                if raw is not None and len(raw) > 0:
                    # OpenBB returns date as index, "close" (lowercase) as column
                    close_col = next((c for c in ["close","Close","adj_close","Adj Close"]
                                       if c in raw.columns), None)
                    if close_col:
                        s = raw[close_col].rename(tkr)
                        s.index = pd.to_datetime(s.index)
                        dfs.append(s)
            except Exception:
                continue
        if dfs:
            return pd.concat(dfs, axis=1).sort_index()
    except Exception:
        pass
    return None

# ── Cached price fetcher — keyed on tickers+dates so each period is separate ──
@st.cache_data(ttl=3600, show_spinner=False)
def get_prices_cached(ticker_tuple, start_str, end_str):
    """Cached wrapper around get_prices. Cache key includes dates so
    1yr, 3yr, 5yr, 10yr each get their own independent cache entry."""
    prices, src = get_prices(list(ticker_tuple), start_str, end_str)
    return prices, src


# ── COMPOSITE OPTIMIZATION ENGINE ─────────────────────────────────────────────
# Blends multiple strategy weight vectors based on user slider intensities.
# Each strategy contributes proportionally to its slider value (0-10 scale).

def blend_strategies(user_knobs, computed_portfolios):
    """
    user_knobs:          {'min_variance': 0.5, 'max_sharpe': 0.3, 'equal_weight': 0.2, ...}
    computed_portfolios: {'min_variance': [w1,w2,...], 'max_sharpe': [w1,w2,...], ...}
    Returns blended weight list normalized to sum=1.0
    """
    if not computed_portfolios:
        return None
    n = len(next(iter(computed_portfolios.values())))
    final_weights = np.zeros(n)
    total_influence = sum(user_knobs.get(k, 0) for k in computed_portfolios)
    if total_influence == 0:
        return None
    for strategy, w_vec in computed_portfolios.items():
        knob_val = user_knobs.get(strategy, 0)
        final_weights += np.array(w_vec) * (knob_val / total_influence)
    # Normalize
    s = final_weights.sum()
    if s > 0:
        final_weights = final_weights / s
    return final_weights.tolist()


# Strategy groups for UI organization
STRATEGY_GROUPS = {
    "🛡️ Safety": {
        "Min Variance":  "Minimize portfolio volatility — shifts toward bonds/utilities",
        "Min CVaR":      "Minimize tail risk losses — most defensive positioning",
        "Min CDaR":      "Minimize drawdown risk — protects against sustained declines",
        "Risk Parity":   "Equal risk contribution — diversified across all assets",
    },
    "🚀 Performance": {
        "Max Sharpe":      "Maximize risk-adjusted return — best return per unit of risk",
        "Max Sortino":     "Maximize downside-adjusted return — focuses on upside capture",
        "Max CVaR Ratio":  "Maximize return vs tail risk — best performance in stress",
        "HRP":             "Hierarchical Risk Parity — data-driven diversification",
    },
    "⚖️ Structural": {
        "Equal Weight":        "Equal allocation to all assets — maximum diversification",
        "NCO":                 "Nested Clusters Optimization — grouped diversification",
        "Max Diversification": "Maximize diversification ratio — spread correlation risk",
        "EF Blend":            "Blended frontier — weighted average of all strategies",
    },
}


# ── PROXY TICKER MAPPING (top-level) ─────────────────────────────────────
# ── PROXY TICKER MAPPING ──────────────────────────────
# For tickers with limited history, map to a proxy with longer history
# The proxy captures the same underlying risk factor / asset class
# ── COMPREHENSIVE PROXY & ASSET CLASS MAP ──────────────────────────────────
# Tier 1: Direct proxies — same asset, older ticker with 10yr+ history
# Tier 2: Asset class proxy — same risk factor if no direct proxy exists
# Tier 3: Risk-free rate (BIL) — last resort for truly novel instruments

PROXY_MAP = {
    # ── Bitcoin / Crypto ──────────────────────────────────────────────
    "FBTC":  "GBTC",   "IBIT":  "GBTC",   "ARKB":  "GBTC",
    "BITB":  "GBTC",   "HODL":  "GBTC",   "BRRR":  "GBTC",
    "EZBC":  "GBTC",   "BTCO":  "GBTC",   "DEFI":  "GBTC",
    "BTCW":  "GBTC",

    # ── Gold & Precious Metals ────────────────────────────────────────
    "GLDM":  "GLD",    "IAUM":  "GLD",    "SGOL":  "GLD",
    "BAR":   "GLD",    "OUNZ":  "GLD",    "PHYS":  "GLD",
    "RING":  "GDX",    "GDXJ":  "GDX",    # Junior miners → senior miners

    # ── Short-Term Treasuries / Cash / Risk-Free ──────────────────────
    "SGOV":  "BIL",    "USFR":  "BIL",    "TFLO":  "BIL",
    "BILS":  "BIL",    "ICSH":  "SHV",    "JPST":  "SHV",
    "CSHI":  "BIL",    "FLOT":  "SHV",    "MINT":  "SHV",
    "NEAR":  "SHV",    "ULST":  "BIL",    "BOXX":  "BIL",

    # ── Intermediate / Long Treasuries ───────────────────────────────
    "VGIT":  "IEF",    "SCHR":  "IEF",    "SPTI":  "IEF",
    "VGLT":  "TLT",    "SPTL":  "TLT",    "ZROZ":  "TLT",
    "EDV":   "TLT",    "GOVZ":  "TLT",

    # ── US Equity — broad market ──────────────────────────────────────
    "VOO":   "SPY",    "IVV":   "SPY",    "FXAIX": "SPY",
    "SPLG":  "SPY",    "CSPX":  "SPY",    "RSP":   "SPY",
    "FNILX": "SPY",

    # ── US Total Market ───────────────────────────────────────────────
    "ITOT":  "VTI",    "SCHB":  "VTI",    "FZROX": "VTI",
    "BKLC":  "VTI",

    # ── US Small Cap Value ────────────────────────────────────────────
    "AVUV":  "VBR",    "DFSV":  "VBR",    "VIOV":  "VBR",
    "SLYV":  "VBR",

    # ── US Small Cap Blend ────────────────────────────────────────────
    "SCHA":  "VB",     "IJR":   "VB",     "VTWO":  "VB",

    # ── US Mid Cap ────────────────────────────────────────────────────
    "IVOO":  "VO",     "IJH":   "VO",     "SPMD":  "VO",

    # ── International Developed ───────────────────────────────────────
    "IDEV":  "EFA",    "SPDW":  "EFA",    "SCHF":  "EFA",
    "VEA":   "EFA",    "DFIC":  "EFA",

    # ── Emerging Markets ──────────────────────────────────────────────
    "AVEM":  "VWO",    "DFEM":  "VWO",    "SPEM":  "VWO",
    "IEMG":  "VWO",    "EEMS":  "VWO",

    # ── Momentum / Factor ETFs ────────────────────────────────────────
    "SPMO":  "MTUM",   "QMOM":  "MTUM",   "VFMO":  "MTUM",
    "IMOM":  "MTUM",   "JMOM":  "MTUM",

    # ── Quality / Dividend ────────────────────────────────────────────
    "DGRW":  "VIG",    "DGRO":  "VIG",    "SDY":   "VIG",
    "SCHD":  "VIG",    "FDVV":  "VIG",

    # ── Sector ETFs (newer → older equivalent) ────────────────────────
    "UTES":  "XLU",    "FIDU":  "XLI",    "FTEC":  "XLK",
    "FHLC":  "XLV",    "FENY":  "XLE",    "FREL":  "VNQ",

    # ── REITs ─────────────────────────────────────────────────────────
    "SCHH":  "VNQ",    "USRT":  "VNQ",    "BBRE":  "VNQ",
    "RWR":   "VNQ",

    # ── High Yield / Corporate Bonds ──────────────────────────────────
    "VWEHX": "HYG",    "FAHDX": "HYG",    "USHY":  "HYG",
    "SHYG":  "HYG",    "BKHY":  "HYG",

    # ── Investment Grade Bonds ────────────────────────────────────────
    "VCIT":  "LQD",    "SPIB":  "LQD",    "IGIB":  "LQD",
    "FBND":  "BND",    "BNDX":  "BND",

    # ── Commodities / Managed Futures ─────────────────────────────────
    "CTA":   "DJP",    "DBMF":  "DJP",    "KMLM":  "DJP",
    "WTMF":  "DJP",    "MNA":   "SPY",    "PDBC":  "DJP",
    "COMT":  "DJP",    "COMB":  "DJP",

    # ── Inflation Protected ───────────────────────────────────────────
    "VTIP":  "TIP",    "STIP":  "TIP",    "SPIP":  "TIP",
    "PBTP":  "TIP",

    # ── Leveraged / Inverse (map to underlying) ───────────────────────
    "SSO":   "SPY",    "UPRO":  "SPY",    "QLD":   "QQQ",
    "TQQQ":  "QQQ",    "TMF":   "TLT",    "UBT":   "TLT",
}

# ── ASSET CLASS FALLBACK MAP ─────────────────────────────────────────────────
# If a ticker isn't in PROXY_MAP and has short history,
# classify by name patterns → assign asset class proxy
ASSET_CLASS_PATTERNS = [
    # (regex pattern, proxy, description)
    (r'bitcoin|btc|crypto|coin',       "GBTC", "crypto"),
    (r'gold|silver|precious|metal',    "GLD",  "precious metals"),
    (r'bond|fixed|income|treasury|tsy',"IEF",  "bonds"),
    (r'short.*term|cash|money.market', "BIL",  "short-term/cash"),
    (r'emerging|develop.*market',      "VWO",  "emerging markets"),
    (r'small.*cap|small.*value',       "VBR",  "small cap"),
    (r'reit|real.estate',              "VNQ",  "real estate"),
    (r'commodity|commodit',            "DJP",  "commodities"),
    (r'dividend|income',               "VIG",  "dividend"),
    (r'momentum|growth',               "MTUM", "momentum"),
    (r'international|global|world',    "EFA",  "international"),
    (r'high.yield|junk',               "HYG",  "high yield"),
]

# Risk-free rate proxy: 3-month T-Bill (BIL)
# Used when no better proxy can be found — represents the minimum expected return
RISK_FREE_PROXY = "BIL"


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_ticker(tkr, start_date_str, min_days=120):
    """Fast single-ticker proxy lookup — uses PROXY_MAP first, no network calls."""
    import re as _re
    # Tier 1: direct proxy in PROXY_MAP (no network needed)
    proxy = PROXY_MAP.get(tkr.upper())
    if proxy and proxy != tkr:
        return proxy, f"proxy ({proxy})"
    # Tier 2: asset-class pattern match on symbol
    tkr_lower = tkr.lower()
    for pattern, cls_proxy, cls_name in ASSET_CLASS_PATTERNS:
        if _re.search(pattern, tkr_lower):
            return cls_proxy, f"class proxy ({cls_proxy} = {cls_name})"
    # Tier 3: return as-is (will be handled by fill_missing_history)
    return tkr, "direct"

def _stitch_series(orig_series, proxy_series):
    """Splice proxy history onto the start of orig_series so the combined series
    spans the full proxy range while keeping original's real prices where available.

    The proxy segment is scaled so its last pre-join value equals orig's first real
    value — preserves return continuity across the splice. Returns a single pd.Series.
    """
    orig_clean  = orig_series.dropna()
    proxy_clean = proxy_series.dropna()
    if orig_clean.empty:
        return proxy_clean if not proxy_clean.empty else orig_series
    if proxy_clean.empty:
        return orig_clean
    first_orig_date = orig_clean.index[0]
    # Proxy points strictly BEFORE orig's first date
    proxy_before = proxy_clean.loc[proxy_clean.index < first_orig_date]
    if proxy_before.empty:
        return orig_clean
    # Scale factor so proxy's last pre-join value = orig's first real value
    scale = orig_clean.iloc[0] / proxy_before.iloc[-1]
    stitched_proxy = proxy_before * scale
    import pandas as _pd
    return _pd.concat([stitched_proxy, orig_clean])


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices_with_proxies(ticker_list_or_tuple, start_date, end_date, min_days=120):
    """Batch download with intelligent proxy substitution and STITCHING.

    Strategy for each ticker:
      1. If the original has enough history for the full window, use it as-is.
      2. If the original has SOME history but not enough, stitch the proxy's
         earlier history onto it (scaled at the join point for continuity).
      3. If the original has no history at all, fall back to the proxy directly.
      4. If no proxy, try the asset-class pattern proxy.
      5. If nothing else works, fall back to risk-free rate (BIL) — represents
         the minimum expected return, so it doesn't distort the backtest.
    """
    import re as _re
    ticker_list = list(ticker_list_or_tuple)
    start_str   = str(start_date)
    end_str     = str(end_date)

    # Step 1: resolve proxies (no network — PROXY_MAP + pattern match only)
    resolved    = {}
    proxy_notes = {}
    for tkr in ticker_list:
        actual, note = resolve_ticker(tkr, start_str, min_days)
        resolved[tkr] = actual
        if actual != tkr:
            proxy_notes[tkr] = note

    # Step 2: batch download originals + proxies + asset-class proxies + BIL
    class_proxies = [p for _, p, _ in ASSET_CLASS_PATTERNS]
    all_fetch = list(set(ticker_list + list(resolved.values()) + class_proxies + [RISK_FREE_PROXY]))
    try:
        prices, _ = get_prices_cached(tuple(sorted(all_fetch)), start_str, end_str)
    except Exception:
        return pd.DataFrame(), {}
    if prices.empty:
        return pd.DataFrame(), {}

    # Step 3: for each ticker, build the best possible series
    result = pd.DataFrame(index=prices.index)
    for orig in ticker_list:
        orig_series  = prices[orig]  if orig in prices.columns  else pd.Series(dtype=float)
        orig_len     = len(orig_series.dropna())
        proxy        = resolved.get(orig, orig)
        proxy_series = prices[proxy] if proxy in prices.columns else pd.Series(dtype=float)
        proxy_len    = len(proxy_series.dropna())

        # Case 1: original has enough full-window history
        if orig_len >= min_days and orig_len >= 0.85 * prices.shape[0]:
            result[orig] = orig_series
            continue

        # Case 2: original has some data — stitch proxy history to fill earlier period
        if orig_len >= 20 and proxy != orig and proxy_len >= min_days:
            stitched = _stitch_series(orig_series, proxy_series)
            # Reindex to shared frame, forward-fill weekends/holidays
            result[orig] = stitched.reindex(prices.index).ffill()
            proxy_notes[orig] = (
                f"stitched ({proxy} before {orig_series.dropna().index[0].date()})"
                if orig_len > 0 else f"proxy ({proxy})"
            )
            continue

        # Case 3: no original data but PROXY_MAP proxy has enough
        if proxy != orig and proxy_len >= min_days:
            result[orig] = proxy_series
            proxy_notes[orig] = f"proxy ({proxy})"
            continue

        # Case 4: try asset-class pattern match
        matched = False
        for pattern, cls_proxy, cls_name in ASSET_CLASS_PATTERNS:
            if _re.search(pattern, orig.lower()):
                if cls_proxy in prices.columns and len(prices[cls_proxy].dropna()) >= min_days:
                    cls_series = prices[cls_proxy]
                    if orig_len >= 20:
                        # Prefer stitching even with class proxy
                        result[orig] = _stitch_series(orig_series, cls_series).reindex(prices.index).ffill()
                        proxy_notes[orig] = f"stitched class proxy ({cls_proxy}={cls_name})"
                    else:
                        result[orig] = cls_series
                        proxy_notes[orig] = f"class proxy ({cls_proxy}={cls_name})"
                    matched = True
                    break
        if matched:
            continue

        # Case 5: last resort — risk-free rate (BIL). Flat-ish low-return series
        # so the backtest doesn't blow up, but user is flagged via proxy_notes.
        if RISK_FREE_PROXY in prices.columns:
            rf_series = prices[RISK_FREE_PROXY]
            if orig_len >= 20:
                result[orig] = _stitch_series(orig_series, rf_series).reindex(prices.index).ffill()
            else:
                result[orig] = rf_series
            proxy_notes[orig] = f"risk-free proxy (BIL — no reliable history found)"

    return result.ffill().dropna(how="all"), proxy_notes


# ── PORTFOLIO STATS FUNCTION (top-level) ──────────────────────────────────
def port_stats_from_prices(price_df, weights_dict, period_years):
    """Compute portfolio stats from a price DataFrame + weight dict.
    Returns full stats dict or None if insufficient data.

    NOTE: tickers in `weights_dict` that are missing from `price_df.columns`
    are silently dropped and the remaining weights renormalized. Callers
    must ensure all needed tickers are in price_df, otherwise the portfolio
    stats will reflect only the subset present (which can make different
    portfolios look identical if the difference is the missing ticker).
    """
    tklist = list(weights_dict.keys())
    vcols  = [t for t in tklist if t in price_df.columns]
    if not vcols: return None
    # Detect silent drops — log a warning so issues surface in dev mode
    if len(vcols) < len(tklist):
        _missing = [t for t in tklist if t not in price_df.columns]
        try:
            import warnings as _w
            _w.warn(f"port_stats_from_prices: missing prices for {_missing} "
                    f"in weights_dict {tklist} — dropped silently",
                    RuntimeWarning, stacklevel=2)
        except Exception:
            pass
    w  = np.array([weights_dict[t] for t in vcols])
    w  = w / w.sum()
    # Note: previously this did `.ffill().pct_change().dropna()`. ffill on
    # prices before pct_change inflates Sharpe by smoothing gap days into 0%
    # returns. Drop NaNs aligned across tickers (rectangular panel) instead.
    r  = price_df[vcols].pct_change().dropna()
    if len(r) < 10: return None
    s   = pd.Series(r.values @ w, index=r.index)
    cum = (1+s).cumprod()
    dd  = cum / cum.cummax() - 1
    tr  = float(cum.iloc[-1] - 1)
    # Annualize from actual trading days elapsed
    actual_yrs = len(s) / 252.0
    ann = (1+tr)**(1.0/max(actual_yrs,0.08)) - 1
    vol = float(s.std() * np.sqrt(252))
    # Excess-return Sharpe (consistent with all other call sites)
    sh  = _shared_sharpe(ann, vol)
    neg = s[s<0]
    dn  = float(neg.std()*np.sqrt(252)) if len(neg)>1 else vol
    so  = ((ann - DEFAULT_RISK_FREE_RATE) / dn) if dn > 0 else 0
    md  = float(dd.min())
    ca  = abs(ann/md) if md!=0 else 0
    cv  = float(s.nsmallest(max(1,int(len(s)*0.05))).mean())
    # YTD always from Jan 1 this year
    try:
        ytd_s = s[s.index >= pd.Timestamp(date(date.today().year,1,1))]
        ytd   = float((1+ytd_s).prod()-1) if len(ytd_s)>0 else 0.0
    except Exception: ytd = 0.0
    return {"ann_return":ann,"ann_vol":vol,"sharpe":sh,"sortino":so,
            "calmar":ca,"max_drawdown":md,"total_return":tr,
            "ytd_return":ytd,"cvar_5":cv,
            # Holdings preserved for downstream per-holding analytics
            # (proper portfolio risk score with classification + correlation,
            # weighted expense ratio, holdings detail tables)
            "tickers": vcols,
            "weights": [float(x) for x in w],
            "weights_dict": {t: float(wi) for t, wi in zip(vcols, w)}}


# ── Alpha Vantage Integration ─────────────────────────────────────────────────
# Free API key: https://www.alphavantage.co/support/#api-key
# Free tier: 25 requests/day, 5/min
# Use as secondary source for validation or when yfinance fails
# Store key in: C:\Users\tonys\Downloads\av_key.txt  OR  env var AV_API_KEY

import os, requests as _requests

def _load_av_key():
    """Load Alpha Vantage API key from env var, file, or st.secrets.

    Order of precedence:
      1. AV_API_KEY environment variable
      2. ~/av_key.txt or av_key.txt next to app.py
      3. st.secrets["ALPHA_VANTAGE_API_KEY"]
    Returns None if no key is configured. NEVER hardcode a key here —
    a leaked key gets the entire user base rate-limited.
    """
    # Try environment variable first
    key = os.environ.get("AV_API_KEY", "").strip()
    if key and key not in ("demo",""):
        return key
    # Try key file next to app.py
    for path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "av_key.txt"),
        os.path.expanduser("~/av_key.txt"),
    ]:
        try:
            with open(path) as f:
                key = f.read().strip()
                if key and key not in ("demo",""):
                    return key
        except Exception:
            pass
    # Try st.secrets
    try:
        k = st.secrets.get("ALPHA_VANTAGE_API_KEY")
        if k and str(k).strip() not in ("demo", ""):
            return str(k).strip()
    except Exception:
        pass
    # No key configured — Alpha Vantage features will be disabled
    return None

_AV_KEY = _load_av_key()
_AV_AVAILABLE = _AV_KEY is not None

def _av_get_prices(ticker, start_date, end_date):
    """Fetch daily prices from Alpha Vantage for a single ticker.
    Returns a pd.Series with date index, or None on failure.

    NOTE: TIME_SERIES_DAILY returns RAW (unadjusted) close on the free tier;
    the `5. adjusted close` field is only populated by TIME_SERIES_DAILY_ADJUSTED,
    which is a premium endpoint. This function therefore returns RAW close —
    callers must NOT treat it as dividend/split-adjusted. yfinance with
    auto_adjust=True is the correct source for adjusted prices.
    """
    if not _AV_AVAILABLE:
        return None
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY"
            f"&symbol={ticker}"
            f"&outputsize=full"
            f"&datatype=json"
            f"&apikey={_AV_KEY}"
        )
        resp = _requests.get(url, timeout=15)
        data = resp.json()
        # Check for error or rate limit
        if "Note" in data or "Information" in data or "Error Message" in data:
            return None
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        prices = {}
        for date_str, vals in ts.items():
            # `5. adjusted close` is premium-only; use raw close on free tier
            close = vals.get("4. close")
            if close:
                prices[pd.Timestamp(date_str)] = float(close)
        s = pd.Series(prices).sort_index()
        # Filter to date range
        s = s[(s.index >= pd.Timestamp(start_date)) & (s.index <= pd.Timestamp(end_date))]
        return s if len(s) > 10 else None
    except Exception:
        return None


def get_prices(tickers, start_date, end_date):
    """Unified price fetch: tries OpenBB first, falls back to yfinance.
    Always returns auto-adjusted prices (splits + dividends reinvested).
    Returns (DataFrame with ticker columns, source_label).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # ── Source priority: OpenBB → yfinance → Alpha Vantage ───────────────────

    # 1. Try OpenBB first (multi-source, most reliable)
    if _OBB_AVAILABLE:
        obb_result = _obb_get_prices(tickers, start_date, end_date)
        if obb_result is not None and not obb_result.empty and len(obb_result) > 5:
            return obb_result, "openbb"

    # 2. Try yfinance (fast, free, handles multi-ticker well)
    yf_result = pd.DataFrame()
    try:
        raw = yf.download(tickers, start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                yf_result = raw["Close"]
            elif "Close" in raw.columns:
                yf_result = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers)==1 else raw["Close"]
            else:
                yf_result = raw
            if isinstance(yf_result, pd.Series):
                yf_result = yf_result.to_frame(tickers[0] if tickers else "price")
            yf_result.index = pd.to_datetime(yf_result.index)
            yf_result = yf_result.dropna(how="all")
    except Exception:
        yf_result = pd.DataFrame()

    if not yf_result.empty:
        # NOTE: Previously this branch did a yfinance vs Alpha Vantage cross-check
        # and switched to AV if values diverged >5%. That logic was BROKEN:
        # yfinance with auto_adjust=True returns split+dividend-adjusted prices,
        # while AV's TIME_SERIES_DAILY returns RAW close on the free tier (the
        # adjusted endpoint is premium-only). For dividend-paying assets the
        # divergence is expected, and the old code would silently switch to the
        # WORSE (raw) data source. Cross-check removed — yfinance auto-adjusted
        # is the correct source.
        return yf_result, "yfinance"

    # 3. Fallback: Alpha Vantage (slower, 25 req/day limit, RAW close only)
    # Only used when yfinance fails entirely. RAW close is degraded data —
    # users should know if this path is taken.
    if _AV_AVAILABLE and len(tickers) == 1:
        av_s = _av_get_prices(tickers[0], start_date, end_date)
        if av_s is not None:
            try:
                _warnings.warn(
                    f"get_prices: yfinance failed for {tickers[0]}; using "
                    "Alpha Vantage RAW (unadjusted) close as fallback. "
                    "Long-term backtests may be inaccurate for dividend-paying assets.",
                    RuntimeWarning, stacklevel=2,
                )
            except Exception:
                pass
            return av_s.to_frame(tickers[0]), "alpha_vantage_raw"

    return pd.DataFrame(), "failed"


# NOTE: imports below were previously here (post-function-defs); moved to top
# of file alongside other stdlib/third-party imports. Only skfolio kept here
# since it's a heavy import and only used by run_backtest.
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import (
    MeanRisk, RiskBudgeting, EqualWeighted,
    HierarchicalRiskParity, NestedClustersOptimization,
    MaximumDiversification, ObjectiveFunction
)
from skfolio.moments import (
    LedoitWolf, GerberCovariance,
    DenoiseCovariance, EWMu, ShrunkMu
)
from skfolio.prior import EmpiricalPrior, BlackLitterman
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio import RiskMeasure, RatioMeasure
from sklearn.model_selection import train_test_split

# ── DATA FILE LOCATIONS ──────────────────────────────────────────────────────
# Anchor every JSON store to the directory app.py lives in. Previously these
# were bare filenames, which meant Streamlit read/wrote them in whatever folder
# you launched it from — so saving from `C:\Users\tonys\` and later launching
# from `C:\Users\tonys\Downloads\` would silently lose your saved portfolios.
# Paths are now absolute and immune to launch-directory drift.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _data_path(name: str) -> str:
    return os.path.join(_APP_DIR, name)

SAVE_FILE              = _data_path("saved_portfolios.json")
HOLDINGS_FILE          = _data_path("holdings.json")
WATCHLIST_FILE         = _data_path("watchlist.json")
CLIENT_PROFILES_FILE   = _data_path("risk_profiles.json")
CLIENT_PROPOSALS_FILE  = _data_path("client_proposals.json")
CLIENT_HOLDINGS_FILE   = _data_path("client_holdings.json")

# ── FIRM BRANDING ─────────────────────────────────────────────
# Global firm-level branding used on every generated PDF. Set once in the
# Client Records → Firm Branding panel; persisted to firm_settings.json
# (text fields) plus firm_logo.png / advisor_photo.png (image bytes saved
# directly so the PDF builder can embed them with reportlab.platypus.Image).
FIRM_SETTINGS_FILE     = _data_path("firm_settings.json")
FIRM_LOGO_PATH         = _data_path("firm_logo.png")
ADVISOR_PHOTO_PATH     = _data_path("advisor_photo.png")

def load_firm_settings() -> dict:
    """Return saved firm branding (text fields). Image paths are checked
    separately via os.path.exists so a missing image just falls back to
    a text-only header rather than crashing the PDF build.

    Routes through data_store so firm_settings.json is read from the
    shared GitHub repo, making the portal see changes made here."""
    val = _shared_load_json(FIRM_SETTINGS_FILE, default={})
    return val if isinstance(val, dict) else {}


def save_firm_settings(settings: dict) -> None:
    """Routes through data_store so firm_settings.json is written to the
    shared GitHub repo (where the portal can read it)."""
    _shared_save_json(FIRM_SETTINGS_FILE, settings)

# ── POPULAR PUBLIC PORTFOLIOS ─────────────────────────────────────────────────
# Lifted to module scope so the sidebar (which runs before the main tab body)
# can render these as quick-load buttons. The Securities section also reads
# this same dict — keep both in sync if you add/remove presets here.
#
# Value shape — supports two formats:
#   • None  — separator/placeholder (rendered greyed out)
#   • str   — comma-separated tickers, equal-weighted (legacy)
#   • dict  — {"tickers": [...], "weights": [...pct]} for portfolios with
#             specific allocations (e.g. Schwab Core ETF 48/52 has weights
#             27.2 / 2.9 / 11.6 / ... not equal-weighted).
#
# The Schwab section is built dynamically from schwab_portfolios.json so
# that editing the JSON updates the dropdown without code changes. If the
# Schwab module isn't loadable, the dict still has the structural keys
# (Custom + Saved separator) so the dropdown doesn't disappear.

def _build_schwab_preset_entries():
    """Return an OrderedDict-style dict of Schwab preset entries to splice
    into POPULAR_PORTFOLIOS. Format:
      {
        "── Schwab Core ETF ──": None,
        "Schwab Core ETF 8/92":  {"tickers":[...], "weights":[...]},
        ...
      }
    Returns {} if schwab_portfolios isn't available — caller handles it.
    """
    if not _SCHWAB_AVAILABLE or _schwab_portfolios is None:
        return {}

    entries = {}
    # Series order + their human-readable section headers and short prefixes
    series_cfg = [
        ("core_etf",              "Schwab Core ETF",              "Schwab Core ETF"),
        ("core_income",           "Schwab Core Income",           "Schwab Core Income"),
        ("core_enhanced_income",  "Schwab Core Enhanced Income",  "Schwab Enhanced Income"),
        ("passive_active_income", "Schwab Passive-Active Income", "Schwab Passive-Active"),
    ]

    try:
        available_series = set(_schwab_portfolios.get_all_series())
    except Exception:
        return {}

    for series_key, header, short_prefix in series_cfg:
        if series_key not in available_series:
            continue
        # Section header (no value → renders as separator)
        entries[f"── {header} ──"] = None

        try:
            tier_labels = _schwab_portfolios.list_tiers(series_key)
        except Exception:
            continue

        for tier_label in tier_labels:
            try:
                holdings = _schwab_portfolios.get_holdings(
                    series_key, tier_label, drop_zero=True,
                )
            except Exception:
                continue

            tickers, weights = [], []
            for h in holdings:
                # Replace the "CASH" placeholder with SGOV — same substitution
                # as in _build_schwab_alternate_tier above. Keeps the cash
                # sleeve materialized as a tradeable ticker the rest of the
                # app already understands.
                t = "SGOV" if h["ticker"] == "CASH" else h["ticker"]
                w = float(h.get("weight", 0))
                if w <= 0:
                    continue
                tickers.append(t)
                weights.append(round(w, 2))

            # Normalize to exactly 100 (rounding artifacts)
            _total = sum(weights)
            if _total > 0:
                weights = [round(w * 100.0 / _total, 2) for w in weights]

            if tickers:
                # Display label — terse form per Tony's preference, e.g.
                # "Schwab Core ETF 64/36"
                display = f"{short_prefix} {tier_label}"
                entries[display] = {"tickers": tickers, "weights": weights}

    return entries


# Base structural entries (always present)
POPULAR_PORTFOLIOS = {
    "Custom — Enter Your Own Tickers": None,
    "── Saved Portfolios ──":          None,
}
# Splice in the Schwab presets (replaces the prior generic preset list per
# Tony's call — Schwab-only going forward). If the Schwab module fails to
# load, this is a no-op and the dropdown shows just Custom + Saved.
POPULAR_PORTFOLIOS.update(_build_schwab_preset_entries())


def _resolve_preset(label):
    """Resolve a POPULAR_PORTFOLIOS label to (tickers, weights_dict_pct).

    Handles all three value shapes (None / str / dict). Returns
    (None, None) for separators or unknown labels. Weights are returned
    as percentages (0–100), matching what the rest of app.py uses.

    This is the single point of truth — every reader of POPULAR_PORTFOLIOS
    should go through this so the str-vs-dict format details are isolated.
    """
    if not label or label not in POPULAR_PORTFOLIOS:
        return None, None
    val = POPULAR_PORTFOLIOS[label]
    if val is None:
        return None, None

    # Dict form: explicit tickers + weights
    if isinstance(val, dict):
        tks = [t.strip().upper() for t in val.get("tickers", []) if t and str(t).strip()]
        ws  = list(val.get("weights", []))
        if not tks:
            return None, None
        # Pad / truncate weights to match tickers (defensive — shouldn't happen)
        if len(ws) != len(tks):
            ws = ([float(w) for w in ws] + [0.0] * len(tks))[: len(tks)]
        # Build dict and normalize to sum to 100
        wmap = {t: float(w) for t, w in zip(tks, ws)}
        total = sum(wmap.values())
        if total > 0:
            wmap = {t: round(w * 100.0 / total, 2) for t, w in wmap.items()}
        return tks, wmap

    # Legacy string form: comma-separated tickers, equal-weighted
    if isinstance(val, str):
        tks = [t.strip().upper() for t in val.split(",") if t.strip()]
        if not tks:
            return None, None
        per = round(100.0 / len(tks), 4)
        return tks, {t: per for t in tks}

    return None, None


def load_portfolio_into_session(source_label: str) -> bool:
    """Load the named portfolio into session state so the Securities section
    picks it up. `source_label` is one of:
      • "📁 <name>"  — a user-saved portfolio (key in saved_portfolios.json)
      • "<key>"      — a key in POPULAR_PORTFOLIOS
      • "Custom — Enter Your Own Tickers" — leave inputs alone
    Returns True if state was changed.

    Used by both the Securities-section dropdown AND the sidebar quick-load
    panel, so the two paths stay byte-for-byte identical.
    """
    if not source_label:
        return False
    st.session_state.portfolio_source = source_label

    # Safe writer for the main-area selectbox key. When called from the sidebar
    # (before the main selectbox is instantiated this run), this pre-seeds the
    # widget so it renders with the right selection. When called from the
    # main-area on-change handler (after the widget is already instantiated),
    # Streamlit raises StreamlitAPIException — but in that case the widget's
    # own return value is already `source_label`, so the write is redundant
    # and we can safely swallow the exception.
    def _sync_main_selectbox(label):
        try:
            st.session_state["portfolio_source_sel"] = label
        except Exception:
            # Widget already instantiated this run; its value is already correct.
            pass

    if source_label.startswith("📁 "):
        sp = load_saved().get(source_label[2:])
        if not sp:
            return False
        tickers_str = ", ".join(sp.get("tickers", []))
        st.session_state.ticker_input_val     = tickers_str
        st.session_state["ticker_text_input"] = tickers_str
        # Saved portfolios store weights as decimals (0.6 = 60%); the input
        # widget expects percentages, so convert.
        st.session_state["loaded_weights"] = {
            t: round(w * 100, 1)
            for t, w in zip(sp.get("tickers", []), sp.get("weights", []))
        }
        # Keep dropdown widget in sync (so it reflects what the sidebar set)
        _sync_main_selectbox(source_label)
        return True

    if source_label in POPULAR_PORTFOLIOS and POPULAR_PORTFOLIOS[source_label]:
        # Use the resolver — handles both legacy str presets and new dict
        # presets (Schwab portfolios) uniformly.
        tks, wmap = _resolve_preset(source_label)
        if tks:
            tickers_str = ", ".join(tks)
            st.session_state.ticker_input_val     = tickers_str
            st.session_state["ticker_text_input"] = tickers_str
            # If preset has explicit weights (Schwab), pass them through;
            # else empty dict → equal-weight default in the input widget.
            st.session_state["loaded_weights"]    = wmap or {}
            _sync_main_selectbox(source_label)
            return True
        return False

    if source_label == "Custom — Enter Your Own Tickers":
        _sync_main_selectbox(source_label)
        return True

    return False

# ── UNASSOCIATED REPORTS ────────────────────────────────────────────────
# Special "client" key under which proposals generated without a selected
# client are stored. Renders as a special 📂 folder in Client Records so
# advisors can later move proposals into a real client profile.
UNASSOCIATED_CLIENT_KEY = "__unassociated__"


def ensure_unassociated_profile():
    """Make sure the synthetic 'Unassociated Reports' profile exists in the
    client profiles store. Idempotent — safe to call on every save.

    Returns the profile dict.
    """
    profiles = _load_json_safe(CLIENT_PROFILES_FILE)
    if UNASSOCIATED_CLIENT_KEY not in profiles:
        profiles[UNASSOCIATED_CLIENT_KEY] = {
            "client_name":     "📂 Unassociated Reports",
            "client_email":    "",
            "client_phone":    "",
            "client_age":      "",
            "overall_score":   "—",
            "tolerance_score": "—",
            "capacity_score":  "—",
            "risk_label":      "—",
            "priorities":      [],
            "date":            "",
            "completed_at":    "",
            "advisor":         "",
            "_is_unassociated": True,      # marker flag for UI rendering
        }
        _save_json_safe(CLIENT_PROFILES_FILE, profiles)
    return profiles[UNASSOCIATED_CLIENT_KEY]


# ── CLIENT PROPOSAL STORAGE ──────────────────────────────────────────────
def _load_json_safe(path):
    """Atomic-aware JSON loader. Wraps shared.load_json for compatibility with
    existing call sites — all new code should import shared.load_json directly."""
    return _shared_load_json(path, default={})


def _save_json_safe(path, data):
    """Atomic JSON writer with file locking. Wraps shared.save_json.

    Replaces the old non-atomic write that lost data under concurrent writers
    (multiple Streamlit reruns / multiple users editing simultaneously).
    """
    _shared_save_json(path, data)


def load_all_proposals():
    """Returns {client_key: {version_id: proposal_dict}}."""
    return _load_json_safe(CLIENT_PROPOSALS_FILE)


def save_proposal(client_key, version_id, proposal_dict):
    """Persist a versioned proposal atomically. Creates the client entry if missing.
    Ensures the proposal carries a secure random share token (`_share_token`).
    """
    # Ensure share token exists — used by find_proposal_by_token. Random, not
    # derived from email/version_id (those would be guessable).
    if "_share_token" not in proposal_dict:
        proposal_dict = dict(proposal_dict)
        proposal_dict["_share_token"] = make_secure_token()

    def _mutate(all_p):
        all_p.setdefault(client_key, {})[version_id] = proposal_dict
    _shared_update_json(CLIENT_PROPOSALS_FILE, _mutate)


def delete_proposal(client_key, version_id):
    """Delete a proposal atomically."""
    def _mutate(all_p):
        if client_key in all_p and version_id in all_p[client_key]:
            del all_p[client_key][version_id]
            if not all_p[client_key]:
                del all_p[client_key]
    _shared_update_json(CLIENT_PROPOSALS_FILE, _mutate)


def make_proposal_token(client_key, version_id):
    """Look up (or create) the secure random share token for a proposal.

    Previously this *derived* a token from sha256(client_key::version_id), which
    was guessable if version IDs followed any predictable pattern (timestamps,
    sequential, etc.). Now reads the random token stored on the proposal at
    creation time. If a legacy proposal lacks one, mints and persists a new
    token atomically.
    """
    proposals = load_all_proposals()
    versions = proposals.get(client_key, {})
    proposal = versions.get(version_id)
    if not proposal:
        return None
    if proposal.get("_share_token"):
        return proposal["_share_token"]
    # Legacy proposal — backfill a token atomically
    new_token = make_secure_token()
    def _mutate(all_p):
        if client_key in all_p and version_id in all_p[client_key]:
            all_p[client_key][version_id]["_share_token"] = new_token
    _shared_update_json(CLIENT_PROPOSALS_FILE, _mutate)
    return new_token


def find_proposal_by_token(token):
    """Reverse lookup: token → (client_key, version_id, proposal) or (None,None,None).

    Uses constant-time comparison on each stored token to prevent timing attacks
    on the lookup. O(N×M) over all proposals — acceptable for current scale; if
    it grows, build an index dict {token: (ck, vid)} on save.
    """
    if not token:
        return None, None, None
    import hmac as _hmac
    for ck, versions in load_all_proposals().items():
        for vid, proposal in versions.items():
            stored = proposal.get("_share_token", "") if isinstance(proposal, dict) else ""
            if stored and _hmac.compare_digest(stored, token):
                return ck, vid, proposal
    return None, None, None


# ── TARGET-RISK → ALLOCATION ─────────────────────────────────────────────
def allocation_for_risk_score(score):
    """Map 1-99 risk score to a (equity, bonds, cash) split.

    Anchors:
      score ≤ 10  → 20 / 70 / 10   (very conservative)
      score = 30  → 40 / 55 / 5
      score = 50  → 60 / 38 / 2
      score = 70  → 80 / 20 / 0
      score ≥ 90  → 95 /  5 / 0    (very aggressive)
    """
    s = max(1, min(99, int(score)))
    if s <= 10:
        eq, bd, cs = 20, 70, 10
    elif s <= 30:
        t = (s - 10) / 20;   eq = 20 + t*20; bd = 70 - t*15; cs = 10 - t*5
    elif s <= 50:
        t = (s - 30) / 20;   eq = 40 + t*20; bd = 55 - t*17; cs = 5  - t*3
    elif s <= 70:
        t = (s - 50) / 20;   eq = 60 + t*20; bd = 38 - t*18; cs = 2  - t*2
    elif s <= 90:
        t = (s - 70) / 20;   eq = 80 + t*15; bd = 20 - t*15; cs = 0
    else:
        eq, bd, cs = 95, 5, 0
    return round(eq, 1), round(bd, 1), round(cs, 1)


def implied_risk_score_from_allocation(eq_pct):
    """Inverse of allocation_for_risk_score: given an equity percentage,
    return the approximate implied risk score on the 1-99 scale.

    Uses piecewise-linear interpolation matching the same anchors.
    """
    eq = max(0, min(100, float(eq_pct)))
    # Anchors: (eq_pct → score)
    # 20→10, 40→30, 60→50, 80→70, 95→90
    if eq <= 20:
        s = 1 + (eq - 0) / 20 * 9           # 0%→1, 20%→10
    elif eq <= 40:
        s = 10 + (eq - 20) / 20 * 20        # 20%→10, 40%→30
    elif eq <= 60:
        s = 30 + (eq - 40) / 20 * 20        # 40%→30, 60%→50
    elif eq <= 80:
        s = 50 + (eq - 60) / 20 * 20        # 60%→50, 80%→70
    elif eq <= 95:
        s = 70 + (eq - 80) / 15 * 20        # 80%→70, 95%→90
    else:
        s = 90 + (eq - 95) / 5 * 9          # 95%→90, 100%→99
    return int(round(max(1, min(99, s))))


# ── EXPENSE RATIO LOOKUPS ─────────────────────────────────────────────
# Ticker → annual expense ratio (decimal). Used as a fast/offline source
# before falling back to yfinance .info. Numbers reflect publicly disclosed
# rates as of late 2025 and are kept conservative (round up where uncertain
# so portfolio cost figures aren't understated).
_EXPENSE_RATIO_OVERRIDES = {
    # ─────────────────────────────────────────────────────────────────
    # LAST-RESORT FALLBACK ONLY — primary source is the Alpha Vantage
    # ETF_PROFILE API with a 30-day disk cache. These hardcoded values
    # are consulted only when both API providers are unreachable, so
    # graceful degradation kicks in during outages without showing
    # "—" for popular tickers.
    #
    # Last validated against SEC filings: April 30, 2026
    # Major fee changes incorporated:
    #   • Vanguard Feb 2, 2026 round (53 funds, ~$250M annual savings)
    #   • Vanguard Feb 1, 2025 round (87 funds)
    #   • Invesco QQQ reclassification (Oct 2024): 0.20% → 0.18%
    #   • Schwab core ETF reductions (post-2024 rebrand)
    # When updating: cross-reference SEC EDGAR 497/485BPOS filings,
    # then bump the validation date above.
    # ─────────────────────────────────────────────────────────────────
    # ── Vanguard core equity ──
    # Updated 2026-02-02: VTV, VUG, VBR, VBK reduced. VWO reduced.
    # VTI, VOO, VEA NOT on the 2026 cut list — current values stand.
    "VTI": 0.0003, "VOO": 0.0003, "VEA": 0.0006, "VWO": 0.0006,
    "VTV": 0.0003, "VUG": 0.0003, "VBR": 0.0005, "VBK": 0.0005,
    "VEU": 0.0006, "VXUS": 0.0007, "VT": 0.0006, "VNQ": 0.0012,
    "VV":  0.0003,  # Vanguard Large-Cap ETF (added — popular TLH partner for VTI)
    # ── Vanguard bond ──
    # 2026 round did NOT touch BND, BSV, BIV, BLV, VCSH/VCIT/VCLT,
    # VGSH/VGIT/VGLT, VTIP. Values are unchanged.
    "BND": 0.0003, "BNDX": 0.0007, "BSV": 0.0004, "BIV": 0.0004,
    "BLV": 0.0004, "VCSH": 0.0004, "VCIT": 0.0004, "VCLT": 0.0004,
    "VGSH": 0.0004, "VGIT": 0.0004, "VGLT": 0.0004, "VTIP": 0.0004,
    "VTEB": 0.0005,
    # ── iShares core ──
    "IVV": 0.0003, "IJH": 0.0005, "IJR": 0.0006, "IEFA": 0.0007,
    "IEMG": 0.0009, "AGG": 0.0003, "IEF": 0.0015, "TLT": 0.0015,
    "SHY": 0.0015, "TIP": 0.0019, "LQD": 0.0014, "HYG": 0.0049,
    "IWM": 0.0019, "IWR": 0.0019, "IWB": 0.0015, "ITOT": 0.0003,
    "IXUS": 0.0007, "EFA": 0.0033, "EEM": 0.0070, "GOVT": 0.0005,
    "MUB": 0.0005, "EMB": 0.0039,
    # ── State Street SPDR ──
    "SPY": 0.0009, "SPLG": 0.0002, "SPYG": 0.0004, "SPYV": 0.0004,
    "SPMD": 0.0003, "SPSM": 0.0003, "MDY": 0.0023, "DIA": 0.0016,
    "GLD": 0.0040, "GLDM": 0.0010, "SLV": 0.0050,
    "JNK": 0.0040,
    # ── Schwab ──
    # SCHF: 0.03% (reduced from 0.06% post-2024 rebrand). Other Schwab
    # core ETFs verified current as of 2026-04-30.
    "SCHB": 0.0003, "SCHX": 0.0003, "SCHA": 0.0004, "SCHF": 0.0003,
    "SCHE": 0.0011, "SCHZ": 0.0003, "SCHD": 0.0006, "SCHO": 0.0003,
    "SCHR": 0.0003, "SCHP": 0.0003,
    # ── Invesco ──
    # QQQ: 0.18% (reduced from 0.20% effective Oct 2024 when Invesco
    # reclassified the trust). Reference: SEC 485BPOS FY2025.
    "QQQ": 0.0018, "QQQM": 0.0015, "RSP": 0.0020, "SPHD": 0.0030,
    "SPLV": 0.0025,
    # ── Cash / money market ETFs ──
    "SGOV": 0.0007, "BIL": 0.0014, "SHV": 0.0015, "ICSH": 0.0008,
    "JPST": 0.0018, "FLOT": 0.0015, "USFR": 0.0015, "GBIL": 0.0012,
    # ── Sector / thematic ──
    "XLK": 0.0009, "XLF": 0.0009, "XLE": 0.0009, "XLV": 0.0009,
    "XLY": 0.0009, "XLP": 0.0009, "XLI": 0.0009, "XLB": 0.0009,
    "XLRE": 0.0009, "XLU": 0.0009, "XLC": 0.0009,
    "ARKK": 0.0075, "ARKG": 0.0075, "ARKW": 0.0075, "ARKQ": 0.0075,
    # ── Leveraged / inverse ETFs ──
    "TQQQ": 0.0086, "SQQQ": 0.0095, "UPRO": 0.0091, "SPXU": 0.0091,
    "SOXL": 0.0089, "SOXS": 0.0093, "TMF": 0.0106, "TMV": 0.0106,
    "SSO": 0.0089, "SDS": 0.0090, "TBT": 0.0089,
    # ── Dividend / income ──
    # 2026-02-02: VIG and VYM both reduced from 0.06% to 0.04%.
    "VYM": 0.0004, "DGRO": 0.0008, "NOBL": 0.0035, "HDV": 0.0008,
    "VIG": 0.0004, "DVY": 0.0038,
    # ── Money market funds ──
    "VMFXX": 0.0011, "VUSXX": 0.0009, "SPAXX": 0.0042, "FZDXX": 0.0030,
    "SWVXX": 0.0034,
}


# ── EXPENSE RATIO CACHE (GitHub-backed via data_store, 30-day TTL) ────
# Previously wrote to a local JSON file in the container working directory,
# which is ephemeral on Streamlit Community Cloud — every container restart
# wiped the cache. With a 30-fund portfolio and Alpha Vantage's free-tier
# limit of 25 calls/day, cold starts could exhaust the daily quota before
# the portfolio finished loading.
#
# Routing through data_store.py (same as firm_settings.json, client
# proposals, etc.) gives us:
#   • Persistence across container restarts (writes to shared GitHub repo)
#   • Cache sharing between advisor app and client portal — when one
#     looks up VTI's ER, the other gets the cached value for free
#   • Graceful local-disk fallback when GitHub secrets aren't configured
#     (data_store handles this transparently per its docstring)
#
# In-memory memoization is preserved on the function attribute so we don't
# hit data_store on every lookup within a single render.
EXPENSE_RATIO_CACHE_FILE = "expense_ratios_cache.json"
EXPENSE_RATIO_CACHE_TTL_DAYS = 30

def _load_er_cache():
    """Load the expense-ratio cache from the shared data store.

    Returns dict keyed by ticker → {er, fetched (ISO date), source}.
    Tolerates missing/corrupt remote state by falling back to {}.

    Memoized in module attribute — data_store is hit once per process,
    then kept in memory. _save_er_cache() updates the in-memory copy too.
    """
    if hasattr(_load_er_cache, "_cache"):
        return _load_er_cache._cache
    try:
        cache = _shared_load_json(EXPENSE_RATIO_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
    except Exception:
        cache = {}
    _load_er_cache._cache = cache
    return cache

def _save_er_cache(cache):
    """Persist the expense-ratio cache via data_store (shared GitHub repo
    when configured, local-disk fallback otherwise) and update the
    in-memory copy.

    Failures are swallowed — a transient GitHub outage shouldn't break
    expense-ratio lookups. The in-memory cache still holds the value
    for the rest of the session, and the next session will retry.
    """
    _load_er_cache._cache = cache
    try:
        _shared_save_json(EXPENSE_RATIO_CACHE_FILE, cache)
    except Exception:
        pass

def _er_cache_is_fresh(entry):
    """Check whether a cache entry is still within the 30-day TTL."""
    if not entry or "fetched" not in entry:
        return False
    try:
        from datetime import datetime as _dt, timedelta as _td
        fetched = _dt.fromisoformat(entry["fetched"])
        return (_dt.now() - fetched) < _td(days=EXPENSE_RATIO_CACHE_TTL_DAYS)
    except Exception:
        return False

def _fmp_fetch_expense_ratio(ticker, api_key):
    """Fetch expense ratio from FMP, trying free-tier compatible endpoints
    in order. Returns decimal ER or None.

    Endpoint waterfall (FMP changed their tiers; some endpoints that were
    free are now paid). We try in this order, stopping at the first that
    returns valid data:
      1. /api/v3/profile/{symbol}      — has expenseRatio for ETFs/MF (free)
      2. /api/v3/etf-info/{symbol}     — paid as of late 2024 but still
                                          works for some tiers
      3. /stable/etf-info/?symbol=X    — newer Stable API (free-eligible)

    FMP's percent format: a value of 0.03 means "0.03%" (3 bps), so we
    divide by 100 when value > 0.5 (anything that looks like a percent).
    """
    if not api_key or not ticker:
        return None

    import requests as _req

    def _normalize_er(raw):
        """Convert raw FMP value to decimal. Returns None if invalid."""
        if raw is None:
            return None
        try:
            er = float(raw)
        except (TypeError, ValueError):
            return None
        if er == 0 or er < 0:
            return None
        # FMP returns as percent (0.03 means 0.03%). Convert to decimal.
        # Heuristic: anything > 0.5 is almost certainly already a percent.
        if er > 0.5:
            er = er / 100.0
        return er

    endpoints = [
        # Stable API (newer, generally available on free tier)
        ("https://financialmodelingprep.com/stable/etf-info",
         {"symbol": ticker, "apikey": api_key},
         "expenseRatio"),
        # Profile endpoint (free tier — has expense ratio for funds)
        (f"https://financialmodelingprep.com/api/v3/profile/{ticker}",
         {"apikey": api_key},
         "expenseRatio"),
        # Original etf-info (paid as of late 2024 but try anyway)
        (f"https://financialmodelingprep.com/api/v3/etf-info/{ticker}",
         {"apikey": api_key},
         "expenseRatio"),
    ]

    for url, params, field_name in endpoints:
        try:
            resp = _req.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            # FMP returns either a list of dicts or sometimes a single dict
            if isinstance(data, dict):
                data = [data]
            if not data or not isinstance(data, list) or len(data) == 0:
                continue
            row = data[0]
            if not isinstance(row, dict):
                continue
            er = _normalize_er(row.get(field_name))
            if er is not None:
                return er
        except Exception:
            continue

    return None


def _alphavantage_fetch_expense_ratio(ticker, api_key):
    """Fetch expense ratio from Alpha Vantage's ETF_PROFILE function.

    Free tier: 25 calls/day. AV returns net_expense_ratio as a percent
    (e.g. "0.0003" for 3 bps). Returns decimal ER or None.

    AV embeds errors inside HTTP 200 responses (Error Message, Note for
    rate limits, Information for premium-only) so we check those.
    """
    if not api_key or not ticker:
        return None
    try:
        import requests as _req
        url = "https://www.alphavantage.co/query"
        params = {"function": "ETF_PROFILE", "symbol": ticker,
                  "apikey": api_key}
        resp = _req.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Detect AV's in-band errors
        if (not isinstance(data, dict) or "Error Message" in data
                or "Note" in data or "Information" in data):
            return None
        raw = data.get("net_expense_ratio")
        if raw is None or raw == "" or raw == "None":
            return None
        try:
            er = float(raw)
        except (TypeError, ValueError):
            return None
        if er <= 0:
            return None
        # AV returns as decimal already (0.0003 = 3 bps), but sometimes as
        # percent (0.03 means 0.03%). Heuristic: > 0.5 → was percent.
        if er > 0.5:
            er = er / 100.0
        return er
    except Exception:
        return None


# ── ALPHA VANTAGE COMPANY/FUND PROFILE FETCHER ──────────────────────────
# Used to populate the PDF holdings table — replaces slow/unreliable
# yfinance .info calls. AV's OVERVIEW (stocks) and ETF_PROFILE (funds)
# return name, sector, dividend yield, etc. Cached on the function itself
# (in-memory) for the session. ETF/MF tickers route to ETF_PROFILE.
@functools.lru_cache(maxsize=512)
def _alphavantage_fetch_profile(ticker, api_key):
    """Fetch company/fund profile from Alpha Vantage.

    Returns dict {name, type, sec_yield} or None.

    Routes ETF/MF tickers (per _classify_ticker) to ETF_PROFILE since
    OVERVIEW only works for stocks. Uses dict mapping for AV's varied
    field names across endpoints.

    RATE LIMITING: tracks the timestamp of the last AV call on the
    function attribute and sleeps if needed to stay under 5 req/min.
    Free tier limits are 25/day AND 5/minute — we respect both.
    """
    if not api_key or not ticker:
        return None
    cls, _ = _classify_ticker(ticker)

    def _av_throttle():
        """Sleep just long enough to keep us under 5 calls/minute."""
        import time as _time
        if not hasattr(_alphavantage_fetch_profile, "_last_call"):
            _alphavantage_fetch_profile._last_call = 0
        _now = _time.time()
        _elapsed = _now - _alphavantage_fetch_profile._last_call
        # 12s spacing keeps us at ~5/min with safety margin
        if _elapsed < 12 and _alphavantage_fetch_profile._last_call > 0:
            _time.sleep(12 - _elapsed)
        _alphavantage_fetch_profile._last_call = _time.time()

    try:
        import requests as _req
        url = "https://www.alphavantage.co/query"

        # Pick endpoint based on classification
        if cls in ("bond", "cash"):
            _av_throttle()
            params = {"function": "ETF_PROFILE", "symbol": ticker,
                      "apikey": api_key}
            resp = _req.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if (not isinstance(data, dict) or "Error Message" in data
                    or "Note" in data or "Information" in data):
                return None
            return {
                "name": data.get("name") or ticker,
                "type": "ETF",
                "sec_yield": _av_to_float(data.get("dividend_yield")),
            }
        elif cls in ("crypto_btc", "crypto_alt"):
            return {"name": ticker, "type": "CRYPTO", "sec_yield": None}
        elif cls == "leveraged":
            _av_throttle()
            params = {"function": "ETF_PROFILE", "symbol": ticker,
                      "apikey": api_key}
            resp = _req.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if (isinstance(data, dict) and "Error Message" not in data
                        and "Note" not in data and "Information" not in data
                        and data.get("name")):
                    return {
                        "name": data.get("name") or ticker,
                        "type": "ETF",
                        "sec_yield": _av_to_float(data.get("dividend_yield")),
                    }
            return {"name": ticker, "type": "ETF (LEVERAGED)", "sec_yield": None}
        else:
            # Equity (stock OR ETF that wasn't classified as bond/cash)
            # Try OVERVIEW first; if it returns empty, fall back to ETF_PROFILE.
            _av_throttle()
            params = {"function": "OVERVIEW", "symbol": ticker,
                      "apikey": api_key}
            resp = _req.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if (isinstance(data, dict) and data
                        and "Error Message" not in data
                        and "Note" not in data
                        and "Information" not in data
                        and (data.get("Name") or data.get("Symbol"))):
                    return {
                        "name": data.get("Name") or ticker,
                        "type": (data.get("AssetType") or "EQUITY").upper(),
                        "sec_yield": _av_to_float(data.get("DividendYield")),
                    }
            # Fall through to ETF_PROFILE for funds OVERVIEW didn't recognize
            _av_throttle()
            params = {"function": "ETF_PROFILE", "symbol": ticker,
                      "apikey": api_key}
            resp = _req.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if (isinstance(data, dict) and data
                        and "Error Message" not in data
                        and "Note" not in data
                        and "Information" not in data
                        and data.get("name")):
                    return {
                        "name": data.get("name") or ticker,
                        "type": "ETF",
                        "sec_yield": _av_to_float(data.get("dividend_yield")),
                    }
            return None
    except Exception:
        return None


def _av_to_float(v):
    """Coerce a string AV value to float decimal, or None.

    AV returns things like "0.0042" (decimal) or sometimes "0.42%" (string).
    """
    if v is None or v == "" or v == "None" or v == "-":
        return None
    try:
        s = str(v).strip().rstrip("%")
        f = float(s)
        # If it's a percent string, divide by 100
        if "%" in str(v):
            f = f / 100.0
        return f
    except (TypeError, ValueError):
        return None


def _resolve_av_key():
    """Single source of truth for AV API key resolution.
    Order: session manual override → secrets.toml → env var.
    """
    try:
        k = st.session_state.get("av_api_key_manual")
        if k: return k
    except Exception:
        pass
    try:
        k = st.secrets.get("ALPHA_VANTAGE_API_KEY")
        if k: return k
    except Exception:
        pass
    return os.environ.get("ALPHA_VANTAGE_API_KEY")


def _expense_ratio_for_ticker(ticker):
    """Return the annual expense ratio (decimal) for a ticker.

    Decision tree (Schwab JSON > shared cache > APIs > hardcoded):
      1. In-session cache (per-process, fastest)
      2. Schwab model portfolios JSON (authoritative for the ~30 tickers
         in the four Schwab series — Core ETF, Core Income, Enhanced
         Income, Passive-Active. Schwab publishes these alongside model
         rebalances, so they're always current and trump everything else.)
      3. Stocks (via _classify_ticker) → 0.0  (no ER applies)
      4. Crypto → 0.0
      5. Curated mutual fund table (_MUTUAL_FUND_TABLE) — covers Vanguard,
         Fidelity, etc. active funds whose ERs the Alpha Vantage endpoint
         doesn't reliably return.
      6. Shared cache via data_store (30-day TTL) — survives restarts
      7. Alpha Vantage ETF_PROFILE → cache + return  (PRIMARY API)
      8. FMP profile API → cache + return  (secondary API)
      9. Hardcoded `_EXPENSE_RATIO_OVERRIDES` — last-resort fallback
     10. None for genuinely unknown tickers (rare)

    Why Schwab JSON sits above APIs: the four Schwab model series
    include active mutual funds (PRFDX, PDBZX, CPXIX, RPIFX, HFQIX,
    CSJIX, etc.) that Alpha Vantage's ETF_PROFILE endpoint doesn't
    consistently return data for — those would silently resolve to
    0.0% and understate portfolio cost. By promoting the JSON above
    the API tiers we get correct numbers for those funds without
    waiting for AV/FMP coverage to improve.

    Stocks return 0.0 (not None) so they count as "fully covered" in
    the weighted ER calculation.
    """
    if ticker is None:
        return None
    t = ticker.upper().strip()
    if not t:
        return None

    # 1. In-memory cache for the current session
    if not hasattr(_expense_ratio_for_ticker, "_session_cache"):
        _expense_ratio_for_ticker._session_cache = {}
    sess = _expense_ratio_for_ticker._session_cache
    if t in sess:
        return sess[t]

    # 2. Schwab JSON — highest-priority external source.
    # Promotes the per-fund ERs Tony curates in schwab_portfolios.json
    # above the API path so active MFs without API coverage still
    # display the correct cost. The lookup is module-level and seeded
    # once at startup, so this check is O(1).
    if _SCHWAB_ER_LOOKUP and t in _SCHWAB_ER_LOOKUP:
        sess[t] = float(_SCHWAB_ER_LOOKUP[t])
        return sess[t]

    # 3+4. Stocks/crypto via classifier → 0.0
    cls, _credit = _classify_ticker(t)
    if cls in ("crypto_btc", "crypto_alt"):
        sess[t] = 0.0
        return 0.0
    # For "equity" class: probably a stock but might be an uncommon ETF —
    # go through the API path before falling back to 0.0 below.

    # 6. Shared cache (30-day TTL) — fast path for repeat lookups.
    # Backed by data_store so it survives container restarts and is
    # shared between the advisor app and client portal.
    disk_cache = _load_er_cache()
    if t in disk_cache and _er_cache_is_fresh(disk_cache[t]):
        sess[t] = disk_cache[t].get("er")
        return sess[t]

    # 5. Curated mutual fund table — Alpha Vantage's ETF_PROFILE
    # endpoint has gaps for many active mutual funds (it's named for
    # ETFs and doesn't always return data for MF tickers). To avoid
    # silently returning 0% expense ratio for popular MFs like
    # Contrafund (FCNTX), Wellington (VWELX), Growth Fund of America
    # (AGTHX), etc., we consult the curated _MUTUAL_FUND_TABLE which
    # carries verified expense ratios alongside asset classification.
    # Returns the value WITHOUT writing to disk cache so that future
    # API improvements (or fee cuts) propagate when the table is
    # refreshed in code.
    _mf_lookup = _mutual_fund_lookup(t)
    if _mf_lookup is not None:
        sess[t] = float(_mf_lookup[2])
        return sess[t]

    # 4. Alpha Vantage ETF_PROFILE — PRIMARY SOURCE.
    # Free tier supports this endpoint reliably and pulls from SEC
    # filings (same primary source as FINRA Fund Analyzer).
    from datetime import datetime as _dt
    av_key = None
    try:
        av_key = st.session_state.get("av_api_key_manual")
    except Exception:
        av_key = None
    if not av_key:
        try:
            av_key = st.secrets.get("ALPHA_VANTAGE_API_KEY")
        except Exception:
            av_key = None
    if not av_key:
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

    if av_key:
        er = _alphavantage_fetch_expense_ratio(t, av_key)
        if er is not None:
            disk_cache[t] = {
                "er": float(er),
                "fetched": _dt.now().isoformat(),
                "source": "alphavantage",
            }
            _save_er_cache(disk_cache)
            sess[t] = float(er)
            return sess[t]

    # 5. FMP profile API — secondary source. Free tier doesn't always
    # support these endpoints; included for users with paid plans.
    fmp_key = None
    try:
        fmp_key = st.session_state.get("fmp_api_key_manual")
    except Exception:
        fmp_key = None
    if not fmp_key:
        try:
            fmp_key = st.secrets.get("FMP_API_KEY")
        except Exception:
            fmp_key = None
    if not fmp_key:
        fmp_key = os.environ.get("FMP_API_KEY")

    if fmp_key:
        er = _fmp_fetch_expense_ratio(t, fmp_key)
        if er is not None:
            disk_cache[t] = {
                "er": float(er),
                "fetched": _dt.now().isoformat(),
                "source": "fmp",
            }
            _save_er_cache(disk_cache)
            sess[t] = float(er)
            return sess[t]

    # 6. EMERGENCY FALLBACK — hardcoded reference table.
    # Only consulted when both APIs are unreachable / no API key set.
    # Values verified against SEC filings periodically; see header
    # comment on _EXPENSE_RATIO_OVERRIDES for last-validated date.
    if t in _EXPENSE_RATIO_OVERRIDES:
        sess[t] = _EXPENSE_RATIO_OVERRIDES[t]
        # Don't write to disk cache — keep this branch ephemeral so
        # next session's API call gets a fresh value if connectivity
        # has been restored.
        return sess[t]

    # 7. All sources failed. For equity-classified tickers, return 0.0
    # (almost certainly a stock with no expense ratio). Otherwise None.
    if cls == "equity":
        disk_cache[t] = {
            "er": 0.0,
            "fetched": _dt.now().isoformat(),
            "source": "stock-fallback",
        }
        _save_er_cache(disk_cache)
        sess[t] = 0.0
        return 0.0
    sess[t] = None
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_portfolio_vol(tickers_tuple, weights_tuple, years=10):
    """Compute realized annualized portfolio vol from price history.

    Cached for 1hr, keyed by (tickers, rounded weights, years). Used as the
    `portfolio_vol` input to compute_portfolio_risk_score so the
    correlation/diversification adjustment fires.

    NOTE on the years default (10):
        Previously this function was hardcoded to a 3-year lookback. PCM
        (the Portfolio Composition table) uses a separate 10-year vol path
        for its Risk Score row, while every other consumer of this helper
        — the optimizer tier gauges, the PDF cover-page score, the
        Broad-ETF Alternate calibration, and the saved-portfolio gauges
        on Client Records — fell through to the 3-year cache. That
        produced a visible discrepancy: a portfolio that scored 62 in
        PCM might score 46 in the optimizer because the 3yr correlation
        adjustment differs from the 10yr one.

        Switching the default to 10 years aligns ALL gauges with the PCM
        methodology Tony validated as the source of truth. Per-ticker
        scoring (security_risk_score) still uses 3 years and is unchanged
        — only the portfolio-level diversification adjustment shifts.

    Args:
        tickers_tuple: tuple of ticker symbols
        weights_tuple: tuple of weights (rounded to 4 decimals for stable hashing)
        years: lookback window in years (default 10 = PCM parity).

    Returns:
        Float annualized vol (decimal) or None if computation fails.
    """
    if not tickers_tuple or not weights_tuple:
        return None
    try:
        from datetime import date as _date
        from dateutil.relativedelta import relativedelta as _rd
        _end_d = _date.today()
        _start_d = _end_d - _rd(years=int(years))
        _w_sum = sum(float(w) for w in weights_tuple)
        if _w_sum <= 0:
            return None
        _w_norm = {t: float(w) / _w_sum
                   for t, w in zip(tickers_tuple, weights_tuple)
                   if float(w) > 0}
        if not _w_norm:
            return None
        _pp, _ = get_prices_with_proxies(
            tuple(_w_norm.keys()), str(_start_d), str(_end_d))
        if _pp.empty:
            return None
        _pst = port_stats_from_prices(_pp, _w_norm, int(years))
        if _pst:
            return float(_pst.get("ann_vol", 0))
    except Exception:
        return None
    return None


def weighted_expense_ratio(tickers, weights):
    """Compute weighted expense ratio for a portfolio.

    Args:
        tickers: list of ticker symbols
        weights: list of weights (decimal or percent — auto-normalized)
    Returns:
        (weighted_er, coverage_pct) tuple where:
            weighted_er = expense ratio as decimal (e.g. 0.0008 = 0.08%)
            coverage_pct = % of portfolio weight for which we found an ER
                           (or determined definitively that none applies, i.e. stocks).

    Stocks count as fully covered (er=0.0). Coverage drops below 100% only
    when a fund's ER is genuinely unknown (FMP returned nothing AND no
    override entry).
    """
    if not tickers or not weights:
        return (0.0, 0.0)
    w = np.array([float(x or 0) for x in weights], dtype=float)
    total = w.sum()
    if total <= 0:
        return (0.0, 0.0)
    w = w / total

    weighted = 0.0
    coverage = 0.0
    for tk, wi in zip(tickers, w):
        er = _expense_ratio_for_ticker(tk)
        if er is not None:
            weighted += wi * er
            coverage += wi
    return (weighted, coverage * 100.0)


def build_tier_proposal(target_score, label, equity_universe=None, bond_universe=None,
                        cash_ticker="SGOV", priorities=None,
                        user_tickers=None, user_weights=None):
    """Build a proposal dict for a given target risk score.

    Two modes:
      1. **User-ticker mode** (when `user_tickers` is provided):
         Allocate across the client's actual submitted holdings, classified
         into equity/bond buckets via _is_bond. Missing buckets are filled
         with a single sensible default (VTI for equity, AGG for bonds).
         If `user_weights` is provided (dict of ticker→submitted %), weights
         within each bucket are proportional to the user's submitted weights.

      2. **Broad-ETF mode** (legacy, when `user_tickers` is None):
         Build the proposal from priority-driven ETF universes (VTI/VEA/VWO,
         AGG/TIP, etc.). Used for the "Broad-ETF Alternate" proposal.

    `priorities` is an optional list of priority keys from the client's
    Goals & Priorities multi-select. Priority tilts always apply.
    `label` is shown to the client (e.g. "Conservative", "Balanced").

    Returns {label, target_score, tickers, weights, equity_pct, bond_pct, cash_pct,
             rationale, priority_tilts, flags, save_to_profile}.
    """
    priorities = priorities or []
    eq_pct, bd_pct, cs_pct = allocation_for_risk_score(target_score)

    # ── PRIORITY TILTS ──────────────────────────────────────
    tilts_applied = []   # human-readable explanations for the proposal/notes

    # Capital preservation → reduce equity 7%, shift to bonds & cash
    if "capital_preservation" in priorities:
        shift = min(7.0, eq_pct)
        eq_pct -= shift; bd_pct += shift * 0.6; cs_pct += shift * 0.4
        tilts_applied.append("−7% equity → bonds/cash (capital preservation)")

    # Capital appreciation → +5% equity (if room)
    if "capital_appreciation" in priorities and eq_pct < 95:
        shift = min(5.0, bd_pct * 0.5)
        eq_pct += shift; bd_pct -= shift
        tilts_applied.append("+5% equity from bonds (capital appreciation)")

    # Liquidity → raise cash floor to at least 5%
    if "liquidity" in priorities and cs_pct < 5:
        need = 5 - cs_pct
        take_bd = min(need, bd_pct)
        bd_pct -= take_bd;       cs_pct += take_bd
        remainder = need - take_bd
        if remainder > 0:
            eq_pct -= remainder; cs_pct += remainder
        tilts_applied.append("cash raised to ≥5% (liquidity)")

    # Round & clamp
    eq_pct = round(max(0, eq_pct), 1)
    bd_pct = round(max(0, bd_pct), 1)
    cs_pct = round(max(0, cs_pct), 1)

    # ══════════════════════════════════════════════════════════════
    # MODE 1: USER-TICKER mode — allocate across submitted holdings
    # ══════════════════════════════════════════════════════════════
    if user_tickers:
        # Classify submitted tickers into equity vs bond buckets
        user_equity = [t for t in user_tickers if not _is_bond(t)]
        user_bonds  = [t for t in user_tickers if _is_bond(t)]
        tilts_applied.append(f"allocated across {len(user_tickers)} submitted ticker(s)")

        # If the user is missing a bucket we need, fill with single sensible default
        if eq_pct > 0 and not user_equity:
            user_equity = ["VTI"]
            tilts_applied.append("+VTI (no equity in submitted portfolio)")
        if bd_pct > 0 and not user_bonds:
            user_bonds = ["AGG"]
            tilts_applied.append("+AGG (no bonds in submitted portfolio)")

        # Distribute a percent across a bucket, weighted by user_weights if given
        def _distribute(bucket, pct):
            if not bucket or pct <= 0:
                return []
            if user_weights:
                shares = [max(0.0, float(user_weights.get(t, 0) or 0)) for t in bucket]
                total_s = sum(shares)
                if total_s <= 0:
                    shares = [1.0] * len(bucket); total_s = float(len(bucket))
                return [round(pct * s / total_s, 2) for s in shares]
            per = pct / len(bucket)
            return [round(per, 2) for _ in bucket]

        tickers, weights = [], []
        for t, w in zip(user_equity, _distribute(user_equity, eq_pct)):
            tickers.append(t); weights.append(w)
        for t, w in zip(user_bonds,  _distribute(user_bonds,  bd_pct)):
            tickers.append(t); weights.append(w)
        if cs_pct > 0:
            tickers.append(cash_ticker); weights.append(round(cs_pct, 2))

        # Normalize to exactly 100
        total = sum(weights)
        if total > 0:
            weights = [round(w * 100.0 / total, 2) for w in weights]

        flags = []
        if "insurance_planning" in priorities:
            flags.append("Review insurance coverage alongside this portfolio.")
        if "legacy_planning" in priorities:
            flags.append("Consider beneficiary designations / trust structures.")
        if "tax_efficiency" in priorities:
            flags.append("Consider asset location and tax-loss harvesting.")

        rationale = (
            f"{label} allocation targeting a risk score of ~{int(target_score)}, "
            f"built from the client's submitted holdings. "
            f"{eq_pct:.0f}% equity / {bd_pct:.0f}% bonds / {cs_pct:.0f}% cash."
        )
        if tilts_applied:
            rationale += " Tilts: " + "; ".join(tilts_applied) + "."

        return {
            "label":          label,
            "target_score":   int(target_score),
            "tickers":        tickers,
            "weights":        weights,
            "equity_pct":     eq_pct,
            "bond_pct":       bd_pct,
            "cash_pct":       cs_pct,
            "rationale":      rationale,
            "priority_tilts": tilts_applied,
            "priority_flags": flags,
            "save_to_profile": False,
        }

    # ══════════════════════════════════════════════════════════════
    # MODE 2: BROAD-ETF mode (priority-driven universes)
    # ══════════════════════════════════════════════════════════════
    # ── UNIVERSE SELECTION driven by priorities ──────────────
    if equity_universe is None:
        if "social_impact" in priorities:
            # ESG-focused equity sleeve
            equity_universe = ["ESGV", "VSGX", "ESGU"]
            tilts_applied.append("ESG equity ETFs (social/impact)")
        elif "income_generation" in priorities:
            # Dividend-tilted equity sleeve
            equity_universe = ["SCHD", "VIG", "VYM"]
            tilts_applied.append("dividend-tilt equity (income)")
        elif "capital_appreciation" in priorities:
            # Growth-leaning
            equity_universe = ["VTI", "QQQ", "VUG"]
            tilts_applied.append("growth-tilt equity (capital appreciation)")
        elif "diversification" in priorities:
            # Broader geo + REIT + commodity sleeve
            equity_universe = ["VTI", "VEA", "VWO", "VNQ", "GLD"]
            tilts_applied.append("expanded equity universe (diversification)")
        else:
            equity_universe = ["VTI", "VEA", "VWO"]

    if bond_universe is None:
        if "tax_efficiency" in priorities:
            # Municipal bonds favored for after-tax income
            bond_universe = ["MUB", "VTEB"]
            tilts_applied.append("municipal bonds (tax efficiency)")
        elif "income_generation" in priorities:
            # Higher-yield bond mix
            bond_universe = ["AGG", "HYG", "TIP"]
            tilts_applied.append("income-tilt bonds (income)")
        elif "capital_preservation" in priorities:
            # Short-duration / treasuries
            bond_universe = ["BIL", "SHY", "AGG"]
            tilts_applied.append("short-duration bonds (preservation)")
        else:
            bond_universe = ["AGG", "TIP"]

    tickers, weights = [], []
    if eq_pct > 0 and equity_universe:
        per_eq = eq_pct / len(equity_universe)
        for t in equity_universe:
            tickers.append(t); weights.append(round(per_eq, 2))
    if bd_pct > 0 and bond_universe:
        per_bd = bd_pct / len(bond_universe)
        for t in bond_universe:
            tickers.append(t); weights.append(round(per_bd, 2))
    if cs_pct > 0:
        tickers.append(cash_ticker); weights.append(round(cs_pct, 2))

    # Normalize to exactly 100
    total = sum(weights)
    if total > 0:
        weights = [round(w * 100.0 / total, 2) for w in weights]

    # ── Advisor notes (flags that don't change the math) ─────
    flags = []
    if "insurance_planning" in priorities:
        flags.append("Review insurance coverage (life, disability, umbrella) alongside this portfolio.")
    if "legacy_planning" in priorities:
        flags.append("Consider beneficiary designations, TOD accounts, or trust structures for wealth transfer.")
    if "tax_efficiency" in priorities:
        flags.append("Consider asset location (tax-advantaged vs. taxable) and tax-loss harvesting opportunities.")

    rationale = (
        f"{label} allocation targeting a risk score of ~{int(target_score)}. "
        f"{eq_pct:.0f}% equity / {bd_pct:.0f}% bonds / {cs_pct:.0f}% cash."
    )
    if tilts_applied:
        rationale += " Tilts: " + "; ".join(tilts_applied) + "."

    return {
        "label":          label,
        "target_score":   int(target_score),
        "tickers":        tickers,
        "weights":        weights,
        "equity_pct":     eq_pct,
        "bond_pct":       bd_pct,
        "cash_pct":       cs_pct,
        "rationale":      rationale,
        "priority_tilts": tilts_applied,
        "priority_flags": flags,
    }


def _schwab_engine_score_adapter(tickers, weights_pct):
    """Adapter passed to portfolios.calibrate_core_etf_tiers().

    Computes the same engine score that the analysis-table gauge uses for
    the Broad-ETF Alternate tier — security_risk_score per holding, plus
    the diversification adjustment via portfolio_vol / weighted_sum_vol.

    Returns an integer 1-99 or None if any required data is missing.
    """
    try:
        if not tickers or not weights_pct:
            return None
        # Per-ticker risk scores + vols (3yr lookback, cached for 1hr each)
        h_scores, h_vols = [], []
        for tk in tickers:
            r = security_risk_score(tk)
            if r is None:
                return None
            h_scores.append(r["score"])
            h_vols.append(r["ann_vol"])
        # Portfolio vol via the same cached path the optimizer uses
        try:
            p_vol = _cached_portfolio_vol(
                tuple(tickers),
                tuple(round(float(w) / 100.0, 6) for w in weights_pct),
            )
        except Exception:
            p_vol = None
        return compute_portfolio_risk_score(
            tickers, weights_pct,
            holding_scores=h_scores,
            holding_vols=h_vols,
            portfolio_vol=p_vol,
        )
    except Exception:
        return None


def _build_schwab_alternate_tier(client_risk_score, label, priorities=None):
    """Build the Broad-ETF Alternate tier from the Schwab Core ETF model.

    Replaces the prior generic broad-ETF construction with a registry-
    driven recommendation: the floor-match rule picks the highest-scoring
    Schwab Core ETF tier whose engine score doesn't exceed the client's
    risk score. If no Core ETF tier is within the gap threshold (10 pts),
    the AI synthesizer fires and returns a Sharpe-optimized broad-ETF
    portfolio targeting the client's score.

    Returned dict matches build_tier_proposal()'s shape so downstream code
    (PDF, pie charts, saved-proposal expanders) needs no changes.

    Returns None if the schwab_portfolios module/file isn't available;
    the caller should fall back to the legacy build_tier_proposal in that
    case.
    """
    if not _SCHWAB_AVAILABLE or _schwab_portfolios is None:
        return None

    priorities = priorities or []

    # Use the registry recommender, narrowed to Schwab Core ETF first.
    # If no Core ETF tier floor-matches within tolerance, the recommender
    # falls back through (Schwab any-series) → AI synthesizer.
    try:
        rec = _schwab_portfolios.recommend_portfolio(
            int(client_risk_score),
            live_engine_score_fn=_schwab_engine_score_adapter,
            prefer_source="schwab",
            prefer_series="core_etf",
        )
    except Exception as _e:
        _warnings.warn(f"recommend_portfolio failed: {_e}", RuntimeWarning)
        return None

    model = rec.get("model")
    if model is None:
        return None

    tickers = list(model.get("tickers", []))
    weights = list(model.get("weights", []))

    # Normalize to exactly 100% (rounding artifacts from the registry)
    _total = sum(weights) if weights else 0
    if _total > 0:
        weights = [round(w * 100.0 / _total, 2) for w in weights]

    eq_pct = float(model.get("equity_pct", 0))
    bd_pct = float(model.get("bond_pct", 0))
    cs_pct = float(model.get("cash_pct", 0))

    # Tilts/flags — model is used verbatim, allocations not tweaked by
    # priorities. We surface advisor flags only so they show up in notes
    # alongside the other tiers.
    flags = []
    if "insurance_planning" in priorities:
        flags.append("Review insurance coverage (life, disability, umbrella) alongside this portfolio.")
    if "legacy_planning" in priorities:
        flags.append("Consider beneficiary designations, TOD accounts, or trust structures for wealth transfer.")
    if "tax_efficiency" in priorities:
        flags.append("Consider asset location (tax-advantaged vs. taxable) and tax-loss harvesting opportunities.")

    strategy = rec.get("strategy", "model")
    matched_score = rec.get("matched_score")
    gap = rec.get("gap")

    # Build a rationale that's transparent about what the recommender did.
    # Two flavors: (a) standard model match, (b) AI synthesized fallback.
    if strategy == "ai_synthesized":
        tilts = [f"AI synthesized broad-ETF portfolio · target score {int(client_risk_score)}"]
        rationale = (
            f"Broad-ETF alternate synthesized to target the client's risk score "
            f"of {int(client_risk_score)} (matched at engine score {matched_score}). "
            f"None of the Schwab Core ETF tiers floor-matched within "
            f"{getattr(_schwab_portfolios, 'AI_FALLBACK_GAP_THRESHOLD', 10)} points "
            f"of the client, so the system synthesized a Sharpe-optimized broad-ETF "
            f"mix using a low-cost ETF universe (VTI, VXUS, VWO, VNQ, AGG, VGIT, "
            f"TLT, TIP, SGOV). "
            f"Allocation: {eq_pct:.0f}% equity / {bd_pct:.0f}% fixed income / "
            f"{cs_pct:.0f}% cash. "
            f"Average expense ratio: {model.get('expense_ratio_avg', 0)*100:.2f}%."
        )
    else:
        # Standard Schwab Core ETF (or other source) match
        avg_er = model.get("expense_ratio_avg")
        er_text = f" (avg net expense ratio {avg_er*100:.2f}%)" if avg_er else ""
        tier_label = model.get("tier", "")
        tilts = [f"{model.get('series_display_name', 'Model')} · tier {tier_label}"]
        gap_text = ""
        if gap is not None and gap != 0:
            gap_text = f" (engine score {matched_score}, {abs(gap)} pts {'below' if gap > 0 else 'above'} client score)"
        rationale = (
            f"Broad-ETF alternate floor-matched to {model.get('label', 'a Schwab model')}{er_text}. "
            f"Selected to match the client's risk score of "
            f"{int(client_risk_score)}{gap_text}. "
            f"Allocation: {eq_pct:.0f}% equity / {bd_pct:.0f}% fixed income / "
            f"{cs_pct:.0f}% cash. "
            f"Uses a diversified ETF lineup instead of the client's submitted "
            f"tickers — a 'clean-slate' comparison anchored on a published "
            f"model rather than on the submitted holdings."
        )

    return {
        "label":          label,
        "target_score":   int(client_risk_score),
        "tickers":        tickers,
        "weights":        weights,
        "equity_pct":     round(eq_pct, 1),
        "bond_pct":       round(bd_pct, 1),
        "cash_pct":       round(cs_pct, 1),
        "rationale":      rationale,
        "priority_tilts": tilts,
        "priority_flags": flags,
        # Markers for downstream code (PDF, expanders) — registry-aware.
        "model_id":             model.get("id"),
        "model_source":         model.get("source"),
        "model_strategy":       strategy,        # "model" or "ai_synthesized"
        "model_matched_score":  matched_score,
        "model_gap":            gap,
        # Legacy fields kept for back-compat
        "schwab_model":      model.get("series") if model.get("source") == "schwab" else None,
        "schwab_tier_label": model.get("tier") if model.get("source") == "schwab" else None,
    }


def generate_three_tier_proposals(client_risk_score, advisor_notes="", priorities=None,
                                  user_tickers=None, user_weights=None):
    """Generate FOUR proposal options for a client, anchored on the client's
    CURRENT portfolio (per new design):

      1. conservative  — derived from balanced by adding bonds (shift toward
                         bonds proportional to how much above the client's
                         risk score balanced's implied score is, with a floor
                         to always add a meaningful bond tilt).
      2. balanced      — the CLIENT'S CURRENT PORTFOLIO VERBATIM. This is the
                         core proposal. user_tickers + user_weights become the
                         balanced tier with no recomputation.
      3. aggressive    — derived from balanced by reducing bonds / adding
                         equity (opposite direction from conservative).
      4. alternate     — broad-ETF clean-slate proposal at the client's risk
                         score, ignoring submitted tickers.

    The shift magnitude for conservative/aggressive is based on the delta
    between the client's risk score and balanced's implied risk score. When
    balanced already matches the client's score, the shift is a minimum of
    12 percentage points. When the delta is large, the shift scales up.

    If user_tickers is None/empty, the three primary tiers fall back to the
    old risk-score-targeted behavior (legacy flow).
    """
    priorities = priorities or []

    # ── Legacy fallback: no user tickers provided ─────────────
    if not user_tickers:
        # Keep backward-compatible behavior: build all tiers from risk score
        s = int(client_risk_score)
        lower   = max(1, s - 15)
        matched = s
        higher  = min(99, s + 15)
        # Alternate: Schwab Core ETF model at the matched score, with a
        # legacy build_tier_proposal fallback if schwab_portfolios isn't
        # importable for any reason.
        _alt = _build_schwab_alternate_tier(matched, "Broad-ETF Alternate",
                                            priorities=priorities)
        if _alt is None:
            _alt = build_tier_proposal(matched, "Broad-ETF Alternate",
                                       priorities=priorities)
        return {
            "conservative": build_tier_proposal(lower,   "Option 2 (slightly more conservative)",
                                                priorities=priorities),
            "balanced":     build_tier_proposal(matched, "Option 1 (proposed)",
                                                priorities=priorities),
            "aggressive":   build_tier_proposal(higher,  "Option 3 (slightly more aggressive)",
                                                priorities=priorities),
            "alternate":    _alt,
        }

    # ── MODE 1: BALANCED = current portfolio verbatim ─────────
    balanced_tier = _build_balanced_from_current(
        client_risk_score, priorities, user_tickers, user_weights,
    )

    # Derive Conservative + Aggressive from balanced via bond-tilt shift
    bal_eq = balanced_tier.get("equity_pct", 0)
    bal_bd = balanced_tier.get("bond_pct",   0)
    bal_cs = balanced_tier.get("cash_pct",   0)
    bal_implied = implied_risk_score_from_allocation(bal_eq)

    # Shift magnitude scaled by (client_score - balanced_implied)
    # If balanced already matches the score → still apply 12pt min shift so
    # tiers are visibly distinct.
    score_delta = int(client_risk_score) - bal_implied
    base_shift  = 12.0
    # Larger delta → larger shift (max ~20pts so we don't rewrite allocation wholesale)
    extra_shift = min(8.0, abs(score_delta) * 0.4)
    shift_pct   = base_shift + extra_shift

    # ── Options 1 (conservative) and 3 (aggressive) via corridor optimizer ──
    # Run a constrained re-optimization of the user's existing holdings:
    #   • Option 1: minimize variance, weights within ±50% of submitted
    #   • Option 3: maximize Sharpe ratio, weights within ±50% of submitted
    # Same tickers as Option 2, just rebalanced toward each objective.
    # If the optimization can't run (no price data, scipy issues, etc.),
    # fall back to the legacy ±15% bond/equity tilt via _shift_tier.
    def _tier_from_opt(opt_result, base_tier, label, target_score, direction_text,
                       priorities, fallback_shift_pct):
        """Wrap optimizer output (tickers, weights_pct) into a tier dict
        matching _shift_tier's output shape. Falls back to _shift_tier if
        opt_result is None."""
        if opt_result is None:
            # Fallback: legacy bond/equity tilt
            return _shift_tier(
                base_tier,
                direction="conservative" if "conservative" in direction_text else "aggressive",
                shift_pct=fallback_shift_pct,
                priorities=priorities,
                label=label,
            )
        opt_tickers, opt_weights = opt_result
        # Recompute equity/bond/cash buckets for the new allocation
        _CASH_TICKERS = {"SGOV", "BIL", "SHV", "VUSXX", "USFR", "SHY"}
        eq_pct = bd_pct = cs_pct = 0.0
        for t, w in zip(opt_tickers, opt_weights):
            tu = t.upper()
            if tu in _CASH_TICKERS:
                cs_pct += w
            elif _is_bond(t):
                bd_pct += w
            else:
                eq_pct += w
        rationale = (
            f"{direction_text} — re-optimized within ±50% of the submitted "
            f"allocation. Same holdings as Option 2; weights tilted to "
            f"{'minimize volatility' if 'conservative' in direction_text else 'maximize Sharpe ratio'}."
        )
        return {
            "label":         label,
            "target_score":  int(target_score),
            "tickers":       opt_tickers,
            "weights":       opt_weights,
            "equity_pct":    round(eq_pct, 1),
            "bond_pct":      round(bd_pct, 1),
            "cash_pct":      round(cs_pct, 1),
            "rationale":     rationale,
            "priority_tilts": [
                ("Min-volatility re-optimization within ±50% corridor"
                 if "conservative" in direction_text
                 else "Max-Sharpe re-optimization within ±50% corridor")
            ],
            "priority_flags": [],
            "save_to_profile": False,
        }

    _cons_opt = _optimize_within_corridor(
        user_tickers, user_weights, objective="min_vol", corridor=0.5,
    )
    conservative_tier = _tier_from_opt(
        _cons_opt, balanced_tier,
        label="Option 2 (slightly more conservative)",
        target_score=max(1, int(client_risk_score) - 15),
        direction_text="Slightly more conservative",
        priorities=priorities,
        fallback_shift_pct=shift_pct,
    )

    _aggr_opt = _optimize_within_corridor(
        user_tickers, user_weights, objective="max_sharpe", corridor=0.5,
    )
    aggressive_tier = _tier_from_opt(
        _aggr_opt, balanced_tier,
        label="Option 3 (slightly more aggressive)",
        target_score=min(99, int(client_risk_score) + 15),
        direction_text="Slightly more aggressive",
        priorities=priorities,
        fallback_shift_pct=shift_pct,
    )

    # Fourth proposal — Schwab Core ETF model portfolio at the tier matched
    # to the client's risk score. Falls back to the legacy build_tier_proposal
    # path if schwab_portfolios isn't loadable. The returned dict carries
    # the same keys as the other tiers, so the saved-proposal expander, PDF
    # builder, and pie charts work unchanged.
    alternate_tier = _build_schwab_alternate_tier(
        int(client_risk_score), "Broad-ETF Alternate",
        priorities=priorities,
    )
    if alternate_tier is None:
        # Fallback: legacy clean-slate broad-ETF build
        alternate_tier = build_tier_proposal(
            int(client_risk_score), "Broad-ETF Alternate",
            priorities=priorities,
            user_tickers=None, user_weights=None,
        )
        alternate_tier["rationale"] = (
            "Broad-ETF alternate built from scratch at the client's risk score. "
            "Uses diversified index-ETF sleeves instead of the client's submitted "
            "tickers — a 'clean-slate' comparison. " + alternate_tier.get("rationale", "")
        )

    return {
        "conservative": conservative_tier,
        "balanced":     balanced_tier,
        "aggressive":   aggressive_tier,
        "alternate":    alternate_tier,
    }


def _optimize_within_corridor(user_tickers, user_weights, objective="min_vol",
                              corridor=0.5, lookback_years=10):
    """Re-optimize the user's existing holdings within a ±corridor multiplicative
    bound around their submitted weights.

    Why this exists: the simplest "more conservative" / "more aggressive"
    variants for Options 1 and 3 should *recognize* the client's current
    portfolio — same tickers, similar proportions — but tilt the math
    toward an objective. A corridor of ±50% means a holding at 20% can
    range 10%-30%, a holding at 5% can range 2.5%-7.5%, etc.

    Parameters
    ----------
    user_tickers : list[str]
        Step 1 holdings.
    user_weights : dict[str, float]
        Submitted weights as percentages (sum to 100). Tickers missing from
        the dict are treated as zero (and therefore can stay at zero).
    objective : str
        "min_vol"     → minimize variance (Option 1, conservative)
        "max_sharpe"  → maximize Sharpe ratio (Option 3, aggressive)
    corridor : float
        ±multiplicative deviation from base weight, expressed as a fraction.
        Default 0.5 = ±50%. So a holding at 20% gets bounds [10%, 30%].
    lookback_years : int
        How much history to fetch for covariance/returns estimation.

    Returns
    -------
    (tickers, weights_pct) tuple, or None if optimization couldn't run
    (e.g., no price data, scipy convergence failure, etc.). Caller should
    fall back to a simpler tier-shift method when None is returned.

    Implementation notes
    --------------------
    Uses skfolio's MeanRisk with per-asset min/max weight constraints.
    Both objectives use the same Ledoit-Wolf shrunk covariance — what
    differs is the objective function (minimize vs maximize_ratio).
    Always returns weights summing to 1.0 (skfolio's invariant).
    """
    if not user_tickers or not user_weights:
        return None
    # Build base weights as decimals (skfolio expects 0-1, not 0-100)
    base_w = []
    for t in user_tickers:
        w = float(user_weights.get(t, 0) or 0) / 100.0
        base_w.append(max(0.0, w))
    if sum(base_w) <= 0:
        return None
    # Renormalize base weights so they sum to 1 (handles rounding noise)
    _bsum = sum(base_w)
    base_w = [w / _bsum for w in base_w]

    # Compute corridor bounds. For zero-weight holdings, allow them to
    # stay at zero but also allow a small non-zero band so the optimizer
    # has flexibility (otherwise zero stays zero forever).
    min_w, max_w = [], []
    for w in base_w:
        if w <= 0.001:
            # Zero-weight holding: optimizer can give it 0-1% if it wants
            min_w.append(0.0)
            max_w.append(0.01)
        else:
            min_w.append(max(0.0, w * (1 - corridor)))
            max_w.append(min(1.0, w * (1 + corridor)))

    # Sanity: the sum of mins must be ≤ 1 and sum of maxes must be ≥ 1
    # (otherwise no feasible solution exists). Loosen if needed.
    if sum(min_w) > 1.0:
        # Scale all mins down proportionally
        _scale = 0.99 / sum(min_w)
        min_w = [m * _scale for m in min_w]
    if sum(max_w) < 1.0:
        # Scale all maxes up; if a single position has max < its base, this
        # signals a numerical issue — give up and let the caller fall back.
        return None

    # Fetch price data + build returns matrix
    try:
        end_dt = date.today()
        start_dt = end_dt - relativedelta(years=lookback_years)
        prices, _ = get_prices_with_proxies(
            tuple(user_tickers), start_dt, end_dt,
            min_days=max(60, lookback_years * 60),
        )
        if prices is None or prices.empty:
            return None
        # Restrict to tickers we actually have data for; rebuild bounds in same order
        present = [t for t in user_tickers if t in prices.columns]
        if len(present) < 2:
            # Need at least 2 holdings for a meaningful optimization
            return None
        # If we lost any tickers, restrict everything to the present subset
        idx_keep = [i for i, t in enumerate(user_tickers) if t in present]
        base_w_k = [base_w[i] for i in idx_keep]
        min_w_k  = [min_w[i]  for i in idx_keep]
        max_w_k  = [max_w[i]  for i in idx_keep]
        # Renormalize after dropping
        _b = sum(base_w_k)
        if _b > 0:
            base_w_k = [w / _b for w in base_w_k]
        prices_k = prices[present].dropna(how="all").ffill()
        if len(prices_k) < 60:
            return None
        X = prices_to_returns(prices_k)
        if X is None or len(X) < 60:
            return None
        X = X.dropna(axis=0, how="any")
        if len(X) < 60:
            return None

        # Build skfolio model with per-asset corridor constraints
        prior = EmpiricalPrior(
            mu_estimator=ShrunkMu(),
            covariance_estimator=LedoitWolf(),
        )
        if objective == "max_sharpe":
            model = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                risk_measure=RiskMeasure.VARIANCE,
                prior_estimator=prior,
                min_weights=dict(zip(present, min_w_k)),
                max_weights=dict(zip(present, max_w_k)),
            )
        else:  # default: min_vol
            model = MeanRisk(
                risk_measure=RiskMeasure.VARIANCE,
                prior_estimator=prior,
                min_weights=dict(zip(present, min_w_k)),
                max_weights=dict(zip(present, max_w_k)),
            )

        model.fit(X)
        ptf = model.predict(X)
        opt_w = list(ptf.weights)

        # Convert to percentages (sum to 100, rounded to 2dp)
        out_pct = [round(w * 100.0, 2) for w in opt_w]
        # Renormalize to exactly 100 (handle rounding)
        s = sum(out_pct)
        if s > 0:
            out_pct = [round(w * 100.0 / s, 2) for w in out_pct]

        # Reassemble in original ticker order — tickers we dropped get 0
        out_tickers = []
        out_weights = []
        out_map = dict(zip(present, out_pct))
        for t in user_tickers:
            w = out_map.get(t, 0.0)
            if w > 0.05:  # drop dust
                out_tickers.append(t)
                out_weights.append(w)
        # Final renormalize
        s = sum(out_weights)
        if s > 0:
            out_weights = [round(w * 100.0 / s, 2) for w in out_weights]
        if not out_tickers:
            return None
        return out_tickers, out_weights
    except Exception:
        return None


def _build_balanced_from_current(client_risk_score, priorities, user_tickers, user_weights):
    """Construct the Balanced tier as the user's CURRENT PORTFOLIO verbatim.

    Takes user_weights directly (no re-allocation). Classifies existing
    tickers into equity/bond/cash buckets to compute the equity_pct /
    bond_pct / cash_pct summary used elsewhere (PDF, rationale, etc.).
    """
    priorities = priorities or []
    # Normalize weights to percentages
    tickers, weights = [], []
    _raw_total = 0.0
    for t in user_tickers:
        w = float(user_weights.get(t, 0) or 0) if user_weights else 0
        if w <= 0:
            continue
        tickers.append(t); weights.append(w)
        _raw_total += w

    # Equal-weight fallback if user_weights didn't contain values
    if not weights:
        tickers = list(user_tickers)
        per = 100.0 / max(1, len(tickers))
        weights = [per for _ in tickers]
        _raw_total = sum(weights)

    if _raw_total > 0 and abs(_raw_total - 100.0) > 0.01:
        weights = [round(w * 100.0 / _raw_total, 2) for w in weights]

    # Classify into equity / bond / cash buckets for summary stats
    eq_pct = bd_pct = cs_pct = 0.0
    _CASH_TICKERS = {"SGOV", "BIL", "SHV", "VUSXX", "USFR", "SHY"}
    for t, w in zip(tickers, weights):
        tu = t.upper()
        if tu in _CASH_TICKERS:
            cs_pct += w
        elif _is_bond(t):
            bd_pct += w
        else:
            eq_pct += w

    rationale = (
        f"Proposed allocation — your submitted holdings, verbatim. "
        f"Classification: {eq_pct:.0f}% equity / "
        f"{bd_pct:.0f}% bonds / {cs_pct:.0f}% cash. "
        f"Options 1 and 3 are derived from this allocation by re-optimizing "
        f"the same holdings within ±50% of these weights — Option 1 minimizes "
        f"volatility, Option 3 maximizes Sharpe ratio."
    )

    flags = []
    if "insurance_planning" in priorities:
        flags.append("Review insurance coverage alongside this portfolio.")
    if "legacy_planning" in priorities:
        flags.append("Consider beneficiary designations / trust structures.")
    if "tax_efficiency" in priorities:
        flags.append("Consider asset location and tax-loss harvesting.")

    return {
        "label":           "Option 1 (proposed)",
        "target_score":    int(client_risk_score),
        "tickers":         tickers,
        "weights":         weights,
        "equity_pct":      round(eq_pct, 1),
        "bond_pct":        round(bd_pct, 1),
        "cash_pct":        round(cs_pct, 1),
        "rationale":       rationale,
        "priority_tilts":  ["balanced = current portfolio (Step 1)"],
        "priority_flags":  flags,
        "save_to_profile": False,
    }


def _shift_tier(balanced, direction, shift_pct, priorities, label):
    """Derive a Conservative or Aggressive tier by shifting balanced's weights.

    GUARANTEES (per user UX requirement):
      • Aggressive result has equity_pct > Balanced's equity_pct by at least 10pts
      • Conservative result has equity_pct < Balanced's equity_pct by at least 10pts
      • When Balanced is at extremes (e.g. 95% equity), we INJECT the needed
        sleeve (AGG for conservative / VTI for aggressive) to force the spread.

    direction='conservative': take shift_pct from equity → add to bonds.
    direction='aggressive':   take shift_pct from bonds → add to equity.

    Returns a fresh proposal dict with the same shape as build_tier_proposal.
    """
    new_tickers = list(balanced["tickers"])
    new_weights = [float(w) for w in balanced["weights"]]

    _CASH_TICKERS = {"SGOV", "BIL", "SHV", "VUSXX", "USFR", "SHY"}

    def _reclassify():
        """Recompute index buckets after tickers are added/removed."""
        eq_idx = [i for i, t in enumerate(new_tickers)
                  if t.upper() not in _CASH_TICKERS and not _is_bond(t)]
        bd_idx = [i for i, t in enumerate(new_tickers)
                  if _is_bond(t) and t.upper() not in _CASH_TICKERS]
        eq_total = sum(new_weights[i] for i in eq_idx)
        bd_total = sum(new_weights[i] for i in bd_idx)
        return eq_idx, bd_idx, eq_total, bd_total

    # Balanced's equity % — the reference point we must separate from
    balanced_eq = float(balanced.get("equity_pct", 0))

    # Minimum equity-pct delta required for tier distinctness
    MIN_EQ_DELTA = 10.0

    # Target equity pct for this tier
    if direction == "conservative":
        # Want equity to drop by AT LEAST shift_pct, and no more than MIN_EQ_DELTA below 0
        target_eq = max(0.0, balanced_eq - max(shift_pct, MIN_EQ_DELTA))
    else:  # aggressive
        # Want equity to rise by AT LEAST shift_pct, capped at 98
        target_eq = min(98.0, balanced_eq + max(shift_pct, MIN_EQ_DELTA))

    tilts = []
    eq_idx, bd_idx, eq_total, bd_total = _reclassify()

    if direction == "conservative":
        # How much equity to remove to hit target_eq
        needed_remove = balanced_eq - target_eq
        # Can we take all of it from existing equity? If equity exhausts,
        # the rest must come from cash (if any).
        take = min(needed_remove, eq_total)
        if eq_total > 0 and take > 0:
            for i in eq_idx:
                new_weights[i] -= new_weights[i] / eq_total * take
        tilts.append(f"−{take:.1f}% equity (toward bonds) for conservative tilt")

        # Add equivalent to bonds; inject AGG if no bonds exist
        if bd_idx:
            if bd_total > 0:
                for i in bd_idx:
                    new_weights[i] += new_weights[i] / bd_total * take
            else:
                per = take / len(bd_idx)
                for i in bd_idx:
                    new_weights[i] += per
        else:
            new_tickers.append("AGG"); new_weights.append(round(take, 2))
            tilts.append("+AGG added (no bonds in balanced)")

    elif direction == "aggressive":
        needed_add = target_eq - balanced_eq
        # Can we take the shift from existing bonds? If not, we'll need to
        # pull from cash (last resort) or just inject more equity.
        take = min(needed_add, bd_total)
        if bd_total > 0 and take > 0:
            for i in bd_idx:
                new_weights[i] -= new_weights[i] / bd_total * take
        tilts.append(f"−{take:.1f}% bonds (toward equity) for aggressive tilt")

        # Sometimes balanced has so few bonds that take < needed_add. In that
        # case we also pull from cash to reach the target.
        remaining = needed_add - take
        if remaining > 0.5:
            cs_idx = [i for i, t in enumerate(new_tickers)
                      if t.upper() in _CASH_TICKERS]
            cs_total = sum(new_weights[i] for i in cs_idx)
            cs_take = min(remaining, cs_total)
            if cs_total > 0 and cs_take > 0:
                for i in cs_idx:
                    new_weights[i] -= new_weights[i] / cs_total * cs_take
                take += cs_take
                tilts.append(f"−{cs_take:.1f}% cash (toward equity) — bonds exhausted")

        # Add equivalent to equity; inject VTI if no equity exists
        if eq_idx:
            if eq_total > 0:
                for i in eq_idx:
                    new_weights[i] += new_weights[i] / eq_total * take
            else:
                per = take / len(eq_idx)
                for i in eq_idx:
                    new_weights[i] += per
        else:
            new_tickers.append("VTI"); new_weights.append(round(take, 2))
            tilts.append("+VTI added (no equity in balanced)")

    # Drop any tickers that ended up with ~0 weight
    _cleaned = [(t, round(w, 2)) for t, w in zip(new_tickers, new_weights) if w > 0.05]
    new_tickers = [t for t, _ in _cleaned]
    new_weights = [w for _, w in _cleaned]

    # Renormalize to 100
    total = sum(new_weights)
    if total > 0:
        new_weights = [round(w * 100.0 / total, 2) for w in new_weights]

    # Recompute bucket stats
    eq_pct = bd_pct = cs_pct = 0.0
    for t, w in zip(new_tickers, new_weights):
        tu = t.upper()
        if tu in _CASH_TICKERS:
            cs_pct += w
        elif _is_bond(t):
            bd_pct += w
        else:
            eq_pct += w

    # ── HARD-ENFORCE SEPARATION (belt-and-suspenders) ─────────
    # If for any reason we didn't achieve the required delta, inject directly.
    # EDGE CASE: when balanced is already at an extreme (≥90% equity for
    # aggressive, ≤10% equity for conservative), there's no bond/equity
    # sleeve to move FROM, so the standard shift can't produce separation.
    # In that case we tilt the *composition* of the equity sleeve instead:
    #   • Aggressive on equity-heavy balanced: inject QQQ (higher-beta
    #     NASDAQ-100) and pull proportionally from broad-market — same
    #     equity %, but visibly more aggressive composition.
    #   • Conservative on bond-heavy balanced: inject SHV (short Treasuries,
    #     the lowest-beta fixed-income proxy) and pull proportionally
    #     from longer-duration bonds — same bond %, but lower duration
    #     and lower volatility.
    achieved_delta = (balanced_eq - eq_pct) if direction == "conservative" \
                     else (eq_pct - balanced_eq)
    if achieved_delta < MIN_EQ_DELTA - 0.5:
        shortfall = MIN_EQ_DELTA - achieved_delta
        # Detect the "extreme balanced" case where standard shift broke down
        _extreme_balanced = (
            (direction == "aggressive"   and balanced_eq >= 90.0) or
            (direction == "conservative" and balanced_eq <= 10.0)
        )
        if _extreme_balanced:
            if direction == "aggressive":
                # Inject ~15% QQQ for high-beta tilt; pull proportionally
                # from existing equity holdings. Net equity % unchanged,
                # composition shifted toward higher-beta names.
                _tilt_amt = max(15.0, shortfall)
                _tilt_ticker = "QQQ"
                if _tilt_ticker in new_tickers:
                    idx = new_tickers.index(_tilt_ticker)
                    new_weights[idx] += _tilt_amt
                else:
                    new_tickers.append(_tilt_ticker)
                    new_weights.append(round(_tilt_amt, 2))
                # Pull proportionally from non-QQQ equity holdings
                cur_eq_idx = [i for i, t in enumerate(new_tickers)
                              if t.upper() not in _CASH_TICKERS
                              and not _is_bond(t)
                              and t.upper() != _tilt_ticker]
                cur_eq_total = sum(new_weights[i] for i in cur_eq_idx)
                if cur_eq_total > 0:
                    for i in cur_eq_idx:
                        new_weights[i] -= new_weights[i] / cur_eq_total * _tilt_amt
                tilts.append(
                    f"+{_tilt_amt:.0f}% {_tilt_ticker} (high-beta tilt) — "
                    f"balanced is already equity-heavy, so 'more aggressive' "
                    f"means a higher-beta composition rather than more equity %"
                )
            else:  # conservative
                # Inject ~15% SHV for low-duration tilt; pull proportionally
                # from existing bonds.
                _tilt_amt = max(15.0, shortfall)
                _tilt_ticker = "SHV"
                if _tilt_ticker in new_tickers:
                    idx = new_tickers.index(_tilt_ticker)
                    new_weights[idx] += _tilt_amt
                else:
                    new_tickers.append(_tilt_ticker)
                    new_weights.append(round(_tilt_amt, 2))
                cur_bd_idx = [i for i, t in enumerate(new_tickers)
                              if _is_bond(t)
                              and t.upper() not in _CASH_TICKERS
                              and t.upper() != _tilt_ticker]
                cur_bd_total = sum(new_weights[i] for i in cur_bd_idx)
                if cur_bd_total > 0:
                    for i in cur_bd_idx:
                        new_weights[i] -= new_weights[i] / cur_bd_total * _tilt_amt
                tilts.append(
                    f"+{_tilt_amt:.0f}% {_tilt_ticker} (short-duration tilt) — "
                    f"balanced is already bond-heavy, so 'more conservative' "
                    f"means lower duration rather than more bond %"
                )
        elif direction == "conservative":
            # Inject AGG at a weight that covers the shortfall
            if "AGG" in new_tickers:
                idx = new_tickers.index("AGG")
                new_weights[idx] += shortfall
            else:
                new_tickers.append("AGG"); new_weights.append(round(shortfall, 2))
            # Pull proportionally from equity
            cur_eq_idx = [i for i, t in enumerate(new_tickers)
                          if t.upper() not in _CASH_TICKERS and not _is_bond(t)]
            cur_eq_total = sum(new_weights[i] for i in cur_eq_idx)
            if cur_eq_total > 0:
                for i in cur_eq_idx:
                    new_weights[i] -= new_weights[i] / cur_eq_total * shortfall
            tilts.append(f"+{shortfall:.1f}% AGG injection to enforce tier separation")
        else:  # aggressive
            if "VTI" in new_tickers:
                idx = new_tickers.index("VTI")
                new_weights[idx] += shortfall
            else:
                new_tickers.append("VTI"); new_weights.append(round(shortfall, 2))
            cur_bd_idx = [i for i, t in enumerate(new_tickers)
                          if _is_bond(t) and t.upper() not in _CASH_TICKERS]
            cur_bd_total = sum(new_weights[i] for i in cur_bd_idx)
            if cur_bd_total > 0:
                for i in cur_bd_idx:
                    new_weights[i] -= new_weights[i] / cur_bd_total * shortfall
            else:
                cur_cs_idx = [i for i, t in enumerate(new_tickers)
                              if t.upper() in _CASH_TICKERS]
                cur_cs_total = sum(new_weights[i] for i in cur_cs_idx)
                if cur_cs_total > 0:
                    for i in cur_cs_idx:
                        new_weights[i] -= new_weights[i] / cur_cs_total * shortfall
            tilts.append(f"+{shortfall:.1f}% VTI injection to enforce tier separation")

        # Re-clean + renormalize + recompute
        _cleaned = [(t, round(w, 2)) for t, w in zip(new_tickers, new_weights) if w > 0.05]
        new_tickers = [t for t, _ in _cleaned]
        new_weights = [w for _, w in _cleaned]
        total = sum(new_weights)
        if total > 0:
            new_weights = [round(w * 100.0 / total, 2) for w in new_weights]
        eq_pct = bd_pct = cs_pct = 0.0
        for t, w in zip(new_tickers, new_weights):
            tu = t.upper()
            if tu in _CASH_TICKERS:
                cs_pct += w
            elif _is_bond(t):
                bd_pct += w
            else:
                eq_pct += w

    implied = implied_risk_score_from_allocation(eq_pct)

    rationale = (
        f"{label} proposal — derived from Balanced by shifting "
        f"{'equity → bonds' if direction == 'conservative' else 'bonds → equity'}. "
        f"Result: {eq_pct:.0f}% equity / {bd_pct:.0f}% bonds / {cs_pct:.0f}% cash "
        f"(implied risk score ≈ {implied}; Balanced implied ≈ "
        f"{implied_risk_score_from_allocation(balanced_eq)})."
    )
    if tilts:
        rationale += " Adjustments: " + "; ".join(tilts) + "."

    flags = []
    if "insurance_planning" in priorities:
        flags.append("Review insurance coverage alongside this portfolio.")
    if "legacy_planning" in priorities:
        flags.append("Consider beneficiary designations / trust structures.")
    if "tax_efficiency" in priorities:
        flags.append("Consider asset location and tax-loss harvesting.")

    return {
        "label":           label,
        "target_score":    implied,
        "tickers":         new_tickers,
        "weights":         new_weights,
        "equity_pct":      round(eq_pct, 1),
        "bond_pct":        round(bd_pct, 1),
        "cash_pct":        round(cs_pct, 1),
        "rationale":       rationale,
        "priority_tilts":  tilts,
        "priority_flags":  flags,
        "save_to_profile": False,
    }


# ── BOND TICKER PATTERNS for Manual Tilt heuristic ────────────
_BOND_PATTERNS = ("AGG","BND","TLT","IEF","SHY","LQD","HYG","MUB","VTEB","TIP","BIL","SGOV")

# Balanced/allocation funds split between equity and bond buckets at
# their stated weights. We don't try to fetch the live mix per fund —
# instead we use a typical-mix table for the most common balanced
# funds and default to 60/40 for anything else the classifier flagged
# as balanced.
_BALANCED_FUND_SPLITS = {
    # Vanguard
    "VWELX": (0.65, 0.35),  "VWENX": (0.65, 0.35),  # Wellington
    "VWINX": (0.40, 0.60),  "VWIAX": (0.40, 0.60),  # Wellesley
    "VGSTX": (0.60, 0.40),                           # STAR
    # Vanguard LifeStrategy
    "VASIX": (0.20, 0.80),  # Income
    "VSCGX": (0.40, 0.60),  # Conservative Growth
    "VSMGX": (0.60, 0.40),  # Moderate Growth
    "VASGX": (0.80, 0.20),  # Growth
    # Target Retirement glide path approximations (rough; actual mix shifts each year)
    "VTINX": (0.30, 0.70),  # Income
    "VTTVX": (0.45, 0.55),  # 2025
    "VTHRX": (0.60, 0.40),  # 2030
    "VTTHX": (0.70, 0.30),  # 2035
    "VFORX": (0.78, 0.22),  # 2040
    "VTIVX": (0.85, 0.15),  # 2045
    "VFIFX": (0.88, 0.12),  # 2050
    "VFFVX": (0.90, 0.10),  # 2055
    "VTTSX": (0.90, 0.10),  # 2060
    "VLXVX": (0.90, 0.10),  # 2065
    # Fidelity
    "FPURX": (0.60, 0.40),  # Puritan
    "FBALX": (0.65, 0.35),  # Balanced
    # American Funds
    "ABALX": (0.65, 0.35),  # American Balanced
    "AMECX": (0.55, 0.45),  # Income Fund of America
    "CAIBX": (0.55, 0.45),  # Capital Income Builder
    # T. Rowe Price
    "PRWCX": (0.65, 0.35),  # Capital Appreciation
    "PRPFX": (0.50, 0.50),  # Permanent Portfolio
}


def _is_bond(ticker):
    """True if the ticker is bond-classified.

    Uses the full _classify_ticker lookup (which consults the curated
    mutual fund table and yfinance fallback) rather than just prefix
    matching against a small set of common ETF tickers. This is what
    fixes mutual-fund miscategorization downstream — a bond MF like
    VBTLX or PIMIX now correctly returns True instead of being
    treated as equity.

    Falls back to the original prefix check if the classifier returns
    nothing usable.
    """
    if not ticker:
        return False
    t = ticker.upper()
    try:
        cls, _ = _classify_ticker(t)
        if cls == "bond":
            return True
        if cls in ("equity", "balanced", "cash", "crypto_btc",
                   "crypto_alt", "leveraged"):
            return False
    except Exception:
        pass
    return any(t.startswith(p) or t == p for p in _BOND_PATTERNS)


def _is_balanced(ticker):
    """True if the ticker is a balanced/allocation fund."""
    if not ticker:
        return False
    try:
        cls, _ = _classify_ticker(ticker.upper())
        return cls == "balanced"
    except Exception:
        return False


def _balanced_split(ticker):
    """Return (equity_pct, bond_pct) for a balanced fund as decimals.

    Looks up the curated _BALANCED_FUND_SPLITS table; defaults to 60/40
    for unknown balanced funds. Cash sleeve is folded into bonds for
    the purpose of this split since most balanced funds keep <2% cash.
    """
    if not ticker:
        return (0.60, 0.40)
    return _BALANCED_FUND_SPLITS.get(ticker.upper(), (0.60, 0.40))


def generate_three_tier_from_blend(basis, client_risk_score, priorities=None,
                                   mode="manual", opt_results=None):
    """Turn a blended portfolio (from Composite Optimizer) into 3 tiers.

    `basis` is a dict: {"balanced_tickers": [...], "balanced_weights": [...]}
    `mode` is "manual" (±15% equity/bond tilt) or "smart"
           (pick best-matching optimizer strategies for each tier).
    `opt_results` is the optimizer's strategy dict — required for smart mode.

    Returns the same shape as generate_three_tier_proposals.
    """
    bal_t = list(basis.get("balanced_tickers", []))
    bal_w = [float(w) for w in basis.get("balanced_weights", [])]
    if not bal_t or not bal_w or sum(bal_w) <= 0:
        # Fallback to standard generator if the basis is empty
        return generate_three_tier_proposals(client_risk_score, priorities=priorities)

    # Normalize to percentages (0-100)
    total = sum(bal_w)
    bal_pct = [round(w * 100.0 / total, 2) for w in bal_w]

    # Balanced tier is always the blend as-is
    balanced_tier = {
        "label":         "Option 1 (proposed)",
        "target_score":  int(client_risk_score),
        "tickers":       bal_t,
        "weights":       bal_pct,
        "equity_pct":    round(sum(w for t, w in zip(bal_t, bal_pct) if not _is_bond(t)), 1),
        "bond_pct":      round(sum(w for t, w in zip(bal_t, bal_pct) if _is_bond(t)), 1),
        "cash_pct":      0.0,
        "rationale":     "Balanced allocation — uses the current Composite Optimizer blend as-is.",
        "priority_tilts": [],
        "priority_flags": [],
        "save_to_profile": False,
    }

    if mode == "smart" and opt_results:
        # Pick best strategies from opt_results for each tier
        # Scoring: Conservative → lowest |DD| with Sharpe > 0;
        #         Aggressive → highest ann_return with Sharpe > 0.5 or max Sharpe fallback
        try:
            # Filter out strategies with missing metrics
            valid = {k: r for k, r in opt_results.items()
                     if r and isinstance(r, dict) and "sharpe" in r and "weights" in r}
            # Conservative: lowest drawdown (abs), prefer positive Sharpe
            cons_choice = min(
                ((k, r) for k, r in valid.items() if r.get("sharpe", 0) >= 0),
                key=lambda kv: abs(kv[1].get("max_drawdown", -1)),
                default=None,
            )
            # Aggressive: highest annualized return
            aggr_choice = max(
                valid.items(),
                key=lambda kv: kv[1].get("ann_return", -99),
                default=None,
            )

            def _make_tier_from_strat(choice, label, target):
                if not choice:
                    return None
                name, r = choice
                w_arr = r.get("weights", [])
                if not w_arr:
                    return None
                # Normalize to percentages
                tot = sum(float(x) for x in w_arr)
                if tot <= 0:
                    return None
                ws = [round(float(x) * 100.0 / tot, 2) for x in w_arr]
                return {
                    "label":         label,
                    "target_score":  int(target),
                    "tickers":       list(bal_t[:len(ws)]),
                    "weights":       ws,
                    "equity_pct":    round(sum(w for t,w in zip(bal_t[:len(ws)], ws) if not _is_bond(t)),1),
                    "bond_pct":      round(sum(w for t,w in zip(bal_t[:len(ws)], ws) if _is_bond(t)),1),
                    "cash_pct":      0.0,
                    "rationale":     f"{label} — smart-matched to {name} "
                                     f"(Sharpe {r.get('sharpe',0):.2f}, "
                                     f"DD {r.get('max_drawdown',0):.1%})",
                    "priority_tilts": [f"smart-matched: {name}"],
                    "priority_flags": [],
                    "save_to_profile": False,
                }

            cons_tier = _make_tier_from_strat(
                cons_choice, "Option 2 (slightly more conservative)", max(1, client_risk_score - 15))
            aggr_tier = _make_tier_from_strat(
                aggr_choice, "Option 3 (slightly more aggressive)", min(99, client_risk_score + 15))

            if cons_tier and aggr_tier:
                return {
                    "conservative": cons_tier,
                    "balanced":     balanced_tier,
                    "aggressive":   aggr_tier,
                }
            # If smart-matching failed, fall through to manual tilt below
        except Exception:
            pass

    # Manual tilt — shift weight from equities → bonds (conservative) and vice versa
    TILT = 15.0

    def _tilt(tickers, pct, direction):
        """direction: +1 = conservative (bonds up), -1 = aggressive (equity up).
        Always returns (new_tickers, new_pct).
        """
        eq_mass = sum(w for t, w in zip(tickers, pct) if not _is_bond(t))
        bd_mass = sum(w for t, w in zip(tickers, pct) if _is_bond(t))
        tickers_out = list(tickers)
        pct_out     = list(pct)

        if direction > 0:
            # Conservative: equities down, bonds up
            shift = min(TILT, eq_mass)
            if eq_mass <= 0:
                return tickers_out, pct_out
            eq_scale = (eq_mass - shift) / eq_mass
            bd_scale = ((bd_mass + shift) / bd_mass) if bd_mass > 0 else 1.0
            pct_out = []
            for t, w in zip(tickers_out, pct):
                if _is_bond(t):
                    pct_out.append(w * bd_scale if bd_mass > 0 else w)
                else:
                    pct_out.append(w * eq_scale)
            # If no bonds exist yet, append a generic aggregate bond slice
            if bd_mass <= 0:
                tickers_out = tickers_out + ["AGG"]
                pct_out     = pct_out + [shift]
        else:
            # Aggressive: bonds down, equities up
            shift = min(TILT, bd_mass)
            if bd_mass <= 0:
                return tickers_out, pct_out
            bd_scale = (bd_mass - shift) / bd_mass
            eq_scale = ((eq_mass + shift) / eq_mass) if eq_mass > 0 else 1.0
            pct_out = []
            for t, w in zip(tickers_out, pct):
                if _is_bond(t):
                    pct_out.append(w * bd_scale)
                else:
                    pct_out.append(w * eq_scale)

        return tickers_out, pct_out

    cons_t, cons_pct = _tilt(bal_t, bal_pct, +1)
    aggr_t, aggr_pct = _tilt(bal_t, bal_pct, -1)

    # Normalize each to exactly 100
    def _norm(pct):
        s = sum(pct)
        return [round(p * 100.0 / s, 2) for p in pct] if s > 0 else pct

    cons_pct = _norm(cons_pct)
    aggr_pct = _norm(aggr_pct)

    cons_tier = {
        "label":         "Option 2 (slightly more conservative)",
        "target_score":  max(1, int(client_risk_score) - 15),
        "tickers":       cons_t,
        "weights":       cons_pct,
        "equity_pct":    round(sum(w for t,w in zip(cons_t, cons_pct) if not _is_bond(t)),1),
        "bond_pct":      round(sum(w for t,w in zip(cons_t, cons_pct) if _is_bond(t)),1),
        "cash_pct":      0.0,
        "rationale":     "Conservative — +15% bond tilt from the Composite blend.",
        "priority_tilts": ["+15% bonds / -15% equity from blend"],
        "priority_flags": [],
        "save_to_profile": False,
    }
    aggr_tier = {
        "label":         "Option 3 (slightly more aggressive)",
        "target_score":  min(99, int(client_risk_score) + 15),
        "tickers":       aggr_t,
        "weights":       aggr_pct,
        "equity_pct":    round(sum(w for t,w in zip(aggr_t, aggr_pct) if not _is_bond(t)),1),
        "bond_pct":      round(sum(w for t,w in zip(aggr_t, aggr_pct) if _is_bond(t)),1),
        "cash_pct":      0.0,
        "rationale":     "Aggressive — -15% bond tilt / +15% equity from the Composite blend.",
        "priority_tilts": ["-15% bonds / +15% equity from blend"],
        "priority_flags": [],
        "save_to_profile": False,
    }

    return {
        "conservative": cons_tier,
        "balanced":     balanced_tier,
        "aggressive":   aggr_tier,
    }

st.set_page_config(
    page_title="Foresight Portfolio Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─────────────────────────────────────────────────────────────
       FORESIGHT — CLINICAL THEME (teal on white)
       Mirrors client_portal.py palette so advisor + client UIs
       feel like the same product.
       Tokens (kept here for reference; values inlined below):
         bg           #F4F7F9     surface       #FFFFFF
         surface2     #EEF3F6     line          #E1E8EE
         ink          #0B1F2A     ink2          #3F5260
         muted        #6B7E8A     primary       #0E5C5E
         primary_soft #D8ECEC     accent        #0E7C86
         healthy      #16A34A     caution       #C2700A
         risk         #C2410C
       ───────────────────────────────────────────────────────────── */

    /* ── BASE ────────────────────────────────────────────────── */
    html, body, .stApp {
        background: #F4F7F9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0B1F2A !important;
        -webkit-font-smoothing: antialiased;
    }
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1360px !important;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────── */
    [data-testid="collapsedControl"] { display: none !important; }

    /* Hide Streamlit's auto-generated header-anchor link icons — the
       small chain-link that appears next to every <h1>/<h2>/<h3> in
       markdown blocks. They look like dead links to users on a
       client-facing app. */
    .stApp h1 a, .stApp h2 a, .stApp h3 a,
    .stApp h4 a, .stApp h5 a, .stApp h6 a,
    [data-testid="stHeaderActionElements"],
    [data-testid="stMarkdownHeadingActionElements"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E1E8EE !important;
        padding: 1.5rem 1rem !important;
    }
    section[data-testid="stSidebar"] * { color: #3F5260 !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #0B1F2A !important; font-size: 0.85rem !important;
        text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;
    }

    /* ── TYPOGRAPHY ──────────────────────────────────────────── */
    h1, h2, h3 {
        color: #0B1F2A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }
    h4 { color: #3F5260 !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
    p, .stMarkdown p { color: #3F5260 !important; font-size: 0.9rem !important; line-height: 1.6 !important; }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #6B7E8A !important; font-size: 0.78rem !important;
    }
    label { color: #0B1F2A !important; font-weight: 500 !important; font-size: 0.85rem !important; }

    /* ── METRICS ─────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 14px !important;
        padding: 20px 22px !important;
        box-shadow: 0 1px 2px rgba(11,31,42,0.04) !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 16px rgba(11,31,42,0.07) !important;
        border-color: #0E5C5E !important;
    }
    div[data-testid="stMetric"] label {
        color: #6B7E8A !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0B1F2A !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────────── */
    /* Default (secondary) buttons: light card with teal hover.
       Mirrors client_portal — the loud blue-on-everything look is gone. */
    .stButton > button,
    div[data-testid="stBaseButton-secondary"] > button,
    button[kind="secondary"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #0B1F2A !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 10px 22px !important;
        letter-spacing: -0.01em !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(11,31,42,0.04) !important;
        opacity: 1 !important;
    }
    .stButton > button:hover,
    div[data-testid="stBaseButton-secondary"] > button:hover,
    button[kind="secondary"]:hover {
        background: #EEF3F6 !important;
        background-color: #EEF3F6 !important;
        border-color: #0E5C5E !important;
        color: #0E5C5E !important;
    }
    .stButton > button:focus,
    .stButton > button:active {
        box-shadow: 0 0 0 3px #D8ECEC !important;
        outline: none !important;
    }

    /* Primary buttons: solid teal */
    div[data-testid="stBaseButton-primary"] > button,
    button[kind="primary"],
    .stButton > button[kind="primary"] {
        background: #0E5C5E !important;
        background-color: #0E5C5E !important;
        color: #FFFFFF !important;
        border: 1px solid #0E5C5E !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(14,92,94,0.25) !important;
    }
    div[data-testid="stBaseButton-primary"] > button:hover,
    button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: #0E7C86 !important;
        background-color: #0E7C86 !important;
        border-color: #0E7C86 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(14,92,94,0.30) !important;
    }
    div[data-testid="stBaseButton-primary"] > button p,
    div[data-testid="stBaseButton-primary"] > button span,
    button[kind="primary"] p, button[kind="primary"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* ── TABS ────────────────────────────────────────────────── */
    /* Pill-style tabs were too heavy; client portal uses underline tabs.
       Underline-on-teal matches and lets the content breathe. */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-radius: 0 !important;
        border-bottom: 1px solid #E1E8EE !important;
        padding: 0 4px !important;
        gap: 28px !important;
        margin-bottom: 18px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 0 !important;
        color: #6B7E8A !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 12px 4px !important;
        min-height: auto !important;
        letter-spacing: 0.01em !important;
        transition: color 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #3F5260 !important; }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #0B1F2A !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0E5C5E !important;
        background: #0E5C5E !important;
        height: 2.5px !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #E1E8EE !important;
        background: #E1E8EE !important;
    }

    /* ── INPUTS ──────────────────────────────────────────────── */
    input, .stTextInput input,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #0B1F2A !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    input:focus, .stTextInput input:focus,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #0E5C5E !important;
        box-shadow: 0 0 0 3px #D8ECEC !important;
        outline: none !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 10px !important;
        color: #0B1F2A !important;
        font-size: 0.875rem !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: #D8ECEC !important;
        border-color: rgba(14,92,94,0.4) !important;
        color: #0E5C5E !important;
    }
    .stNumberInput > div > div > input {
        font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 0.9rem !important;
    }
    textarea, .stTextArea textarea {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 10px !important;
        color: #0B1F2A !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Tabular numerals inside radios + selects so $ amounts line up */
    .stRadio label, .stRadio [data-baseweb="radio"] div,
    .stSelectbox div, .stMultiSelect div {
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1, "lnum" 1;
    }

    /* ── DATAFRAME ───────────────────────────────────────────── */
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid #E1E8EE !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(11,31,42,0.04) !important;
    }
    .stDataFrame table {
        font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 0.8rem !important;
    }
    .stDataFrame thead th {
        background: #EEF3F6 !important;
        color: #6B7E8A !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        border-bottom: 1px solid #E1E8EE !important;
        padding: 10px 14px !important;
    }
    .stDataFrame tbody td {
        color: #0B1F2A !important;
        border-bottom: 1px solid #EEF3F6 !important;
        padding: 9px 14px !important;
    }
    .stDataFrame tbody tr:hover { background: #F4F7F9 !important; }
    .stDataFrame tbody tr:last-child td { border-bottom: none !important; }

    /* ── EXPANDER ────────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(11,31,42,0.04) !important;
    }
    div[data-testid="stExpander"] summary {
        color: #3F5260 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #0B1F2A !important;
    }

    /* ── ALERTS ──────────────────────────────────────────────── */
    /* Client-portal style: white surface, left rail in the
       semantic color, rounded 12. */
    .stAlert {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-radius: 12px !important;
        color: #0B1F2A !important;
    }
    .stSuccess > div, [data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-left: 3px solid #16A34A !important;
        border-radius: 12px !important;
        color: #0B1F2A !important;
    }
    .stWarning > div {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-left: 3px solid #C2700A !important;
        border-radius: 12px !important;
        color: #0B1F2A !important;
    }
    .stInfo > div {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-left: 3px solid #0E5C5E !important;
        border-radius: 12px !important;
        color: #0B1F2A !important;
    }
    .stError > div {
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-left: 3px solid #C2410C !important;
        border-radius: 12px !important;
        color: #0B1F2A !important;
    }

    /* ── PROGRESS / PLOTS ────────────────────────────────────── */
    .stProgress { display: none !important; }
    .js-plotly-plot, .plot-container { background: transparent !important; }

    /* ── SPINNER OVERLAY (advisor-side analyzer) ─────────────── */
    #portfolio-spinner {
        display: none;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(244,247,249,0.88);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        gap: 20px;
    }
    #portfolio-spinner.active { display: flex; }
    .spinner-wrap { position: relative; width: 72px; height: 72px; }
    .spinner-ring {
        width: 72px; height: 72px;
        border-radius: 50%;
        border: 2.5px solid #E1E8EE;
        border-top-color: #0E5C5E;
        animation: spin 0.75s cubic-bezier(0.4,0,0.2,1) infinite;
        position: absolute; top: 0; left: 0;
    }
    .spinner-ring-inner {
        width: 48px; height: 48px;
        border-radius: 50%;
        border: 2px solid #EEF3F6;
        border-top-color: #0E7C86;
        animation: spin 0.55s linear infinite reverse;
        position: absolute;
        top: 12px; left: 12px;
    }
    .spinner-dot {
        width: 8px; height: 8px;
        background: #0E5C5E;
        border-radius: 50%;
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner-text {
        color: #0B1F2A;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .spinner-subtext {
        color: #6B7E8A;
        font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        animation: pulse 1.8s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }

    /* ── HEADER ──────────────────────────────────────────────── */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E1E8EE;
        border-radius: 18px;
        padding: 32px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(11,31,42,0.05);
    }
    .app-header::before {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 100%;
        background: linear-gradient(135deg, transparent 40%, #EEF3F6 100%);
        pointer-events: none;
    }
    .app-header h1 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #0E5C5E !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.025em !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .app-header p {
        color: #6B7E8A !important;
        font-size: 0.875rem !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #D8ECEC;
        border: 1px solid rgba(14,92,94,0.18);
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.68rem;
        color: #0E5C5E;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* ── SECTION LABELS ──────────────────────────────────────── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 32px 0 14px 0;
    }
    .section-num {
        width: 24px; height: 24px;
        background: #0E5C5E;
        border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.68rem; font-weight: 700; color: #FFFFFF;
        font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
    }
    .section-title {
        font-size: 0.8rem; font-weight: 600;
        color: #3F5260; letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ── CALLOUT ─────────────────────────────────────────────── */
    .callout-box {
        background: #FFFFFF;
        border: 1px solid #E1E8EE;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(11,31,42,0.04);
    }

    /* ── DIVIDER ─────────────────────────────────────────────── */
    hr { border: none !important; border-top: 1px solid #E1E8EE !important; margin: 28px 0 !important; }

    /* ── RADIO / CHECKBOX ────────────────────────────────────── */
    .stRadio label, .stCheckbox label {
        color: #3F5260 !important;
        font-size: 0.875rem !important;
    }

    /* ── CLIENT-PORTAL UTILITY CLASSES ───────────────────────── */
    /* Ported from client_portal.py so inline markup that uses these
       classes (cards, vitals, eyebrows, chips, dark CTAs) renders
       identically across both surfaces. */
    .fr-card {
        background: #FFFFFF;
        border: 1px solid #E1E8EE;
        border-radius: 18px;
        padding: 22px 22px;
        margin-bottom: 16px;
    }
    .fr-eyebrow {
        font-size: 0.69rem; font-weight: 600;
        color: #6B7E8A;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .fr-vital {
        background: #FFFFFF;
        border: 1px solid #E1E8EE;
        border-radius: 14px;
        padding: 14px 14px;
        min-height: 102px;
        display: flex; flex-direction: column; gap: 8px;
        margin-bottom: 10px;
    }
    .fr-vital-label {
        font-size: 0.65rem; font-weight: 600;
        color: #6B7E8A;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .fr-vital-value {
        font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace;
        font-size: 1.4rem; font-weight: 600; color: #0B1F2A;
        letter-spacing: -0.01em; line-height: 1;
    }
    .fr-vital-detail {
        display: flex; align-items: center; justify-content: space-between;
        font-size: 0.72rem; color: #6B7E8A;
    }
    .fr-mono {
        font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace;
        font-weight: 600;
    }
    .fr-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 10px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .fr-chip::before {
        content: ""; width: 6px; height: 6px; border-radius: 999px;
        background: currentColor;
    }
    .fr-greeting {
        font-size: 0.85rem; color: #3F5260; margin-bottom: 4px;
    }
    .fr-headline {
        font-size: 1.5rem; font-weight: 600; color: #0B1F2A;
        letter-spacing: -0.015em; line-height: 1.18; margin: 0 0 14px 0;
    }
    .fr-headline-accent { color: #0E5C5E; }
    .fr-cta-dark {
        background: #0B1F2A;
        color: #FFFFFF;
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: 14px;
        display: flex; align-items: center; gap: 14px;
    }
    .fr-cta-icon {
        width: 40px; height: 40px; border-radius: 12px;
        background: rgba(255,255,255,0.12);
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 1.2rem;
    }

    /* ── SCROLLBAR ───────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #C8D4DD; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6B7E8A; }
</style>

<!-- Spinner overlay -->
<div id="portfolio-spinner">
    <div class="spinner-wrap">
        <div class="spinner-ring"></div>
        <div class="spinner-ring-inner"></div>
        <div class="spinner-dot"></div>
    </div>
    <div class="spinner-text">Running Analysis</div>
    <div class="spinner-subtext">Optimizing portfolios</div>
</div>
""", unsafe_allow_html=True)

# ── FILE HELPERS ──────────────────────────────────────────────
# These wrap shared.load_json / save_json so all JSON IO in this file goes
# through the atomic, locked, race-condition-free implementation. The original
# versions used non-atomic open()+json.dump() which lost data when two
# Streamlit reruns wrote simultaneously.
def load_json(path):
    return _shared_load_json(path, default={})

def save_json(path, data):
    _shared_save_json(path, data)

def load_saved():      return load_json(SAVE_FILE)
def save_all(d):       save_json(SAVE_FILE, d)
def load_holdings():   return load_json(HOLDINGS_FILE)
def save_holdings(d):  save_json(HOLDINGS_FILE, d)
def load_watchlist():  return load_json(WATCHLIST_FILE)
def save_watchlist(d): save_json(WATCHLIST_FILE, d)

# ── LIVE QUOTES ───────────────────────────────────────────────
def get_live_quotes(tickers):
    results = {}
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                prev  = float(hist["Close"].iloc[-2])
                price = float(hist["Close"].iloc[-1])
            elif len(hist) == 1:
                price = prev = float(hist["Close"].iloc[-1])
            else:
                price = prev = 0
            chg     = price - prev
            chg_pct = (chg / prev * 100) if prev else 0
            try:
                info = t.info
            except Exception:
                info = {}
            results[ticker] = {
                "name":        info.get("shortName") or info.get("longName") or ticker,
                "price":       price, "prev_close": prev,
                "change":      chg, "change_pct": chg_pct,
                "volume":      info.get("volume") or info.get("regularMarketVolume") or 0,
                "avg_volume":  info.get("averageVolume") or 0,
                "mkt_cap":     info.get("marketCap") or 0,
                "pe":          info.get("trailingPE") or info.get("forwardPE") or None,
                "week52_high": info.get("fiftyTwoWeekHigh") or None,
                "week52_low":  info.get("fiftyTwoWeekLow") or None,
                "sector":      info.get("sector") or info.get("industry") or "—",
            }
        except Exception:
            results[ticker] = {
                "name": ticker, "price": 0, "prev_close": 0,
                "change": 0, "change_pct": 0, "volume": 0,
                "avg_volume": 0, "mkt_cap": 0, "pe": None,
                "week52_high": None, "week52_low": None, "sector": "—"
            }
    return results

def get_sparkline(ticker, period="1mo"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"] if not hist.empty else None
    except Exception:
        return None

# ── FULL BACKTEST ENGINE ──────────────────────────────────────
def run_backtest(tickers, years, custom_weights=None, custom_weights_valid=False,
                 cov_estimator="Ledoit-Wolf", mu_estimator="Shrunk",
                 use_hrp=True, use_nco=True, use_maxdiv=True,
                 use_walkforward=False):
    end_dt   = date.today()
    start_dt = end_dt - relativedelta(years=years)

    # ── Use proxy-aware fetch so short-history tickers get full history ──────
    # e.g. FBTC→GBTC (2015), GLDM→GLD (2004), SGOV→BIL (2007)
    # min_days gates whether a ticker's own history is accepted (else proxy is used).
    # Previously years*200 was too strict — a holding with ~2yr of history would fall
    # through to risk-free proxy on a 3y run, polluting the backtest. Using a floor of
    # ~50 trading days per year of window lets real tickers in, and gap-alignment
    # (dropna below) handles the ragged start gracefully.
    _min_days = max(60, min(years * 60, 400))
    try:
        prices, _proxy_notes = get_prices_with_proxies(
            tuple(tickers), start_dt, end_dt,
            min_days=_min_days,
        )
        if prices.empty:
            raise ValueError("empty")
    except Exception:
        # Hard fallback: plain fetch without proxies
        try:
            prices, _ = get_prices(tickers, start_dt, end_dt)
            if isinstance(prices, pd.Series): prices = prices.to_frame()
        except Exception:
            return None

    # ── Drop rows that are entirely NaN; do NOT ffill ────────────────────────
    # Previously this did `prices.dropna(how="all").ffill()`. The `ffill`
    # was a methodology bug: it carried prior prices forward across NaN gaps,
    # producing 0% returns on gap days and inflating apparent Sharpe (lower
    # measured volatility, same returns). For a backtest, the right fix is to
    # keep gaps as gaps and let `dropna(axis=0, how='any')` on the return
    # matrix below align the panel to days where every ticker traded.
    prices = prices.dropna(how="all")
    if prices.shape[0] < 60:
        return None

    X = prices_to_returns(prices)
    if X is None or len(X) < 60:
        return None
    # Drop columns that are 100% NaN, then drop rows where any ticker is NaN
    # (keeps the panel rectangular without smearing prices across gaps).
    X = X.dropna(axis=1, how='all').dropna(axis=0, how='any')
    if len(X) < 60 or X.shape[1] == 0:
        return None
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)
    if len(X_train) < 10 or len(X_test) < 10:
        return None

    # ── Covariance estimator ──────────────────────────────────
    cov_map = {
        "Empirical":    None,
        "Ledoit-Wolf":  LedoitWolf(),
        "Gerber":       GerberCovariance(),
        "Denoised":     DenoiseCovariance(),
    }
    cov_est = cov_map.get(cov_estimator)

    # ── Mean estimator ────────────────────────────────────────
    mu_map = {
        "Empirical": None,
        "Shrunk":    ShrunkMu(),
        "EWM":       EWMu(alpha=0.2),
    }
    mu_est = mu_map.get(mu_estimator)

    # Build prior
    if cov_est is not None and mu_est is not None:
        prior = EmpiricalPrior(mu_estimator=mu_est, covariance_estimator=cov_est)
    elif cov_est is not None:
        prior = EmpiricalPrior(covariance_estimator=cov_est)
    elif mu_est is not None:
        prior = EmpiricalPrior(mu_estimator=mu_est)
    else:
        prior = EmpiricalPrior()

    # ── Models ────────────────────────────────────────────────
    models = {
        "Equal Weight":    EqualWeighted(),
        "Min Variance":    MeanRisk(risk_measure=RiskMeasure.VARIANCE, prior_estimator=prior),
        "Max Sharpe":      MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                                    risk_measure=RiskMeasure.VARIANCE, prior_estimator=prior),
        "Max Sortino":     MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                                    risk_measure=RiskMeasure.SEMI_VARIANCE, prior_estimator=prior),
        "Risk Parity":     RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, prior_estimator=prior),
        "Min CVaR":        MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior),
        "Min CDaR":        MeanRisk(risk_measure=RiskMeasure.CDAR, prior_estimator=prior),
        "Max CVaR Ratio":  MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                                    risk_measure=RiskMeasure.CVAR, prior_estimator=prior),
        "Risk Parity CVaR":RiskBudgeting(risk_measure=RiskMeasure.CVAR, prior_estimator=prior),
    }

    if use_hrp:
        models["HRP"]         = HierarchicalRiskParity(prior_estimator=prior)
        models["HRP CVaR"]    = HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR,
                                                        prior_estimator=prior)
    if use_maxdiv:
        try:
            models["Max Diversification"] = MaximumDiversification(prior_estimator=prior)
        except Exception:
            pass
    if use_nco and len(tickers) >= 4:
        try:
            models["NCO"] = NestedClustersOptimization(
                inner_estimator=MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior),
                outer_estimator=RiskBudgeting(risk_measure=RiskMeasure.VARIANCE),
            )
        except Exception:
            pass

    # ── EFFICIENT FRONTIER BLEND ──────────────────────────────
    # A meta-strategy that averages weights across all strategies
    # weighted by their Sharpe ratios — the "blend of all strategies"
    models["EF Blend"] = None  # placeholder, computed post-fit

    # ── Helper: compute consistent stats over a return series ─
    # Used for ALL strategies + custom + blend so every "ann_return", "sharpe",
    # etc. uses the SAME definition (CAGR + excess-return Sharpe).
    def _series_stats(rets_series: pd.Series) -> dict:
        cum = (1 + rets_series).cumprod()
        dd  = cum / cum.cummax() - 1
        neg = rets_series[rets_series < 0]
        actual_yrs = max(len(rets_series) / 252.0, 0.08)
        tr_val = float(cum.iloc[-1] - 1) if len(cum) else 0.0
        ann_r  = (1 + tr_val) ** (1.0 / actual_yrs) - 1
        ann_v  = float(rets_series.std() * np.sqrt(252)) if len(rets_series) > 1 else 0.0
        # Sharpe = excess return / vol, with consistent risk-free rate
        sharpe  = _shared_sharpe(ann_r, ann_v)
        downside_vol = float(neg.std() * np.sqrt(252)) if len(neg) > 1 else 0.0
        sortino = ((ann_r - DEFAULT_RISK_FREE_RATE) / downside_vol) if downside_vol > 0 else 0.0
        max_dd = float(dd.min()) if len(dd) else 0.0
        calmar = abs(ann_r / max_dd) if max_dd != 0 else 0
        return {
            "ann_return":   ann_r,
            "ann_vol":      ann_v,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "max_drawdown": max_dd,
            "total_return": tr_val,
            "calmar":       calmar,
        }

    results = {}
    for name, model in models.items():
        try:
            if use_walkforward and len(X) > 200:
                cv = WalkForward(train_size=min(252, len(X_train)), test_size=21)
                mmp = cross_val_predict(model, X, cv=cv)
                rets = pd.Series(mmp.returns)
                # Walkforward returns align to the END of X
                rets_index = X.index[-len(rets):]
                # Fit one more time on X_train just to extract weights for display
                try:
                    model.fit(X_train)
                    weights = model.predict(X_test).weights.tolist()
                except Exception:
                    weights = [1.0/X.shape[1]] * X.shape[1]
            else:
                model.fit(X_train)
                ptf  = model.predict(X_test)
                weights = ptf.weights.tolist()
                # ── OUT-OF-SAMPLE returns: apply train-fitted weights to X_test ONLY.
                # Previously this code applied weights back to FULL X (X_train+X_test),
                # which produced in-sample lookahead bias AND a length mismatch with
                # the X_test.index that gets stored as "index" — guaranteed crash on
                # any chart that does pd.Series(returns, index=index).
                w_arr = np.array(weights)
                if len(w_arr) == X_test.shape[1]:
                    rets = pd.Series(X_test.values @ w_arr, index=X_test.index)
                else:
                    rets = pd.Series(ptf.returns, index=X_test.index[:len(ptf.returns)])
                rets_index = rets.index

            stats = _series_stats(rets)

            results[name] = {
                "weights":      weights,
                **stats,
                "returns":      rets.tolist(),
                # ── Length invariant: returns and index always match ─────────
                "index":        pd.Index(rets_index).astype(str).tolist(),
            }
            # Sanity check the invariant — if this ever fires we have a regression
            assert len(results[name]["returns"]) == len(results[name]["index"]), \
                f"{name}: returns/index length mismatch ({len(results[name]['returns'])} vs {len(results[name]['index'])})"
        except Exception:
            pass

    # ── EFFICIENT FRONTIER BLEND ─────────────────────────────
    # Compute blend: weighted average of all strategy weights by Sharpe.
    # Uses the SAME stats helper (CAGR + excess Sharpe) as every other strategy
    # — previously this computed `mean*252` which was inconsistent with the
    # per-strategy CAGR and made EF Blend's Sharpe non-comparable.
    if results:
        try:
            # Only use strategies with positive Sharpe — losing strategies shouldn't
            # contribute to the optimal blend.
            sharpes = {n: max(0.01, r["sharpe"]) for n, r in results.items()
                       if n != "EF Blend" and r.get("sharpe") is not None}
            total_sh = sum(sharpes.values())
            if total_sh > 0:
                blend_weights = np.zeros(X_test.shape[1])
                for n, r in results.items():
                    if n != "EF Blend" and sharpes.get(n, 0) > 0:
                        w = np.array(r["weights"])
                        if len(w) == X_test.shape[1]:
                            blend_weights += w * (sharpes[n] / total_sh)
                ws = blend_weights.sum()
                if ws > 0:
                    blend_weights = blend_weights / ws
                    bs = pd.Series(X_test.values @ blend_weights, index=X_test.index)
                    bstats = _series_stats(bs)
                    results["EF Blend"] = {
                        "weights": blend_weights.tolist(),
                        **bstats,
                        "returns": bs.tolist(),
                        "index":   X_test.index.astype(str).tolist(),
                    }
        except Exception:
            pass
    # Remove None placeholder if blend wasn't actually computed
    if results.get("EF Blend") is None:
        results.pop("EF Blend", None)

    # ── Custom portfolio — uses FULL date range for context ─────────────────
    # This is fine: the user-supplied weights weren't fitted to data, so there's
    # no in-sample bias. Using the full X provides a longer history for context.
    if custom_weights and custom_weights_valid:
        # Use all tickers present in full X (not just X_test)
        valid_tickers = [t for t in tickers if t in X.columns] if hasattr(X, 'columns') else tickers
        if not valid_tickers:
            valid_tickers = list(X.columns) if hasattr(X, 'columns') else tickers
        w_array = np.array([custom_weights.get(t, 0) / 100.0 for t in valid_tickers])
        if w_array.sum() > 0:
            w_array = w_array / w_array.sum()
        else:
            w_array = np.ones(len(valid_tickers)) / len(valid_tickers)

        X_full = X[valid_tickers] if hasattr(X, 'columns') else X
        cr_full = pd.Series(X_full.values @ w_array, index=X_full.index)
        cstats = _series_stats(cr_full)

        _src = st.session_state.get("portfolio_source", "Custom — Enter Your Own Tickers")
        if _src.startswith("📁 "):
            _port_label = f"⭐ {_src[2:]}"
        else:
            _port_label = st.session_state.get("run_port_label", "⭐ My Portfolio")

        results[_port_label] = {
            "weights":      w_array.tolist(),
            **cstats,
            "returns":      cr_full.tolist(),
            "index":        X_full.index.astype(str).tolist(),
        }

    # ── Efficient frontier points ─────────────────────────────
    ef_points = []
    try:
        for target_r in np.linspace(
            min(r["ann_return"] for r in results.values()),
            max(r["ann_return"] for r in results.values()), 30):
            try:
                m = MeanRisk(
                    risk_measure=RiskMeasure.VARIANCE,
                    min_return=target_r / 252,
                    prior_estimator=prior
                )
                m.fit(X_train)
                p = m.predict(X_test)
                ef_points.append({
                    "vol":    p.annualized_standard_deviation,
                    "ret":    p.annualized_mean,
                    "sharpe": p.annualized_sharpe_ratio,
                })
            except Exception:
                pass
    except Exception:
        pass

    return results, ef_points

# ── ONE-CLICK PDF AUTO-DOWNLOAD HELPER ──────────────────────────────────
def trigger_pdf_download(pdf_bytes, filename):
    """Trigger an immediate browser download of PDF bytes — no second click.

    Streamlit's st.download_button requires the user to click twice (Generate,
    then Download). This helper renders an invisible anchor with a data URL
    and auto-clicks it via JS, producing a true one-click download-and-save.

    Usage:
        if st.button("Save PDF"):
            pdf_bytes = build_my_pdf(...)
            trigger_pdf_download(pdf_bytes, "report.pdf")
    """
    import base64
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    # Sanitize filename — no slashes, quotes, control chars
    safe_name = "".join(c for c in filename if c.isalnum() or c in "-_.() ").strip() or "report.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    html = f"""
    <a id="auto_dl_{hash(b64) & 0xffffff:x}" download="{safe_name}"
       href="data:application/pdf;base64,{b64}" style="display:none">dl</a>
    <script>
      (function() {{
        const a = document.getElementById("auto_dl_{hash(b64) & 0xffffff:x}");
        if (a) {{ a.click(); a.remove(); }}
      }})();
    </script>
    """
    import streamlit.components.v1 as _components
    _components.html(html, height=0, width=0)


# ════════════════════════════════════════════════════════════════════════
# PROPOSAL RETURN SERIES + CHART RENDERING (Increment 1)
# ════════════════════════════════════════════════════════════════════════
# Three chart helpers that take a daily-return Series and return PNG bytes
# (BytesIO, ready to drop into ReportLab via Image(...)). All render in the
# app's teal palette and use IBM Plex Mono for numbers.
#
# Soft-imports — if quantstats or matplotlib aren't installed, the helpers
# return None and the PDF builder just skips the chart and shows a small
# caption. The rest of the app keeps working.
# ════════════════════════════════════════════════════════════════════════

# Brand palette (kept in sync with the Streamlit theme defined near line 2340)
_PROP_TEAL       = "#0E5C5E"
_PROP_TEAL_SOFT  = "#D8ECEC"
_PROP_ACCENT     = "#0E7C86"
_PROP_INK        = "#0B1F2A"
_PROP_INK2       = "#3F5260"
_PROP_MUTED      = "#6B7E8A"
_PROP_LINE       = "#E1E8EE"
_PROP_BG         = "#FFFFFF"
_PROP_BG_SOFT    = "#F4F7F9"
_PROP_RISK       = "#C2410C"
_PROP_HEALTHY    = "#16A34A"


def _apply_proposal_chart_style(fig, ax_or_axes):
    """Apply the teal/Plex theme to a matplotlib figure produced by QuantStats
    (or any matplotlib chart). Mutates in place."""
    import matplotlib as _mpl
    import logging as _logging
    # Inter may not be installed on every system; suppress the noisy
    # "Font family 'Inter' not found" warnings — matplotlib will fall back
    # to DejaVu Sans transparently and the chart still renders fine.
    _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
    _mpl.rcParams["font.family"] = ["Inter", "DejaVu Sans", "sans-serif"]

    fig.patch.set_facecolor(_PROP_BG)

    axes = ax_or_axes if isinstance(ax_or_axes, (list, tuple)) else [ax_or_axes]
    try:
        # Some QS plots return ndarray of axes
        import numpy as _np_local
        if hasattr(ax_or_axes, "ravel"):
            axes = list(ax_or_axes.ravel())
    except Exception:
        pass

    for _ax in axes:
        if _ax is None:
            continue
        _ax.set_facecolor(_PROP_BG)
        for _sp_name, _sp in _ax.spines.items():
            if _sp_name in ("top", "right"):
                _sp.set_visible(False)
            else:
                _sp.set_color(_PROP_LINE)
                _sp.set_linewidth(0.6)
        _ax.tick_params(colors=_PROP_INK2, labelsize=8.5, length=3, width=0.6)
        _ax.grid(True, color=_PROP_LINE, linewidth=0.5, alpha=0.7)
        _ax.set_axisbelow(True)
        if _ax.get_title():
            # Re-set at left position with our typography. (Chart-build funcs
            # set it at default/center; we move it to left for that "data card"
            # look matching the rest of the app.)
            _existing_title = _ax.get_title()
            _ax.set_title("")
            _ax.set_title(
                _existing_title,
                color=_PROP_INK, fontsize=10.5, fontweight="600", pad=10,
                loc="left",
            )
        if _ax.get_xlabel():
            _ax.set_xlabel(_ax.get_xlabel(), color=_PROP_INK2, fontsize=8.5)
        if _ax.get_ylabel():
            _ax.set_ylabel(_ax.get_ylabel(), color=_PROP_INK2, fontsize=8.5)
        # Recolor any plotted lines/areas to teal palette — but only if the
        # caller hasn't explicitly tagged a line/collection with `gid="keep"`,
        # which is how chart functions opt out of the recoloring (used for
        # multi-color fills like positive-vs-negative regions).
        for _line in _ax.get_lines():
            if _line.get_gid() == "keep":
                continue
            _line.set_color(_PROP_TEAL)
            _line.set_linewidth(1.6)
        for _coll in _ax.collections:
            if hasattr(_coll, "get_gid") and _coll.get_gid() == "keep":
                continue
            try:
                _coll.set_facecolor(_PROP_TEAL_SOFT)
                _coll.set_edgecolor(_PROP_TEAL)
            except Exception:
                pass


def _save_fig_to_png_bytes(fig, dpi=150):
    """Render a matplotlib figure to PNG bytes (BytesIO) and close the figure
    so we don't leak handles across many proposals."""
    from io import BytesIO as _BytesIO
    import matplotlib.pyplot as _plt
    buf = _BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    _plt.close(fig)
    return buf


# (render_drawdown_chart_png, render_rolling_sharpe_png, and render_forward_mc_png were removed May 2026 — they were used only by the prior Risk Analysis page that has been replaced by the Notable Market Periods table. Live on-screen charts use run_monte_carlo directly and were unaffected.)


def compute_tier_returns(tickers, weights, years=5):
    """Backtest a tier (tickers + weights as percentages) and return a daily
    return Series with a DatetimeIndex. Used at proposal-save time so the
    saved proposal carries the data the PDF charts need.

    Returns None on failure (no prices available, mismatched lengths, etc.) —
    caller should treat None as "no charts available for this tier."
    """
    try:
        import pandas as _pd
        import numpy as _np
        from datetime import date as _date
        from dateutil.relativedelta import relativedelta as _rd

        if not tickers or not weights or len(tickers) != len(weights):
            return None

        # Normalize weights to fractions (input is in % per the rest of the app)
        w = _np.array(weights, dtype=float)
        if w.sum() <= 0:
            return None
        w = w / w.sum()

        end_dt   = _date.today()
        start_dt = end_dt - _rd(years=years)

        # Use the proxy-aware fetch — handles short-history tickers like FBTC, GLDM, SGOV
        try:
            prices, _ = get_prices_with_proxies(
                tuple(tickers), start_dt, end_dt,
                min_days=max(60, min(years * 60, 400)),
            )
        except Exception:
            prices, _ = get_prices(tickers, start_dt, end_dt)

        if prices is None or (hasattr(prices, "empty") and prices.empty):
            return None
        if isinstance(prices, _pd.Series):
            prices = prices.to_frame()

        # Align columns to the requested ticker order; drop tickers we couldn't fetch
        cols = [t for t in tickers if t in prices.columns]
        if not cols:
            return None
        prices = prices[cols].dropna(how="all").ffill().dropna()
        if prices.empty or len(prices) < 30:
            return None

        # Re-normalize weights against the tickers we actually have
        idx = [tickers.index(c) for c in cols]
        w_eff = w[idx]
        if w_eff.sum() <= 0:
            return None
        w_eff = w_eff / w_eff.sum()

        daily_ret = prices.pct_change().dropna()
        portfolio = (daily_ret * w_eff).sum(axis=1)
        portfolio.name = "portfolio"
        return portfolio
    except Exception:
        return None


def attach_returns_to_proposal(proposal_dict, years=5):
    """Mutate `proposal_dict` in place: backtest each tier and attach a
    `returns` list + `index` list (ISO date strings) + summary stats. Run
    once at save time. Best-effort — any tier that can't be backtested simply
    skips this enrichment; the PDF builder will gracefully degrade for it.
    """
    import numpy as _np
    tiers = proposal_dict.get("tiers") or {}
    for tier_key, tier in tiers.items():
        if not isinstance(tier, dict):
            continue
        # Already has returns? leave alone (idempotent).
        if tier.get("returns") and tier.get("index"):
            continue
        rets = compute_tier_returns(
            tier.get("tickers") or [],
            tier.get("weights") or [],
            years=years,
        )
        if rets is None or len(rets) < 30:
            continue
        try:
            tier["returns"] = [float(x) for x in rets.values]
            tier["index"]   = [d.strftime("%Y-%m-%d") for d in rets.index]
            equity = (1.0 + rets).cumprod()
            peak   = equity.cummax()
            dd     = (equity / peak - 1.0)
            tier["stats"] = {
                "total_return": float(equity.iloc[-1] - 1.0),
                "ann_return":   float(((1 + rets.mean()) ** 252) - 1.0),
                "ann_vol":      float(rets.std() * (252 ** 0.5)),
                "sharpe":       float((rets.mean() / rets.std()) * (252 ** 0.5))
                                if rets.std() > 0 else 0.0,
                "max_dd":       float(dd.min()),
                "n_days":       int(len(rets)),
                "years":        int(years),
            }
        except Exception:
            # Don't let a stats glitch block save — partial data is fine.
            pass
    return proposal_dict


# ── CLIENT PROPOSAL PDF BUILDER ──────────────────────────────────────────
# Notable historical event windows for the "Notable Market Periods" PDF
# section. Each entry is (label, start_date, end_date, description).
# Windows are FULL EVENT periods (crash + recovery, not just peak-to-trough)
# so the table tells a story: how did the portfolio do during the storm
# AND how quickly did it recover. End dates are chosen to capture the
# return to (or near) the prior peak; if recovery wasn't complete by the
# nominal end date, the period still shows real-world experience.
NOTABLE_PERIODS = [
    ("2008 Financial Crisis", "2008-01-01", "2009-12-31",
     "Lehman collapse, credit crunch, and recovery year"),
    ("2015–16 China & Oil Selloff", "2015-07-01", "2016-06-30",
     "Yuan devaluation, oil crash to $26/bbl, China growth fears"),
    ("2018 Q4 Selloff", "2018-10-01", "2019-03-31",
     "Fed tightening fears, trade-war escalation, and Q1 2019 rebound"),
    ("2020 COVID Crash", "2020-02-01", "2020-08-31",
     "Pandemic lockdowns, fastest-ever 30%+ drawdown, V-shaped recovery"),
    ("2022 Bear Market", "2022-01-01", "2022-12-31",
     "Inflation surge, Fed rate hikes, simultaneous stock + bond decline"),
]


def _compute_period_returns(portfolios, periods=None):
    """Compute total return for each (name, tickers, weights) across each
    notable historical window.

    Args:
        portfolios: list of (name, tickers, weights) tuples. Weights are
            in percent (0-100); will be normalized.
        periods:   optional override list of (label, start, end, desc)
            tuples. Defaults to NOTABLE_PERIODS.

    Returns:
        dict of {portfolio_name: {period_label: float_or_None}} where
        float is the cumulative total return (decimal, e.g. -0.18 = -18%)
        over the window, or None if data is unavailable for that
        portfolio in that window.

        Includes a "_periods" key with the list of period labels used
        (in order) so the caller can build a table without re-deriving.
    """
    import pandas as _pd
    from datetime import datetime as _dt

    periods = periods or NOTABLE_PERIODS
    out = {"_periods": [p[0] for p in periods]}

    for name, tickers, weights in portfolios:
        out[name] = {}
        if not tickers or not weights:
            for label, *_ in periods:
                out[name][label] = None
            continue

        # Normalize weights to fractions
        total_w = sum(weights)
        if total_w <= 0:
            for label, *_ in periods:
                out[name][label] = None
            continue
        w_norm = [w / total_w for w in weights]

        for label, start, end, _desc in periods:
            try:
                prices, _ = get_prices_with_proxies(
                    tuple(tickers), start, end, min_days=20,
                )
                if prices is None or (hasattr(prices, "empty") and prices.empty):
                    out[name][label] = None
                    continue
                if isinstance(prices, _pd.Series):
                    prices = prices.to_frame()

                # Align to requested ticker order; drop any we couldn't fetch
                cols = [t for t in tickers if t in prices.columns]
                if not cols:
                    out[name][label] = None
                    continue
                prices = prices[cols].dropna(how="all").ffill().dropna()
                if prices.empty or len(prices) < 10:
                    out[name][label] = None
                    continue

                # Re-normalize weights against the tickers we actually have
                idx = [tickers.index(c) for c in cols]
                w_eff = [w_norm[i] for i in idx]
                w_sum = sum(w_eff)
                if w_sum <= 0:
                    out[name][label] = None
                    continue
                w_eff = [w / w_sum for w in w_eff]

                # Per-ticker total return = last/first - 1
                ticker_rets = [
                    (prices[c].iloc[-1] / prices[c].iloc[0]) - 1.0
                    for c in cols
                ]
                # Portfolio return = weighted sum of per-ticker returns.
                # This is a "buy-and-hold" approximation (no rebalancing
                # within the window). For most of these short windows the
                # error vs daily-rebalanced is small; if perfect accuracy
                # is needed later we can switch to compounding daily
                # weighted returns instead.
                portfolio_ret = sum(w * r for w, r in zip(w_eff, ticker_rets))
                out[name][label] = float(portfolio_ret)
            except Exception:
                out[name][label] = None

    return out


def _resolve_advisory_fee_pct(proposal: dict | None,
                              client_profile: dict | None,
                              firm_settings: dict | None) -> float:
    """Resolve the advisory fee (annual %, e.g. 1.00) for SEC Marketing Rule
    disclosure rendering.

    Fallback chain (most-specific wins):
      1. proposal["advisory_fee_pct"]              — override per-proposal
      2. client_profile["advisory_fee_pct"]        — per-client override
      3. firm_settings["default_advisory_fee_pct"] — firm-wide default
      4. 1.00                                      — last-resort default

    Returns a float in percent units (NOT decimal) — i.e. 1.00 means 1%
    so callers can format it as f"{fee:.2f}%". The 1.00% default is the
    most common AUM fee for small RIAs and lets the PDF generate without
    crashing if no fee has been configured yet, but a calling RIA SHOULD
    set the firm default in firm_settings to ensure the disclosure
    accurately reflects their actual fee schedule.
    """
    for src in (proposal, client_profile, firm_settings):
        if not isinstance(src, dict):
            continue
        # accept either name; firm uses "default_advisory_fee_pct"
        for k in ("advisory_fee_pct", "default_advisory_fee_pct"):
            v = src.get(k)
            if v is not None:
                try:
                    fv = float(v)
                    if 0 <= fv <= 10:  # sanity bounds
                        return fv
                except (TypeError, ValueError):
                    continue
    return 1.00


def _fee_impact_table_data(fee_levels: list[float] | None = None,
                           years: list[int] | None = None,
                           gross_return: float = 0.07,
                           initial: float = 100.0) -> list[list[str]]:
    """Build the rows for the fee-impact disclosure table.

    Returns a list-of-lists suitable for ReportLab's Table:
      row 0   = header (["Fee", "Year 1", "Year 3", ...])
      row 1+  = ["X.XX%", "$YYY.YY", "$YYY.YY", ...] for each fee level

    Mirrors FinMason's table format. Computes the future value of `initial`
    after compound monthly returns of `gross_return - fee` over each
    horizon, so a reader can see how a 0.75% vs 2.00% advisory fee
    changes their realized balance over 1/3/5/7/10 years.

    Args:
        fee_levels:   list of annual fee percentages (e.g. [0.0, 0.75, 1.0]).
                      Defaults to [0.0, 0.75, 1.0, 1.5, 2.0, 2.5].
        years:        time horizons to show. Defaults to [1, 3, 5, 7, 10].
        gross_return: assumed gross annual return (decimal). 7% is
                      the SEC's standard illustration figure.
        initial:      starting balance (display only — proportional).
    """
    fee_levels = fee_levels if fee_levels is not None else [0.0, 0.75, 1.0, 1.5, 2.0, 2.5]
    years      = years      if years      is not None else [1, 3, 5, 7, 10]

    rows = [["Fee"] + [f"Year {y}" for y in years]]
    for fee_pct in fee_levels:
        net_annual = gross_return - (fee_pct / 100.0)
        # Compound monthly to match FinMason's note ("compounded monthly")
        net_monthly = net_annual / 12.0
        row = [f"{fee_pct:.2f}%"]
        for y in years:
            n_months = y * 12
            fv = initial * ((1.0 + net_monthly) ** n_months)
            row.append(f"${fv:,.2f}")
        rows.append(row)
    return rows


def _render_pdf_builder(
    proposal: dict,
    client_profile: dict | None = None,
    *,
    key_prefix: str,
    download_filename: str,
    title: str = "📄 Build PDF Report",
):
    """Unified PDF section picker + download trigger.

    Renders an inline expander with the 11 section checkboxes and a
    "Download PDF" button. Used by every PDF-generation entry point in
    the app so the advisor sees a consistent UI regardless of whether
    the proposal is associated with a client or sitting in Unassociated.

    Defaults: all sections checked. Chart-bearing sections (drawdown,
    rolling, forward) are auto-disabled when the proposal lacks the
    underlying return-series data — same logic the previous Report
    Builder panel used, just inlined.

    Args:
        proposal: the proposal dict (from client_proposals.json).
        client_profile: optional client profile dict. When None,
            a synthesized "Unassociated Proposal" stub is used.
        key_prefix: unique prefix for Streamlit widget keys (e.g.
            f"saved_{client_key}_{version_id}" or f"unassoc_{version_id}").
            Required because Streamlit needs unique keys per widget.
        download_filename: filename used when the browser saves the PDF.
        title: label for the expander header. Defaults to "📄 Build PDF Report".

    No return value — the function manages its own UI state and
    triggers the download via trigger_pdf_download() when the user
    clicks the button.
    """
    # Detect whether chart-data is present so we can gate the chart sections.
    # The proposal carries returns on the balanced tier when it was saved
    # via the Optimizer "Generate Proposal" flow. The Notable Market Periods
    # section (which replaced the prior drawdown / rolling Sharpe / forward
    # Monte Carlo charts) no longer reads from this attached data — it
    # fetches live price history at PDF-build time, so it works whether or
    # not the proposal carries `returns` data.

    with st.expander(title, expanded=False):
        st.caption("Select the sections to include, then click Download PDF.")

        rb1, rb2 = st.columns(2)
        # Always-available sections (no underlying data dependency)
        sec_cover     = rb1.checkbox("Cover page + client summary",  True, key=f"{key_prefix}_cov")
        sec_profile   = rb1.checkbox("Risk profile results",         True, key=f"{key_prefix}_prof")
        sec_proposals = rb1.checkbox("3-tier proposal summary",      True, key=f"{key_prefix}_prop")
        sec_hist      = rb1.checkbox("Historical performance table", True, key=f"{key_prefix}_hist")
        sec_fee_comp  = rb1.checkbox(
            "Fee comparison table",  True, key=f"{key_prefix}_feecmp",
            help="Shows the impact of different annual fee levels (0%, 0.75%, "
                 "1%, 1.5%, 2%, 2.5%) on a $100 starting balance over 1/3/5/7/10 years. "
                 "Useful for clients comparing your firm's fee against industry alternatives.",
        )

        # Notable Market Periods — replaces the prior Drawdown / Rolling
        # Sharpe / Forward Monte Carlo trio. Compares each portfolio
        # (Recommended, Current, SPY, BND) across five historical event
        # windows (2008 GFC, 2015–16 China/oil, 2018 Q4, 2020 COVID, 2022).
        # Pulls live price history at PDF-build time, so works whether or
        # not the proposal has chart-data attached.
        sec_notable = rb2.checkbox(
            "Notable Market Periods",  True, key=f"{key_prefix}_notable",
            help="A single-page table comparing how each portfolio performed "
                 "during five notable historical events (2008 Financial Crisis, "
                 "2015–16 China/oil selloff, 2018 Q4 selloff, 2020 COVID crash, "
                 "2022 bear market). Compares Recommended, Current, S&P 500, and "
                 "Aggregate Bond benchmark across the full event windows.",
        )
        sec_alloc   = rb2.checkbox("Allocation pie charts",      True, key=f"{key_prefix}_alloc")
        sec_metrics = rb2.checkbox("Risk metrics table",         True, key=f"{key_prefix}_met")
        sec_notes   = rb2.checkbox("Advisor notes + disclaimer", True, key=f"{key_prefix}_notes")

        if st.button("📥 Download PDF", type="primary",
                     use_container_width=True,
                     key=f"{key_prefix}_dl"):
            sections = {
                "cover":     sec_cover,    "profile":   sec_profile,
                "proposals": sec_proposals, "historical": sec_hist,
                "notable_periods": sec_notable,
                "allocation": sec_alloc,
                "metrics":   sec_metrics,  "notes":     sec_notes,
                "fee_comparison": sec_fee_comp,
                # Legacy keys kept off — the rendering paths for these
                # were removed when Notable Market Periods replaced them.
                # Setting False so any saved-section dicts referencing
                # these old keys don't trigger removed code.
                "drawdown": False, "rolling": False, "forward": False,
            }
            try:
                with st.spinner("Building PDF…"):
                    pdf_bytes = build_client_proposal_pdf(
                        client_profile=client_profile or {"client_name": "Unassociated Proposal"},
                        proposal=proposal,
                        sections=sections,
                    )
                trigger_pdf_download(pdf_bytes, download_filename)
                st.success(f"✅ {download_filename} downloading…")
            except Exception as _pe:
                # Don't let a builder error disappear silently — show it
                # along with a traceback so we can diagnose why the PDF
                # didn't generate.
                import traceback as _tb
                st.error(f"PDF generation failed: {_pe}")
                with st.expander("Debug info"):
                    st.code(_tb.format_exc())


def build_client_proposal_pdf(client_profile, proposal, sections):
    """Generate a professional client-facing investment proposal PDF.

    REVAMPED STRUCTURE (Snapshot Report style):
        1. Snapshot Report header — big RISK N badge, narrative, client vs portfolio
           comparison chips at top right
        2. Current Portfolio card  — account summary with risk score
        3. Up to THREE Proposed Portfolios (driven by Step 4 "Select Final
           Proposal for Report" dropdowns: option_1, option_2, option_3).
           Each has its own card with pie chart, holdings legend, and metrics.
        4. Client Profile  — contact + risk assessment + priorities
        5. Methodology     — how the analysis was constructed
        6. Risk Analysis   — factors + mitigations
        7. Implementation  — rebalancing cadence
        8. Advisor Notes
        9. Appendix        — glossary + disclaimers

    Palette: navy (#274c77) + mid-blue (#6096ba) + sky-blue (#a3cef1)
             + stone (#e7ecef) + warm-gray (#8b8c89).

    Pies use the same PIE_PALETTE and sort-by-weight logic as on-screen.

    Arguments:
        client_profile: dict from risk_profiles.json
        proposal:       dict from client_proposals.json (has tiers +
                        optionally final_picks mapping option_1/2/3 → tier key
                        or preset/saved name)
        sections:       dict of bool flags — legacy keys preserved
    Returns PDF bytes.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, PageBreak,
                                    KeepTogether, Flowable, Image)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Wedge, Rect, String, Circle
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics import renderPDF
    from datetime import datetime as _dt

    # ── CLIENT PROPOSAL PALETTE ────────────────────────────────
    NAVY        = colors.HexColor("#274C77")
    NAVY_DEEP   = colors.HexColor("#1A3454")
    NAVY_MID    = colors.HexColor("#3D6A9B")
    ACCENT      = colors.HexColor("#6096BA")   # mid blue
    ACCENT_SOFT = colors.HexColor("#A3CEF1")   # sky blue
    CHARCOAL    = colors.HexColor("#2A3541")
    SLATE       = colors.HexColor("#4A5563")
    GRAY        = colors.HexColor("#8B8C89")   # warm gray
    GRAY_SOFT   = colors.HexColor("#B0B1AE")
    BORDER      = colors.HexColor("#D4D9DB")
    BORDER_SOFT = colors.HexColor("#E7ECEF")
    BG_SOFT     = colors.HexColor("#E7ECEF")
    BG_LIGHT    = colors.HexColor("#F3F5F7")
    WHITE       = colors.white
    BLACK       = colors.black

    # Pie palette — mirrors the on-screen Sasha Trubetskoy 19-color set
    # (minus brown/beige/maroon per user spec) so PDFs match the app.
    PDF_PIE_PALETTE = [
        colors.HexColor("#e6194B"), colors.HexColor("#3cb44b"),
        colors.HexColor("#ffe119"), colors.HexColor("#4363d8"),
        colors.HexColor("#f58231"), colors.HexColor("#911eb4"),
        colors.HexColor("#42d4f4"), colors.HexColor("#f032e6"),
        colors.HexColor("#bfef45"), colors.HexColor("#fabed4"),
        colors.HexColor("#469990"), colors.HexColor("#dcbeff"),
        colors.HexColor("#aaffc3"), colors.HexColor("#808000"),
        colors.HexColor("#ffd8b1"), colors.HexColor("#000075"),
        colors.HexColor("#a9a9a9"),
    ]

    TIER_COLORS = {
        "conservative": NAVY,
        "balanced":     ACCENT,
        "aggressive":   NAVY_DEEP,
        "alternate":    GRAY,
    }

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=0.55*inch, leftMargin=0.55*inch,
        topMargin=0.55*inch,   bottomMargin=0.65*inch,
        title=f"Portfolio Snapshot — {client_profile.get('client_name','Client')}",
        author="Portfolio Intelligence",
    )

    # ── TYPOGRAPHY ─────────────────────────────────────────────
    snapshot_title = ParagraphStyle(
        "snap_title", fontSize=26, leading=30, textColor=CHARCOAL,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=2,
    )
    h1 = ParagraphStyle(
        "h1", fontSize=16, leading=20, textColor=NAVY,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
        spaceBefore=8, spaceAfter=3,
    )
    h1_eyebrow = ParagraphStyle(
        "h1_eyebrow", fontSize=8, leading=10, textColor=ACCENT,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=2,
    )
    h2 = ParagraphStyle(
        "h2", fontSize=13, leading=16, textColor=NAVY,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
        spaceBefore=8, spaceAfter=3,
    )
    h3 = ParagraphStyle(
        "h3", fontSize=10, leading=13, textColor=NAVY_MID,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
        spaceBefore=6, spaceAfter=3,
    )
    body = ParagraphStyle(
        "body", fontSize=10, leading=14, textColor=CHARCOAL,
        fontName="Helvetica", alignment=TA_LEFT, spaceAfter=5,
    )
    body_justify = ParagraphStyle(
        "body_justify", parent=body, alignment=TA_JUSTIFY,
    )
    body_small = ParagraphStyle(
        "body_small", fontSize=9, leading=12, textColor=CHARCOAL,
        fontName="Helvetica", alignment=TA_LEFT, spaceAfter=3,
    )
    caption = ParagraphStyle(
        "caption", fontSize=8, leading=10, textColor=GRAY,
        fontName="Helvetica-Oblique", alignment=TA_LEFT, spaceAfter=2,
    )
    eyebrow_cap = ParagraphStyle(
        "eyebrow_cap", fontSize=7.5, leading=9, textColor=GRAY,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=1,
    )
    tip_style = ParagraphStyle(
        "tip", fontSize=9, leading=13, textColor=NAVY_MID,
        fontName="Helvetica", alignment=TA_LEFT,
    )

    # ── HELPERS ────────────────────────────────────────────────

    def risk_badge(score, label="RISK", size=0.55*inch, needle_color=None):
        """Render the boxed 'RISK N' badge from the reference design.

        Small bordered square with 'RISK' label on top and big number
        below. Returns a Drawing.
        """
        d = Drawing(size * 1.05, size * 1.05)
        # Colored ring/border
        border_color = needle_color or NAVY
        d.add(Rect(0, 0, size, size, strokeColor=border_color,
                   strokeWidth=1.2, fillColor=WHITE))
        # Tiny colored bar on top with RISK label
        bar_h = size * 0.26
        d.add(Rect(0, size - bar_h, size, bar_h,
                   strokeColor=border_color, strokeWidth=0,
                   fillColor=border_color))
        d.add(String(size/2, size - bar_h*0.72, label,
                     fontName="Helvetica-Bold", fontSize=7,
                     fillColor=WHITE, textAnchor="middle"))
        # Big score number
        try:
            n = str(int(score)) if score != "—" else "—"
        except (ValueError, TypeError):
            n = "—"
        d.add(String(size/2, (size - bar_h)/2 - 5, n,
                     fontName="Helvetica-Bold", fontSize=18,
                     fillColor=CHARCOAL, textAnchor="middle"))
        return d

    def pie_drawing(tickers, weights, size=2.2*inch):
        """Render a vector pie chart with the same styling as on-screen.

        Sorts by weight descending, uses PDF_PIE_PALETTE cycling, tiny
        donut hole, white slice separators.
        """
        # Clean + sort
        data = [(t, float(w)) for t, w in zip(tickers, weights)
                if t and float(w or 0) > 0.05]
        data.sort(key=lambda x: -x[1])
        if not data:
            d = Drawing(size, size)
            d.add(String(size/2, size/2, "No data",
                         fontName="Helvetica", fontSize=9,
                         fillColor=GRAY, textAnchor="middle"))
            return d

        # Stride through palette so adjacent slices contrast
        n = len(data)
        if n <= len(PDF_PIE_PALETTE):
            stride = max(1, len(PDF_PIE_PALETTE) // max(n, 1))
            colors_list = [PDF_PIE_PALETTE[(i * stride) % len(PDF_PIE_PALETTE)]
                           for i in range(n)]
        else:
            colors_list = [PDF_PIE_PALETTE[i % len(PDF_PIE_PALETTE)]
                           for i in range(n)]

        d = Drawing(size, size)
        p = Pie()
        p.x = 0
        p.y = 0
        p.width = size
        p.height = size
        p.data = [v for _, v in data]
        p.labels = None              # labels rendered in separate legend
        p.slices.strokeColor = WHITE
        p.slices.strokeWidth = 1.3
        p.startAngle = 90
        p.direction = "clockwise"
        for i, c in enumerate(colors_list):
            p.slices[i].fillColor = c
        # Tiny donut hole — ReportLab Pie doesn't directly support,
        # so overlay a white circle at center (18% of diameter)
        d.add(p)
        cx, cy = size/2, size/2
        d.add(Circle(cx, cy, size * 0.09, fillColor=WHITE,
                     strokeColor=None, strokeWidth=0))
        return d

    def pie_legend_table(tickers, weights, max_rows_per_col=8):
        """Static legend rendered as a ReportLab Table: color swatch +
        ticker + weight. Mirrors the on-screen key column.

        Auto-splits into 2 columns when > max_rows_per_col holdings.
        """
        data = [(t, float(w)) for t, w in zip(tickers, weights)
                if t and float(w or 0) > 0.05]
        data.sort(key=lambda x: -x[1])
        if not data:
            return Paragraph("—", body_small)

        # Match colors to pie ordering
        n = len(data)
        if n <= len(PDF_PIE_PALETTE):
            stride = max(1, len(PDF_PIE_PALETTE) // max(n, 1))
            row_colors = [PDF_PIE_PALETTE[(i * stride) % len(PDF_PIE_PALETTE)]
                          for i in range(n)]
        else:
            row_colors = [PDF_PIE_PALETTE[i % len(PDF_PIE_PALETTE)]
                          for i in range(n)]

        def _row(color, t, w):
            swatch = Drawing(10, 10)
            swatch.add(Rect(0, 0, 10, 10, fillColor=color,
                            strokeColor=None, strokeWidth=0))
            return [swatch,
                    Paragraph(f"<b>{t}</b>", body_small),
                    Paragraph(f"{w:.2f}%", body_small)]

        # One column
        if n <= max_rows_per_col:
            rows = [_row(row_colors[i], t, w) for i, (t, w) in enumerate(data)]
            tbl = Table(rows, colWidths=[0.22*inch, 0.7*inch, 0.6*inch])
            tbl.setStyle(TableStyle([
                ("LEFTPADDING",    (0,0), (-1,-1), 2),
                ("RIGHTPADDING",   (0,0), (-1,-1), 4),
                ("TOPPADDING",     (0,0), (-1,-1), 2),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
                ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
                ("ALIGN",          (2,0), (2,-1),  "RIGHT"),
            ]))
            return tbl
        # Two columns
        half = (n + 1) // 2
        left = [_row(row_colors[i], t, w) for i, (t, w) in enumerate(data[:half])]
        right = [_row(row_colors[half + i], t, w)
                 for i, (t, w) in enumerate(data[half:])]
        while len(right) < len(left):
            right.append([Drawing(1, 1), Paragraph("", body_small),
                          Paragraph("", body_small)])

        merged = []
        for lr, rr in zip(left, right):
            merged.append(lr + [Drawing(8, 1)] + rr)
        tbl = Table(merged, colWidths=[0.22*inch, 0.6*inch, 0.55*inch,
                                        0.15*inch,
                                        0.22*inch, 0.6*inch, 0.55*inch])
        tbl.setStyle(TableStyle([
            ("LEFTPADDING",    (0,0), (-1,-1), 2),
            ("RIGHTPADDING",   (0,0), (-1,-1), 3),
            ("TOPPADDING",     (0,0), (-1,-1), 2),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
            ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",          (2,0), (2,-1),  "RIGHT"),
            ("ALIGN",          (6,0), (6,-1),  "RIGHT"),
        ]))
        return tbl

    def thin_rule(color=ACCENT, thickness=1, width="100%"):
        return HRFlowable(width=width, thickness=thickness, color=color,
                          spaceBefore=2, spaceAfter=6)

    def section_header(eyebrow, title):
        return KeepTogether([
            Paragraph(eyebrow.upper(), h1_eyebrow),
            Paragraph(title, h1),
            thin_rule(ACCENT, 1),
        ])

    # Page background / footer (later pages)
    def _on_page(canvas, _doc):
        canvas.saveState()
        # Thin nav stripe at top
        canvas.setFillColor(NAVY)
        canvas.rect(0, letter[1] - 0.20*inch, letter[0], 0.20*inch, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, letter[1] - 0.24*inch, letter[0], 0.04*inch, fill=1, stroke=0)
        # Footer
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            0.55*inch, 0.35*inch,
            f"Confidential — {client_profile.get('client_name','Client')}  ·  "
            f"Prepared {_dt.now().strftime('%B %d, %Y')}"
        )
        canvas.drawRightString(
            letter[0] - 0.55*inch, 0.35*inch,
            f"Page {_doc.page}"
        )
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.55*inch, 0.50*inch, letter[0] - 0.55*inch, 0.50*inch)
        canvas.restoreState()

    def _on_first_page(canvas, _doc):
        # Dotted separator line across the top (reference-style)
        canvas.saveState()
        canvas.setStrokeColor(CHARCOAL)
        canvas.setLineWidth(0.8)
        canvas.setDash(1, 2)
        canvas.line(0.55*inch, letter[1] - 0.55*inch,
                    letter[0] - 0.55*inch, letter[1] - 0.55*inch)
        # Footer
        canvas.setDash()
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            0.55*inch, 0.35*inch,
            f"Confidential — {client_profile.get('client_name','Client')}  ·  "
            f"Prepared {_dt.now().strftime('%B %d, %Y')}"
        )
        canvas.drawRightString(
            letter[0] - 0.55*inch, 0.35*inch,
            f"Page {_doc.page}"
        )
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.55*inch, 0.50*inch, letter[0] - 0.55*inch, 0.50*inch)
        canvas.restoreState()

    story = []
    tiers = proposal.get("tiers", {}) or {}

    # ── RESOLVE WHICH 3 PORTFOLIOS GO IN THE REPORT ───────────
    # Priority: proposal['final_picks'] (option_1/2/3 per Step 4) →
    # fall back to whatever tiers are available in order.
    # Labels here are what gets printed in the PDF ribbon next to "OPTION N ·",
    # so we drop the "Option N" prefix from each label to avoid double-billing
    # ("OPTION 1 · OPTION 1 (SLIGHTLY MORE CONSERVATIVE)" → "OPTION 1 ·
    # SLIGHTLY MORE CONSERVATIVE").
    TIER_LABEL_MAP = {
        "conservative": ("Slightly more conservative",
                         "Min-volatility re-optimization within ±50% corridor"),
        "balanced":     ("Proposed",
                         "Your submitted holdings, verbatim"),
        "aggressive":   ("Slightly more aggressive",
                         "Max-Sharpe re-optimization within ±50% corridor"),
        "alternate":    ("Broad-ETF Alternate",
                         "Clean-slate diversified option"),
    }

    def _real_score(tks, wts):
        """Compute actual portfolio risk score from holdings — same math as
        PCM (per-holding weighted average minus diversification adjustment).
        Returns None if it can't be computed; callers fall back to
        target_score in that case."""
        if not tks or not wts:
            return None
        try:
            _hs, _hv = [], []
            for _t in tks:
                _r = security_risk_score(_t)
                if _r:
                    _hs.append(_r.get("score", 50))
                    _hv.append(_r.get("ann_vol", 0.15))
                else:
                    _hs.append(50)
                    _hv.append(0.15)
            try:
                _pv = _cached_portfolio_vol(
                    tuple(tks),
                    tuple(round(float(w), 4) for w in wts),
                )
            except Exception:
                _pv = None
            return compute_portfolio_risk_score(
                tks, wts,
                holding_scores=_hs,
                holding_vols=_hv,
                portfolio_vol=_pv,
            )
        except Exception:
            return None

    def _resolve_option(slot_name):
        """Parse a final_picks slot value (e.g. '⭐ Recommended — Balanced',
        '📊 Subject Portfolio', '📁 Saved Portfolio', '🧩 60/40 Classic',
        or a tier key) into (label, subtitle, tier_key_or_None, tickers,
        weights, score)."""
        raw = (proposal.get("final_picks") or {}).get(slot_name, "") or ""
        raw = raw.strip()
        if not raw or raw in ("— none —", "Custom — Enter Your Own Tickers"):
            return None
        # 📊 Subject Portfolio — same holdings as the Balanced tier
        # (which is the user's submitted portfolio verbatim). Use the
        # canonical "Proposed" label so the PDF ribbon reads consistently
        # with the other two option slots ("OPTION 2 · PROPOSED" rather
        # than "OPTION 2 · SUBJECT PORTFOLIO").
        if raw.startswith("📊 Subject Portfolio"):
            t = tiers.get("balanced", {})
            lbl, sub = TIER_LABEL_MAP["balanced"]
            _tks = t.get("tickers", []); _wts = t.get("weights", [])
            _score = _real_score(_tks, _wts) or t.get("target_score")
            return (lbl, sub, "balanced", _tks, _wts, _score)
        # ⭐ Recommended — {label} → find the matching tier
        if raw.startswith("⭐ Recommended"):
            for tk, t in tiers.items():
                if t.get("label", "").lower() in raw.lower():
                    lbl, sub = TIER_LABEL_MAP.get(tk, (t.get("label", tk), ""))
                    _tks = t.get("tickers", []); _wts = t.get("weights", [])
                    _score = _real_score(_tks, _wts) or t.get("target_score")
                    return (lbl, sub, tk, _tks, _wts, _score)
        # 🧩 prefix → preset portfolio (e.g. "🧩 Schwab Core ETF 64/36").
        # Strip the prefix and resolve the underlying ticker/weight pair via
        # _resolve_preset so the PDF can render its pie chart and risk score
        # rather than treating it as an opaque external portfolio.
        if raw.startswith("🧩 "):
            preset_label = raw[2:].strip()
            _tks, _wmap = _resolve_preset(preset_label)
            if _tks:
                # Convert pct → decimal for the PDF (which downstream divides
                # again — _real_score handles either, but we keep parity with
                # tier dicts which carry decimals here).
                _total = sum(_wmap.values()) or 1.0
                _wts = [_wmap.get(t, 0.0) / _total for t in _tks]
                _score = _real_score(_tks, _wts)
                return (preset_label, "Preset portfolio",
                        None, _tks, _wts, _score)
            # Fallback: opaque external label
            return (preset_label, "External portfolio",
                    None, [], [], None)
        # 📁 — saved portfolios. Tickers may not be on the proposal object,
        # so we just label it and skip pie rendering.
        if raw.startswith("📁 "):
            return (raw[2:].strip(), "External portfolio",
                    None, [], [], None)
        # Treat as literal tier key if matches
        if raw in tiers:
            t = tiers[raw]
            lbl, sub = TIER_LABEL_MAP.get(raw, (t.get("label", raw), ""))
            _tks = t.get("tickers", []); _wts = t.get("weights", [])
            _score = _real_score(_tks, _wts) or t.get("target_score")
            return (lbl, sub, raw, _tks, _wts, _score)
        return (raw, "", None, [], [], None)

    picks = []
    for slot in ("option_1", "option_2", "option_3"):
        opt = _resolve_option(slot)
        if opt:
            picks.append(opt)

    # Fallback if no final_picks set — use tiers in display order:
    # Option 1 = balanced (proposed), Option 2 = conservative, Option 3 = aggressive.
    # alternate slots into a leftover position only if one is open.
    if not picks:
        order = ["balanced", "conservative", "aggressive", "alternate"]
        for tk in order:
            if tk in tiers and len(picks) < 3:
                t = tiers[tk]
                lbl, sub = TIER_LABEL_MAP.get(tk, (t.get("label", tk), ""))
                _tks = t.get("tickers", []); _wts = t.get("weights", [])
                _score = _real_score(_tks, _wts) or t.get("target_score")
                picks.append((lbl, sub, tk, _tks, _wts, _score))

    # ═══════════════════════════════════════════════════════════
    # PAGE 1 — SNAPSHOT REPORT
    # ═══════════════════════════════════════════════════════════
    client_score = client_profile.get("overall_score", "—")
    client_name  = client_profile.get("client_name", "Client")
    client_label = client_profile.get("risk_label", "—")

    # Current portfolio risk score
    # ─────────────────────────────
    # The "current portfolio" on page 1 should reflect what the CLIENT
    # actually holds today — which is Step 2's client_current_portfolio
    # snapshot — NOT the balanced tier (which is Option 1, the proposed
    # allocation). When Step 2 wasn't set, fall back to the balanced tier.
    #
    # Scoring uses the SAME math as the PCM risk score row: per-holding
    # weighted average minus a diversification adjustment based on
    # portfolio_vol / weighted-sum-of-holding-vols. Without the volatility
    # data the score loses ~5-10 points of precision on diversified
    # portfolios.
    _curr_snap_pg1 = (proposal or {}).get("client_current_portfolio") or {}
    _curr_snap_tks_pg1 = list(_curr_snap_pg1.get("tickers") or [])
    _curr_snap_w_pg1   = _curr_snap_pg1.get("weights") or {}
    if _curr_snap_tks_pg1 and isinstance(_curr_snap_w_pg1, dict):
        # Step 2 stores weights as a dict (ticker → percent). Convert to
        # parallel-list shape that compute_portfolio_risk_score expects.
        _cur_tickers = _curr_snap_tks_pg1
        _cur_weights = [
            float(_curr_snap_w_pg1.get(t, 0) or 0) for t in _cur_tickers
        ]
        # Drop zero-weight slots
        _pairs = [(t, w) for t, w in zip(_cur_tickers, _cur_weights) if w > 0]
        if _pairs:
            _cur_tickers = [t for t, _ in _pairs]
            _cur_weights = [w for _, w in _pairs]
    else:
        # No Step 2 snapshot — fall back to the balanced tier (Option 1).
        # This is what the legacy behavior was.
        current_port = tiers.get("balanced", {})
        _cur_tickers = current_port.get("tickers") or []
        _cur_weights = current_port.get("weights") or []

    # Always keep current_port pointing somewhere for downstream code that
    # references it (target_score, holdings list, etc.)
    current_port = tiers.get("balanced", {})

    try:
        if _cur_tickers and _cur_weights and len(_cur_tickers) == len(_cur_weights):
            # Build per-holding score + vol arrays, then call
            # compute_portfolio_risk_score with all three knobs so the
            # diversification adjustment fires (matching PCM math exactly).
            _h_scores_pg1 = []
            _h_vols_pg1   = []
            for _tt in _cur_tickers:
                _rr = security_risk_score(_tt)
                if _rr:
                    _h_scores_pg1.append(_rr.get("score", 50))
                    _h_vols_pg1.append(_rr.get("ann_vol", 0.15))
                else:
                    _h_scores_pg1.append(50)
                    _h_vols_pg1.append(0.15)
            # Portfolio vol from cache (same helper PCM uses).
            try:
                _port_vol_pg1 = _cached_portfolio_vol(
                    tuple(_cur_tickers),
                    tuple(round(float(w), 4) for w in _cur_weights),
                )
            except Exception:
                _port_vol_pg1 = None
            current_score = compute_portfolio_risk_score(
                _cur_tickers, _cur_weights,
                holding_scores=_h_scores_pg1,
                holding_vols=_h_vols_pg1,
                portfolio_vol=_port_vol_pg1,
            )
        else:
            current_score = current_port.get("target_score", client_score)
    except Exception:
        # Never let a scoring hiccup take down the PDF — fall back gracefully.
        current_score = current_port.get("target_score", client_score)

    # ── FIRM BRANDING HEADER (page 1) ──────────────────────────
    # Pulls from firm_settings.json + firm_logo.png + advisor_photo.png
    # (set in Client Records → 🎨 Firm Branding). Anything missing simply
    # collapses out — a firm with no logo gets a clean text-only header,
    # a firm with everything set gets logo + firm/advisor block.
    _firm_settings = {}
    try:
        _firm_settings = load_firm_settings() or {}
    except Exception:
        _firm_settings = {}

    _has_logo = os.path.exists(FIRM_LOGO_PATH)
    _firm_name     = (_firm_settings.get("firm_name")     or "").strip()
    _adv_name      = (_firm_settings.get("advisor_name")  or "").strip()
    _adv_title     = (_firm_settings.get("advisor_title") or "").strip()
    _adv_email     = (_firm_settings.get("advisor_email") or "").strip()
    _adv_phone     = (_firm_settings.get("advisor_phone") or "").strip()
    _firm_website  = (_firm_settings.get("firm_website")  or "").strip()
    _has_any_firm_text = any([_firm_name, _adv_name, _adv_title,
                              _adv_email, _adv_phone, _firm_website])

    if _has_logo or _has_any_firm_text:
        # Left cell: logo (if present).
        # Sized to 2.8" × 1.1" — 4× the area of the original 1.4" × 0.55"
        # box, while staying within sensible bounds for a pro letterhead
        # (the page is 7.5" usable width; logo column is ~3.0" leaving
        # ~4.5" for firm/advisor text on the right).
        if _has_logo:
            try:
                _logo_flow = Image(FIRM_LOGO_PATH, width=2.8*inch, height=1.1*inch,
                                   kind="proportional")
            except Exception:
                _logo_flow = Paragraph("", body_small)
        else:
            _logo_flow = Paragraph("", body_small)

        # Right cell: firm + advisor lines, right-aligned
        _firm_right_style = ParagraphStyle(
            "firm_right", fontSize=8.5, leading=11, textColor=CHARCOAL,
            fontName="Helvetica", alignment=TA_RIGHT, spaceAfter=0,
        )
        _firm_right_bold = ParagraphStyle(
            "firm_right_bold", parent=_firm_right_style,
            fontName="Helvetica-Bold", textColor=NAVY, fontSize=9.5, leading=12,
        )
        _firm_lines = []
        if _firm_name:
            _firm_lines.append(Paragraph(_firm_name, _firm_right_bold))
        if _adv_name or _adv_title:
            _adv_line = _adv_name + (f" · {_adv_title}" if _adv_title and _adv_name else _adv_title)
            _firm_lines.append(Paragraph(_adv_line, _firm_right_style))
        _contact_bits = []
        if _adv_email: _contact_bits.append(_adv_email)
        if _adv_phone: _contact_bits.append(_adv_phone)
        if _contact_bits:
            _firm_lines.append(Paragraph(" · ".join(_contact_bits), _firm_right_style))
        if _firm_website:
            _firm_lines.append(Paragraph(_firm_website, _firm_right_style))

        _branding_band = Table(
            [[_logo_flow, _firm_lines or Paragraph("", body_small)]],
            colWidths=[3.0*inch, 4.3*inch],
        )
        _branding_band.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING",   (0,0), (-1,-1), 0),
            ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ]))
        story.append(_branding_band)
        story.append(Spacer(1, 0.04*inch))
        story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER,
                                 spaceBefore=2, spaceAfter=4))

    # Header row: big "Snapshot Report" title on left, client vs current
    # portfolio badges on the right
    def _mini_badge_cell(score, label_top, label_bot, color):
        return Table(
            [[risk_badge(score, size=0.45*inch, needle_color=color),
              Table([[Paragraph(f"<font color='{GRAY.hexval()}' size='7'><b>"
                                f"{label_top.upper()}</b></font>", body_small)],
                     [Paragraph(f"<font color='{CHARCOAL.hexval()}' size='9'><b>"
                                f"{label_bot}</b></font>", body_small)]],
                    colWidths=[1.3*inch])]],
            colWidths=[0.55*inch, 1.3*inch],
        )
    _mini_badge_cell1 = _mini_badge_cell(client_score, "YOUR RISK", "Client Profile", NAVY)
    _mini_badge_cell2 = _mini_badge_cell(current_score, "CURRENT", "Your Portfolio", ACCENT)

    # ── Risk Alignment % ──────────────────────────────────────
    # Compares the client's questionnaire-derived risk profile (client_score)
    # to the engine score of their actual current portfolio (current_score).
    # Direction-aware: shows the alignment % AND the magnitude/direction of
    # the gap so the advisor can explain WHY a recommendation is needed.
    #
    # Formula: 100% - |gap|, clamped to [0, 100].
    #   gap = current_score - client_score
    #   gap > 0  → portfolio runs HOTTER than profile (over-aggressive)
    #   gap < 0  → portfolio runs COOLER than profile (over-conservative)
    #   gap = 0  → perfectly aligned
    #
    # Only renders when both scores are numeric. If client_score is "—" or
    # current_score couldn't be computed, the badge collapses out and the
    # header keeps just the two risk badges.
    _alignment_cell = None
    try:
        _cs = float(client_score)  if client_score  not in ("—", None, "") else None
        _ps = float(current_score) if current_score not in ("—", None, "") else None
    except (TypeError, ValueError):
        _cs = _ps = None

    if _cs is not None and _ps is not None:
        _gap = _ps - _cs
        _alignment_pct = max(0.0, min(100.0, 100.0 - abs(_gap)))

        # Color the alignment % by tier — green for tight, amber for moderate,
        # red for poor. Thresholds match a typical advisor's framing of
        # "actionable" gaps: ≤5pt is essentially aligned, 6-15pt is worth
        # discussing, 16+ pt is a clear mismatch worth recommending against.
        if abs(_gap) <= 5:
            _align_color = colors.HexColor("#15803d")  # darker green
            _align_verdict = "Aligned"
        elif abs(_gap) <= 15:
            _align_color = colors.HexColor("#d97706")  # amber
            _align_verdict = "Slightly Misaligned"
        else:
            _align_color = colors.HexColor("#b91c1c")  # darker red
            _align_verdict = "Misaligned"

        # Direction phrasing: how the portfolio sits relative to the profile
        if abs(_gap) < 1:
            _direction_text = "matches profile"
        elif _gap > 0:
            _direction_text = f"+{abs(_gap):.0f} pts (more aggressive than profile)"
        else:
            _direction_text = f"−{abs(_gap):.0f} pts (more conservative than profile)"

        # Build the alignment badge — same visual shape as the two risk
        # badges so the trio reads as a unit. Shows the percentage prominently
        # with the verdict label and direction text below.
        _align_pct_para = Paragraph(
            f"<font color='{_align_color.hexval()}' size='16'><b>"
            f"{_alignment_pct:.0f}%</b></font>",
            ParagraphStyle("align_pct", fontSize=16, leading=18,
                           alignment=TA_CENTER, fontName="Helvetica-Bold"),
        )
        _align_label_para = Paragraph(
            f"<font color='{GRAY.hexval()}' size='7'><b>"
            f"RISK ALIGNMENT</b></font>",
            body_small,
        )
        _align_verdict_para = Paragraph(
            f"<font color='{CHARCOAL.hexval()}' size='9'><b>"
            f"{_align_verdict}</b></font>",
            body_small,
        )
        _align_direction_para = Paragraph(
            f"<font color='{GRAY.hexval()}' size='7'>"
            f"{_direction_text}</font>",
            body_small,
        )

        _alignment_cell = Table(
            [[_align_pct_para,
              Table([[_align_label_para],
                     [_align_verdict_para],
                     [_align_direction_para]],
                    colWidths=[1.4*inch])]],
            colWidths=[0.55*inch, 1.4*inch],
        )

    # Header row layout: title on left, risk badges + alignment on right.
    # When alignment is available we have 4 cells (title + 3 badges),
    # otherwise the original 3-cell layout (title + 2 badges).
    if _alignment_cell is not None:
        header_row = Table([[
            Paragraph("Snapshot Report", snapshot_title),
            _mini_badge_cell1,
            _mini_badge_cell2,
            _alignment_cell,
        ]], colWidths=[2.0*inch, 1.7*inch, 1.7*inch, 1.95*inch])
    else:
        header_row = Table([[
            Paragraph("Snapshot Report", snapshot_title),
            _mini_badge_cell1,
            _mini_badge_cell2,
        ]], colWidths=[3.6*inch, 1.9*inch, 1.9*inch])
    header_row.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(Spacer(1, 0.08*inch))
    story.append(header_row)
    story.append(Spacer(1, 0.05*inch))
    # Dotted line below header
    story.append(HRFlowable(width="100%", thickness=0.8, color=CHARCOAL,
                             dash=(1, 2), spaceBefore=2, spaceAfter=10))

    # ── "Your Risk Number" hero block ─────────────────────
    hero_left = Table([
        [risk_badge(client_score, size=0.8*inch, needle_color=NAVY)],
    ], colWidths=[0.9*inch])
    hero_right = [
        Paragraph("<b>Your Risk Number</b>", h2),
        Paragraph(
            f"On a scale of 1–99 where 99 is the highest risk tolerance and 1 is the lowest, "
            f"<b>{client_name}</b> has been profiled at a Risk Number of "
            f"<b>{client_score}</b> "
            f"(<b>{client_label}</b>). "
            f"This rating reflects the combination of risk <i>tolerance</i> (willingness) "
            f"and risk <i>capacity</i> (ability) to bear investment volatility. "
            f"The portfolio recommendations on the following pages are aligned to this "
            f"profile, with room to adjust based on goals, time horizon, and life-stage changes.",
            body_justify,
        ),
    ]
    hero_tbl = Table(
        [[hero_left, hero_right]],
        colWidths=[0.95*inch, 6.4*inch],
    )
    hero_tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ]))
    story.append(hero_tbl)
    story.append(Spacer(1, 0.12*inch))
    story.append(thin_rule(BORDER, 0.6))

    # ── COMPACT CLIENT PROFILE (page 1) ──────────────────────
    # Two-column band: contact/demographics on the left, risk-assessment
    # numbers + priorities on the right. Replaces the old "Section 2"
    # full-page profile that lived on page 2 — having client identity on
    # the cover page is what advisors expect when they scan a proposal.
    if sections.get("profile", True):
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("CLIENT PROFILE", eyebrow_cap))

        _PRIORITY_LBL = {
            "capital_preservation": "Capital Preservation",
            "insurance_planning":   "Insurance Planning",
            "income_generation":    "Income Generation",
            "capital_appreciation": "Capital Appreciation",
            "diversification":      "Diversification",
            "tax_efficiency":       "Tax Efficiency",
            "liquidity":            "Liquidity",
            "social_impact":        "Social/Impact",
            "legacy_planning":      "Legacy Planning",
        }
        _pcs = client_profile.get("priorities") or []

        _profile_left_rows = [
            ["Name",     client_profile.get("client_name", "—") or "—"],
            ["Email",    client_profile.get("client_email", "—") or "—"],
            ["Phone",    client_profile.get("client_phone", "—") or "—"],
            ["Age",      str(client_profile.get("client_age", "—"))],
            ["As of",    client_profile.get("date",
                            client_profile.get("completed_at", "—")) or "—"],
        ]
        _profile_right_rows = [
            ["Risk Score",       f"{client_profile.get('overall_score','—')} / 99"],
            ["Risk Tolerance",   f"{client_profile.get('tolerance_score','—')} / 99"],
            ["Risk Capacity",    f"{client_profile.get('capacity_score','—')} / 99"],
            ["Classification",   client_profile.get("risk_label","—") or "—"],
        ]

        def _profile_subtable(rows):
            t = Table(rows, colWidths=[0.95*inch, 2.55*inch])
            t.setStyle(TableStyle([
                ("TEXTCOLOR",    (0,0), (0,-1),  GRAY),
                ("FONTNAME",     (0,0), (0,-1),  "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (0,-1),  7.5),
                ("TEXTCOLOR",    (1,0), (1,-1),  CHARCOAL),
                ("FONTNAME",     (1,0), (1,-1),  "Helvetica"),
                ("FONTSIZE",     (1,0), (1,-1),  9),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING",  (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING",   (0,0), (-1,-1), 3),
                ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                ("LINEBELOW",    (0,0), (-1,-2), 0.3, BORDER_SOFT),
            ]))
            return t

        _profile_band = Table(
            [[_profile_subtable(_profile_left_rows),
              _profile_subtable(_profile_right_rows)]],
            colWidths=[3.65*inch, 3.65*inch],
        )
        _profile_band.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("BOX",          (0,0), (-1,-1), 0.5, BORDER),
            ("LINEAFTER",    (0,0), (0,0),   0.3, BORDER_SOFT),
            ("BACKGROUND",   (0,0), (-1,-1), BG_LIGHT),
        ]))
        story.append(_profile_band)

        if _pcs:
            story.append(Spacer(1, 0.05*inch))
            _chips_html = "&nbsp;&nbsp;".join(
                f"<font color='{ACCENT.hexval()}'>●</font> "
                f"<font color='{CHARCOAL.hexval()}'>{_PRIORITY_LBL.get(p, p)}</font>"
                for p in _pcs
            )
            _priorities_para = Paragraph(
                f"<font color='{GRAY.hexval()}' size='7'><b>PRIORITIES</b></font>"
                f" &nbsp; {_chips_html}",
                body_small,
            )
            story.append(_priorities_para)

        story.append(Spacer(1, 0.08*inch))
        story.append(thin_rule(BORDER, 0.6))

    # ── CURRENT PORTFOLIO CARD ────────────────────────────
    # Shows what the client ACTUALLY HOLDS TODAY (Step 2 snapshot), not
    # the proposed/balanced tier. The current_score variable was already
    # computed correctly above from the Step 2 snapshot; here we mirror
    # that for the holdings list, pie chart, and equity/bond/cash split.
    # Falls back to the balanced tier (the legacy behavior) only when
    # Step 2 wasn't set when the proposal was saved.
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("CURRENT PORTFOLIO", eyebrow_cap))
    cur_badge = risk_badge(current_score, size=0.55*inch, needle_color=ACCENT)

    # Re-resolve from the Step 2 snapshot (same logic as the score block
    # above so they stay in sync).
    _cur_snap_card = (proposal or {}).get("client_current_portfolio") or {}
    _cur_snap_tks_card = list(_cur_snap_card.get("tickers") or [])
    _cur_snap_w_card   = _cur_snap_card.get("weights") or {}
    if _cur_snap_tks_card and isinstance(_cur_snap_w_card, dict):
        _cur_tickers = _cur_snap_tks_card
        _cur_weights = [
            float(_cur_snap_w_card.get(t, 0) or 0) for t in _cur_tickers
        ]
        _pairs_card = [(t, w) for t, w in zip(_cur_tickers, _cur_weights) if w > 0]
        if _pairs_card:
            _cur_tickers = [t for t, _ in _pairs_card]
            _cur_weights = [w for _, w in _pairs_card]
        # Compute equity/bond/cash classification on the actual current
        # holdings. Uses _classify_ticker (which consults the curated
        # mutual fund table + yfinance fallback for long-tail MFs)
        # rather than prefix matching, so bond MFs like VBTLX/PIMIX
        # correctly count toward bonds. Balanced funds (Wellington,
        # Wellesley, Target Retirement, etc.) split between equity
        # and bond buckets at their stated mix via _balanced_split.
        _w_sum_card = sum(_cur_weights) or 1.0
        _cur_eq = _cur_bd = _cur_cs = 0.0
        for _t, _w in zip(_cur_tickers, _cur_weights):
            _frac = (_w / _w_sum_card) * 100.0
            try:
                _cls, _ = _classify_ticker(_t.upper())
            except Exception:
                _cls = "equity"
            if _cls == "cash":
                _cur_cs += _frac
            elif _cls == "bond":
                _cur_bd += _frac
            elif _cls == "balanced":
                _eq_share, _bd_share = _balanced_split(_t)
                _cur_eq += _frac * _eq_share
                _cur_bd += _frac * _bd_share
            else:
                # equity / leveraged / crypto all bucket as equity exposure
                _cur_eq += _frac
    else:
        # Legacy fallback — Step 2 wasn't snapshotted on this proposal.
        _cur_tickers = current_port.get("tickers", [])
        _cur_weights = current_port.get("weights", [])
        _cur_eq = current_port.get("equity_pct", 0)
        _cur_bd = current_port.get("bond_pct",   0)
        _cur_cs = current_port.get("cash_pct",   0)

    cur_info = [
        [Paragraph(
            f"<font color='{CHARCOAL.hexval()}' size='12'><b>Your Current Holdings</b></font>",
            body_small,
        )],
        [Paragraph(
            f"<font color='{GRAY.hexval()}' size='8'>{len(_cur_tickers)} HOLDINGS</font>",
            body_small,
        )],
        [Paragraph(
            f"<font size='9'><b>{_cur_eq:.0f}%</b> Equity &nbsp;·&nbsp; "
            f"<b>{_cur_bd:.0f}%</b> Bonds &nbsp;·&nbsp; "
            f"<b>{_cur_cs:.0f}%</b> Cash</font>",
            body_small,
        )],
    ]
    cur_left = Table([[cur_badge, Table(cur_info, colWidths=[5.5*inch])]],
                      colWidths=[0.6*inch, 5.5*inch])
    cur_left.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(cur_left)

    # ── Current portfolio pie + legend ─────────────────────
    if _cur_tickers and _cur_weights and sum(float(w or 0) for w in _cur_weights) > 0:
        story.append(Spacer(1, 0.06*inch))
        _cur_pie = pie_drawing(_cur_tickers, _cur_weights, size=2.0*inch)
        _cur_legend = pie_legend_table(_cur_tickers, _cur_weights, max_rows_per_col=8)
        _cur_card = Table(
            [[_cur_pie, _cur_legend]],
            colWidths=[2.3*inch, 4.9*inch],
        )
        _cur_card.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",         (0,0), (0,-1),  "CENTER"),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("BOX",           (0,0), (-1,-1), 0.4, BORDER),
            ("BACKGROUND",    (0,0), (-1,-1), BG_LIGHT),
        ]))
        story.append(_cur_card)

    # ── HOLDINGS DETAIL TABLE (dedicated page 2) ─────────────
    # Per-holding breakdown matching the reference design: small RISK badge,
    # ticker + name, amount, % of portfolio, 6-month historical range.
    # On its own page so the page-1 cover stays uncluttered and the holdings
    # always have room to render in full without splitting.
    if _cur_tickers and _cur_weights:
        story.append(PageBreak())
        story.append(section_header("Section 1", "Current Holdings"))
        story.append(Paragraph(
            "The full breakdown of the client's current portfolio: position "
            "size, weight, recent risk profile, and 6-month price range.",
            body,
        ))
        story.append(Spacer(1, 0.08*inch))
        try:
            import yfinance as _yf
            import pandas as _pd
            import numpy as _np
            from datetime import timedelta as _td2

            _holding_total_usd = float(proposal.get("portfolio_value",
                                         client_profile.get("portfolio_value", 10000.0)))

            # Fetch 6-month price history for each ticker (and get info)
            _end2  = _dt.now()
            _start_6m = _end2 - _td2(days=200)
            _upper_tks = [t.upper() for t in _cur_tickers]
            _info_cache = {}
            _range_cache = {}  # {ticker: (low_6m_pct, high_6m_pct)}
            _ticker_vol   = {}  # {ticker: annualized_vol}

            try:
                _hist = _yf.download(
                    _upper_tks, start=_start_6m, end=_end2,
                    auto_adjust=True, progress=False, threads=True,
                )["Close"]
                if isinstance(_hist, _pd.Series):
                    _hist = _hist.to_frame(name=_upper_tks[0])
                _hist = _hist.dropna(how="all")
            except Exception:
                _hist = _pd.DataFrame()

            for _tk_u in _upper_tks:
                # Range: min/max cumulative return over 6 months
                if _tk_u in _hist.columns and len(_hist[_tk_u].dropna()) > 10:
                    _s = _hist[_tk_u].dropna()
                    _start_px = _s.iloc[0]
                    _cum = (_s / _start_px - 1.0)
                    _range_cache[_tk_u] = (float(_cum.min() * 100),
                                           float(_cum.max() * 100))
                    # Annualized vol for risk-score assignment
                    _r = _s.pct_change().dropna()
                    _ticker_vol[_tk_u] = float(_r.std() * _np.sqrt(252)) if len(_r) > 5 else 0.15
                else:
                    _range_cache[_tk_u] = (None, None)
                    _ticker_vol[_tk_u] = 0.15

            # Fetch names + SEC yields. Prefer Alpha Vantage (faster,
            # more reliable) — fall back to yfinance .info if AV unavailable
            # for a given ticker.
            _av_key_pdf = _resolve_av_key()
            for _tk_u in _upper_tks:
                _profile = None
                if _av_key_pdf:
                    _profile = _alphavantage_fetch_profile(_tk_u, _av_key_pdf)
                if _profile:
                    _info_cache[_tk_u] = _profile
                    continue
                # Fallback to yfinance
                try:
                    _yt = _yf.Ticker(_tk_u)
                    _inf = _yt.info or {}
                    _info_cache[_tk_u] = {
                        "name":      _inf.get("longName") or _inf.get("shortName") or _tk_u,
                        "sec_yield": _inf.get("yield") or _inf.get("dividendYield"),
                        "type":      _inf.get("quoteType", "").upper(),
                    }
                except Exception:
                    _info_cache[_tk_u] = {"name": _tk_u, "sec_yield": None, "type": ""}

            # Per-ticker risk score using the unified scoring system —
            # classifies by ticker (cash/bond/equity/crypto/leveraged) and
            # applies vol × 4.2 with class-specific caps + log compression
            # above 80 for equities. Uses 6mo DD already computed earlier.
            def _score_from_vol_and_ticker(tk_u, v, dd):
                if v is None and dd is None:
                    return 50
                v = float(v or 0)
                dd = float(dd or 0)
                # Use the unified compute_risk_score (sharpe ignored in new spec)
                return compute_risk_score(v, dd, 0, ticker=tk_u)

            # Small risk badge drawing (compact version of the hero badge)
            def _small_risk_badge(score, size=0.35*inch):
                s = size
                d = Drawing(s, s)
                # Color-code by zone
                if score <= 33:   c = colors.HexColor("#00B36B")   # green
                elif score <= 66: c = colors.HexColor("#E8A317")   # amber
                else:             c = colors.HexColor("#CC3B3B")   # red
                d.add(Rect(0, 0, s, s, strokeColor=c, strokeWidth=0.9,
                           fillColor=WHITE))
                bar_h = s * 0.32
                d.add(Rect(0, s - bar_h, s, bar_h, strokeColor=c,
                           strokeWidth=0, fillColor=c))
                d.add(String(s/2, s - bar_h*0.72, "RISK",
                             fontName="Helvetica-Bold", fontSize=4.5,
                             fillColor=WHITE, textAnchor="middle"))
                d.add(String(s/2, (s - bar_h)/2 - 4, str(score),
                             fontName="Helvetica-Bold", fontSize=11,
                             fillColor=CHARCOAL, textAnchor="middle"))
                return d

            # Build table rows
            holdings_hdr = [
                Paragraph("<b>RISK</b>", eyebrow_cap),
                Paragraph("<b>HOLDING</b>", eyebrow_cap),
                Paragraph("<b>AMOUNT</b>", eyebrow_cap),
                Paragraph("<b>% OF<br/>PORTFOLIO</b>", eyebrow_cap),
                Paragraph("<b>SEC<br/>YIELD</b>", eyebrow_cap),
                Paragraph("<b>95% HISTORICAL<br/>RANGE (6 MOS)</b>", eyebrow_cap),
            ]
            holdings_rows = [holdings_hdr]

            # Sort by weight descending to match the pie
            _sorted_holdings = sorted(
                zip(_cur_tickers, _cur_weights),
                key=lambda x: -float(x[1] or 0),
            )

            for _tkr, _wt in _sorted_holdings:
                _tk_u = _tkr.upper()
                _info = _info_cache.get(_tk_u, {})
                _name = _info.get("name") or _tkr
                _name_short = _name if len(_name) <= 42 else _name[:40] + "…"
                _type = _info.get("type") or ""
                # Pull the worst-case 6mo drawdown from the range we computed
                _lo_pct, _hi_pct = _range_cache.get(_tk_u, (None, None))
                _dd_for_score = abs(_lo_pct / 100.0) if _lo_pct is not None else 0
                _score = _score_from_vol_and_ticker(
                    _tk_u, _ticker_vol.get(_tk_u), _dd_for_score,
                )
                _amount_usd = float(_wt or 0) / 100.0 * _holding_total_usd
                _sec_y = _info.get("sec_yield")
                _sec_str = (f"{_sec_y*100:.2f}%" if isinstance(_sec_y, (int, float))
                            and _sec_y and _sec_y < 1 else
                            (f"{_sec_y:.2f}%" if isinstance(_sec_y, (int, float))
                             and _sec_y else "—"))
                _lo, _hi = _range_cache.get(_tk_u, (None, None))
                if _lo is not None and _hi is not None:
                    _range_cell = Paragraph(
                        f"<font color='#CC3B3B'>{_lo:+.2f}%</font>  |  "
                        f"<font color='#00B36B'>{_hi:+.2f}%</font>",
                        body_small,
                    )
                else:
                    _range_cell = Paragraph("—", body_small)

                holdings_rows.append([
                    _small_risk_badge(_score, size=0.35*inch),
                    Paragraph(
                        f"<b>{_tkr}</b> · {_name_short}<br/>"
                        f"<font color='{GRAY.hexval()}' size='7'>{_type}</font>",
                        body_small,
                    ),
                    Paragraph(f"${_amount_usd:,.2f}", body_small),
                    Paragraph(f"{float(_wt or 0):.2f}%", body_small),
                    Paragraph(_sec_str, body_small),
                    _range_cell,
                ])

            holdings_tbl = Table(
                holdings_rows,
                colWidths=[0.5*inch, 2.4*inch, 0.85*inch, 0.85*inch,
                           0.65*inch, 1.55*inch],
            )
            holdings_tbl.setStyle(TableStyle([
                ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("ALIGN",         (0,0), (0,-1),  "CENTER"),
                ("ALIGN",         (2,0), (-2,-1), "RIGHT"),
                ("ALIGN",         (2,0), (-2,0),  "LEFT"),   # header cell left-align
                ("LEFTPADDING",   (0,0), (-1,-1), 4),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LINEBELOW",     (0,0), (-1,0),  1.0, NAVY),
                ("LINEBELOW",     (0,1), (-1,-2), 0.25, BORDER_SOFT),
            ]))
            story.append(holdings_tbl)
        except Exception as _he:
            # Graceful fallback: just skip the detailed table
            story.append(Paragraph(
                f"<i>Detailed holdings breakdown unavailable: {_he}</i>",
                caption,
            ))

    story.append(HRFlowable(width="100%", thickness=2, color=NAVY,
                             spaceBefore=6, spaceAfter=10))

    # ═══════════════════════════════════════════════════════════
    # PROPOSED PORTFOLIOS (max 3, from Step 4 final_picks)
    # All 3 fit on one page with compact layout. Each portfolio block is
    # wrapped in KeepTogether so a tier never breaks across pages.
    # ═══════════════════════════════════════════════════════════
    if picks:
        story.append(PageBreak())   # all 3 on a dedicated page
        story.append(section_header("Recommendations",
                                    f"Proposed Portfolios ({len(picks)})"))
        story.append(Paragraph(
            "The recommendations below have been selected by your advisor from "
            "the optimizer. Each allocation is aligned to your risk profile and "
            "goals.",
            body,
        ))
        story.append(Spacer(1, 0.04*inch))

        for idx, (lbl, sub, tk, ptks, pws, pscore) in enumerate(picks):
            tier_block = []  # everything for this tier — wrapped in KeepTogether

            # Tier ribbon — slimmer padding
            tcolor = TIER_COLORS.get(tk, NAVY)
            ribbon = Table(
                [[
                    Paragraph(f"<font color='white' size='9'><b>OPTION {idx+1} · {lbl.upper()}</b></font>",
                              body_small),
                    Paragraph(f"<font color='#A3CEF1' size='8'>{sub}</font>",
                              body_small),
                    Paragraph(
                        f"<font color='white' size='9'><b>"
                        f"RISK {pscore if pscore else '—'}</b></font>",
                        body_small,
                    ),
                ]],
                colWidths=[2.4*inch, 3.3*inch, 1.5*inch],
            )
            ribbon.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), tcolor),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("ALIGN",         (2,0), (2,0),   "RIGHT"),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            tier_block.append(ribbon)

            # Pie + legend — smaller pie (1.6") so 3 cards fit on one page
            if ptks and pws and sum(float(w or 0) for w in pws) > 0:
                pie_d = pie_drawing(ptks, pws, size=1.6*inch)
                legend_tbl = pie_legend_table(ptks, pws, max_rows_per_col=6)
                card_row = Table(
                    [[pie_d, legend_tbl]],
                    colWidths=[1.85*inch, 5.35*inch],
                )
                card_row.setStyle(TableStyle([
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                    ("ALIGN",         (0,0), (0,-1),  "CENTER"),
                    ("LEFTPADDING",   (0,0), (-1,-1), 3),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 3),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("BOX",           (0,0), (-1,-1), 0.4, BORDER),
                    ("BACKGROUND",    (0,0), (-1,-1), BG_LIGHT),
                ]))
                tier_block.append(card_row)
            else:
                tier_block.append(Paragraph(
                    f"<i>External portfolio — holdings not embedded in this proposal.</i>",
                    body_small,
                ))

            # Rationale below
            if tk and tk in tiers:
                rat = tiers[tk].get("rationale", "")
                if rat:
                    tier_block.append(Paragraph(f"<i>{rat}</i>", caption))
            tier_block.append(Spacer(1, 0.06*inch))

            # Wrap entire tier so it never breaks across pages
            story.append(KeepTogether(tier_block))

    # ═══════════════════════════════════════════════════════════
    # BACKTEST ANALYSIS — 1/3/5/10 year summary table + 3yr line chart
    # ═══════════════════════════════════════════════════════════
    # Pulls historical data via yfinance and compares each of the picks
    # (plus current portfolio) across 4 horizons. Chart shows the last
    # 3 years as growth-of-$10k curves.
    if picks and sections.get("proposals", True):
        try:
            story.append(PageBreak())
            story.append(section_header("Performance", "Historical Backtest"))
            story.append(Paragraph(
                "The table below compares total return, annualized volatility, "
                "Sharpe ratio, and maximum drawdown across each proposed "
                "portfolio and your current portfolio over 1, 3, 5, and 10 year "
                "horizons. The chart shows the growth of $10,000 invested three "
                "years ago.",
                body,
            ))
            story.append(Spacer(1, 0.08*inch))

            # Build the set of portfolios to backtest.
            # Current Portfolio comes from Step 2 (client's current selector
            # at analysis time, snapshotted into the proposal at save time).
            # If Step 2 wasn't set when the proposal was saved, the Current
            # line is omitted entirely — per advisor spec, we don't want to
            # fall back to Step 1 because Option 1 is Step 1 verbatim and
            # the chart would just show overlapping lines.
            bt_portfolios = []
            _curr_snap = (proposal or {}).get("client_current_portfolio") or {}
            _curr_snap_tks = list(_curr_snap.get("tickers") or [])
            _curr_snap_w   = _curr_snap.get("weights") or {}
            # Step 2 snapshot stores weights as a dict (ticker → percent).
            # Convert to the (tickers, weights) parallel-list shape the
            # backtester expects.
            if _curr_snap_tks and _curr_snap_w and isinstance(_curr_snap_w, dict):
                _curr_w_list = [
                    float(_curr_snap_w.get(t, 0) or 0) for t in _curr_snap_tks
                ]
                if sum(_curr_w_list) > 0:
                    bt_portfolios.append(
                        ("Current Portfolio", _curr_snap_tks, _curr_w_list)
                    )
            for idx, (lbl, sub, tk, ptks, pws, pscore) in enumerate(picks):
                if ptks and pws:
                    bt_portfolios.append((f"Option {idx+1}: {lbl}", ptks, pws))

            # Fetch price data for all unique tickers across ALL portfolios
            all_tickers = set()
            for _, tks, _ in bt_portfolios:
                for t in tks:
                    if t: all_tickers.add(t.upper())

            if all_tickers:
                import yfinance as _yf
                import pandas as _pd
                import numpy as _np
                from datetime import timedelta as _td

                _end = _dt.now()
                _start_10y = _end - _td(days=365*10 + 30)
                try:
                    _prices = _yf.download(
                        list(all_tickers), start=_start_10y, end=_end,
                        auto_adjust=True, progress=False, threads=True,
                    )["Close"]
                    if isinstance(_prices, _pd.Series):
                        _prices = _prices.to_frame()
                    _prices = _prices.dropna(how="all")
                except Exception:
                    _prices = None

                if _prices is not None and len(_prices) > 30:
                    _rets = _prices.pct_change().dropna(how="all").fillna(0)

                    def _port_returns(tickers, weights):
                        """Compute portfolio daily returns as weighted sum."""
                        w_arr = _np.array([float(w or 0) for w in weights])
                        tot = w_arr.sum()
                        if tot <= 0: return None
                        w_arr = w_arr / tot
                        cols = [t.upper() for t in tickers if t.upper() in _rets.columns]
                        if not cols: return None
                        # Re-normalize weights over available cols
                        aligned = _np.array([
                            float(weights[i] or 0)
                            for i, t in enumerate(tickers)
                            if t.upper() in _rets.columns
                        ])
                        aligned = aligned / aligned.sum() if aligned.sum() > 0 else aligned
                        return (_rets[cols] * aligned).sum(axis=1)

                    def _stats_over_window(r, days):
                        """Return (total%, vol%, sharpe, maxdd%) over last N days.

                        Sharpe is excess-return Sharpe ((CAGR - rf) / vol) so it
                        matches the rest of the codebase's definition. Old code
                        used (mean*√252)/std which biased Sharpe upward by both
                        omitting rf and using arithmetic mean instead of CAGR.
                        """
                        if r is None or len(r) < days * 0.3:
                            return (None, None, None, None)
                        r_w = r.iloc[-days:] if len(r) > days else r
                        if len(r_w) < 10: return (None, None, None, None)
                        total = float((1 + r_w).prod() - 1)
                        vol = float(r_w.std() * _np.sqrt(252))
                        # CAGR over the actual window
                        actual_yrs = max(len(r_w) / 252.0, 0.08)
                        ann_r = (1 + total) ** (1.0 / actual_yrs) - 1
                        sharpe = _shared_sharpe(ann_r, vol)
                        equity = (1 + r_w).cumprod()
                        dd = float(((equity / equity.cummax()) - 1).min())
                        return (total, vol, sharpe, dd)

                    # ── Build the table: rows = metric × period, cols = portfolios ──
                    # New layout (per advisor feedback): one number per cell
                    # makes the table much easier to scan than the old
                    # 4-portfolios × 5-windows grid with three metrics
                    # crammed into each cell. Row groups are:
                    #   • Total Return (1y / 3y / 5y / 10y)
                    #   • Annualized Volatility (1y / 3y / 5y / 10y)
                    #   • Sharpe Ratio (1y / 3y / 5y / 10y)
                    #   • Max Drawdown (1y / 3y / 5y / 10y)

                    # Compute all stats up front for every portfolio × period
                    # so the table builder is a simple lookup.
                    _periods = [("1-Year", 252), ("3-Year", 252*3),
                                ("5-Year", 252*5), ("10-Year", 252*10)]
                    # _stats[portfolio_idx][period_idx] = (tot, vol, sh, dd) or None
                    _stats = []
                    for name, tks, wts in bt_portfolios:
                        r = _port_returns(tks, wts)
                        if r is None:
                            _stats.append([None] * len(_periods))
                            continue
                        _row_stats = []
                        for _plbl, _days in _periods:
                            _row_stats.append(_stats_over_window(r, _days))
                        _stats.append(_row_stats)

                    # Build header row: blank top-left, then portfolio names
                    _portfolio_names = [name for name, _, _ in bt_portfolios]
                    hdr = ["Metric / Period"] + _portfolio_names
                    bt_rows = [hdr]

                    def _fmt(value, kind):
                        """Format a single metric value with the right unit."""
                        if value is None:
                            return "—"
                        if kind == "pct":
                            return f"{value*100:+.1f}%"
                        if kind == "pct_abs":
                            return f"{value*100:.1f}%"
                        if kind == "ratio":
                            return f"{value:.2f}"
                        return str(value)

                    # Group label rows (full-width navy header strips)
                    # and per-period rows under each.
                    _metric_groups = [
                        ("Total Return",          0, "pct"),
                        ("Annualized Volatility", 1, "pct_abs"),
                        ("Sharpe Ratio",          2, "ratio"),
                        ("Maximum Drawdown",      3, "pct"),
                    ]

                    # Track which rows are group headers (for styling)
                    _group_header_rows = []
                    for _grp_label, _stat_idx, _kind in _metric_groups:
                        # Group header row: spans all columns, just the label
                        _group_header_rows.append(len(bt_rows))
                        bt_rows.append([_grp_label] + [""] * len(_portfolio_names))
                        # Then one row per period
                        for _plbl, _ in _periods:
                            _period_idx = [p[0] for p in _periods].index(_plbl)
                            _row = [f"   {_plbl}"]
                            for _pi, _ in enumerate(bt_portfolios):
                                _stat_tuple = _stats[_pi][_period_idx]
                                if _stat_tuple is None:
                                    _row.append("—")
                                else:
                                    _row.append(_fmt(_stat_tuple[_stat_idx], _kind))
                            bt_rows.append(_row)

                    # Column widths: first col wider for the metric label,
                    # remaining cols split evenly across the portfolio columns.
                    _n_port_cols = len(_portfolio_names)
                    _label_w = 1.5*inch
                    _port_w  = (5.2*inch) / max(1, _n_port_cols)
                    bt_tbl = Table(bt_rows,
                                   colWidths=[_label_w] + [_port_w] * _n_port_cols)
                    _tbl_style = [
                        # Top header row (portfolio names)
                        ("BACKGROUND",   (0,0), (-1,0), NAVY),
                        ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
                        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE",     (0,0), (-1,0), 9),
                        # Body cells
                        ("FONTSIZE",     (0,1), (-1,-1), 9),
                        ("TEXTCOLOR",    (1,1), (-1,-1), CHARCOAL),
                        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
                        ("ALIGN",        (0,0), (0,-1),  "LEFT"),
                        ("LEFTPADDING",  (0,0), (-1,-1), 8),
                        ("RIGHTPADDING", (0,0), (-1,-1), 8),
                        ("TOPPADDING",   (0,0), (-1,-1), 5),
                        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
                        ("BOX",          (0,0), (-1,-1), 0.5, BORDER),
                        ("LINEBELOW",    (0,0), (-1,0), 1.2, ACCENT),
                    ]
                    # Style the metric-group header rows: light navy band,
                    # navy bold text spanning all columns.
                    for _gi in _group_header_rows:
                        _tbl_style.append(("BACKGROUND", (0,_gi), (-1,_gi), BG_SOFT))
                        _tbl_style.append(("FONTNAME",   (0,_gi), (-1,_gi), "Helvetica-Bold"))
                        _tbl_style.append(("TEXTCOLOR",  (0,_gi), (-1,_gi), NAVY))
                        _tbl_style.append(("FONTSIZE",   (0,_gi), (-1,_gi), 9.5))
                        _tbl_style.append(("SPAN",       (0,_gi), (-1,_gi)))
                        _tbl_style.append(("ALIGN",      (0,_gi), (-1,_gi), "LEFT"))
                        _tbl_style.append(("TOPPADDING", (0,_gi), (-1,_gi), 7))
                        _tbl_style.append(("BOTTOMPADDING",(0,_gi),(-1,_gi), 4))
                    bt_tbl.setStyle(TableStyle(_tbl_style))
                    # Wrap table + caption together so they never split
                    story.append(KeepTogether([
                        bt_tbl,
                        Paragraph(
                            "<i>Each row reports a single metric over 1, 3, 5, "
                            "and 10-year windows. Sharpe ratio uses excess return "
                            "over the risk-free rate.</i>",
                            caption,
                        ),
                    ]))
                    story.append(Spacer(1, 0.18*inch))

                    # The 3-year growth-of-$10K chart that previously lived
                    # here was removed in May 2026. Per advisor request, the
                    # chart was redundant with the table immediately above
                    # (which already shows total return, vol, Sharpe, and
                    # drawdown across 1/3/5/10 year windows) and overlapping
                    # lines made comparisons hard to read. The replacement
                    # for visual context is the Notable Market Periods
                    # section, which shows performance across specific
                    # historical event windows rather than a smoothed line.
                else:
                    story.append(Paragraph(
                        "<i>Backtest data unavailable — unable to fetch price history.</i>",
                        caption,
                    ))
        except Exception as _be:
            story.append(Paragraph(
                f"<i>Backtest section unavailable: {_be}</i>",
                caption,
            ))

    # Methodology (formerly Section 3 here) has been moved to the final
    # page, directly above the Disclosures, so the methodology and the
    # SEC Marketing Rule disclosure language read together. See the
    # APPENDIX block below.

    # ═══════════════════════════════════════════════════════════
    # SECTION 4 — NOTABLE MARKET PERIODS
    # ═══════════════════════════════════════════════════════════
    # Replaced the prior Risk Analysis page (drawdown chart / rolling
    # Sharpe / 10yr forward Monte Carlo) with a single-page comparison
    # of how each portfolio (recommended, current, SPY benchmark, BND
    # benchmark) performed across five notable historical event windows.
    # The advisor's reasoning: a static drawdown chart of one portfolio
    # is less informative than a side-by-side cross-portfolio comparison
    # during the moments clients actually remember (2008, COVID, 2022).
    # The 10yr Monte Carlo is being replaced by a proper retirement-goal
    # projection in a later iteration.
    if sections.get("notable_periods", True):
        story.append(PageBreak())
        story.append(section_header("Section 4", "Notable Market Periods"))
        story.append(Paragraph(
            "How each portfolio performed during five notable market events. "
            "Periods cover the full event window (crash + recovery) so the "
            "table reflects real-world investor experience, not just the "
            "peak-to-trough decline. Returns are total returns, gross of "
            "advisory fees. The S&amp;P 500 (SPY) and Aggregate Bond (BND) "
            "are shown as reference benchmarks.",
            body,
        ))
        story.append(Spacer(1, 0.10*inch))

        # Build the portfolio set: recommended (balanced tier), current
        # (from the saved snapshot if present), plus SPY and BND benchmarks.
        # The recommended portfolio comes from the proposal's "balanced"
        # tier — that's the same tier the rest of the PDF shows.
        _bal_tier = (proposal.get("tiers") or {}).get("balanced") or {}
        _bal_tickers = list(_bal_tier.get("tickers") or [])
        _bal_weights = list(_bal_tier.get("weights") or [])

        _curr_snap = (proposal or {}).get("client_current_portfolio") or {}
        _curr_snap_tks = list(_curr_snap.get("tickers") or [])
        _curr_snap_w   = _curr_snap.get("weights") or {}
        _curr_w_list = []
        if _curr_snap_tks and _curr_snap_w and isinstance(_curr_snap_w, dict):
            _curr_w_list = [
                float(_curr_snap_w.get(t, 0) or 0) for t in _curr_snap_tks
            ]

        _np_portfolios = []
        if _bal_tickers and _bal_weights:
            _np_portfolios.append(("Recommended", _bal_tickers, _bal_weights))
        if _curr_snap_tks and _curr_w_list and sum(_curr_w_list) > 0:
            _np_portfolios.append(("Current", _curr_snap_tks, _curr_w_list))
        # Always include the two benchmarks.
        _np_portfolios.append(("S&P 500 (SPY)", ["SPY"], [100.0]))
        _np_portfolios.append(("Agg Bond (BND)", ["BND"], [100.0]))

        try:
            _period_data = _compute_period_returns(_np_portfolios)
        except Exception as _np_err:
            _period_data = None
            story.append(Paragraph(
                f"<i>Notable Market Periods unavailable: {_np_err}</i>",
                body_small,
            ))

        if _period_data:
            # Build the table: rows = periods, cols = portfolios (1 row per
            # period plus a description row beneath each label).
            _period_labels = _period_data["_periods"]
            _portfolio_names = [n for n, *_ in _np_portfolios]

            # Header row: "Period" + portfolio names
            np_rows = [["Period"] + _portfolio_names]

            def _fmt_pct(value):
                if value is None:
                    return "—"
                # Color-code is applied via cell-level styling below
                return f"{value*100:+.1f}%"

            # One row per period — period label on left, returns on right
            for period in NOTABLE_PERIODS:
                _label, _start, _end, _desc = period
                # Show date range under the label
                _label_html = (
                    f"<b>{_label}</b><br/>"
                    f"<font size='7' color='#8b8c89'>{_start[:7]} – {_end[:7]}</font>"
                )
                row = [Paragraph(_label_html, body_small)]
                for pname in _portfolio_names:
                    val = _period_data.get(pname, {}).get(_label)
                    row.append(_fmt_pct(val))
                np_rows.append(row)

            np_tbl = Table(
                np_rows,
                colWidths=[2.0*inch] + [(5.3 / max(1, len(_portfolio_names))) * inch] * len(_portfolio_names),
            )
            # Style: header dark, alternating row backgrounds, color-code
            # the numeric cells based on +/- (green/red). We compute color
            # cell-by-cell so positive returns are clearly differentiated
            # from drawdowns at a glance.
            _np_styles = [
                ("BACKGROUND",   (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0), 9),
                ("FONTSIZE",     (0, 1), (-1, -1), 9),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",        (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN",        (0, 0), (0, -1),  "LEFT"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING",   (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, BG_SOFT]),
                ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
                ("LINEBELOW",    (0, 0), (-1, 0),  1.2, ACCENT),
            ]
            # Color-code positive/negative cells. Skip header row.
            _green = colors.HexColor("#15803d")  # darker green for visibility
            _red   = colors.HexColor("#b91c1c")  # darker red for visibility
            for ri, period in enumerate(NOTABLE_PERIODS, start=1):
                _label, *_ = period
                for ci, pname in enumerate(_portfolio_names, start=1):
                    val = _period_data.get(pname, {}).get(_label)
                    if val is None:
                        continue
                    color = _green if val >= 0 else _red
                    _np_styles.append(("TEXTCOLOR", (ci, ri), (ci, ri), color))
                    _np_styles.append(("FONTNAME",  (ci, ri), (ci, ri), "Helvetica-Bold"))
            np_tbl.setStyle(TableStyle(_np_styles))
            story.append(np_tbl)
            story.append(Spacer(1, 0.15*inch))

            # Brief description block — list each period with what happened
            # so the client (and their advisor's compliance officer) has
            # context for the numbers above. Important: the periods are
            # FULL event windows, not just drawdowns, so investors see the
            # recovery side too.
            story.append(Paragraph("<b>About these periods</b>", h3))
            for label, start, end, desc in NOTABLE_PERIODS:
                story.append(Paragraph(
                    f"&bull; <b>{label}</b> ({start[:7]} – {end[:7]}): {desc}",
                    body_small,
                ))
            story.append(Spacer(1, 0.10*inch))
            story.append(Paragraph(
                "<i>Returns are total returns over the full event window (crash and "
                "subsequent recovery). Past performance does not guarantee future results. "
                "Hypothetical comparisons assume the proposed allocation was held throughout "
                "each window without rebalancing or trading.</i>",
                caption,
            ))
        else:
            # Notable Market Periods table couldn't be built — fall back to
            # the original risk-mitigation strategy table so the section
            # never appears empty.
            risk_rows = [
                ["Risk Factor",        "Mitigation Strategy"],
                ["Market Drawdown",
                 "Diversification across equities, bonds, and cash reduces exposure to any single asset class."],
                ["Interest-Rate Risk",
                 "Bond allocations blend short- and intermediate-duration instruments."],
                ["Inflation Risk",
                 "Equity exposure provides long-run inflation hedge; TIPS may be included based on priorities."],
                ["Liquidity Needs",
                 "Cash reserve sized to priority; all holdings are liquid ETFs/equities."],
                ["Behavioral Risk",
                 "Quarterly rebalancing schedule removes emotional timing decisions from the process."],
            ]
            tbl = Table(risk_rows, colWidths=[1.7*inch, 5.2*inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0), NAVY),
                ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
                ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,-1), 9),
                ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
                ("TEXTCOLOR",    (0,1), (-1,-1), CHARCOAL),
                ("VALIGN",       (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",  (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, BG_SOFT]),
                ("BOX",          (0,0), (-1,-1), 0.5, BORDER),
                ("LINEBELOW",    (0,0), (-1,0), 1.2, ACCENT),
            ]))
            story.append(tbl)

    # Implementation Plan (formerly Section 5 here) has been moved to the
    # second-to-last page, beneath the Advisor signature card. See the
    # ADVISOR INFO + IMPLEMENTATION block further down.

    # ═══════════════════════════════════════════════════════════
    # ADVISOR NOTES
    # ═══════════════════════════════════════════════════════════
    if sections.get("notes", True):
        _notes_block = []
        _notes_block.append(Spacer(1, 0.15*inch))
        _notes_block.append(section_header("Section 6", "Advisor Notes"))
        _notes = (proposal.get("advisor_notes") or "").strip()
        if _notes:
            _notes_block.append(Paragraph(_notes, body))
        else:
            _notes_block.append(Paragraph(
                "<i>No additional notes were attached to this proposal version.</i>",
                body_small,
            ))
        story.append(KeepTogether(_notes_block))

    # ═══════════════════════════════════════════════════════════
    # SECOND-TO-LAST PAGE — Advisor Info (top) + Implementation (below)
    # ═══════════════════════════════════════════════════════════
    # The signature card that previously trailed the Advisor Notes block
    # has been promoted to the top of its own page, with the Implementation
    # Plan moved underneath it. Layout per advisor request: clear "who's
    # accountable" identification visible alongside the action plan, on the
    # page right before the methodology + disclosures.
    _has_photo = os.path.exists(ADVISOR_PHOTO_PATH)
    _show_advisor_card = (_has_photo or _has_any_firm_text)
    _show_implementation = sections.get("proposals", True)

    if _show_advisor_card or _show_implementation:
        story.append(PageBreak())

        if _show_advisor_card:
            story.append(section_header("Section 7", "Your Advisor"))

            if _has_photo:
                try:
                    _photo_flow = Image(ADVISOR_PHOTO_PATH,
                                        width=0.95*inch, height=0.95*inch,
                                        kind="proportional")
                except Exception:
                    _photo_flow = Paragraph("", body_small)
            else:
                _photo_flow = Paragraph("", body_small)

            _sig_label = ParagraphStyle(
                "sig_label", fontSize=7.5, leading=9, textColor=GRAY,
                fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=2,
            )
            _sig_name = ParagraphStyle(
                "sig_name", fontSize=12, leading=15, textColor=NAVY,
                fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=1,
            )
            _sig_role = ParagraphStyle(
                "sig_role", fontSize=9, leading=12, textColor=CHARCOAL,
                fontName="Helvetica", alignment=TA_LEFT, spaceAfter=4,
            )
            _sig_contact = ParagraphStyle(
                "sig_contact", fontSize=8.5, leading=11, textColor=SLATE,
                fontName="Helvetica", alignment=TA_LEFT, spaceAfter=1,
            )

            _sig_right = [Paragraph("YOUR ADVISOR", _sig_label)]
            if _adv_name:
                _sig_right.append(Paragraph(_adv_name, _sig_name))
            _role_bits = []
            if _adv_title: _role_bits.append(_adv_title)
            if _firm_name: _role_bits.append(_firm_name)
            if _role_bits:
                _sig_right.append(Paragraph(" · ".join(_role_bits), _sig_role))
            if _adv_email:
                _sig_right.append(Paragraph(f"📧 &nbsp;{_adv_email}", _sig_contact))
            if _adv_phone:
                _sig_right.append(Paragraph(f"📞 &nbsp;{_adv_phone}", _sig_contact))
            if _firm_website:
                _sig_right.append(Paragraph(f"🌐 &nbsp;{_firm_website}", _sig_contact))

            _sig_card = Table(
                [[_photo_flow, _sig_right]],
                colWidths=[1.15*inch, 6.15*inch],
            )
            _sig_card.setStyle(TableStyle([
                ("VALIGN",       (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",  (0,0), (-1,-1), 12),
                ("RIGHTPADDING", (0,0), (-1,-1), 12),
                ("TOPPADDING",   (0,0), (-1,-1), 12),
                ("BOTTOMPADDING",(0,0), (-1,-1), 12),
                ("BOX",          (0,0), (-1,-1), 0.6, BORDER),
                ("BACKGROUND",   (0,0), (-1,-1), BG_LIGHT),
            ]))
            story.append(KeepTogether([Spacer(1, 0.10*inch), _sig_card]))

        if _show_implementation:
            _impl_block = []
            _impl_block.append(Spacer(1, 0.25*inch))
            _impl_block.append(section_header("Section 8", "Implementation Plan"))
            impl_rows = [
                ["Stage", "Cadence", "Action"],
                ["Initial Funding", "Day 0",
                 "Fund account and execute initial allocation per selected option."],
                ["First Review", "30 days",
                 "Confirm execution, verify holdings match proposal."],
                ["Rebalancing", "Quarterly",
                 "Drift threshold 5% per position; tax-aware rebalancing where applicable."],
                ["Performance Review", "Semi-Annual",
                 "Review against benchmarks; discuss changes in goals."],
                ["Full Re-Assessment", "Annual",
                 "Update risk profile; refresh proposal if score or goals change."],
            ]
            tbl = Table(impl_rows, colWidths=[1.5*inch, 1.1*inch, 4.3*inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0), NAVY),
                ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
                ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,-1), 9),
                ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
                ("TEXTCOLOR",    (0,1), (0,-1), NAVY_MID),
                ("TEXTCOLOR",    (1,1), (-1,-1), CHARCOAL),
                ("VALIGN",       (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",  (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, BG_SOFT]),
                ("BOX",          (0,0), (-1,-1), 0.5, BORDER),
                ("LINEBELOW",    (0,0), (-1,0), 1.2, ACCENT),
            ]))
            _impl_block.append(tbl)
            story.append(KeepTogether(_impl_block))

    # ═══════════════════════════════════════════════════════════
    # OPTIONAL — FEE COMPARISON TABLE (own page)
    # ═══════════════════════════════════════════════════════════
    # When the advisor ticks "Fee comparison table" in the section picker,
    # render the table on its own page so the client can study it without
    # the visual weight of disclosure paragraphs around it. The table
    # itself stays in the disclosures section as well (smaller, tighter)
    # for compliance — this version is the larger comparison-focused one.
    if sections.get("fee_comparison", False):
        story.append(PageBreak())
        story.append(section_header("Section 9", "Fee Comparison"))

        _adv_fee_pct_for_compare = _resolve_advisory_fee_pct(
            proposal, client_profile, _firm_settings,
        )

        story.append(Paragraph(
            f"This table illustrates how different annual advisory fee levels "
            f"affect the growth of a hypothetical $100 starting balance over "
            f"common time horizons. Your firm's fee of "
            f"<b>{_adv_fee_pct_for_compare:.2f}%</b> is shown alongside "
            f"industry benchmark levels (0%, 0.75%, 1%, 1.5%, 2%, 2.5%) "
            f"so you can compare. All figures assume a 7% gross annual return "
            f"compounded monthly. Actual returns will differ.",
            body,
        ))
        story.append(Spacer(1, 0.15*inch))

        # Build a fee table that includes the firm's actual rate alongside
        # the standard benchmarks. Inserts the firm rate in sorted order so
        # the table reads as a smooth gradient. If the firm rate matches a
        # benchmark exactly (e.g. 1.00%), no duplicate is inserted.
        _benchmark_fees = [0.0, 0.75, 1.0, 1.5, 2.0, 2.5]
        _all_fees = sorted(set(_benchmark_fees + [round(_adv_fee_pct_for_compare, 2)]))
        _fee_rows_full = _fee_impact_table_data(fee_levels=_all_fees)

        # Highlight the firm's row by inserting an asterisk marker into the
        # fee column. ReportLab will pick it up via Paragraph rendering;
        # for plain string cells we just append " ←" so the row stands out.
        _firm_fee_str = f"{_adv_fee_pct_for_compare:.2f}%"
        for i, row in enumerate(_fee_rows_full):
            if i > 0 and row[0] == _firm_fee_str:
                row[0] = f"{_firm_fee_str}  ★"

        _fee_tbl_full = Table(
            _fee_rows_full,
            colWidths=[0.95*inch] + [1.05*inch] * (len(_fee_rows_full[0]) - 1),
        )
        _fee_tbl_full.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN",         (0, 0), (-1, -1), "RIGHT"),
            ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, BG_SOFT]),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.2, ACCENT),
        ]))
        story.append(_fee_tbl_full)
        story.append(Spacer(1, 0.10*inch))
        story.append(Paragraph(
            "<i>★ marks your firm's actual fee. Performance figures are "
            "hypothetical and for illustration only. Past performance does "
            "not guarantee future results.</i>",
            caption,
        ))

    # ═══════════════════════════════════════════════════════════
    # FINAL PAGE — Methodology + Disclosures
    # ═══════════════════════════════════════════════════════════
    story.append(PageBreak())

    # ── Methodology (final page top) ────────────────────────────
    # Moved here from its prior position between Historical and Risk
    # Analysis so it reads alongside the disclosure language. Keeps the
    # client's eye on "how was this built and what are the limitations"
    # in one place rather than spread across the document.
    story.append(section_header("Methodology", "How This Proposal Was Built"))
    story.append(Paragraph(
        "Each recommended portfolio is constructed using institutional-grade "
        "optimization techniques. Allocations are informed by risk-score-targeted "
        "equity/bond/cash splits with priority-driven tilts applied on top.",
        body,
    ))
    story.append(Paragraph(
        "&bull; <b>Risk-targeted base allocation</b> — a mapping from the client's "
        "1-99 risk score to a target equity / bond / cash split forms the starting point.<br/>"
        "&bull; <b>Priority tilts</b> — client-stated goals (e.g. capital preservation, "
        "income, social impact) adjust both the asset-class mix and the ticker universe.<br/>"
        "&bull; <b>Holdings selection</b> — where possible, proposals use the client's "
        "own submitted securities; gaps are filled with broadly-diversified index ETFs.",
        body,
    ))
    story.append(Spacer(1, 0.20*inch))

    # ── Disclosures (final page below methodology) ──────────────
    story.append(section_header("Disclosures", "Important Information"))

    story.append(Paragraph("Key Definitions", h3))
    glossary = [
        ["Risk Number",
         "Integer 1-99 summarizing combined risk tolerance (willingness) and capacity (ability)."],
        ["Equity / Bond / Cash",
         "High-level split across growth, stability, and liquidity objectives."],
        ["Sharpe Ratio",
         "Return per unit of total volatility. Values above 1.0 indicate strong risk-adjusted performance."],
        ["Maximum Drawdown",
         "Largest peak-to-trough decline over the measurement period."],
        ["Priority Tilt",
         "Adjustment to base allocation based on client-stated goals."],
    ]
    gl = Table(glossary, colWidths=[1.5*inch, 5.4*inch])
    gl.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,0), (0,-1),  NAVY),
        ("TEXTCOLOR",     (1,0), (-1,-1), CHARCOAL),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-1), 0.4, BORDER_SOFT),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
    ]))
    story.append(gl)

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Important Performance Disclosures", h3))

    # Resolve the advisory fee using the fallback chain so the disclosure
    # accurately reflects what the client is being charged. If nothing is
    # configured anywhere, falls back to 1.00% (a typical small-RIA AUM
    # fee) — but the firm default SHOULD be set in firm_settings so the
    # disclosure matches the actual ADV.
    _adv_fee_pct = _resolve_advisory_fee_pct(proposal, client_profile, _firm_settings)

    story.append(Paragraph(
        "<b>Past performance is no guarantee of future results.</b> "
        "Investment return and principal value of an investment will fluctuate; "
        "therefore, you may have a gain or loss when you sell your shares. "
        "Current performance may be higher or lower than the performance data quoted.",
        body_small,
    ))
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph(
        "<b>Net of Fees Performance.</b> Performance figures shown in this "
        "report reflect the underlying funds' net expense ratios but are "
        f"<b>gross of advisory fees</b>. Your actual return would be reduced "
        f"by the firm's advisory fee of <b>{_adv_fee_pct:.2f}%</b> per year, "
        "as well as any brokerage commissions, custodial costs, and other "
        f"expenses. {'See the Fee Comparison section above for a side-by-side illustration of the impact at common fee levels.' if sections.get('fee_comparison', False) else 'A fee comparison table illustrating the impact at common fee levels is available on request.'}",
        body_small,
    ))
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph(
        "<b>Hypothetical and Backtested Data.</b> Where the analysis includes "
        "performance for portfolio combinations the client did not actually "
        "hold during the measurement period, those results are <b>hypothetical "
        "and backtested</b>. Such results are achieved by retroactively applying "
        "a model to historical data and do not represent actual trading. "
        "Hypothetical performance has inherent limitations: it does not reflect "
        "the impact that material economic and market factors might have had "
        "on an advisor's decision-making if the advisor were actually managing "
        "client money. Forward-looking projections (e.g., Monte Carlo) are "
        "estimates, not guarantees.",
        body_small,
    ))
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph(
        "<b>Benchmark Comparisons.</b> Where a benchmark portfolio (such as "
        "the S&amp;P 500 Index, a Schwab Core ETF model, or another reference) "
        "is shown for comparative purposes, it is illustrative only. Indexes "
        "are unmanaged; you cannot invest directly in an index, although you "
        "may be able to invest in a fund that tracks an index. Benchmark "
        "performance reflects only the underlying index or model and not the "
        "deduction of advisory fees.",
        body_small,
    ))
    story.append(Spacer(1, 0.08*inch))

    story.append(Paragraph(
        "<b>Limitations and Risk.</b> All investing involves risk, including "
        "possible loss of principal. Diversification and asset allocation do "
        "not guarantee a profit or protect against loss. This report is "
        "informational only and does not constitute tax, legal, or accounting "
        "advice. It is intended for use in a one-on-one discussion with your "
        "advisor so that you can ask questions and fully understand the analysis.",
        body_small,
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(thin_rule(BORDER, 0.5))
    story.append(Paragraph(
        f"Proposal <b>{proposal.get('version_id','—')}</b> &nbsp;·&nbsp; "
        f"Prepared for <b>{client_profile.get('client_name','—')}</b> &nbsp;·&nbsp; "
        f"Generated <b>{_dt.now().strftime('%B %d, %Y at %I:%M %p')}</b>",
        caption,
    ))

    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.getvalue()


# (generate_pdf_report removed May 2026 — Save PDF Report panel was its only consumer.)


# ── RISK SCORING ──────────────────────────────────────────────
@functools.lru_cache(maxsize=2048)
def compute_risk_score(ann_vol, max_drawdown, sharpe, ticker=None,
                       duration=None, asset_class=None, credit_tier=None):
    """Score from 1 (very low risk) to 99 (very high risk).

    Wrapper around shared.compute_risk_score that injects this file's
    `_classify_ticker` for auto-classification when asset_class is not
    supplied. The scoring math itself lives in shared.py — both this app
    and risk_assessment.py call into the same function so the same security
    gets the same score in both views.
    """
    return _shared_compute_risk_score(
        ann_vol, max_drawdown, sharpe,
        ticker=ticker,
        duration=duration,
        asset_class=asset_class,
        credit_tier=credit_tier,
        classifier=_classify_ticker,
    )


# ─────────────────────────────────────────────────────────────────────
# CURATED MUTUAL FUND TABLE
# ─────────────────────────────────────────────────────────────────────
# Common active mutual funds and index MFs from the major fund families
# (Vanguard, Fidelity, American Funds, T. Rowe Price, PIMCO, DFA).
# Single source of truth for both classification AND expense ratio
# fallback — keeps the two in sync.
#
# Format: ticker → (asset_class, credit_tier, expense_ratio_decimal)
#   asset_class ∈ {"cash","bond","equity","balanced"}
#     - "balanced" funds are split downstream by their stated equity/bond mix
#   credit_tier ∈ {"govt","ig","hy","em"} — only meaningful for bonds
#   expense_ratio is a decimal (e.g. 0.0025 = 25 bps)
#
# yfinance covers the long tail (see _classify_ticker fallback). This
# table just front-loads the most common funds your clients hold so we
# don't burn a network call on every render.
#
# Last validated: April 30, 2026. Source: each fund family's website +
# SEC 497 filings.
# ─────────────────────────────────────────────────────────────────────
_MUTUAL_FUND_TABLE = {
    # ── Vanguard active equity ──
    "VWELX": ("balanced", "ig",   0.0025),  # Wellington (Investor)
    "VWENX": ("balanced", "ig",   0.0017),  # Wellington (Admiral)
    "VWINX": ("balanced", "ig",   0.0023),  # Wellesley Income (Investor)
    "VWIAX": ("balanced", "ig",   0.0016),  # Wellesley Income (Admiral)
    "VPMCX": ("equity",   "govt", 0.0031),  # PRIMECAP
    "VPMAX": ("equity",   "govt", 0.0031),  # PRIMECAP (Admiral)
    "VWNDX": ("equity",   "govt", 0.0029),  # Windsor (Investor)
    "VWNEX": ("equity",   "govt", 0.0019),  # Windsor (Admiral)
    "VGHCX": ("equity",   "govt", 0.0032),  # Health Care
    "VGHAX": ("equity",   "govt", 0.0028),  # Health Care (Admiral)
    "VGENX": ("equity",   "govt", 0.0033),  # Energy
    "VEXPX": ("equity",   "govt", 0.0044),  # Explorer
    "VEXRX": ("equity",   "govt", 0.0030),  # Explorer (Admiral)
    "VGSTX": ("balanced", "ig",   0.0031),  # STAR
    "VWIGX": ("equity",   "govt", 0.0042),  # International Growth
    "VDIGX": ("equity",   "govt", 0.0029),  # Dividend Growth
    "VHCOX": ("equity",   "govt", 0.0040),  # Capital Opportunity
    "VHCAX": ("equity",   "govt", 0.0035),  # Capital Opportunity (Admiral)
    "VASVX": ("equity",   "govt", 0.0033),  # Selected Value
    "VEIPX": ("equity",   "govt", 0.0028),  # Equity Income
    "VEIRX": ("equity",   "govt", 0.0019),  # Equity Income (Admiral)
    "VMGRX": ("equity",   "govt", 0.0034),  # Mid-Cap Growth
    "VTCLX": ("equity",   "govt", 0.0009),  # Tax-Managed Capital Appreciation
    # ── Vanguard active bond ──
    "VFIIX": ("bond", "govt", 0.0019),  # GNMA
    "VFIJX": ("bond", "govt", 0.0010),  # GNMA (Admiral)
    "VWEHX": ("bond", "hy",   0.0023),  # High-Yield Corporate
    "VWEAX": ("bond", "hy",   0.0013),  # High-Yield Corporate (Admiral)
    "VWESX": ("bond", "ig",   0.0022),  # Long-Term Investment-Grade
    "VWETX": ("bond", "ig",   0.0012),  # Long-Term Investment-Grade (Admiral)
    "VFICX": ("bond", "ig",   0.0020),  # Intermediate-Term Investment-Grade
    "VFIDX": ("bond", "ig",   0.0010),  # Intermediate-Term IG (Admiral)
    "VFSTX": ("bond", "ig",   0.0020),  # Short-Term Investment-Grade
    "VFSUX": ("bond", "ig",   0.0010),  # Short-Term IG (Admiral)
    "VFITX": ("bond", "govt", 0.0020),  # Intermediate-Term Treasury
    "VFIUX": ("bond", "govt", 0.0010),  # Intermediate-Term Treasury (Admiral)
    "VUSTX": ("bond", "govt", 0.0020),  # Long-Term Treasury
    "VUSUX": ("bond", "govt", 0.0010),  # Long-Term Treasury (Admiral)
    "VFISX": ("bond", "govt", 0.0020),  # Short-Term Treasury
    "VFIRX": ("bond", "govt", 0.0010),  # Short-Term Treasury (Admiral)
    # ── Vanguard index MF (mutual fund share classes — ETFs handled elsewhere) ──
    "VFIAX": ("equity", "govt", 0.0004),  # 500 Index (Admiral)
    "VFINX": ("equity", "govt", 0.0014),  # 500 Index (Investor — discontinued, legacy holdings)
    "VTSAX": ("equity", "govt", 0.0004),  # Total Stock Market (Admiral)
    "VTSMX": ("equity", "govt", 0.0014),  # Total Stock Market (Investor — legacy)
    "VTIAX": ("equity", "govt", 0.0009),  # Total International Stock (Admiral)
    "VBTLX": ("bond",   "ig",   0.0005),  # Total Bond Market (Admiral)
    "VBMFX": ("bond",   "ig",   0.0015),  # Total Bond Market (Investor — legacy)
    "VTABX": ("bond",   "ig",   0.0011),  # Total International Bond (Admiral)
    "VFWAX": ("equity", "govt", 0.0010),  # FTSE All-World ex-US (Admiral)
    "VEMAX": ("equity", "em",   0.0014),  # Emerging Markets (Admiral)
    "VEUSX": ("equity", "govt", 0.0011),  # European Stock (Admiral)
    "VPACX": ("equity", "govt", 0.0011),  # Pacific Stock (Admiral)
    "VGSLX": ("equity", "govt", 0.0013),  # REIT Index (Admiral)
    "VSMAX": ("equity", "govt", 0.0005),  # Small-Cap Index (Admiral)
    "VEXAX": ("equity", "govt", 0.0006),  # Extended Market Index (Admiral)
    "VIMAX": ("equity", "govt", 0.0005),  # Mid-Cap Index (Admiral)
    "VVIAX": ("equity", "govt", 0.0005),  # Value Index (Admiral)
    "VIGAX": ("equity", "govt", 0.0005),  # Growth Index (Admiral)
    "VTAPX": ("bond",   "govt", 0.0007),  # Short-Term Inflation-Protected (Admiral)
    "VAIPX": ("bond",   "govt", 0.0010),  # Inflation-Protected Securities (Admiral)
    # ── Vanguard LifeStrategy / Target Retirement ──
    "VASIX": ("balanced", "ig", 0.0011),  # LifeStrategy Income
    "VSCGX": ("balanced", "ig", 0.0012),  # LifeStrategy Conservative Growth
    "VSMGX": ("balanced", "ig", 0.0013),  # LifeStrategy Moderate Growth
    "VASGX": ("balanced", "ig", 0.0014),  # LifeStrategy Growth
    "VTINX": ("balanced", "ig", 0.0008),  # Target Retirement Income
    "VTTVX": ("balanced", "ig", 0.0008),  # Target Retirement 2025
    "VTHRX": ("balanced", "ig", 0.0008),  # Target Retirement 2030
    "VTTHX": ("balanced", "ig", 0.0008),  # Target Retirement 2035
    "VFORX": ("balanced", "ig", 0.0008),  # Target Retirement 2040
    "VTIVX": ("balanced", "ig", 0.0008),  # Target Retirement 2045
    "VFIFX": ("balanced", "ig", 0.0008),  # Target Retirement 2050
    "VFFVX": ("balanced", "ig", 0.0008),  # Target Retirement 2055
    "VTTSX": ("balanced", "ig", 0.0008),  # Target Retirement 2060
    "VLXVX": ("balanced", "ig", 0.0008),  # Target Retirement 2065
    # ── Fidelity active equity ──
    "FCNTX": ("equity", "govt", 0.0039),  # Contrafund
    "FCNKX": ("equity", "govt", 0.0055),  # Contrafund (Class K)
    "FMAGX": ("equity", "govt", 0.0044),  # Magellan
    "FLPSX": ("equity", "govt", 0.0055),  # Low-Priced Stock
    "FDGRX": ("equity", "govt", 0.0073),  # Growth Company
    "FBGRX": ("equity", "govt", 0.0050),  # Blue Chip Growth
    "FDIVX": ("equity", "govt", 0.0058),  # Diversified International
    "FAGIX": ("bond",   "hy",   0.0064),  # Capital & Income
    "SPHIX": ("bond",   "hy",   0.0067),  # High Income
    "FPURX": ("balanced","ig",  0.0049),  # Puritan
    "FBALX": ("balanced","ig",  0.0050),  # Balanced
    # ── Fidelity bond ──
    "FTBFX": ("bond", "ig",   0.0044),  # Total Bond
    "FBNDX": ("bond", "ig",   0.0044),  # Investment Grade Bond
    "FXNAX": ("bond", "ig",   0.0003),  # US Bond Index
    "FUAMX": ("bond", "govt", 0.0003),  # Intermediate Treasury Index
    "FUMBX": ("bond", "govt", 0.0003),  # Short-Term Treasury Index
    # ── Fidelity index MFs ──
    "FXAIX": ("equity", "govt", 0.00015),  # 500 Index
    "FSKAX": ("equity", "govt", 0.00015),  # Total Market Index
    "FZROX": ("equity", "govt", 0.0000),   # Zero Total Market
    "FZILX": ("equity", "govt", 0.0000),   # Zero International
    "FNILX": ("equity", "govt", 0.0000),   # Zero Large Cap
    "FZIPX": ("equity", "govt", 0.0000),   # Zero Extended Market
    "FTIHX": ("equity", "govt", 0.0006),   # Total International Index
    "FSPSX": ("equity", "govt", 0.00035),  # International Index
    "FSMDX": ("equity", "govt", 0.00025),  # Mid Cap Index
    "FSSNX": ("equity", "govt", 0.00025),  # Small Cap Index
    "FNCMX": ("equity", "govt", 0.0029),   # NASDAQ Composite Index
    "FSPGX": ("equity", "govt", 0.00035),  # Large Cap Growth Index
    "FSPHX": ("equity", "govt", 0.0067),   # Select Health Care
    "FREL":  ("equity", "govt", 0.0008),   # Real Estate Index (this one's an ETF, included for completeness)
    # ── American Funds (very common in 401(k)s; A-share classes shown — F/R shares cheaper) ──
    "AGTHX": ("equity",   "govt", 0.0061),  # Growth Fund of America (A)
    "AIVSX": ("equity",   "govt", 0.0057),  # Investment Co. of America (A)
    "AWSHX": ("equity",   "govt", 0.0057),  # Washington Mutual (A)
    "CAIBX": ("balanced", "ig",   0.0057),  # Capital Income Builder (A)
    "AMECX": ("balanced", "ig",   0.0055),  # Income Fund of America (A)
    "ABALX": ("balanced", "ig",   0.0054),  # American Balanced (A)
    "ANCFX": ("equity",   "govt", 0.0058),  # Fundamental Investors (A)
    "CWGIX": ("equity",   "govt", 0.0073),  # Capital World G&I (A)
    "AEPGX": ("equity",   "govt", 0.0080),  # EuroPacific Growth (A)
    "AMCPX": ("equity",   "govt", 0.0067),  # AMCAP (A)
    "ANEFX": ("equity",   "govt", 0.0072),  # New Economy (A)
    "ANWPX": ("equity",   "govt", 0.0073),  # New Perspective (A)
    "SMCWX": ("equity",   "govt", 0.0068),  # SMALLCAP World (A)
    "AHITX": ("bond",     "hy",   0.0067),  # American High-Income Trust (A)
    "ABNDX": ("bond",     "ig",   0.0057),  # Bond Fund of America (A)
    "AMBFX": ("bond",     "ig",   0.0057),  # American Balanced (A) (alt ticker)
    # ── T. Rowe Price ──
    "PRGFX": ("equity",   "govt", 0.0064),  # Growth Stock
    "PRDGX": ("equity",   "govt", 0.0063),  # Dividend Growth
    "TRBCX": ("equity",   "govt", 0.0070),  # Blue Chip Growth
    "PRWCX": ("balanced", "ig",   0.0069),  # Capital Appreciation
    "PRHSX": ("equity",   "govt", 0.0074),  # Health Sciences
    "TRREX": ("equity",   "govt", 0.0074),  # Real Estate
    "PRNHX": ("equity",   "govt", 0.0075),  # New Horizons
    "PRFDX": ("equity",   "govt", 0.0066),  # Equity Income
    "PRMTX": ("equity",   "govt", 0.0078),  # Media & Telecommunications
    "TROSX": ("equity",   "govt", 0.0083),  # Overseas Stock
    "PRFHX": ("bond",     "hy",   0.0072),  # High Yield
    "PRPFX": ("balanced", "ig",   0.0078),  # Permanent Portfolio (3rd-party, but commonly held)
    # ── PIMCO ──
    "PIMIX": ("bond", "ig", 0.0050),  # Income (Institutional)
    "PONAX": ("bond", "ig", 0.0085),  # Income (A)
    "PTTAX": ("bond", "ig", 0.0085),  # Total Return (A)
    "PTTRX": ("bond", "ig", 0.0046),  # Total Return (Institutional)
    "PRRIX": ("bond", "ig", 0.0046),  # Real Return (Institutional)
    "PRTNX": ("bond", "ig", 0.0091),  # Real Return (A)
    "PHIYX": ("bond", "hy", 0.0055),  # High Yield (Institutional)
    # ── DFA (advisor channel — primarily institutional class) ──
    "DFLVX": ("equity", "govt", 0.0026),  # US Large Cap Value
    "DFSVX": ("equity", "govt", 0.0030),  # US Small Cap Value
    "DFEVX": ("equity", "em",   0.0042),  # Emerging Markets Value
    "DFEMX": ("equity", "em",   0.0034),  # Emerging Markets
    "DFIEX": ("equity", "govt", 0.0026),  # International Core Equity
    "DFISX": ("equity", "govt", 0.0035),  # International Small Company
    "DFIVX": ("equity", "govt", 0.0042),  # International Value
    "DFGFX": ("bond",   "ig",   0.0021),  # Two-Year Global Fixed Income
}


def _mutual_fund_lookup(ticker):
    """Return (asset_class, credit_tier, expense_ratio) for a mutual fund
    ticker, or None if not in the curated table.
    """
    return _MUTUAL_FUND_TABLE.get((ticker or "").upper().strip())


@functools.lru_cache(maxsize=4096)
def _classify_ticker(ticker):
    """Return (asset_class, credit_tier) for a ticker.

    asset_class ∈ {"cash","bond","equity","balanced","crypto_btc",
                   "crypto_alt","leveraged"}
    credit_tier ∈ {"govt","ig","hy","em"}  (only meaningful for bonds)

    Priority order:
      1. Crypto / leveraged hardcoded sets
      2. Cash, govt-bond, IG-bond, HY-bond, EM-bond hardcoded sets
      3. Curated mutual fund table (_MUTUAL_FUND_TABLE)
      4. yfinance Ticker.info fallback (categorizes long-tail MFs)
      5. Default: ("equity", "govt")

    LRU-cached since this is a pure function with no I/O for steps 1-3
    and gets called from many tight loops (per-holding scoring, PCM
    rendering, PDF tables). Step 4 uses a network call but is cached.
    """
    t = (ticker or "").upper().strip()

    if t in ("BTC-USD", "BTC", "GBTC", "IBIT", "FBTC", "ARKB", "BITX",
             "BITO", "BITQ", "BITS"):
        return ("crypto_btc", "govt")
    if t.endswith("-USD") or t in ("ETH","SOL","XRP","ADA","DOGE","SHIB",
                                     "AVAX","DOT","LINK","MATIC","LTC","BCH",
                                     "ATOM","TRX","ETC","XLM","FIL","NEAR","APE"):
        return ("crypto_alt", "govt")

    LEVERAGED = {
        "TQQQ","SQQQ","UPRO","SPXU","SPXL","SPXS","UDOW","SDOW","TNA","TZA",
        "FAS","FAZ","SOXL","SOXS","LABU","LABD","CURE","TECL","TECS","DRN",
        "DRV","ERX","ERY","NUGT","DUST","JNUG","JDST","BOIL","KOLD","UCO",
        "SCO","UVXY","SVXY","VIXY","VXX","SSO","SDS","QLD","QID","DDM","DXD",
        "USD","SSG","ROM","REW","DIG","DUG","UYM","SMN","UYG","SKF","UCC","SCC",
        "RXL","RXD","UPW","SDP","UVT","SZK","UVV","SVZ",
        "TMF","TMV","UBT","TBT","PST","TBF","TYO","DLBR","DLBS",
    }
    if t in LEVERAGED:
        return ("leveraged", "govt")
    if any(s in t for s in ("3X", "2X")) and len(t) <= 6:
        return ("leveraged", "govt")

    CASH = {"SGOV","BIL","SHV","ICSH","JPST","FLOT","USFR","NEAR","GBIL",
            "VMFXX","VUSXX","SPAXX","FZDXX","SWVXX","CASH","MMF"}
    if t in CASH:
        return ("cash", "govt")

    GOVT_BONDS = {
        "SHY","IEF","TLT","TLH","IEI","GOVT","VGSH","VGIT","VGLT","SCHO",
        "SCHR","SPTI","SPTL","SPTS","EDV","ZROZ","TIP","TIPX","SCHP","VTIP",
        "STIP","ITPS","LTPZ","TBIL","VRIG",
    }
    if t in GOVT_BONDS:
        return ("bond", "govt")

    IG_BONDS = {
        "AGG","BND","BNDX","BIV","BLV","BSV","SCHZ","TOTL","FBND","JAGG",
        "LQD","VCIT","VCSH","VCLT","SLQD","IGSB","IGIB","IGLB","SUSC","USIG",
        "USRT","SCHJ","MUB","TFI","VTEB","ITM","SUB","SHM","HYD","HYMB",
    }
    if t in IG_BONDS:
        return ("bond", "ig")

    HY_BONDS = {
        "HYG","JNK","SHYG","SJNK","ANGL","FALN","HYLB","USHY","HYS","SRLN",
        "BKLN","FTSL","SLRN","BSJP","BSJQ","BSJR","BSJS","BSJT","HYDB",
    }
    if t in HY_BONDS:
        return ("bond", "hy")

    EM_BONDS = {
        "EMB","EMLC","CEMB","EBND","EMHY","PCY","VWOB","EMAG","FEMB",
    }
    if t in EM_BONDS:
        return ("bond", "em")

    # Curated mutual fund table — covers the most common active and
    # index MFs from Vanguard, Fidelity, American Funds, T. Rowe Price,
    # PIMCO, and DFA. Drop the expense-ratio component; classifier only
    # needs (asset_class, credit_tier).
    _mf = _mutual_fund_lookup(t)
    if _mf is not None:
        return (_mf[0], _mf[1])

    # yfinance fallback for long-tail mutual funds NOT in the curated
    # table. Reads `Ticker.info` for the fund's stated category /
    # quoteType and classifies accordingly. Network call, but the
    # result is cached by the @lru_cache decorator on this function.
    # Wrapped in try/except so any failure falls through to the
    # equity default — same behavior as before this fallback existed.
    try:
        import yfinance as _yf_cls
        _info = _yf_cls.Ticker(t).info or {}
        _qt = (_info.get("quoteType") or "").upper()
        # Only run categorization for actual mutual funds. ETFs and
        # equities should already be caught by hardcoded sets above
        # or fall through to the equity default cleanly.
        if _qt in ("MUTUALFUND", "MONEYMARKET"):
            _cat = (_info.get("category") or "").lower()
            _legal = (_info.get("legalType") or "").lower()
            # Money market funds → cash
            if _qt == "MONEYMARKET" or "money market" in _cat:
                return ("cash", "govt")
            # Bond categorization by Morningstar category strings
            if any(_kw in _cat for _kw in (
                "bond", "fixed income", "treasury", "tips",
                "tax-free", "muni", "credit", "ultrashort",
            )):
                # Subdivide by credit
                if any(_kw in _cat for _kw in ("high yield", "junk")):
                    return ("bond", "hy")
                if any(_kw in _cat for _kw in ("emerging", "international",
                                                "global bond")):
                    return ("bond", "em")
                if any(_kw in _cat for _kw in ("government", "treasury",
                                                "tips", "agency")):
                    return ("bond", "govt")
                # Default bond → IG
                return ("bond", "ig")
            # Balanced / allocation funds
            if any(_kw in _cat for _kw in (
                "allocation", "balanced", "target", "lifestyle",
                "world allocation",
            )):
                return ("balanced", "ig")
            # Otherwise treat as equity (most non-bond MFs are)
            return ("equity", "govt")
    except Exception:
        # Network failure or yfinance API change — fall through to default.
        pass

    return ("equity", "govt")


def compute_portfolio_risk_score(holdings, weights, holding_scores=None,
                                 holding_vols=None, portfolio_vol=None):
    """Portfolio score = weighted-avg of holding scores − correlation adj.

    `holding_scores` should be a list of pre-computed per-ticker scores.
    `holding_vols` and `portfolio_vol` (decimal) enable diversification adj:
        div_ratio = portfolio_vol / sum(w_i × vol_i)
        adj       = (1 − div_ratio) × 10
        final     = base_avg − adj
    """
    if not holdings or not weights:
        return 50
    w = np.array([float(x or 0) for x in weights], dtype=float)
    total = w.sum()
    if total <= 0:
        return 50
    w = w / total

    if holding_scores is None:
        holding_scores = []
        for tk in holdings:
            r = security_risk_score(tk)
            holding_scores.append(r["score"] if r and "score" in r else 50)
    base = float(np.dot(w, np.array(holding_scores, dtype=float)))

    div_ratio = 1.0
    if holding_vols is not None and portfolio_vol is not None:
        weighted_sum_vol = float(np.dot(w, np.array(holding_vols, dtype=float)))
        if weighted_sum_vol > 1e-6:
            div_ratio = max(0.3, min(1.0, portfolio_vol / weighted_sum_vol))

    final = base - (1.0 - div_ratio) * 10.0
    return int(round(min(99, max(1, final))))


def risk_label(score):
    if score <= 15:  return "🟢 Very Low"
    elif score <= 35: return "🟢 Low"
    elif score <= 50: return "🟡 Moderate"
    elif score <= 65: return "🟠 Moderately High"
    elif score <= 80: return "🔴 High"
    else:             return "🔴 Very High"

def risk_color(score):
    if score <= 15:   return "#4ade80"
    elif score <= 35: return "#86efac"
    elif score <= 50: return "#fde047"
    elif score <= 65: return "#fb923c"
    elif score <= 80: return "#f87171"
    else:             return "#dc2626"

@st.cache_data(ttl=3600, show_spinner=False)
def security_risk_score(ticker):
    """Compute annualized vol and max drawdown for a single ticker over 3 years.

    Cached for 1 hour: this function gets called many times per render (PCM
    rows, optimizer tier gauges, PDF holdings table) for the same tickers.
    Caching here eliminates 80%+ of redundant price fetches and makes the
    optimizer tab feel snappy after the first load.
    """
    try:
        end_dt   = date.today()
        start_dt = end_dt - relativedelta(years=3)

        # get_prices returns a DataFrame with ticker symbol as the column name
        # (NOT "Close", and NOT a MultiIndex — that's flattened in get_prices).
        # The previous version had unreachable MultiIndex/"Close" branches that
        # only worked by accident via the iloc[:,0] fallback.
        raw, _src = get_prices([ticker], start_dt, end_dt)
        if raw is None or raw.empty:
            return None

        # Pick the ticker's column; defensive fallback to first column if missing
        if ticker in raw.columns:
            hist = raw[ticker].dropna()
        else:
            hist = raw.iloc[:, 0].dropna()

        if isinstance(hist, pd.DataFrame):
            hist = hist.squeeze()
        if len(hist) < 60:
            return None

        rets  = hist.pct_change().dropna()
        ann_v = float(rets.std() * np.sqrt(252))
        if ann_v == 0:
            return None
        cum   = (1 + rets).cumprod()
        dd    = float((cum / cum.cummax() - 1).min())
        # CAGR over the realized window, not arithmetic-mean × 252
        actual_yrs = max(len(rets) / 252.0, 0.08)
        tr    = float(cum.iloc[-1] - 1)
        ann_r = (1 + tr) ** (1.0 / actual_yrs) - 1
        sh    = _shared_sharpe(ann_r, ann_v)
        score = compute_risk_score(ann_v, dd, sh, ticker=ticker)
        return {
            "score":   score,
            "label":   risk_label(score),
            "ann_vol": ann_v,
            "max_dd":  dd,
            "sharpe":  sh,
        }
    except Exception:
        return None


def fetch_benchmark_returns(benchmark_ticker, years):
    """Fetch benchmark returns aligned to test period."""
    try:
        end_dt   = date.today()
        start_dt = end_dt - relativedelta(years=years)
        prices, _src = get_prices([benchmark_ticker], start_dt, end_dt)
        prices = prices[benchmark_ticker] if benchmark_ticker in prices.columns else prices.iloc[:,0]
        prices = prices.dropna()
        if len(prices) < 60:
            return None
        rets = prices_to_returns(prices.to_frame())
        _, X_test = train_test_split(rets, test_size=0.33, shuffle=False)
        return X_test.iloc[:, 0].tolist()
    except Exception:
        return None


# (Removed: make_gauge_chart — unused. Risk scores are displayed in the
# Portfolio Comparison Matrix table, not as separate gauge widgets.)


# ── SHARED PIE CHART HELPER ────────────────────────────────────────────
# Rainbow spectrum palette — cycles through the color wheel like the reference
# image. Colors flow smoothly adjacent to each other on a color wheel so
# neighboring slices feel like a natural gradient around the pie.
PIE_PALETTE = [
    "#e6194B",  # bright red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#dcbeff",  # lavender
    "#aaffc3",  # mint
    "#808000",  # olive
    "#ffd8b1",  # apricot
    "#000075",  # navy
    "#a9a9a9",  # gray
    "#ffffff",  # white
    "#000000",  # black
]


def make_risk_gauge(risk_score, height=140):
    """Compact standalone risk-score gauge (1-99 scale) with colored zones.

    Returns a Plotly Figure with a single Indicator trace. Designed to be
    rendered next to (or above) an allocation pie chart via st.plotly_chart.
    """
    _rs = int(max(1, min(99, risk_score)))
    # Color the needle based on score zone
    if _rs <= 33:   _needle_color = "#00D95F"   # emerald — conservative
    elif _rs <= 66: _needle_color = "#FFB800"   # amber — balanced
    else:           _needle_color = "#FF3B30"   # red — aggressive

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=_rs,
        number=dict(
            font=dict(size=28, color="#0D1B2E",
                      family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        ),
        title=dict(
            text="<b>RISK SCORE</b>",
            font=dict(size=11, color="#64748b",
                      family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 99],
                tickwidth=1,
                tickcolor="#cbd5e1",
                tickmode="array",
                tickvals=[0, 33, 66, 99],
                ticktext=["0", "33", "66", "99"],
                tickfont=dict(size=9, color="#94a3b8"),
            ),
            bar=dict(color=_needle_color, thickness=0.6),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            steps=[
                {"range": [0,  33], "color": "#D7F5E3"},
                {"range": [33, 66], "color": "#FFEDC2"},
                {"range": [66, 99], "color": "#FED4D0"},
            ],
            threshold=dict(
                line=dict(color="#0D1B2E", width=3),
                thickness=0.9,
                value=_rs,
            ),
        ),
        domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
    ))
    fig.update_layout(
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_allocation_pie(tickers, weights, title=None, height=420, risk_score=None):
    """Rainbow-spectrum allocation pie.

    Design:
      - Small donut hole (0.18)
      - Ticker names + percentages in a compact vertical legend on the right
      - Thin 1.5px white separators between slices
      - Slices sorted descending by weight

    Note: `risk_score` parameter is accepted for backward compatibility but
    no longer embeds a gauge (use make_risk_gauge separately for that).
    """
    _wsum = sum(float(w) for w in weights) if weights else 0
    as_pct = (_wsum > 1.5)  # sum ≈ 1 → fractions; else percentages

    data = []
    for t, w in zip(tickers, weights):
        try:
            v = float(w) * (1.0 if as_pct else 100.0)
        except (TypeError, ValueError):
            continue
        if v >= 0.1:
            data.append((t, round(v, 2)))
    data.sort(key=lambda x: -x[1])

    if not data:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            annotations=[dict(text="No allocation data",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#9ca3af", size=12))],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    _labels = [d[0] for d in data]
    _values = [d[1] for d in data]

    # Evenly distribute rainbow colors around the pie so adjacent slices are
    # visually distinct even with a handful of holdings
    n = len(data)
    if n <= len(PIE_PALETTE):
        _stride = max(1, len(PIE_PALETTE) // max(n, 1))
        _colors = [PIE_PALETTE[(i * _stride) % len(PIE_PALETTE)] for i in range(n)]
    else:
        _colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(n)]

    # Legend labels: ticker + percentage (e.g. "AAPL — 23.4%")
    _legend_labels = [f"{t} — {v:.2f}%" for t, v in data]

    # Pie layout (gauge is rendered separately at call sites via make_risk_gauge)
    _pie_domain_x = [0.0, 0.68]
    _pie_domain_y = [0.0, 1.0]

    fig = go.Figure(go.Pie(
        labels=_legend_labels,
        values=_values,
        hole=0.18,                                       # tiny center hole
        textinfo="none",                                 # percentages live in legend only
        hovertemplate="<b>%{label}</b><extra></extra>",
        marker=dict(
            colors=_colors,
            line=dict(color="#ffffff", width=1.5),
        ),
        sort=False,
        direction="clockwise",
        rotation=90,
        domain=dict(x=_pie_domain_x, y=_pie_domain_y),
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            # Center the title horizontally OVER THE PIE itself (pie occupies
            # x=[0.0, 0.68], so center is at 0.34).
            x=0.34, xanchor="center",
            # Tight y so title sits close to the top of the pie
            y=0.97, yanchor="top",
            pad=dict(t=0, b=0),
            font=dict(
                size=15, color="#111827",
                family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            ),
        ),
        height=height,
        # Tight top margin — barely enough for the title to breathe
        margin=dict(t=24 if title else 6, b=6, l=12, r=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left",   x=0.72,
            font=dict(
                size=13, color="#1f2937",
                family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            ),
            bgcolor="rgba(255,255,255,0)",
            itemsizing="constant",
            itemclick=False, itemdoubleclick=False,
            tracegroupgap=3,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo(returns_series_tuple, years_forward=10, n_simulations=1000,
                    benchmark_returns_tuple=None, method="bootstrap"):
    """Monte Carlo simulation. Cached 1hr.

    Args:
        returns_series_tuple: tuple of historical daily returns (decimal)
        years_forward: projection horizon in years
        n_simulations: number of paths (default 1000 — increased from 200
            so tail percentiles are more reliable)
        benchmark_returns_tuple: optional benchmark returns for overlay
        method: "bootstrap" (default — resamples from actual history,
            captures fat tails) or "gaussian" (legacy normal sampling)

    Returns:
        days, p5, p10, p25, p50, p75, p90, p95, bm_path,
        prob_loss_by_year (dict {year: probability})

    Bootstrap is preferred because real return distributions have fat tails
    and skew. Gaussian sampling systematically understates tail risk.
    """
    returns_series = list(returns_series_tuple)
    benchmark_returns = list(benchmark_returns_tuple) if benchmark_returns_tuple else None
    rets = np.array(returns_series)
    if len(rets) < 30:
        # Not enough history — fall back to Gaussian on whatever we have
        method = "gaussian"

    mean_r = np.mean(rets)
    std_r  = np.std(rets)
    trading_days = years_forward * 252

    # Simulate paths
    np.random.seed(42)
    if method == "bootstrap":
        # Resample from actual historical returns with replacement.
        # Captures real-world skew, kurtosis, fat tails — produces more
        # realistic loss scenarios than Gaussian assumes.
        idx = np.random.randint(0, len(rets),
                                size=(n_simulations, trading_days))
        simulations = rets[idx]
    else:
        simulations = np.random.normal(mean_r, std_r,
                                       (n_simulations, trading_days))

    cum_paths = np.cumprod(1 + simulations, axis=1)

    # Percentile paths — wider tails (5/95) for downside visibility
    p5   = np.percentile(cum_paths,  5, axis=0)
    p10  = np.percentile(cum_paths, 10, axis=0)
    p25  = np.percentile(cum_paths, 25, axis=0)
    p50  = np.percentile(cum_paths, 50, axis=0)
    p75  = np.percentile(cum_paths, 75, axis=0)
    p90  = np.percentile(cum_paths, 90, axis=0)
    p95  = np.percentile(cum_paths, 95, axis=0)

    # Probability of loss at year boundaries — % of paths below starting value
    prob_loss = {}
    for yr in (1, 3, 5, 10):
        if yr * 252 <= trading_days:
            day_idx = yr * 252 - 1
            prob_loss[yr] = float((cum_paths[:, day_idx] < 1.0).mean())

    # Benchmark projection (kept simple — Gaussian for benchmark is fine
    # since we only show the median path)
    bm_path = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm_mean = np.mean(benchmark_returns)
        bm_std  = np.std(benchmark_returns)
        bm_sims = np.random.normal(bm_mean, bm_std, (200, trading_days))
        bm_cum  = np.cumprod(1 + bm_sims, axis=1)
        bm_path = np.percentile(bm_cum, 50, axis=0)

    days_range = np.arange(1, trading_days + 1)
    return days_range, p5, p10, p25, p50, p75, p90, p95, bm_path, prob_loss


def build_projection_chart(strategy_name, returns, benchmark_returns, bm_label,
                           years=10, comparison_list=None):
    """Build the 10-year forward projection chart with Monte Carlo bands.
    comparison_list: list of (name, returns_series) tuples for comparison overlays.

    Bands rendered (outermost to innermost): 90% (5th-95th pct), 80% (10th-90th),
    50% (25th-75th), median path. The 90% band makes downside scenarios
    visually obvious — losses are shaded below the dashed 0% line.
    """
    # Strip emoji prefixes from labels for legend display, preserving the 👤
    # silhouette on the client-current portfolio. Same convention as the
    # _clean_label helper used by the other charts.
    import re as _re_proj
    def _clean_proj(s):
        if not isinstance(s, str):
            return s
        if s.startswith("👤"):
            return s
        return _re_proj.sub(
            r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*",
            "", s
        ).strip() or s

    strategy_name = _clean_proj(strategy_name)
    bm_label      = _clean_proj(bm_label)
    days, p5, p10, p25, p50, p75, p90, p95, bm_path, prob_loss = run_monte_carlo(
        tuple(returns) if not isinstance(returns, tuple) else returns,
        years_forward=years,
        benchmark_returns_tuple=tuple(benchmark_returns) if benchmark_returns else None
    )

    tick_positions = list(range(0, years * 252 + 1, 252))
    tick_labels    = [f"Yr {i}" for i in range(years + 1)]

    fig = go.Figure()

    # Convert cumulative $ values to % returns
    p5_pct  = [v-1 for v in p5];  p10_pct = [v-1 for v in p10]
    p25_pct = [v-1 for v in p25]; p50_pct = [v-1 for v in p50]
    p75_pct = [v-1 for v in p75]; p90_pct = [v-1 for v in p90]
    p95_pct = [v-1 for v in p95]
    bm_pct  = [v-1 for v in bm_path] if bm_path is not None else None

    # ── 90% confidence band (5th–95th pct) — widest, captures tail risk ─
    fig.add_trace(go.Scatter(
        x=list(days), y=p95_pct, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(days), y=p5_pct, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(220,38,38,0.05)",  # very light red — visualizes downside
        name="90% Confidence Band (5th–95th pct)", showlegend=True,
        hovertemplate="90% band: %{y:+.1%}<extra></extra>",
    ))
    # ── 80% band ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(days), y=p90_pct, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(days), y=p10_pct, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(37,99,235,0.07)",
        name="80% Confidence Band", showlegend=True,
        hovertemplate="80% band: %{y:+.1%}<extra></extra>",
    ))
    # ── 50% band ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(days), y=p75_pct, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(days), y=p25_pct, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(37,99,235,0.14)",
        name="50% Confidence Band", showlegend=True,
        hovertemplate="50% band: %{y:+.1%}<extra></extra>",
    ))

    # ── Zero-return line so losses are visually obvious ──────
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1,
                  annotation_text="Starting value (0% return)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#6b7280"))

    # ── Median path ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(days), y=p50_pct, mode="lines",
        line=dict(color="#2563eb", width=3, dash="solid"),
        name=f"{strategy_name} (Median)",
        legend="legend2",
        hovertemplate=f"<b>{strategy_name}</b><br>Day %{{x}}: %{{y:+.1%}}<extra></extra>",
    ))

    # ── 5th percentile line (downside scenario) — dashed red ─
    # Stays in the upper-left key with the confidence bands, since it's a
    # statistical reference (worst-case scenario) not a portfolio.
    fig.add_trace(go.Scatter(
        x=list(days), y=p5_pct, mode="lines",
        line=dict(color="#dc2626", width=1.5, dash="dot"),
        name="5th pct (worst-case scenario)",
        hovertemplate="5th pct: %{y:+.1%}<extra></extra>",
    ))

    # ── Benchmark median ──────────────────────────────────────
    # Suppressed when SPY is already in the comparison_list (which is the
    # default when the "📈 S&P 500 (SPY)" toggle is on at the top of the
    # Results & Charts tab). Otherwise the chart shows two identical SPY
    # lines — one labeled "SPY (Median)" from this auto-benchmark trace and
    # another labeled "S&P 500 (SPY) (Median)" from the comparison list.
    _spy_in_cmp = False
    if comparison_list:
        for _cn, _ in comparison_list:
            _cn_l = (_cn or "").lower()
            if "s&p 500" in _cn_l or _cn_l.strip().endswith("spy") or "(spy)" in _cn_l:
                _spy_in_cmp = True
                break
    if bm_pct is not None and not _spy_in_cmp:
        fig.add_trace(go.Scatter(
            x=list(days), y=bm_pct, mode="lines",
            line=dict(color="#9ca3af", width=2, dash="solid"),
            name=f"{bm_label} (Median)",
            legend="legend2",
            hovertemplate=f"<b>{bm_label}</b><br>Day %{{x}}: %{{y:+.1%}}<extra></extra>",
        ))

    # ── Comparison overlays — converted to % return ──────────
    cmp_colors = ["#059669","#d97706","#dc2626","#7c3aed","#ea580c"]
    if comparison_list:
        for ci, (cmp_name, cmp_rets) in enumerate(comparison_list):
            try:
                _cmp_disp = _clean_proj(cmp_name)
                # Use new signature (10 returns now)
                _ret = run_monte_carlo(
                    tuple(cmp_rets) if not isinstance(cmp_rets, tuple) else cmp_rets,
                    years_forward=years
                )
                _, _cp5, cp10, cp25, cp50, cp75, cp90, _cp95, _, _ = _ret
                # Convert from $1 cumulative to % return (same as main strategy)
                cp25_pct = [v-1 for v in cp25]
                cp50_pct = [v-1 for v in cp50]
                cp75_pct = [v-1 for v in cp75]
                c_ = cmp_colors[ci % len(cmp_colors)]
                r_int, g_int, b_int = int(c_[1:3],16), int(c_[3:5],16), int(c_[5:7],16)

                # Light band for comparison
                fig.add_trace(go.Scatter(
                    x=list(days), y=cp75_pct, mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=list(days), y=cp25_pct, mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor=f"rgba({r_int},{g_int},{b_int},0.07)",
                    showlegend=False, hoverinfo="skip",
                ))
                # Median line — goes on the SECOND legend (portfolios row below chart)
                fig.add_trace(go.Scatter(
                    x=list(days), y=cp50_pct, mode="lines",
                    line=dict(color=c_, width=2, dash="solid"),
                    name=f"{_cmp_disp} (Median)",
                    legend="legend2",
                    hovertemplate=f"<b>{_cmp_disp}</b><br>Day %{{x}}: %{{y:+.1%}}<extra></extra>",
                ))
            except Exception:
                pass

    # ── Reference lines ────────────────────────────────────────
    fig.add_hline(y=1.0, line_dash="solid", line_color="#e5e7eb", line_width=1,
                   annotation_text="0%", annotation_position="left",
                   annotation_font=dict(size=10, color="#9ca3af"))

    fig.update_layout(
        title=dict(
            # Drop the "vs <bm>" suffix when the benchmark is already in
            # the comparison list (otherwise we'd say "vs SPY" while the
            # comparison legend below shows "S&P 500 (SPY)" — same line,
            # double-billed).
            text=(f"{years}-Year Forward Projection — {strategy_name}"
                  if _spy_in_cmp else
                  f"{years}-Year Forward Projection — {strategy_name} vs {bm_label}"),
            x=0, xanchor="left", font=dict(size=14, color="#111827")
        ),
        xaxis=dict(
            tickvals=tick_positions, ticktext=tick_labels,
            tickfont=dict(size=11, color="#6b7280"),
            # X-axis title removed — the "Yr 1, Yr 2, ..., Yr 10" tick labels
            # are self-explanatory, and a "Projection Timeline" caption was
            # colliding with the disclaimer text below the chart.
            title_text=None,
            gridcolor="#f0f0f0", linecolor="#e5e7eb", showgrid=True,
        ),
        yaxis=dict(
            title_text="Total Return (%)",
            tickformat="+.0%",
            title_font=dict(size=12, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
            gridcolor="#f0f0f0", linecolor="#e5e7eb", showgrid=True,
        ),
        height=520,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafafa",
        font=dict(color="#374151", family="Inter, sans-serif"),
        # Margins equalized left/right so x=0.5 in paper coords lines up
        # visually with the center of the plot area. Previously left=70,
        # right=32 which offset the paper-center rightward, making the
        # bottom-centered portfolios legend look right-shifted.
        margin=dict(t=54, b=180, l=70, r=70),
        # Primary legend — confidence bands + 5th pct (statistical refs)
        # Stays in the upper-left of the chart area.
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=11, color="#374151"),
            orientation="v", x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            title=dict(text="<b>Confidence</b>",
                       font=dict(size=10, color="#6b7280")),
        ),
        # Secondary legend — portfolios. Sits BELOW the disclaimer in a
        # horizontal row. xanchor=center at x=0.5 + entrywidthmode='fraction'
        # forces plotly to size the box to its actual content width and
        # center it on the chart, rather than spanning the chart and
        # left-aligning items.
        legend2=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=11, color="#374151"),
            orientation="h",
            x=0.5, y=-0.36,
            xanchor="center", yanchor="top",
            itemwidth=30,
        ),
        annotations=[dict(
            text="⚠️ Simulated projections — not a guarantee of future returns.",
            xref="paper", yref="paper",
            # Disclaimer sits in its own row between the chart's tick
            # labels (which end around y=-0.10) and the portfolios legend
            # (which starts at y=-0.36).
            x=0.5, y=-0.22,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=10, color="#9ca3af"),
            align="center",
        )],
    )
    return fig

# ── SESSION STATE ─────────────────────────────────────────────
for key in ["bt1","bt3","bt5","bt10","tickers","ran_once","quotes","holdings_list","watchlist","wl_quotes"]:
    if key not in st.session_state:
        st.session_state[key] = None
saved_portfolios = load_saved()
if st.session_state.holdings_list is None:
    st.session_state.holdings_list = load_holdings()
if st.session_state.watchlist is None:
    wl = load_watchlist()
    # Handle both list format (current) and dict format {"tickers": [...]}
    if isinstance(wl, list):
        st.session_state.watchlist = wl
    elif isinstance(wl, dict):
        st.session_state.watchlist = wl.get("tickers", ["AAPL","MSFT","GOOGL","AMZN","NVDA"])
    else:
        st.session_state.watchlist = ["AAPL","MSFT","GOOGL","AMZN","NVDA"]

PLOT_THEME = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafafa",
    font=dict(color="#374151", family="Inter, sans-serif", size=12),
    # Margins equalized left/right so chart content reads as visually
    # centered. Previously l=64, r=32 made every chart look right-shifted
    # — most noticeable on the drawdown / rolling Sharpe / 10y historical
    # charts where the bottom-centered legend looked off-center because
    # the paper-center didn't match the visual chart-area-center.
    margin=dict(t=52, b=48, l=64, r=64),
    xaxis=dict(
        gridcolor="#f0f0f0", linecolor="#e5e7eb", showgrid=True,
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=12, color="#374151"),
        tickcolor="#e5e7eb", showline=True, zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#f0f0f0", linecolor="#e5e7eb", showgrid=True,
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=12, color="#374151"),
        tickcolor="#e5e7eb", showline=True, zeroline=False,
    ),
    title_font=dict(size=14, color="#111827", family="Inter, sans-serif"),
    hoverlabel=dict(
        bgcolor="#111827", bordercolor="#374151",
        font=dict(color="#f9fafb", size=12, family="Inter, sans-serif"),
        namelength=-1,
    ),
)

LEGEND_BOTTOM_RIGHT = dict(
    bgcolor="rgba(255,255,255,0.92)",
    bordercolor="#e5e7eb",
    borderwidth=1,
    font=dict(size=11, color="#374151", family="Inter, sans-serif"),
    orientation="v",
    x=0.99, y=0.02,
    xanchor="right", yanchor="bottom",
)

# Standard legend style to use in charts
LEGEND_STYLE = dict(
    bgcolor="rgba(255,255,255,0.92)",
    bordercolor="#e5e7eb",
    borderwidth=1,
    font=dict(size=11, color="#374151", family="Inter, sans-serif"),
)
COLORS = ["#2563eb","#059669","#d97706","#dc2626","#7c3aed","#ea580c",
          "#0891b2","#db2777","#65a30d","#9333ea","#c026d3","#0d9488"]

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0">
      <div style="font-size:0.65rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Platform</div>
      <div style="font-size:1rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em">Foresight Portfolio Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── PORTFOLIOS QUICK-LOAD PANEL ─────────────────────────────────────────
    # Two dropdowns — Standard (built-in presets) + Custom (saved portfolios).
    # Picking from either loads it into the Securities section (same effect as
    # the in-page dropdown). Picking from one clears the other so there's
    # always exactly one active selection across the pair.
    #
    # Streamlit constraint: once a widget with a given key renders on the
    # current run, you can't write to that key in session_state. So we use the
    # standard "deferred reset" pattern — set a flag on click, then on the
    # NEXT run (top of this block, before widgets render) pre-seed the keys.
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;color:#cbd5e1;'
        'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px">'
        '📁 Portfolios</div>',
        unsafe_allow_html=True,
    )

    _PLACEHOLDER = "— select —"
    _saved_dict  = load_saved()
    _saved_names = sorted(_saved_dict.keys())
    _preset_names = [k for k, v in POPULAR_PORTFOLIOS.items() if v is not None]

    # ── Deferred reset: applied BEFORE the selectboxes are instantiated ──
    # Triggered when picking from one dropdown should clear the other.
    if st.session_state.pop("_sidebar_pf_clear_custom", False):
        st.session_state["sb_custom_select"] = _PLACEHOLDER
    if st.session_state.pop("_sidebar_pf_clear_preset", False):
        st.session_state["sb_preset_select"] = _PLACEHOLDER

    _current_src = st.session_state.get("portfolio_source", "")

    # Initialize dropdown selections from current state if the user hasn't
    # touched them yet (first render after page load / after external load).
    if "sb_custom_select" not in st.session_state:
        st.session_state["sb_custom_select"] = (
            _current_src[2:] if _current_src.startswith("📁 ")
            and _current_src[2:] in _saved_names else _PLACEHOLDER
        )
    if "sb_preset_select" not in st.session_state:
        st.session_state["sb_preset_select"] = (
            _current_src if _current_src in _preset_names else _PLACEHOLDER
        )

    # ── Standard/popular portfolios (built-in presets) ──
    _sel_preset = st.selectbox(
        "Standard",
        [_PLACEHOLDER] + _preset_names,
        key="sb_preset_select",
    )
    if _sel_preset != _PLACEHOLDER and _sel_preset != _current_src:
        load_portfolio_into_session(_sel_preset)
        # User picked a standard one — clear the custom dropdown on next run
        st.session_state["_sidebar_pf_clear_custom"] = True
        st.rerun()

    # ── Custom (user-saved portfolios from saved_portfolios.json) ──
    if _saved_names:
        _sel_custom = st.selectbox(
            "Custom",
            [_PLACEHOLDER] + _saved_names,
            key="sb_custom_select",
        )
        if _sel_custom != _PLACEHOLDER and f"📁 {_sel_custom}" != _current_src:
            load_portfolio_into_session(f"📁 {_sel_custom}")
            # User picked a saved one — clear the preset dropdown on next run
            st.session_state["_sidebar_pf_clear_preset"] = True
            st.rerun()
    else:
        st.caption("_No saved portfolios yet. Save one from the main view to see it here._")

    # ── EXCEL / CSV UPLOAD ─────────────────────────────────────
    # Bulk-load a portfolio from a spreadsheet rather than typing tickers.
    # Expected format: a column of tickers + a column of weights or share
    # counts. We'll attempt to autodetect column names — the parser accepts:
    #   ticker | symbol     →  ticker column
    #   weight | weights | allocation | pct | percent | %  →  weight (%)
    #   shares | quantity | qty                            →  share count
    # Weights or shares both work — we normalize whichever is present.
    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;color:#cbd5e1;'
        'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">'
        '📤 Upload Portfolio</div>',
        unsafe_allow_html=True,
    )
    st.caption("Excel (.xlsx) or CSV with ticker + weight or share columns.")
    _upload = st.file_uploader(
        "Choose file",
        type=["csv", "xlsx", "xls"],
        key="sidebar_pf_upload",
        label_visibility="collapsed",
    )
    if _upload is not None and not st.session_state.get("_last_upload_processed") == _upload.name:
        try:
            # Read into a DataFrame
            if _upload.name.lower().endswith(".csv"):
                _df = pd.read_csv(_upload)
            else:
                _df = pd.read_excel(_upload)

            # Find the ticker + weight columns by fuzzy column name
            _cols_lower = {c.lower().strip(): c for c in _df.columns}
            _ticker_col = next(
                (_cols_lower[k] for k in ("ticker","symbol","tickers","symbols")
                 if k in _cols_lower),
                None,
            )
            _weight_col = next(
                (_cols_lower[k] for k in
                 ("weight","weights","allocation","pct","percent","%","weight (%)")
                 if k in _cols_lower),
                None,
            )
            _share_col = next(
                (_cols_lower[k] for k in ("shares","quantity","qty","units")
                 if k in _cols_lower),
                None,
            )
            _value_col = next(
                (_cols_lower[k] for k in
                 ("value","market value","mv","amount","balance","total value")
                 if k in _cols_lower),
                None,
            )

            if _ticker_col is None:
                st.error(
                    "Couldn't find a ticker column. Add a column named "
                    "'Ticker' or 'Symbol' and re-upload."
                )
            else:
                # Build (ticker, weight%) pairs
                _df = _df.dropna(subset=[_ticker_col]).copy()
                _df[_ticker_col] = _df[_ticker_col].astype(str).str.upper().str.strip()

                # Helper: pandas to_numeric chokes on common spreadsheet
                # formatting like "6.0%", "$1,234.56", or "100,000" — it
                # coerces them to NaN. Strip those symbols before parsing
                # so uploads from real-world files (which almost always
                # have at least percent signs) actually work.
                def _to_num_clean(series):
                    s = series.astype(str).str.strip()
                    s = s.str.replace(r"[%$,\s]", "", regex=True)
                    return pd.to_numeric(s, errors="coerce").fillna(0).values

                if _weight_col is not None:
                    # Could be 0-1 fractions or 0-100 percentages — autodetect
                    _w = _to_num_clean(_df[_weight_col])
                    if _w.sum() > 0:
                        if _w.max() <= 1.05:    # treat as fractions
                            _w = _w * 100.0
                        _w = _w / _w.sum() * 100.0  # normalize to exactly 100%
                elif _value_col is not None:
                    _v = _to_num_clean(_df[_value_col])
                    if _v.sum() > 0:
                        _w = _v / _v.sum() * 100.0
                    else:
                        _w = None
                elif _share_col is not None:
                    # Share counts alone aren't enough (need price × shares),
                    # so we treat them as proportional weights — caller can
                    # adjust afterward in the Securities table.
                    _s = _to_num_clean(_df[_share_col])
                    if _s.sum() > 0:
                        _w = _s / _s.sum() * 100.0
                    else:
                        _w = None
                else:
                    # No weight info — assume equal-weight
                    _n = len(_df)
                    _w = [100.0 / _n] * _n if _n > 0 else None

                if _w is None or len(_w) == 0:
                    st.error(
                        "No usable weight, value, or share data found. "
                        "Add a Weight, Value, or Shares column."
                    )
                else:
                    _tks = list(_df[_ticker_col])
                    _ws  = list(_w)
                    # Filter out zero-weight rows
                    _pairs = [(t, w) for t, w in zip(_tks, _ws) if w > 0]
                    _tks   = [t for t, _ in _pairs]
                    _ws    = [w for _, w in _pairs]
                    if not _tks:
                        st.error("No tickers with positive weight in the file.")
                    else:
                        # Push into session state — same shape load_portfolio_into_session uses
                        st.session_state.ticker_input_val     = ", ".join(_tks)
                        st.session_state["ticker_text_input"] = ", ".join(_tks)
                        st.session_state["loaded_weights"]    = {
                            t: round(float(w), 1) for t, w in zip(_tks, _ws)
                        }
                        st.session_state.portfolio_source = (
                            f"📤 Uploaded — {_upload.name}"
                        )
                        st.session_state["_last_upload_processed"] = _upload.name
                        st.success(
                            f"✅ Loaded {len(_tks)} ticker(s) from "
                            f"`{_upload.name}`. Switch to the Analyzer tab "
                            f"to review."
                        )
                        st.rerun()
        except Exception as _up_err:
            st.error(f"Couldn't parse file: {_up_err}")
            st.caption(
                "Expected columns (any case): Ticker/Symbol + one of: "
                "Weight, Allocation, %, Value, or Shares."
            )

    # Composite Optimizer sliders previously lived here in the sidebar.
    # They've moved to the Optimizer tab itself (main_tab3) — that's where
    # they're actually relevant. The sidebar is now reserved for navigation
    # and quick portfolio loading only.


# ── MAIN ──────────────────────────────────────────────────────
# Header mirrors the client portal's branding — same hexagon-pulse logo,
# same typography, same teal palette — so the advisor and client surfaces
# feel like one product.
_HEADER_LOGO_SVG = (
    '<svg width="44" height="44" viewBox="0 0 24 24" style="display:block">'
    '<path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" fill="none" '
    'stroke="#0E5C5E" stroke-width="1.6" stroke-linejoin="round"/>'
    '<path d="M6 12 L9 12 L10.5 9 L12 15 L13.5 11 L15 13 L18 13" fill="none" '
    'stroke="#0E5C5E" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
st.markdown(f"""
<div class="app-header">
    <div style="display:flex;align-items:center;gap:14px">
        {_HEADER_LOGO_SVG}
        <div>
            <h1>Foresight Portfolio Intelligence</h1>
            <p>Advanced optimization · 13 strategies · Monte Carlo projections · Real-time data</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tab order: Analyzer, Results & Charts, Optimizer, Fee Drag, Client Records, Settings.
# The `with main_tabN:` blocks farther down in this file are keyed by variable
# name, NOT by position — so to move a tab in the UI we just rebind its
# variable to a new st.tabs() position. main_tab6 (Fee Drag) is bound to the
# 4th position; main_tab4 (Client Records) is bound to the 5th. main_tab5
# (Settings) stays rightmost.
main_tab1, main_tab2, main_tab3, main_tab6, main_tab4, main_tab5 = st.tabs([
    "Analyzer", "Results & Charts", "Optimizer", "Fee Drag Analyzer",
    "Client Records", "Settings"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — OPTIMIZER
# ═══════════════════════════════════════════════════════════════
with main_tab1:
    # Sidebar gates the strategy-blend sliders on this flag
    st.session_state["optimizer_tab_active"] = False

    # ── POPULAR PUBLIC PORTFOLIOS ────────────────────────────
    # POPULAR_PORTFOLIOS is now defined at module scope (top of file) so the
    # sidebar can use it too. Kept this comment as a breadcrumb.

    # ── LARGE UNIVERSE OF US STOCKS ──────────────────────────
    STOCK_UNIVERSE = [
        # Mega cap tech
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ORCL","ADBE",
        # Large cap tech
        "CRM","AMD","INTC","QCOM","TXN","MU","AMAT","KLAC","LRCX","SNPS",
        "CDNS","PANW","CRWD","ZS","FTNT","NET","DDOG","SNOW","PLTR","SHOP",
        # Financials
        "JPM","BAC","WFC","GS","MS","BLK","AXP","V","MA","COF",
        "USB","PNC","TFC","SCHW","CB","MET","PRU","AFL","ALL","AIG",
        # Healthcare
        "JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "GILD","VRTX","REGN","ISRG","SYK","BSX","MDT","EW","ZBH","BAX",
        # Consumer
        "AMZN","WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","YUM",
        "PG","KO","PEP","PM","MO","CL","KMB","CHD","EL","ULTA",
        # Energy
        "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
        "KMI","WMB","ET","EPD","MPLX","DVN","FANG","APA","HAL","BKR",
        # Industrials
        "CAT","DE","BA","HON","GE","MMM","LMT","RTX","NOC","GD",
        "UPS","FDX","CSX","UNP","NSC","EMR","ITW","PH","ROK","ETN",
        # Materials & Real Estate
        "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","CF",
        "AMT","PLD","CCI","EQIX","PSA","O","WELL","SPG","VTR","AVB",
        # Utilities & Telecom
        "NEE","DUK","SO","D","AEP","EXC","SRE","XEL","WEC","ES",
        "T","VZ","TMUS","CMCSA","CHTR","DIS","NFLX","WBD","PARA","FOX",
        # Mid cap growth
        "DKNG","RBLX","U","COIN","HOOD","SOFI","AFRM","UPST","LCID","RIVN",
        "ABNB","UBER","LYFT","DASH","GRAB","BMBL","MTCH","IAC","ANGI","Z",
    ]

    # ═══════════════════════════════════════════════════════════════
    # CLIENT PICKER (Step 1, before Securities)
    # Selecting a client populates session_state["_active_client_*"] keys
    # which downstream tabs (Optimizer, Reports) read from. Default is
    # "no client" → proposals save to Unassociated Reports.
    #
    # Moved here in May 2026 from the Optimizer tab so the advisor picks
    # the client BEFORE entering tickers/weights, not after.
    # ═══════════════════════════════════════════════════════════════
    _all_profiles_step1 = _load_json_safe(CLIENT_PROFILES_FILE)
    _pickable_step1 = {k: p for k, p in _all_profiles_step1.items()
                       if k != UNASSOCIATED_CLIENT_KEY
                       and not p.get("_is_unassociated", False)}

    _picker_cols_step1 = st.columns([3, 1, 1])
    with _picker_cols_step1[0]:
        if _pickable_step1:
            _client_options_step1 = {
                f"{p.get('client_name','?')} — score {p.get('overall_score','?')} "
                f"({p.get('client_email', k)})": k
                for k, p in _pickable_step1.items()
            }

            def _on_client_change_step1():
                """Fires when the client picker value changes. Clears any
                working proposal state so fresh recommendations generate
                for the newly-selected client."""
                for k in list(st.session_state.keys()):
                    if (k.startswith("proposals_working_")
                            or k.startswith("proposals_basis_fp_")
                            or k.startswith("final_")):
                        st.session_state.pop(k, None)

            _pick_label_step1 = st.selectbox(
                "**Client** (optional — leave unselected to save as unassociated)",
                options=["— select a client —"] + list(_client_options_step1.keys()),
                key="step1_client_pick",
                on_change=_on_client_change_step1,
            )
            if _pick_label_step1 == "— select a client —":
                for _k in ("_active_client_key", "_active_client_name",
                           "_active_client_score", "_active_client_priorities"):
                    st.session_state.pop(_k, None)
            else:
                _ck = _client_options_step1[_pick_label_step1]
                _cp = _all_profiles_step1[_ck]
                st.session_state["_active_client_key"]   = _ck
                st.session_state["_active_client_name"]  = _cp.get("client_name", "Client")
                st.session_state["_active_client_score"] = int(_cp.get("overall_score", 50))
                st.session_state["_active_client_priorities"] = list(_cp.get("priorities", []) or [])
        else:
            st.info("No clients yet — analyses & proposals will be saved to Unassociated Reports.")

    with _picker_cols_step1[1]:
        _active_score_step1 = st.session_state.get("_active_client_score")
        if st.session_state.get("_active_client_key"):
            st.metric("Risk Score", _active_score_step1)
        else:
            st.metric("Risk Score", "—", help="Default score 50 (no client)")

    with _picker_cols_step1[2]:
        # Spacer / future use — keeps three-column layout consistent with
        # the prior Optimizer-tab placement.
        st.write("")

    st.markdown("")  # vertical spacer before Section 1 heading

    # Securities
    st.markdown('<div class="section-label"><div class="section-num">1</div><div class="section-title">Securities</div></div>', unsafe_allow_html=True)

    if "ticker_input_val" not in st.session_state:
        st.session_state.ticker_input_val = "AAPL, MSFT, JPM, XOM, JNJ"
    if "portfolio_source" not in st.session_state:
        st.session_state.portfolio_source = "Custom — Enter Your Own Tickers"

    # ── PORTFOLIO SOURCE DROPDOWN ─────────────────────────────
    import random

    # If a previous interaction requested a reset (e.g. Random 10 button was clicked
    # last run), apply it here BEFORE the widgets are instantiated this run.
    # This is the only safe way to update widget state in Streamlit.
    if st.session_state.pop("_pending_random_reset", False):
        # Pre-seed the widget keys before they render
        _rand_tickers = st.session_state.pop("_pending_random_tickers", "")
        if _rand_tickers:
            st.session_state["ticker_text_input"]     = _rand_tickers
            st.session_state.ticker_input_val         = _rand_tickers
        st.session_state["portfolio_source_sel"]      = "Custom — Enter Your Own Tickers"
        st.session_state.portfolio_source             = "Custom — Enter Your Own Tickers"
        st.session_state["loaded_weights"]            = {}

    saved_names   = list(load_saved().keys())
    preset_names  = [k for k in POPULAR_PORTFOLIOS if not k.startswith("── ") and k != "Custom — Enter Your Own Tickers"]
    source_opts   = (
        ["Custom — Enter Your Own Tickers"] +
        ([f"📁 {n}" for n in saved_names] if saved_names else []) +
        preset_names
    )
    src_col, rand_col = st.columns([3, 1])
    sel_src = src_col.selectbox(
        "Load portfolio",
        source_opts,
        index=source_opts.index(st.session_state.portfolio_source)
              if st.session_state.portfolio_source in source_opts else 0,
        key="portfolio_source_sel",
        label_visibility="collapsed"
    )
    if sel_src != st.session_state.portfolio_source:
        # Single source of truth — same helper the sidebar quick-load uses
        load_portfolio_into_session(sel_src)
        st.rerun()

    if "ticker_input_val" not in st.session_state:
        st.session_state.ticker_input_val = "AAPL, MSFT, JPM, XOM, JNJ"

    if rand_col.button("🎲 Random 10", key="random_tickers"):
        # Defer the state update — the selectbox and text_input widgets have
        # already run on this pass, so we can't set their keys now.
        # Stash the new tickers and a reset flag; the top-of-function handler
        # will pre-seed the widget state on the next run, BEFORE the widgets
        # are instantiated.
        st.session_state["_pending_random_reset"]    = True
        st.session_state["_pending_random_tickers"]  = ", ".join(
            random.sample(STOCK_UNIVERSE, 10)
        )
        st.rerun()

    tickers_input = st.text_input(
        "Tickers",
        value=st.session_state.ticker_input_val,
        label_visibility="collapsed",
        placeholder="e.g. AAPL, MSFT, GOOGL, AMZN, JPM",
        key="ticker_text_input"
    )
    # Update session state when user types manually
    if tickers_input != st.session_state.ticker_input_val:
        st.session_state.ticker_input_val = tickers_input
        st.session_state.portfolio_source = "Custom — Enter Your Own Tickers"
        st.session_state["loaded_weights"] = {}

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    # Show ticker chips
    if tickers:
        chips_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;margin-bottom:4px'>"
        for tkr in tickers:
            chips_html += f"""<div style='background:#f3f4f6;border:1px solid #e5e7eb;
                border-radius:6px;padding:4px 10px;font-size:0.8rem;font-weight:600;
                color:#374151;font-family:JetBrains Mono,monospace'>{tkr}</div>"""
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown(f"<span style='color:#9ca3af;font-size:0.75rem'>{len(tickers)} securities</span>",
                unsafe_allow_html=True)


    # ── PORTFOLIO WEIGHTS (consolidated into Step 1) ──────────────────────────
    # Was previously a standalone "Step 3 — Portfolio Weights" section. Moved
    # in here so the user enters tickers, sets weights, and saves all in one
    # block. Set weights to total 100% for your portfolio to be included in
    # the analysis.
    custom_weights = {}
    custom_weights_valid = False
    if tickers:
        st.markdown(
            '<div style="margin-top:18px;font-size:0.78rem;font-weight:600;'
            'color:#374151;letter-spacing:0.02em">Portfolio Weights '
            '<span style="color:#9ca3af;font-weight:400">— optional, must total 100%</span></div>',
            unsafe_allow_html=True,
        )

        # ── Auto-reset weights when tickers or portfolio source changes ──
        # Track what tickers were loaded last time to detect changes
        _prev_tickers = st.session_state.get("_prev_tickers", [])
        _prev_src     = st.session_state.get("_prev_src", "")
        _curr_src     = st.session_state.get("portfolio_source", "")
        _loaded_w     = st.session_state.get("loaded_weights", {})

        # If tickers changed OR a new portfolio was loaded, reset weight boxes
        if tickers != _prev_tickers or _curr_src != _prev_src:
            st.session_state["_prev_tickers"] = tickers
            st.session_state["_prev_src"]     = _curr_src
            if _loaded_w and all(t in _loaded_w for t in tickers):
                # Portfolio was uploaded — use its weights
                for ticker in tickers:
                    st.session_state[f"w_{ticker}"] = float(_loaded_w[ticker])
            else:
                # New ticker list — distribute equally, last ticker absorbs rounding
                _n   = len(tickers)
                _eq  = round(100.0 / _n, 1)
                _tot = round(_eq * (_n - 1), 1)
                _last = round(100.0 - _tot, 1)
                for _ti, ticker in enumerate(tickers):
                    st.session_state[f"w_{ticker}"] = _last if _ti == _n - 1 else _eq

        # ── Action row: Distribute Equally + Save Portfolio ─────────────
        # Equal: divides 100 by number of tickers (last absorbs rounding).
        # Save:  persists current tickers + weights to saved_portfolios.json
        #        under a user-supplied name. Inline name input next to button.
        eq_col, name_col, save_col = st.columns([1, 2, 1])
        if eq_col.button("Distribute Equally", key="eq_dist", use_container_width=True):
            n      = len(tickers)
            eq_val = round(100.0 / n, 1)
            total_assigned = round(eq_val * (n - 1), 1)
            last_val       = round(100.0 - total_assigned, 1)
            for idx_t, ticker in enumerate(tickers):
                st.session_state[f"w_{ticker}"] = last_val if idx_t == n - 1 else eq_val
            st.session_state["loaded_weights"] = {}
            st.rerun()

        # Defer-clear the save-name input from a previous successful save.
        # Must run BEFORE the widget below is instantiated — Streamlit forbids
        # writes to a widget's key after that widget has rendered on the
        # current run. Previously this lived after the widget and crashed.
        if st.session_state.pop("_clear_save_name_next", False):
            st.session_state["save_portfolio_name_input"] = ""

        _save_name = name_col.text_input(
            "save_name", placeholder="Name to save as (e.g. All Weather)",
            label_visibility="collapsed", key="save_portfolio_name_input",
        )
        if save_col.button("Save Portfolio", key="save_portfolio_btn",
                           use_container_width=True,
                           help="Save current tickers + weights to your saved portfolios"):
            _name_clean = (_save_name or "").strip()
            if not _name_clean:
                st.warning("Enter a name above before saving.")
            else:
                # Read current weights from session state (rendered widgets below
                # haven't run yet on this pass, so use what's stored).
                _cur_w = {}
                for _tk in tickers:
                    _cur_w[_tk] = float(st.session_state.get(f"w_{_tk}", 0))
                _w_total = sum(_cur_w.values())
                if _w_total <= 0:
                    st.warning("Weights must sum to a positive total before saving.")
                else:
                    # Normalize to decimals so the saved schema (weights 0.0-1.0)
                    # matches what load_portfolio_into_session expects.
                    _decimals = [round(_cur_w[_tk] / _w_total, 6) for _tk in tickers]
                    _payload = {
                        "tickers": list(tickers),
                        "weights": _decimals,
                        "saved_at": datetime.now().isoformat(timespec="minutes"),
                    }
                    # Atomic upsert (single locked read-modify-write)
                    _shared_update_json(
                        SAVE_FILE,
                        lambda d, n=_name_clean, p=_payload: d.update({n: p}),
                    )
                    # Mark as the active source so the sidebar dropdown reflects it
                    st.session_state.portfolio_source = f"📁 {_name_clean}"
                    st.success(f"✅ Saved as **{_name_clean}** — find it in the Custom dropdown.")
                    # Clear the name input on next run so it's empty for next save.
                    # The actual clear is performed at the top of the next run,
                    # BEFORE the widget renders (see _clear_save_name_next handler
                    # above) — Streamlit forbids widget-key writes post-render.
                    st.session_state["_clear_save_name_next"] = True
                    st.rerun()

        # ── Weight inputs ─────────────────────────────────────────────────
        default_w = round(100.0 / len(tickers), 1)
        loaded_w  = _loaded_w
        # Split into rows of 8
        max_per_row = 8
        rows_of_tickers = [tickers[i:i+max_per_row] for i in range(0, len(tickers), max_per_row)]
        for row in rows_of_tickers:
            wcols = st.columns(max_per_row)
            for i, ticker in enumerate(row):
                init_w = st.session_state.get(f"w_{ticker}", loaded_w.get(ticker, default_w))
                w = wcols[i].number_input(
                    f"{ticker} %", min_value=0.0, max_value=100.0,
                    value=float(init_w), step=1.0, key=f"w_{ticker}"
                )
                custom_weights[ticker] = w
        total = sum(custom_weights.values())
        if total == 0:
            st.caption("Set weights above 0 to include your portfolio.")
        elif abs(total - 100.0) < 0.5:
            st.success(f"✅ Total: {total:.1f}% — Your portfolio will be included!")
            custom_weights_valid = True
        else:
            remaining = 100.0 - total
            st.warning(f"⚠️ Total: {total:.1f}% — Need {abs(remaining):.1f}% {'more' if remaining > 0 else 'less'}")


    # Custom weights
    # ── COMPARISON PORTFOLIOS SECTION ────────────────────────
    st.markdown('<div class="section-label"><div class="section-num">2</div><div class="section-title">Client&apos;s Current Portfolio &amp; Comparisons</div></div>', unsafe_allow_html=True)
    st.caption("Specify the client's actual current holdings (used as the baseline in proposals), plus up to 2 additional comparison portfolios.")

    # ── Client's current portfolio picker ─────────────────────
    # This is what the Optimizer treats as "the client's current holdings"
    # when building the three-tier proposal. Separated from Section 1
    # (the *analyzed* portfolio) so the advisor can analyze a model
    # portfolio while still telling the optimizer what the client *actually*
    # holds today. Falls back to the analyzed portfolio if left at "Use
    # analyzed portfolio (Section 1)" so existing single-portfolio workflows
    # keep working unchanged.
    if "client_current_portfolio_sel" not in st.session_state:
        st.session_state.client_current_portfolio_sel = "Use analyzed portfolio (Section 1)"

    saved_for_curr = list(load_saved().keys())
    preset_for_curr = [k for k in POPULAR_PORTFOLIOS if not k.startswith("── ") and k != "Custom — Enter Your Own Tickers"]
    curr_opts = (
        ["Use analyzed portfolio (Section 1)"]
        + [f"📁 {n}" for n in saved_for_curr]
        + preset_for_curr
    )
    _sel_curr = st.selectbox(
        "👤 Client's current portfolio",
        curr_opts,
        index=curr_opts.index(st.session_state.client_current_portfolio_sel)
              if st.session_state.client_current_portfolio_sel in curr_opts else 0,
        key="client_current_sel_widget",
        help=(
            "What the client actually holds today. Used as the comparison "
            "baseline alongside the Optimizer's recommendations — shows up "
            "as a 'Client's Current' column in Results & Charts and on the "
            "proposal cards. Does NOT drive the Optimizer; the Optimizer "
            "always builds Option 1/2/3 variants from the portfolio in "
            "Section 1 above."
        ),
    )
    st.session_state.client_current_portfolio_sel = _sel_curr

    # Resolve the picked portfolio into tickers + weights and stash on
    # session state under a stable key the Optimizer reads from.
    def _resolve_current_portfolio(label):
        """Return (tickers, weights_dict_pct) for the picked label, or
        (None, None) to signal 'fall back to the analyzed portfolio'."""
        if not label or label == "Use analyzed portfolio (Section 1)":
            return None, None
        if label.startswith("📁 "):
            sp = load_saved().get(label[2:])
            if not sp:
                return None, None
            tks = sp.get("tickers", []) or []
            ws  = sp.get("weights", []) or []
            # Saved portfolios store weights as decimals (0.6 = 60%); the
            # Optimizer expects percentages, so convert.
            return tks, {t: round(float(w) * 100, 2) for t, w in zip(tks, ws)}
        if label in POPULAR_PORTFOLIOS and POPULAR_PORTFOLIOS[label]:
            # Delegate to the global resolver — handles both legacy str
            # presets (equal-weighted) and dict presets (Schwab) cleanly.
            tks, wmap = _resolve_preset(label)
            return (tks, wmap) if tks else (None, None)
        return None, None

    _curr_tks, _curr_ws = _resolve_current_portfolio(_sel_curr)
    if _curr_tks:
        st.session_state["client_current_portfolio"] = {
            "tickers": _curr_tks,
            "weights": _curr_ws,
            "source_label": _sel_curr,
        }
        # Show a small chip strip so the advisor can confirm what got loaded
        _chips = "".join(
            f"<span style='background:#D8ECEC;color:#0E5C5E;padding:2px 8px;"
            f"border-radius:10px;font-size:0.72rem;font-weight:600;"
            f"margin-right:4px;font-family:JetBrains Mono,monospace'>"
            f"{t} {_curr_ws[t]:.0f}%</span>"
            for t in _curr_tks
        )
        st.markdown(
            f"<div style='margin-top:6px;font-size:0.78rem;color:#6B7E8A'>"
            f"Current: {_chips}</div>",
            unsafe_allow_html=True,
        )
    else:
        # Clear the override — Optimizer will fall back to analyzed portfolio
        st.session_state.pop("client_current_portfolio", None)

    st.markdown("---")
    st.caption("Optional comparison portfolios (shown alongside proposals in Results & Charts):")

    if "cmp_port1" not in st.session_state: st.session_state.cmp_port1 = "Classic 60/40 (Stocks/Bonds)"
    if "cmp_port2" not in st.session_state: st.session_state.cmp_port2 = "None"

    _CUSTOM_CMP_LABEL = "✏️ Custom (enter tickers)"
    saved_for_cmp = list(load_saved().keys())
    preset_for_cmp = [k for k in POPULAR_PORTFOLIOS if not k.startswith("── ") and k != "Custom — Enter Your Own Tickers"]
    cmp_opts = ["None"] + [f"📁 {n}" for n in saved_for_cmp] + preset_for_cmp + [_CUSTOM_CMP_LABEL]

    cc1, cc2 = st.columns(2)
    cmp1 = cc1.selectbox("Comparison 1", cmp_opts,
                          index=cmp_opts.index(st.session_state.cmp_port1)
                          if st.session_state.cmp_port1 in cmp_opts else 0,
                          key="cmp1_sel_main")
    cmp2 = cc2.selectbox("Comparison 2", cmp_opts,
                          index=cmp_opts.index(st.session_state.cmp_port2)
                          if st.session_state.cmp_port2 in cmp_opts else 0,
                          key="cmp2_sel_main")
    st.session_state.cmp_port1 = cmp1
    st.session_state.cmp_port2 = cmp2

    # Custom-ticker entry for either or both comparisons
    # Each gets a "TICKER:weight, TICKER:weight" text input shown when
    # the dropdown is set to "✏️ Custom". Weights default to equal-weight
    # if the user enters tickers without colons.
    def _parse_custom_cmp(text):
        """Parse 'AAPL:30, MSFT:50, GOOG:20' or just 'AAPL, MSFT, GOOG'.
        Returns ({ticker: weight_decimal}, normalized_label) or (None, None)."""
        if not text or not text.strip():
            return None, None
        items = [t.strip() for t in text.split(",") if t.strip()]
        weights = {}
        had_explicit_weights = False
        for item in items:
            if ":" in item:
                tkr, w = item.split(":", 1)
                try:
                    weights[tkr.strip().upper()] = float(w.strip())
                    had_explicit_weights = True
                except ValueError:
                    weights[tkr.strip().upper()] = 0.0
            else:
                weights[item.strip().upper()] = 0.0
        # If no weights given, equal-weight
        if not had_explicit_weights and weights:
            per = 1.0 / len(weights)
            weights = {t: per for t in weights}
        else:
            # Normalize to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {t: w / total for t, w in weights.items()}
        if not weights:
            return None, None
        # Build a friendly label for display
        if had_explicit_weights:
            label = "Custom: " + ", ".join(f"{t}({w*100:.0f}%)" for t, w in weights.items())
        else:
            label = "Custom: " + ", ".join(weights.keys())
        return weights, label

    if cmp1 == _CUSTOM_CMP_LABEL:
        with cc1:
            _custom_cmp1_text = st.text_input(
                "Comparison 1 tickers",
                value=st.session_state.get("cmp1_custom_text", ""),
                placeholder="AAPL:30, MSFT:50, GOOG:20",
                help=("Format: TICKER:weight%, TICKER:weight% (or just tickers "
                      "for equal-weight). Example: SPY:60, AGG:40"),
                key="cmp1_custom_text",
                label_visibility="collapsed",
            )
            _w1, _lbl1 = _parse_custom_cmp(_custom_cmp1_text)
            if _w1:
                st.session_state["cmp1_custom_weights"] = _w1
                st.session_state["cmp1_custom_label"]   = _lbl1
                st.caption(f"✅ {_lbl1}")
            else:
                st.session_state.pop("cmp1_custom_weights", None)
                st.session_state.pop("cmp1_custom_label",   None)
    else:
        st.session_state.pop("cmp1_custom_weights", None)
        st.session_state.pop("cmp1_custom_label",   None)

    if cmp2 == _CUSTOM_CMP_LABEL:
        with cc2:
            _custom_cmp2_text = st.text_input(
                "Comparison 2 tickers",
                value=st.session_state.get("cmp2_custom_text", ""),
                placeholder="VTI:70, BND:30",
                help=("Format: TICKER:weight%, TICKER:weight% (or just tickers "
                      "for equal-weight)."),
                key="cmp2_custom_text",
                label_visibility="collapsed",
            )
            _w2, _lbl2 = _parse_custom_cmp(_custom_cmp2_text)
            if _w2:
                st.session_state["cmp2_custom_weights"] = _w2
                st.session_state["cmp2_custom_label"]   = _lbl2
                st.caption(f"✅ {_lbl2}")
            else:
                st.session_state.pop("cmp2_custom_weights", None)
                st.session_state.pop("cmp2_custom_label",   None)
    else:
        st.session_state.pop("cmp2_custom_weights", None)
        st.session_state.pop("cmp2_custom_label",   None)

    if cmp1 != "None" or cmp2 != "None":
        active = [c for c in [cmp1, cmp2] if c != "None"]
        st.markdown(
            "<span style='color:#374151;font-size:0.8rem'>Comparing against: " +
            " · ".join(f"<b>{c}</b>" for c in active) + "</span>",
            unsafe_allow_html=True
        )

    # Custom weights are now part of Step 1 (above) — no separate Step 3.
    # See the "Portfolio Weights" subsection inside Securities.

    # Advanced settings
    # Section 3: Optimization Settings.
    # These have moved to the 🎨 Settings tab → Optimization Settings panel.
    # We read the values back from session_state here. Defaults match the
    # legacy in-page widget defaults exactly so behavior is unchanged for
    # advisors who don't touch the Settings tab.
    cov_estimator = st.session_state.get("opt_cov_estimator", "Ledoit-Wolf")
    mu_estimator  = st.session_state.get("opt_mu_estimator",  "Shrunk")
    use_hrp       = st.session_state.get("opt_use_hrp",    True)
    use_nco       = st.session_state.get("opt_use_nco",    True)
    use_maxdiv    = st.session_state.get("opt_use_maxdiv", True)
    use_wf        = st.session_state.get("opt_use_wf",     False)

    st.markdown("---")
    st.markdown('<div class="section-label"><div class="section-num">3</div><div class="section-title">Benchmark Comparison</div></div>', unsafe_allow_html=True)
    st.caption("All cumulative return charts will be shown relative to this benchmark.")

    if "benchmark_ticker" not in st.session_state:
        st.session_state.benchmark_ticker = "SPY"
        st.session_state.benchmark_label  = "SPY (S&P 500)"

    bm_col1, bm_col2, bm_col3 = st.columns([2, 2, 2])
    bm_type = bm_col1.radio("Benchmark type", ["Index/ETF Ticker", "Saved Portfolio"],
                             horizontal=True, key="bm_type")

    if bm_type == "Index/ETF Ticker":
        bm_input = bm_col2.text_input("Benchmark ticker", value="SPY", key="bm_ticker_input",
                                       help="e.g. SPY, QQQ, IWM, DIA, AGG")
        bm_col3.markdown("<br>", unsafe_allow_html=True)
        bm_col3.caption("SPY=S&P500, QQQ=Nasdaq, IWM=Russell2000, DIA=Dow, AGG=Bonds")
        if bm_input.strip():
            st.session_state.benchmark_ticker = bm_input.strip().upper()
            st.session_state.benchmark_label  = bm_input.strip().upper()
    else:
        saved_names = list(load_saved().keys())
        if saved_names:
            sel_saved = bm_col2.selectbox("Select saved portfolio", saved_names, key="bm_saved_sel")
            st.session_state.benchmark_ticker = f"SAVED::{sel_saved}"
            st.session_state.benchmark_label  = f"📁 {sel_saved}"
        else:
            bm_col2.info("No saved portfolios yet.")

    # Comparison portfolios set in Section 2 above


    st.markdown("---")
    st.markdown('<div class="section-label"><div class="section-num">4</div><div class="section-title">Run Analysis</div></div>', unsafe_allow_html=True)
    st.caption(f"Backtests 1, 3, 5, and 10 years ending today — {date.today().strftime('%B %d, %Y')}")

    # Spinner JS — show overlay while processing
    st.markdown("""
    <script>
    function showSpinner(text) {
        var el = document.getElementById('portfolio-spinner');
        if (el) {
            el.classList.add('active');
            var t = el.querySelector('.spinner-text');
            if (t && text) t.textContent = text;
        }
    }
    function hideSpinner() {
        var el = document.getElementById('portfolio-spinner');
        if (el) el.classList.remove('active');
    }
    </script>
    """, unsafe_allow_html=True)

    if st.button("Run Analysis", type="primary"):
        cw = custom_weights if custom_weights_valid else None
        prog = st.progress(0, text="Initializing...")
        prog.progress(5,  text=f"1-year backtest...")
        r1 = run_backtest(tickers, 1, cw, custom_weights_valid, cov_estimator, mu_estimator, use_hrp, use_nco, use_maxdiv, use_wf)
        prog.progress(28, text="📅 Running 3-year backtest...")
        r3 = run_backtest(tickers, 3, cw, custom_weights_valid, cov_estimator, mu_estimator, use_hrp, use_nco, use_maxdiv, use_wf)
        prog.progress(55, text="📅 Running 5-year backtest...")
        r5 = run_backtest(tickers, 5, cw, custom_weights_valid, cov_estimator, mu_estimator, use_hrp, use_nco, use_maxdiv, use_wf)
        prog.progress(78, text="📅 Running 10-year backtest...")
        r10 = run_backtest(tickers, 10, cw, custom_weights_valid, cov_estimator, mu_estimator, use_hrp, use_nco, use_maxdiv, use_wf)
        prog.progress(88, text="🎯 Computing security risk scores...")
        sec_risks = {}
        # ── Per-security risk scores use a FIXED 10-year window ──
        # Risk scores must be invariant to which backtest period the user is
        # viewing — they're a property of the security, not of the analysis
        # horizon. 10 years is the longest window the app uses, giving the
        # most robust vol/drawdown estimate. Tickers with shorter history fall
        # back to security_risk_score (which itself uses 3yr).
        try:
            end_dt_r   = date.today()
            start_dt_r = end_dt_r - relativedelta(years=10)
            bulk_prices, _src = get_prices(tickers, start_dt_r, end_dt_r)
            close_prices = bulk_prices  # already Close prices from get_prices()

            for tkr in tickers:
                try:
                    if tkr in close_prices.columns:
                        hist = close_prices[tkr].dropna()
                    else:
                        hist = close_prices.squeeze().dropna()

                    if len(hist) < 60:
                        sec_risks[tkr] = security_risk_score(tkr)
                        continue

                    rets  = hist.pct_change().dropna()
                    ann_v = float(rets.std() * np.sqrt(252))
                    if ann_v == 0:
                        sec_risks[tkr] = None
                        continue
                    cum   = (1 + rets).cumprod()
                    dd    = float((cum / cum.cummax() - 1).min())
                    # CAGR over realized window + excess-return Sharpe (consistent
                    # with shared/security_risk_score)
                    actual_yrs = max(len(rets) / 252.0, 0.08)
                    tr    = float(cum.iloc[-1] - 1)
                    ann_r = (1 + tr) ** (1.0 / actual_yrs) - 1
                    sh    = _shared_sharpe(ann_r, ann_v)
                    score = compute_risk_score(ann_v, dd, sh, ticker=tkr)
                    sec_risks[tkr] = {
                        "score":   score,
                        "label":   risk_label(score),
                        "ann_vol": ann_v,
                        "max_dd":  dd,
                        "sharpe":  sh,
                    }
                except Exception:
                    sec_risks[tkr] = security_risk_score(tkr)
        except Exception:
            for tkr in tickers:
                sec_risks[tkr] = security_risk_score(tkr)

        # Fetch benchmark returns for all periods
        prog.progress(94, text=f"📊 Fetching benchmark ({st.session_state.benchmark_ticker})...")
        bm_tkr = st.session_state.benchmark_ticker

        def get_bmark(years):
            if bm_tkr.startswith("SAVED::"):
                # Use saved portfolio returns
                pname = bm_tkr.replace("SAVED::", "")
                sp = load_saved().get(pname)
                if not sp:
                    return None
                try:
                    end_dt   = date.today()
                    start_dt = end_dt - relativedelta(years=years)
                    prices, _src = get_prices(sp["tickers"], start_dt, end_dt)
                    prices = prices.dropna(how="all")
                    if len(prices) < 60:
                        return None
                    X = prices_to_returns(prices)
                    _, X_test = train_test_split(X, test_size=0.33, shuffle=False)
                    w = np.array(sp["weights"])
                    if len(w) == X_test.shape[1]:
                        return (X_test.values @ w).tolist()
                except Exception:
                    return None
            else:
                return fetch_benchmark_returns(bm_tkr, years)

        bm1  = get_bmark(1)
        bm3  = get_bmark(3)
        bm5  = get_bmark(5)
        bm10 = get_bmark(10)

        # ── COMPOSITE OPTIMIZER: compute blended portfolio ──────────────────
        prog.progress(95, text="🎛️ Computing blended strategy...")
        blend_knobs = st.session_state.get("blend_knobs", {})
        active_knobs = {k: v for k, v in blend_knobs.items() if v > 0}

        def _apply_blend(bt_results, knobs, ticker_list):
            """Apply blend_strategies to a backtest results dict.
            Adds '⭐ Blended Strategy' key to the results."""
            if not bt_results or not knobs:
                return bt_results
            # Build computed_portfolios from matching strategy names in results
            available = {}
            for strat_name, knob_val in knobs.items():
                if strat_name in bt_results:
                    available[strat_name] = bt_results[strat_name]["weights"]
            if not available:
                return bt_results
            blended_w = blend_strategies(
                {k: v for k, v in knobs.items() if k in available},
                available
            )
            if blended_w is None:
                return bt_results
            # Compute returns for blended weights
            import numpy as np
            try:
                # Get returns matrix from first available strategy's test data
                ref = next(iter(available.keys()))
                rets_idx = pd.to_datetime(bt_results[ref]["index"])
                # Reconstruct daily returns matrix using Equal Weight returns as proxy
                ew_rets = np.array(bt_results.get("Equal Weight", bt_results[ref])["returns"])
                # Better: use the weighted sum of individual strategy returns
                w_arr = np.array(blended_w[:len(ticker_list)])
                if "Equal Weight" in bt_results:
                    # Approximate: weighted combo of Min Var and Max Sharpe returns
                    blend_r_parts = []
                    blend_w_parts = []
                    for sn, sv in available.items():
                        if sn in bt_results:
                            blend_r_parts.append(np.array(bt_results[sn]["returns"]))
                            knob_total = sum(v for v in available.values() if v)
                            blend_w_parts.append(knobs[sn] / knob_total if knob_total else 0)
                    if blend_r_parts:
                        min_len = min(len(r) for r in blend_r_parts)
                        blend_r_arr = np.column_stack([r[:min_len] for r in blend_r_parts])
                        blended_rets = blend_r_arr @ np.array(blend_w_parts)
                        s    = pd.Series(blended_rets)
                        cum  = (1+s).cumprod(); dd = cum/cum.cummax()-1
                        tr   = float(cum.iloc[-1]-1)
                        ar   = float(s.mean()*252)
                        av   = float(s.std()*np.sqrt(252))
                        neg  = s[s<0]
                        sh   = ar/av if av>0 else 0
                        so   = float(ar/(neg.std()*np.sqrt(252))) if len(neg)>1 else 0
                        bt_results["⭐ Blended Strategy"] = {
                            "weights":      blended_w,
                            "ann_return":   ar,
                            "ann_vol":      av,
                            "sharpe":       sh,
                            "sortino":      so,
                            "calmar":       abs(ar/dd.min()) if dd.min()!=0 else 0,
                            "max_drawdown": float(dd.min()),
                            "total_return": tr,
                            "returns":      blended_rets.tolist(),
                            "index":        bt_results[ref]["index"][:min_len],
                        }
            except Exception:
                pass
            return bt_results

        if active_knobs and r1:
            r1  = _apply_blend(r1,  active_knobs, tickers)
            r3  = _apply_blend(r3,  active_knobs, tickers) if r3 else r3
            r5  = _apply_blend(r5,  active_knobs, tickers) if r5 else r5
            r10 = _apply_blend(r10, active_knobs, tickers) if r10 else r10

            # Persist the blend basis so the Optimizer's auto-regen pathway
            # can use it as Option 2 (proposed) when sliders are set.
            # Pulled from bt10's blended-strategy entry if available
            # (10y has the most representative weights), else bt1.
            try:
                _blend_src = (r10 or r1)[0] if isinstance(r10 or r1, tuple) else (r10 or r1)
                _bs = (_blend_src or {}).get("⭐ Blended Strategy") if _blend_src else None
                if _bs and _bs.get("weights"):
                    _bw = list(_bs["weights"])
                    # Normalize lengths in case of tail truncation
                    _btks = list(tickers)[:len(_bw)]
                    st.session_state["proposal_blend_basis"] = {
                        "balanced_tickers": _btks,
                        "balanced_weights": [float(w) for w in _bw[:len(_btks)]],
                    }
            except Exception:
                pass
        else:
            # No active blend → clear basis so Optimizer falls back to Step 1
            st.session_state.pop("proposal_blend_basis", None)

        prog.progress(100, text="✅ Complete!")
        st.session_state.bt1       = r1
        st.session_state.bt3       = r3
        st.session_state.bt5       = r5
        st.session_state.bt10      = r10
        st.session_state.tickers   = tickers
        # Persist the user's submitted allocation (Step 3 custom weights, or
        # equal-weight fallback). Used by the Optimizer tab's "Current Allocation"
        # pie at the top so the advisor always sees the live submitted portfolio.
        if custom_weights and custom_weights_valid:
            st.session_state.submitted_weights = dict(custom_weights)
        else:
            _eq = 100.0 / len(tickers) if tickers else 0
            st.session_state.submitted_weights = {t: _eq for t in tickers}
        st.session_state.ran_once  = True
        st.session_state.cov_est   = cov_estimator
        st.session_state.mu_est    = mu_estimator
        st.session_state.use_wf    = use_wf
        st.session_state.sec_risks = sec_risks
        # Reset the 10-year risk-scoring cache so this run starts fresh.
        # Without this, switching tickers and re-running would still show the
        # previous portfolio's 10yr-scored risk in the matrix.
        st.session_state.pop("_pcm_scoring_stats_10y", None)
        st.session_state["bmark_returns_1 Year"]   = bm1
        st.session_state["bmark_returns_3 Years"]  = bm3
        st.session_state["bmark_returns_5 Years"]  = bm5
        st.session_state["bmark_returns_10 Years"] = bm10
        st.session_state.benchmark_label = st.session_state.get("benchmark_label", "SPY")
        prog.empty()
        n_strats = len(r1[0]) if r1 else 0
        st.success(f"✅ Analysis complete! {n_strats} strategies × 4 periods. Risk scores computed.")

    if st.session_state.ran_once and st.session_state.bt1:
        tickers   = st.session_state.tickers
        bt1_data  = st.session_state.bt1
        bt3_data  = st.session_state.bt3
        bt5_data  = st.session_state.bt5
        bt10_data = st.session_state.bt10
        bt1       = bt1_data[0]  if bt1_data  else {}
        bt3       = bt3_data[0]  if bt3_data  else {}
        bt5       = bt5_data[0]  if bt5_data  else {}
        bt10      = bt10_data[0] if bt10_data else {}
        ef1       = bt1_data[1]  if bt1_data  else []
        sec_risks = st.session_state.get("sec_risks", {})


        # ── SECURITY RISK SCORES (individual tickers) shown inside render_tab at bottom
        # This section intentionally left for render_tab to handle

        def render_tab(results, ef_points, label, mode="results"):
            # Helper to strip emoji prefixes from portfolio labels for display.
            # Keeps the 👤 silhouette on "Client's Current" (per advisor spec),
            # strips everything else. Internal label strings are unchanged —
            # this is purely a render-time cosmetic transform.
            import re as _re_clean
            _CLEAN_KEEP_PREFIX = "👤"
            def _clean_label(s):
                if not isinstance(s, str):
                    return s
                if s.startswith(_CLEAN_KEEP_PREFIX):
                    return s  # preserve the silhouette + the rest
                # Strip a leading emoji + optional whitespace
                # (covers 🟦 ⚖️ 📈 🧩 📁 ⭐ 🛡️ 🚀 etc.)
                return _re_clean.sub(
                    r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*",
                    "", s
                ).strip() or s

            # Guard: results must be a non-empty dict of strategy results
            if not results or not isinstance(results, dict):
                st.warning(f"No results available for {label}.")
                return
            # Guard: keys must be strings (not nested dicts)
            if results and not isinstance(next(iter(results)), str):
                st.warning(f"Unexpected results format for {label}.")
                return
            # Narrow modes — skip PCM/charts, render only the requested section
            # scores_only and alloc_only fall through to their sections below
            if mode in ("scores_only", "alloc_only"):
                pass  # handled by section guards below

            # ── Data-integrity callout: show actual date range and flag shortfalls ─
            _my_k = next((k for k in results if k.startswith("⭐ ")), None)
            if _my_k and results[_my_k].get("index"):
                _idx        = results[_my_k]["index"]
                _n          = len(_idx)
                _yrs_actual = round(_n / 252, 1)
                # Compare against the label's requested period
                _yrs_req = 0
                if "Year"  in label: _yrs_req = 1
                if "3 Years"  in label: _yrs_req = 3
                if "5 Years"  in label: _yrs_req = 5
                if "10 Years" in label: _yrs_req = 10
                if _yrs_req and _yrs_actual < _yrs_req * 0.85:
                    st.warning(
                        f"⚠️ **{label}** — requested {_yrs_req}y but only "
                        f"**{_yrs_actual}y of data** was available for your portfolio "
                        f"(from `{_idx[0][:10]}` to `{_idx[-1][:10]}`). "
                        f"One or more tickers have shorter history. "
                        f"Add longer-history tickers or switch to the 1y/3y tab for more reliable stats."
                    )
                # (Previously emitted a "📅 My Portfolio data: NN trading days"
                # caption here — removed per UI cleanup, redundant with the
                # period selector at the top of the tab.)

            # ── CANONICAL COMPARISON ORDER ─────────────────────────
            # Honors the three Fixed market benchmark toggles at the top of
            # this tab. Replaces the older 60/40 + 90/10 + S&P + Conservative
            # set. Order: bonds → balanced → equity, so the matrix reads from
            # most-conservative to most-aggressive across the benchmark
            # columns left-to-right.
            COMP_ORDER  = []
            COMP_COLORS = {}
            if st.session_state.get("show_bench_bnd", True):
                COMP_ORDER.append("🟦 100% Bonds (BND)")
                COMP_COLORS["🟦 100% Bonds (BND)"] = "#2563eb"
            if st.session_state.get("show_bench_6040", True):
                COMP_ORDER.append("⚖️ 60/40 (SPY+AGG)")
                COMP_COLORS["⚖️ 60/40 (SPY+AGG)"] = "#059669"
            if st.session_state.get("show_bench_spy", True):
                COMP_ORDER.append("📈 S&P 500 (SPY)")
                COMP_COLORS["📈 S&P 500 (SPY)"] = "#dc2626"
            MY_PORT_COLOR = "#7c3aed"  # Purple for My Portfolio

            # PROXY_MAP, resolve_ticker, get_prices_with_proxies defined at top level
            # ── Date range for this tab — used throughout ──────────
            yrs_pcm   = int(label.split()[0]) if label.split()[0].isdigit() else 1
            end_pcm   = date.today()
            start_pcm = end_pcm - relativedelta(years=yrs_pcm)

            # ── METRICS PLACEHOLDER (filled after std_chart_comparisons) ──
            _metrics_box = st.empty()

            if mode not in ("scores_only", "alloc_only"):
                # ── PCM: DEFAULT 6 COLUMNS ──────────────────────────
                st.markdown("#### Portfolio Comparison Matrix")
                # (Previously rendered a "My Portfolio · {benchmark1} ·
                # {benchmark2} · 1yr period" caption beneath the header —
                # removed per UI cleanup, redundant with the legend.)

                pcm_standards = {}

                # ══════════════════════════════════════════════════════════════════
                # CORE DATA INTEGRITY SYSTEM
                # Every metric is computed from ACTUAL price history for EXACT period.
                # YTD, 1yr, 3yr, 5yr, 10yr — each period fetched and computed separately.
                # Uses get_prices_with_proxies so FBTC→GBTC, GLDM→GLD etc. auto-applied.
                # ══════════════════════════════════════════════════════════════════

                # port_stats_from_prices defined at top level
                # ── Period dates ─────────────────────────────────────
                _today     = date.today()
                _ytd_start = date(_today.year, 1, 1)

                # ── Comparison portfolio definitions ──────────────────
                # Built dynamically from the three Fixed market benchmark
                # toggles. Empty dict = all toggles off → no benchmark
                # columns (the loop below silently skips an empty defs
                # dict and only the user's portfolio is shown).
                _CMP_DEFS = {}
                if st.session_state.get("show_bench_bnd", True):
                    _CMP_DEFS["🟦 100% Bonds (BND)"] = {"BND": 1.0}
                if st.session_state.get("show_bench_6040", True):
                    _CMP_DEFS["⚖️ 60/40 (SPY+AGG)"] = {"SPY": 0.60, "AGG": 0.40}
                if st.session_state.get("show_bench_spy", True):
                    _CMP_DEFS["📈 S&P 500 (SPY)"]   = {"SPY": 1.0}

                # ── Fetch comparison prices for this tab's period ─────
                # IMPORTANT: derive the ticker list directly from _CMP_DEFS so
                # we never silently miss a ticker (which would cause that
                # ticker to be filtered out by port_stats_from_prices and
                # cause the portfolio to look identical to one without it —
                # e.g. 90/10 was rendering identically to S&P 500 because
                # BIL was missing from the fetch list).
                try:
                    _cmp_tickers = sorted({
                        t for d in _CMP_DEFS.values() for t in d.keys()
                    })
                    if not _cmp_tickers:
                        # All three benchmark toggles are off — skip the matrix
                        # benchmark fetch entirely; pcm_standards stays empty
                        # so only My Portfolio renders in the matrix.
                        pass
                    else:
                        _cmp_p, _    = get_prices_cached(
                            tuple(_cmp_tickers),
                            str(start_pcm), str(_today))
                        _cmp_p       = _cmp_p.ffill()

                        for _sname, _swts in _CMP_DEFS.items():
                            _st = port_stats_from_prices(_cmp_p, _swts, yrs_pcm)
                            if _st: pcm_standards[_sname] = _st
                except Exception: pass

                # ── PARALLEL 10-YEAR FETCH FOR RISK SCORING ───────────
                # Risk scores must NOT change as the user flips between 1y/3y/
                # 5y/10y tabs. We compute every portfolio's 10-year vol/drawdown
                # once (cached in session_state across reruns) and use those
                # numbers for the Risk Score column in every tab's matrix.
                # Other metrics (return, Sharpe, etc.) remain period-specific.
                _SCORING_YEARS = 10
                _scoring_stats = st.session_state.get("_pcm_scoring_stats_10y", None)
                if _scoring_stats is None:
                    _scoring_stats = {}
                    try:
                        _start_10y = end_pcm - relativedelta(years=_SCORING_YEARS)
                        # Use proxy-stitched prices so short-history tickers
                        # (e.g. SGOV ~2yr, FBTC ~1yr) get back-filled with
                        # their proxy's older history. Without this, PCM's
                        # 10yr vol diverges from every other scoring path
                        # in the app — which all use get_prices_with_proxies
                        # via _cached_portfolio_vol. Two scoring paths
                        # against different price series produce different
                        # diversification-adjusted risk scores for the
                        # SAME tickers + weights.
                        _cmp_p_10y, _ = get_prices_with_proxies(
                            tuple(_cmp_tickers),
                            str(_start_10y), str(_today),
                            min_days=max(60, int(_SCORING_YEARS * 60)),
                        )
                        if not _cmp_p_10y.empty:
                            for _sname, _swts in _CMP_DEFS.items():
                                _s10 = port_stats_from_prices(_cmp_p_10y, _swts, _SCORING_YEARS)
                                if _s10: _scoring_stats[_sname] = _s10
                    except Exception:
                        pass
                    # Persist for the other tabs/render passes; rebuilt only
                    # when the analysis is re-run (key cleared on re-run).
                    st.session_state["_pcm_scoring_stats_10y"] = _scoring_stats

                # (Equal Weight removed — comparison matrix only uses 60/40, 90/10, S&P 500, Conservative)

                # ── My Portfolio: exact per-ticker price history ───────
                _my_label_pcm    = next((k.replace("⭐ ","") for k in results
                                         if k.startswith("⭐ ")), "My Portfolio")
                my_port_keys_pcm = ([k for k in results if k.startswith("⭐ ")]
                                     or list(results.keys())[:1])
                my_port = {}
                if my_port_keys_pcm and tickers:
                    _r = results[my_port_keys_pcm[0]]
                    # Determine weights: prefer loaded portfolio weights
                    _loaded_w = st.session_state.get("loaded_weights", {})
                    if _loaded_w and all(t in _loaded_w for t in tickers):
                        _my_wts = {t: _loaded_w[t]/100.0 for t in tickers}
                    else:
                        _bw = _r.get("weights", [1/len(tickers)]*len(tickers))
                        _my_wts = {t: float(_bw[i]) if i < len(_bw) else 1/len(tickers)
                                   for i, t in enumerate(tickers)}
                    # Normalize
                    _wsum = sum(_my_wts.values())
                    _my_wts = {t: v/_wsum for t, v in _my_wts.items()}
                    try:
                        # Fetch actual prices with proxy substitution
                        _my_p, _proxies = get_prices_with_proxies(tuple(tickers), start_pcm, str(_today))
                        if not _my_p.empty:
                            _st = port_stats_from_prices(_my_p, _my_wts, yrs_pcm)
                            if _st:
                                my_port[_my_label_pcm] = _st
                    except Exception: pass
                    # Fallback to backtest returns
                    if not my_port:
                        try:
                            _bs = pd.Series(_r["returns"], index=pd.to_datetime(_r["index"]))
                            _st = port_stats_from_prices(
                                _bs.to_frame("port"), {"port":1.0}, yrs_pcm)
                            if _st: my_port[_my_label_pcm] = _st
                        except Exception: pass
                    if not my_port:
                        my_port[_my_label_pcm] = {k:_r.get(k,0) for k in
                            ["ann_return","ann_vol","sharpe","sortino","calmar",
                             "max_drawdown","total_return","cvar_5"]}
                        my_port[_my_label_pcm]["ytd_return"] = 0.0
                        # Also record tickers/weights so PCM scoring + ER work
                        my_port[_my_label_pcm]["tickers"] = list(_my_wts.keys())
                        my_port[_my_label_pcm]["weights"] = list(_my_wts.values())
                        my_port[_my_label_pcm]["weights_dict"] = dict(_my_wts)

                    # ── 10yr stats for My Portfolio (risk-score scoring only) ──
                    if _my_label_pcm not in _scoring_stats:
                        try:
                            _start_10y = end_pcm - relativedelta(years=_SCORING_YEARS)
                            _my_p_10y, _ = get_prices_with_proxies(
                                tuple(tickers), _start_10y, str(_today))
                            if not _my_p_10y.empty:
                                _s10 = port_stats_from_prices(_my_p_10y, _my_wts, _SCORING_YEARS)
                                if _s10:
                                    _scoring_stats[_my_label_pcm] = _s10
                                    st.session_state["_pcm_scoring_stats_10y"] = _scoring_stats
                        except Exception:
                            pass

                # ── Step 2 user-selected comparison portfolios ─────────
                # cmp_port1/cmp_port2 can be: "None", "📁 SavedName", a
                # POPULAR_PORTFOLIOS preset key, or "✏️ Custom (enter tickers)".
                # The custom option pulls weights from cmp{1,2}_custom_weights
                # which were parsed in the Analyzer's Step 2 input.
                def _resolve_cmp_selection(sel, idx=None):
                    """Return (tickers_list, weights_list, display_name) or None.
                    `idx` is 1 or 2 — used to pull the right custom-weights
                    state when `sel` is the custom-tickers option."""
                    if not sel or sel == "None":
                        return None
                    if sel.startswith("📁 "):
                        sp = load_saved().get(sel[2:])
                        if sp:
                            tks = sp["tickers"]
                            wts = sp.get("weights") or [1.0/len(tks)] * len(tks)
                            return tks, wts, sel
                    # Custom-tickers comparison from Step 2 text input
                    if sel.startswith("✏️ Custom") and idx in (1, 2):
                        cw = st.session_state.get(f"cmp{idx}_custom_weights")
                        cl = st.session_state.get(f"cmp{idx}_custom_label")
                        if cw:
                            return list(cw.keys()), list(cw.values()), (cl or "✏️ Custom")
                        return None
                    # Popular preset — handles legacy str (equal-weighted)
                    # and Schwab dict (specific weights) uniformly.
                    if sel in POPULAR_PORTFOLIOS and POPULAR_PORTFOLIOS[sel]:
                        tks, wmap = _resolve_preset(sel)
                        if tks:
                            # _resolve_preset returns weights as percentages;
                            # this caller wants decimals (sums to 1.0).
                            _total = sum(wmap.values()) or 1.0
                            wts = [wmap.get(t, 0.0) / _total for t in tks]
                            return tks, wts, f"🧩 {sel[:14]}"
                    return None

                user_cmps = {}

                # ── Client's current portfolio (Step 2) ─────────────────
                # The dedicated "client's current portfolio" picker from
                # Section 2 of the Analyzer. Distinct from cmp_port1/cmp_port2
                # (which are advisor-defined comparisons) — this is what the
                # client actually holds today. Shows up in PCM as its own
                # column so the advisor can compare current vs proposed vs
                # market benchmarks side-by-side.
                _curr_override = st.session_state.get("client_current_portfolio")
                if _curr_override and _curr_override.get("tickers"):
                    _curr_tks = list(_curr_override["tickers"])
                    _curr_w_pct = _curr_override.get("weights") or {}
                    # Convert percentages → decimals for port_stats_from_prices
                    _curr_wmap = {
                        t: float(_curr_w_pct.get(t, 0)) / 100.0
                        for t in _curr_tks
                        if float(_curr_w_pct.get(t, 0)) > 0
                    }
                    if _curr_wmap:
                        _curr_label_src = _curr_override.get("source_label", "Client's Current")
                        # Display label — short enough to fit a PCM column header
                        _curr_dname = "👤 Client's Current"
                        try:
                            _cp, _ = get_prices_with_proxies(
                                tuple(_curr_wmap.keys()), start_pcm, str(_today))
                            if not _cp.empty:
                                _st = port_stats_from_prices(_cp, _curr_wmap, yrs_pcm)
                                if _st:
                                    user_cmps[_curr_dname] = _st
                        except Exception:
                            pass
                        # 10yr stats for risk scoring
                        if _curr_dname not in _scoring_stats:
                            try:
                                _start_10y = end_pcm - relativedelta(years=_SCORING_YEARS)
                                _cp_10y, _ = get_prices_with_proxies(
                                    tuple(_curr_wmap.keys()), _start_10y, str(_today))
                                if not _cp_10y.empty:
                                    _s10 = port_stats_from_prices(_cp_10y, _curr_wmap, _SCORING_YEARS)
                                    if _s10:
                                        _scoring_stats[_curr_dname] = _s10
                                        st.session_state["_pcm_scoring_stats_10y"] = _scoring_stats
                            except Exception:
                                pass

                for _i, _cmp_key in enumerate(("cmp_port1", "cmp_port2"), start=1):
                    _sel = st.session_state.get(_cmp_key, "None")
                    _resolved = _resolve_cmp_selection(_sel, idx=_i)
                    if not _resolved:
                        continue
                    _tks, _wts, _dname = _resolved
                    try:
                        _cp, _ = get_prices_with_proxies(
                            tuple(_tks), start_pcm, str(_today))
                        if not _cp.empty:
                            _wmap = {t: w for t, w in zip(_tks, _wts)}
                            _st = port_stats_from_prices(_cp, _wmap, yrs_pcm)
                            if _st:
                                user_cmps[_dname] = _st
                    except Exception:
                        pass
                    # 10yr stats for risk scoring (cached across periods)
                    if _dname not in _scoring_stats:
                        try:
                            _start_10y = end_pcm - relativedelta(years=_SCORING_YEARS)
                            _cp_10y, _ = get_prices_with_proxies(
                                tuple(_tks), _start_10y, str(_today))
                            if not _cp_10y.empty:
                                _wmap = {t: w for t, w in zip(_tks, _wts)}
                                _s10 = port_stats_from_prices(_cp_10y, _wmap, _SCORING_YEARS)
                                if _s10:
                                    _scoring_stats[_dname] = _s10
                                    st.session_state["_pcm_scoring_stats_10y"] = _scoring_stats
                        except Exception:
                            pass

                default_cols = {k: pcm_standards[k] for k in COMP_ORDER if k in pcm_standards}
                # Order: My Portfolio → Step 2 inputs (client current + cmp1 + cmp2)
                # → fixed benchmarks. Cap at 8 columns so the table stays
                # readable but all Step 2 selections + fixed benchmarks fit
                # (1 my_port + 3 step 2 + 3 benchmarks = 7, plus 1 headroom).
                all_portfolios = {**my_port, **user_cmps, **default_cols}
                all_portfolios = dict(list(all_portfolios.items())[:8])

                # Controls row — optional extra strategies to overlay on charts
                extra_sel = st.multiselect(
                    "Add optimized strategies to comparison charts",
                    options=[k for k in results.keys() if not k.startswith("⭐ ")],
                    default=[], key=f"pcm_extra_{label}",
                    placeholder="Add optimized strategies to charts...",
                )
                # Add selected strategies to PCM columns (max 6)
                for es in extra_sel:
                    if es in results and es not in all_portfolios:
                        if len(all_portfolios) >= 6:
                            last_key = list(all_portfolios.keys())[-1]
                            del all_portfolios[last_key]
                        # Augment with tickers/weights so PCM scoring + ER work
                        _r_es = dict(results[es])
                        _w_es = _r_es.get("weights", [])
                        if _w_es and tickers:
                            _r_es["tickers"] = list(tickers[:len(_w_es)])
                            _r_es["weights"] = list(_w_es)
                            _r_es["weights_dict"] = {
                                t: float(w) for t, w in zip(tickers, _w_es)
                            }
                        all_portfolios[es] = _r_es
                # Store for use in charts below
                st.session_state[f"chart_extra_{label}"] = extra_sel

                bm_rets = st.session_state.get("bmark_returns_" + label)
                bm_lbl  = st.session_state.get("benchmark_label", "SPY")
                bm_ann  = float(pd.Series(bm_rets).mean() * 252) if bm_rets else None

                metric_names = [
                    "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio",
                    "Calmar Ratio", "Max Drawdown", "Total Return", "YTD Return",
                    "CVaR (5%)", "Expense Ratio", "Diversification", "Risk Score",
                ]

                # YTD: Jan 1 of current year to today
                _ytd_start = date(date.today().year, 1, 1)

                matrix_data = {"Metric": metric_names}
                for pname, r in all_portfolios.items():
                    ar=r["ann_return"]; av=r["ann_vol"]; sh=r["sharpe"]
                    so=r.get("sortino",0); ca=r.get("calmar",0); md=r["max_drawdown"]
                    tr=r["total_return"]; cv=r.get("cvar_5", ar/252 - 2*av/np.sqrt(252))

                    # ── Risk Score uses 10-YEAR vol/drawdown, not period-specific ──
                    # Risk is a property of the portfolio, not of which tab the user
                    # is on. Pull 10yr stats from _scoring_stats; fall back to the
                    # period values if 10yr data isn't available (short-history
                    # tickers etc.).
                    _score_st = _scoring_stats.get(pname, {}) if _scoring_stats else {}
                    _av_score = float(_score_st.get("ann_vol", av) or av)
                    _md_score = float(_score_st.get("max_drawdown", md) or md)
                    _sh_score = float(_score_st.get("sharpe",  sh) or sh)

                    # ── Risk Score: per-holding weighted-avg + correlation ──
                    # If we have tickers/weights, use the proper portfolio scorer
                    # (classifies each holding via _classify_ticker, applies caps,
                    # discounts for diversification). Otherwise fall back to the
                    # equity-class composite.
                    _p_tks = r.get("tickers", [])
                    _p_wts = r.get("weights", [])
                    div_ratio = None   # diversification ratio for this portfolio
                    if _p_tks and _p_wts:
                        _h_scores = []
                        _h_vols   = []
                        for _tt in _p_tks:
                            _rr = security_risk_score(_tt)
                            if _rr:
                                _h_scores.append(_rr["score"])
                                _h_vols.append(_rr.get("ann_vol", 0.15))
                            else:
                                _h_scores.append(50)
                                _h_vols.append(0.15)
                        # Use 10yr portfolio vol for scoring (locked across tabs)
                        rs = compute_portfolio_risk_score(
                            _p_tks, _p_wts,
                            holding_scores=_h_scores,
                            holding_vols=_h_vols,
                            portfolio_vol=_av_score,
                        )
                        # Diversification ratio: 1.00 = perfectly correlated,
                        # < 1.0 = some diversification benefit. Uses the period's
                        # `av` (not 10yr) since this is informational, not a score.
                        _w_arr = np.array([float(x or 0) for x in _p_wts])
                        _w_sum = _w_arr.sum()
                        if _w_sum > 0:
                            _w_norm = _w_arr / _w_sum
                            _wsv = float(np.dot(_w_norm,
                                                np.array(_h_vols, dtype=float)))
                            if _wsv > 1e-6:
                                div_ratio = max(0.0, min(1.0, av / _wsv))
                    else:
                        # Fallback path also uses 10yr values for scoring
                        rs = compute_risk_score(_av_score, _md_score, _sh_score,
                                                asset_class="equity")
                    # ── Weighted Expense Ratio ──
                    if _p_tks and _p_wts:
                        _wer, _cov = weighted_expense_ratio(_p_tks, _p_wts)
                        if _cov > 0:
                            er_str = f"{_wer*100:.2f}%"
                            if _cov < 95:
                                er_str += f"  ({_cov:.0f}% covered)"
                        else:
                            er_str = "—"
                    else:
                        er_str = "—"
                    # ── Diversification ratio cell ──
                    # Render as ratio + qualitative tag so the meaning is clear
                    if div_ratio is not None:
                        if div_ratio >= 0.95:
                            _div_tag = "concentrated"
                        elif div_ratio >= 0.80:
                            _div_tag = "moderate"
                        elif div_ratio >= 0.60:
                            _div_tag = "diversified"
                        else:
                            _div_tag = "well-diversified"
                        div_str = f"{div_ratio:.2f}  ({_div_tag})"
                    else:
                        div_str = "—"
                    alpha = f"{ar-bm_ann:+.2%}" if bm_ann is not None else "—"
                    # PCM column header — emoji-stripped per advisor preference
                    # (only the 👤 silhouette on Client's Current is preserved).
                    short = _clean_label(pname.replace("⭐ ",""))[:24]
                    # Order must match metric_names list exactly
                    ytd = r.get("ytd_return", 0)
                    matrix_data[short] = [
                        f"{ar:.2%}", f"{av:.2%}", f"{sh:.2f}", f"{so:.2f}",
                        f"{ca:.2f}", f"{md:.2%}", f"{tr:.2%}", f"{ytd:.2%}",
                        f"{cv:.2%}", er_str, div_str, str(rs),
                    ]

                matrix_df = pd.DataFrame(matrix_data)
                st.dataframe(matrix_df,
                             use_container_width=True, hide_index=True,
                             column_config={"Metric": st.column_config.TextColumn("Metric", width="medium")})
                if bm_ann:
                    st.caption(f"📊 Benchmark: {bm_lbl} · Ann. Return {bm_ann:.2%}")
                st.caption(
                    "💡 **Diversification ratio** = portfolio volatility ÷ weighted "
                    "sum of individual holding volatilities. **1.00** means the "
                    "holdings move perfectly together (no diversification benefit). "
                    "**Lower values** indicate that uncorrelated movements between "
                    "holdings are reducing overall portfolio risk — a "
                    "well-diversified portfolio typically lands in the 0.55–0.75 range."
                )

                # ── DEBUG: per-portfolio breakdown ────────────────────
                # Lets the user verify expense ratio + diversification ratio
                # calculations by seeing which tickers contributed and where
                # data is missing. Open if numbers look off.
                with st.expander("🔍 Show calculation details (per-portfolio breakdown)", expanded=False):
                    for pname, r in all_portfolios.items():
                        _p_tks = r.get("tickers", [])
                        _p_wts = r.get("weights", [])
                        if not _p_tks or not _p_wts:
                            st.markdown(f"**{pname}**: _(no holdings recorded — using portfolio-level fallback)_")
                            continue
                        st.markdown(f"**{pname}** — {len(_p_tks)} holdings")
                        _w_arr = np.array([float(x or 0) for x in _p_wts])
                        _w_arr = _w_arr / _w_arr.sum() if _w_arr.sum() > 0 else _w_arr
                        _rows = []
                        for _t, _w in zip(_p_tks, _w_arr):
                            _rs = security_risk_score(_t)
                            _er = _expense_ratio_for_ticker(_t)
                            _cls, _ct = _classify_ticker(_t)
                            _rows.append({
                                "Ticker": _t,
                                "Weight": f"{_w*100:.1f}%",
                                "Class":  f"{_cls}/{_ct}" if _cls == "bond" else _cls,
                                "Vol":    f"{_rs.get('ann_vol', 0)*100:.1f}%" if _rs else "—",
                                "Score":  _rs["score"] if _rs else "—",
                                "ER":     ("0.00%" if _er == 0.0 else
                                          f"{_er*100:.2f}%" if _er is not None else "—"),
                            })
                        st.dataframe(pd.DataFrame(_rows), hide_index=True,
                                    use_container_width=True)
                        # Also show the aggregated metrics so the user can verify
                        _wer, _cov = weighted_expense_ratio(_p_tks, _p_wts)
                        st.caption(
                            f"→ Weighted ER: **{_wer*100:.3f}%** "
                            f"(coverage {_cov:.0f}%)  ·  "
                            f"Portfolio vol: **{r.get('ann_vol', 0)*100:.2f}%**"
                        )
                        st.markdown("---")

                # ── STANDARD COMPARISON PORTFOLIOS FOR CHARTS ─────────
                std_chart_comparisons = {}
                try:
                    yrs_cmp   = int(label.split()[0]) if label.split()[0].isdigit() else 3
                    end_cmp   = date.today()
                    start_cmp = end_cmp - relativedelta(years=max(1, yrs_cmp))
                    # ── Fixed market benchmarks (BND / 60-40 / SPY) ──────
                    # Three universal reference portfolios. Each is shown
                    # only if its toggle in the Fixed market benchmarks
                    # expander (top of this tab) is on. Replaces the older
                    # 60/40 + 90/10 + S&P + Conservative set.
                    std_cmp_defs = []
                    if st.session_state.get("show_bench_bnd", True):
                        std_cmp_defs.append(("🟦 100% Bonds (BND)",     {"BND": 1.0}))
                    if st.session_state.get("show_bench_6040", True):
                        std_cmp_defs.append(("⚖️ 60/40 (SPY+AGG)",      {"SPY": 0.60, "AGG": 0.40}))
                    if st.session_state.get("show_bench_spy", True):
                        std_cmp_defs.append(("📈 S&P 500 (SPY)",         {"SPY": 1.0}))
                    # Derive ticker list directly from the defs so we never
                    # silently drop one (e.g. BIL was missing previously and
                    # made 90/10 look identical to S&P 500)
                    _need_tks = sorted({t for _, d in std_cmp_defs for t in d.keys()})
                    if not _need_tks:
                        # All three benchmark toggles are off — nothing to fetch.
                        cmp_prices_std = pd.DataFrame()
                    else:
                        cmp_prices_std, _src = get_prices_cached(
                            tuple(_need_tks), str(start_cmp), str(end_cmp))
                        cmp_prices_std = cmp_prices_std.ffill()

                    def cmp_port_rets(wts_dict):
                        cols = [c for c in wts_dict if c in cmp_prices_std.columns]
                        if not cols: return None
                        w    = np.array([wts_dict[c] for c in cols]); w=w/w.sum()
                        rets = cmp_prices_std[cols].pct_change().dropna()
                        return pd.Series((rets.values @ w), index=rets.index)

                    for cname, cwts in std_cmp_defs:
                        r = cmp_port_rets(cwts)
                        if r is not None: std_chart_comparisons[cname] = r

                    # (Equal Weight removed from chart comparisons per advisor preference.)

                    # ── Client's Current Portfolio (Step 2) ─────────
                    # Mirror the PCM treatment: when the advisor has set a
                    # client-current portfolio in Step 2, it should appear
                    # alongside the analyzed portfolio + comparison portfolios
                    # on every period chart (cumulative return, drawdown,
                    # rolling Sharpe). Previously it was only fed to PCM,
                    # which is why the user saw 5 columns in PCM but only 4
                    # lines on each chart.
                    _curr_chart = st.session_state.get("client_current_portfolio")
                    if _curr_chart and _curr_chart.get("tickers"):
                        try:
                            _curr_tks_c = list(_curr_chart["tickers"])
                            _curr_w_pct_c = _curr_chart.get("weights") or {}
                            _curr_w_dec = {
                                t: float(_curr_w_pct_c.get(t, 0)) / 100.0
                                for t in _curr_tks_c
                                if float(_curr_w_pct_c.get(t, 0)) > 0
                            }
                            if _curr_w_dec:
                                _cp_c, _ = get_prices_with_proxies(
                                    tuple(_curr_w_dec.keys()),
                                    str(start_cmp), str(end_cmp))
                                _cp_c = _cp_c.ffill().dropna(how="all")
                                _vcols_c = [t for t in _curr_w_dec if t in _cp_c.columns]
                                if _vcols_c:
                                    _w_c = np.array([_curr_w_dec[t] for t in _vcols_c])
                                    _w_c = _w_c / _w_c.sum() if _w_c.sum() > 0 else _w_c
                                    _r_c = _cp_c[_vcols_c].pct_change().dropna()
                                    std_chart_comparisons["👤 Client's Current"] = pd.Series(
                                        _r_c.values @ _w_c, index=_r_c.index)
                        except Exception:
                            pass

                    # User Step 2 selections (now supports presets in addition to 📁 saved)
                    for cmp_key in ["cmp_port1", "cmp_port2"]:
                        cmp_sel = st.session_state.get(cmp_key, "None")
                        if not cmp_sel or cmp_sel == "None":
                            continue
                        try:
                            _cmp_tks, _cmp_wts = None, None
                            _cmp_label = cmp_sel
                            if cmp_sel.startswith("📁 "):
                                sp = load_saved().get(cmp_sel[2:])
                                if sp:
                                    _cmp_tks = sp["tickers"]
                                    _cmp_wts = sp.get("weights") or [1.0/len(_cmp_tks)]*len(_cmp_tks)
                            elif cmp_sel in POPULAR_PORTFOLIOS and POPULAR_PORTFOLIOS[cmp_sel]:
                                # _resolve_preset returns percentage weights;
                                # this caller wants decimals.
                                _cmp_tks, _wmap = _resolve_preset(cmp_sel)
                                if _cmp_tks:
                                    _total = sum(_wmap.values()) or 1.0
                                    _cmp_wts = [_wmap.get(t, 0.0) / _total for t in _cmp_tks]
                                    _cmp_label = f"🧩 {cmp_sel[:14]}"
                            if _cmp_tks and _cmp_wts:
                                cp, _src = get_prices_with_proxies(
                                    tuple(_cmp_tks), str(start_cmp), str(end_cmp))
                                cp = cp.ffill().dropna(how="all")
                                vcols2 = [t for t in _cmp_tks if t in cp.columns]
                                if vcols2:
                                    cw = np.array([_cmp_wts[_cmp_tks.index(t)] for t in vcols2])
                                    cw = cw / cw.sum()
                                    cr = cp[vcols2].pct_change().dropna()
                                    std_chart_comparisons[_cmp_label] = pd.Series(
                                        cr.values @ cw, index=cr.index)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Sort in canonical order (keep user comparisons at the end, order preserved)
                _ordered = {k: std_chart_comparisons[k] for k in COMP_ORDER if k in std_chart_comparisons}
                _user = {k: v for k, v in std_chart_comparisons.items() if k not in _ordered}
                std_chart_comparisons = {**_ordered, **_user}

                # ── Align comparison series to My Portfolio date range ─
                # results use a test-split window; align comparisons to same period
                if results:
                    _ref_key = next((k for k in results if k.startswith("⭐ ")), list(results.keys())[0])
                    _ref_idx = pd.to_datetime(results[_ref_key]["index"])
                    _ref_start, _ref_end = _ref_idx[0], _ref_idx[-1]

                    def _align(s):
                        """Normalize tz and trim to ref window. yfinance returns tz-aware;
                        results["index"] is tz-naive — must match before comparing."""
                        try:
                            idx = s.index
                            # Strip timezone info if present
                            if getattr(idx, "tz", None) is not None:
                                s = s.copy()
                                s.index = idx.tz_localize(None)
                            return s.loc[(_ref_start <= s.index) & (s.index <= _ref_end)]
                        except Exception:
                            return s

                    std_chart_comparisons = {
                        k: _align(s)
                        for k, s in std_chart_comparisons.items()
                        if len(_align(s)) > 5
                    }

                # ── Extra strategy overlays (from PCM selector above) ─────
                # These are added to charts in addition to comparison portfolios
                _extra_strats = st.session_state.get(f"chart_extra_{label}", [])
                _extra_colors = ["#f59e0b","#10b981","#6366f1","#ef4444","#8b5cf6"]
                _extra_chart_data = {}  # name -> pd.Series of returns
                for _ei, _en in enumerate(_extra_strats):
                    if _en in results:
                        _er = results[_en]
                        _idx = pd.to_datetime(_er["index"])
                        _extra_chart_data[_en] = pd.Series(_er["returns"], index=_idx)

                # ── RENDER METRICS from all_portfolios (same data as PCM) ────
                _m_data2 = {}
                def _strip2(n):
                    for e in ["📐 ","🌦 ","📈 ","🛡️ ","⚖️ ","⭐ "]: n=n.replace(e,"")
                    return n[:22]
                for _pname, _pdata in all_portfolios.items():
                    _m_data2[_strip2(_pname)] = _pdata
                if _m_data2:
                    _bs2  = max(_m_data2.items(), key=lambda x: x[1]["sharpe"])
                    _bso2 = max(_m_data2.items(), key=lambda x: x[1].get("sortino",0))
                    _bc2  = max(_m_data2.items(), key=lambda x: x[1].get("calmar",0))
                    _lv2  = min(_m_data2.items(), key=lambda x: x[1]["ann_vol"])
                    _ld2  = max(_m_data2.items(), key=lambda x: x[1]["max_drawdown"])  # least negative = best
                    with _metrics_box.container():
                        m1,m2,m3,m4,m5,m6 = st.columns(6)
                        m1.metric("Sharpe",    f"{_bs2[1]['sharpe']:.2f}",         _strip2(_bs2[0]))
                        m2.metric("Sortino",   f"{_bso2[1].get('sortino',0):.2f}", _strip2(_bso2[0]))
                        m3.metric("Calmar",    f"{_bc2[1].get('calmar',0):.2f}",   _strip2(_bc2[0]))
                        m4.metric("Low Vol",   f"{_lv2[1]['ann_vol']:.1%}",        _strip2(_lv2[0]))
                        m5.metric("Max DD",    f"{_ld2[1]['max_drawdown']:.1%}",   _strip2(_ld2[0]))
                        m6.metric("Count",     str(len(_m_data2)))

                if mode in ("charts", "all"):
                # ── 10-YEAR HISTORICAL PERFORMANCE ────────────────────
                    st.markdown("---")
                    st.markdown("#### Historical Performance vs Benchmark")
                    bm_name_hist = st.session_state.get("benchmark_label", "SPY")
                    bm_tkr_hist  = st.session_state.get("benchmark_ticker", "SPY")
                    fig_hist = go.Figure()
                    end_full   = date.today()
                    # Use the tab's period for the historical chart (1yr, 3yr, 5yr, 10yr)
                    _hist_yrs  = int(label.split()[0]) if label.split()[0].isdigit() else 10
                    start_full = end_full - relativedelta(years=_hist_yrs)
                    chart_colors_hist = ["#2563eb","#059669","#d97706","#dc2626","#7c3aed","#ea580c"]
    
                    # ── Find actual common start date for this tab's period ──
                    try:
                        _test_prices, _src = get_prices_cached(tuple(tickers), str(start_full), str(end_full))
                        # Fill forward to handle tickers with gaps, then find common start
                        _filled = _test_prices.ffill()
                        _all_valid = _filled.dropna(how="any")
                        if len(_all_valid) > 30:
                            start_full = _all_valid.index[0].date()
                        else:
                            _most_valid = _filled.dropna(thresh=max(1, len(tickers)//2))
                            start_full = _most_valid.index[0].date() if len(_most_valid) > 30 else start_full
                    except Exception:
                        pass
    
                    # Check if we're using a shorter window than the tab's period
                    _tab_start_expected = (date.today() - relativedelta(years=_hist_yrs))
                    _using_relative = (start_full > _tab_start_expected) if hasattr(start_full, "year") else False
                    _actual_years   = round((date.today() - start_full).days / 365.25, 1) if hasattr(start_full, "year") else _hist_yrs
    
                    # Show proxy notice if any tickers used substitutes
                    try:
                        if _hist_proxies:
                            risk_free = [k for k,v in _hist_proxies.items() if "risk-free" in v]
                            class_px  = [k for k,v in _hist_proxies.items() if "class proxy" in v or "name proxy" in v]
                            direct_px = [k for k,v in _hist_proxies.items() if "proxy" in v and k not in risk_free and k not in class_px]
                            msgs = []
                            if direct_px:
                                msgs.append("🔄 **Direct proxy:** " + ", ".join(
                                    f"{k} → {_hist_proxies[k].split('(')[1].rstrip(')')}" for k in direct_px))
                            if class_px:
                                msgs.append("📊 **Asset-class proxy:** " + ", ".join(
                                    f"{k} → {_hist_proxies[k]}" for k in class_px))
                            if risk_free:
                                msgs.append("⚠️ **Risk-free proxy (BIL):** " + ", ".join(risk_free) +
                                    " — no comparable historical data found; T-Bill returns used")
                            if msgs:
                                st.info(" | ".join(msgs) + f" | Showing {_actual_years:.1f}yr window")
                    except Exception: pass
    
    
                    if _using_relative:
                        # Find which tickers are missing full history
                        try:
                            _short_tickers = [
                                t for t in tickers
                                if t in _test_prices.columns and
                                _test_prices[t].first_valid_index() is not None and
                                _test_prices[t].first_valid_index().date() > _ten_yrs_ago
                            ]
                        except Exception:
                            _short_tickers = []
                        _short_str = ", ".join(_short_tickers[:5]) if _short_tickers else "some tickers"
                        # Pre-compute the contraction outside the f-string —
                        # Python 3.12 allowed `'don\'t have'` inside an f-string
                        # expression, but 3.13 (which Streamlit Cloud uses)
                        # rejects backslashes inside the expression part of
                        # f-strings as a hard SyntaxError.
                        _have_phrase = (
                            "don't have" if len(_short_tickers) != 1
                            else "doesn't have"
                        )
                        st.info(
                            f"📅 **Relative comparison period used** — {_short_str} "
                            f"{_have_phrase} "
                            f"{_hist_yrs} years of history. Chart shows the common available window: "
                            f"**{_actual_years:.1f} years** "
                            f"(from {start_full.strftime('%b %Y') if hasattr(start_full,'strftime') else start_full}). "
                            f"All portfolios start at 0% on the same date for a fair comparison."
                        )
    
                    # My Portfolio
                    _mp_hist_keys = [k for k in results if k.startswith("⭐ ")] or (list(results.keys())[:1] if results else [])
                    my_port_hist = {k: results[k] for k in _mp_hist_keys}
                    if my_port_hist:
                        try:
                            r_my       = list(my_port_hist.values())[0]
                            port_lbl_h = list(my_port_hist.keys())[0].replace("⭐ ","")
                            # Fetch ACTUAL historical prices for the full tab period
                            fp, _hist_proxies = get_prices_with_proxies(tuple(tickers), str(start_full), str(end_full))
                            if _hist_proxies:
                                _proxy_msg = ", ".join(f"{k}→{v}" for k,v in _hist_proxies.items())
                            vcols_h = [t for t in tickers if t in fp.columns]
                            if vcols_h:
                                # Use the actual saved/submitted weights
                                loaded_w = st.session_state.get("loaded_weights", {})
                                if loaded_w and all(t in loaded_w for t in tickers):
                                    w_arr_h = np.array([loaded_w[t]/100.0 for t in vcols_h])
                                else:
                                    w_arr_h = np.array(r_my["weights"][:len(vcols_h)])
                                w_arr_h = w_arr_h / w_arr_h.sum()
                                # Fill forward missing data (handles short-history tickers)
                                fp_filled = fp[vcols_h].ffill().dropna(how="all")
                                common_idx = fp_filled.dropna().index
                                if len(common_idx) > 5:
                                    rets_h  = fp_filled.loc[common_idx].pct_change().dropna()
                                    port_r  = pd.Series(rets_h.values @ w_arr_h, index=rets_h.index)
                                    port_pct = (1 + port_r).cumprod() - 1
                                    # Update start_full for comparison alignment
                                    start_full = port_r.index[0].date()
                                    end_full   = port_r.index[-1].date()
                                    fig_hist.add_trace(go.Scatter(x=port_pct.index, y=port_pct.values,
                                        mode="lines", name=port_lbl_h,
                                        line=dict(width=3, color=MY_PORT_COLOR, dash="solid"),
                                        hovertemplate=f"<b>{port_lbl_h}</b><br>%{{x|%b %Y}}<br>%{{y:+.1%}}<extra></extra>"))
                        except Exception: pass
    
                    # Comparison portfolios (10yr) — fixed benchmarks
                    # honoring the three toggles at the top of this tab.
                    hist_cmp_defs = []
                    if st.session_state.get("show_bench_bnd", True):
                        hist_cmp_defs.append(("🟦 100% Bonds (BND)", ["BND"],         [1.0],         "#2563eb"))
                    if st.session_state.get("show_bench_6040", True):
                        hist_cmp_defs.append(("⚖️ 60/40 (SPY+AGG)",   ["SPY", "AGG"], [0.60, 0.40],  "#059669"))
                    if st.session_state.get("show_bench_spy", True):
                        hist_cmp_defs.append(("📈 S&P 500 (SPY)",     ["SPY"],         [1.0],         "#dc2626"))
                    for cname, ctkrs, cwts, ccol in hist_cmp_defs:
                        try:
                            cp, _src = get_prices_cached(tuple(ctkrs), str(start_full), str(end_full))
                            cp = cp.ffill().dropna(how="all")
                            vcols_c = [t for t in ctkrs if t in cp.columns]
                            if vcols_c:
                                cw = np.array([cwts[ctkrs.index(t)] for t in vcols_c]); cw = cw/cw.sum()
                                cr = cp[vcols_c].pct_change().dropna()
                                # Align to the same start date as My Portfolio
                                # Align to My Portfolio date range
                                cr = cr[(cr.index >= pd.Timestamp(start_full)) & 
                                        (cr.index <= pd.Timestamp(end_full))]
                                if len(cr) < 5: continue
                                cc = (1 + pd.Series(cr.values @ cw, index=cr.index)).cumprod()
                                cc_pct = (cc - 1)
                                fig_hist.add_trace(go.Scatter(x=cc_pct.index, y=cc_pct.values, mode="lines",
                                    name=_clean_label(cname),
                                    line=dict(width=1.8, color=ccol, dash="solid"),
                                    hovertemplate=f"<b>{_clean_label(cname)}</b><br>%{{x|%b %Y}}<br>%{{y:+.1%}}<extra></extra>"))
                        except Exception: pass

                    # ── Client's Current Portfolio (Step 2) on 10y hist ─────
                    # Was missing from the 10y historical chart specifically —
                    # the period sub-tabs use std_chart_comparisons which got
                    # the client-current addition, but the 10y hist chart has
                    # its own code path that bypassed std_chart_comparisons.
                    # This block adds it explicitly so the 10y view matches
                    # the other charts.
                    _curr_h = st.session_state.get("client_current_portfolio")
                    if _curr_h and _curr_h.get("tickers"):
                        try:
                            _ch_tks  = list(_curr_h["tickers"])
                            _ch_wpct = _curr_h.get("weights") or {}
                            _ch_wmap = {
                                t: float(_ch_wpct.get(t, 0)) / 100.0
                                for t in _ch_tks
                                if float(_ch_wpct.get(t, 0)) > 0
                            }
                            if _ch_wmap:
                                _ch_cp, _ = get_prices_with_proxies(
                                    tuple(_ch_wmap.keys()),
                                    str(start_full), str(end_full))
                                _ch_cp = _ch_cp.ffill().dropna(how="all")
                                _ch_vcols = [t for t in _ch_wmap if t in _ch_cp.columns]
                                if _ch_vcols:
                                    _ch_w = np.array([_ch_wmap[t] for t in _ch_vcols])
                                    if _ch_w.sum() > 0:
                                        _ch_w = _ch_w / _ch_w.sum()
                                    _ch_r = _ch_cp[_ch_vcols].pct_change().dropna()
                                    _ch_r = _ch_r[(_ch_r.index >= pd.Timestamp(start_full)) &
                                                  (_ch_r.index <= pd.Timestamp(end_full))]
                                    if len(_ch_r) >= 5:
                                        _ch_cum = (1 + pd.Series(
                                            _ch_r.values @ _ch_w, index=_ch_r.index
                                        )).cumprod()
                                        _ch_pct = _ch_cum - 1
                                        # Distinct color: amber/gold so it
                                        # stands out vs portfolio (purple)
                                        # and benchmarks (blue/green/red).
                                        fig_hist.add_trace(go.Scatter(
                                            x=_ch_pct.index, y=_ch_pct.values, mode="lines",
                                            name="👤 Client's Current",
                                            line=dict(width=2.0, color="#d97706", dash="solid"),
                                            hovertemplate=("<b>👤 Client's Current</b><br>"
                                                           "%{x|%b %Y}<br>%{y:+.1%}<extra></extra>")
                                        ))
                        except Exception:
                            pass

                    # ── Advisor cmp_port1 / cmp_port2 on 10y hist ────────────
                    # Same pattern: fetch and overlay the Step 2 advisor-set
                    # comparison portfolios on the 10y chart.
                    _cmp_colors_10y = ["#f59e0b", "#8b5cf6"]  # amber, violet
                    for _ci, (_ckey, _ccol) in enumerate(zip(
                            ("cmp_port1", "cmp_port2"), _cmp_colors_10y)):
                        _csel = st.session_state.get(_ckey, "None")
                        if not _csel or _csel == "None":
                            continue
                        try:
                            _c10_tks, _c10_wts = None, None
                            _c10_label = _csel
                            if _csel.startswith("📁 "):
                                sp = load_saved().get(_csel[2:])
                                if sp:
                                    _c10_tks = sp["tickers"]
                                    _c10_wts = sp.get("weights") or [1.0/len(_c10_tks)]*len(_c10_tks)
                            elif _csel.startswith("✏️ Custom"):
                                _custom_w = st.session_state.get(f"cmp{_ci+1}_custom_weights")
                                _custom_l = st.session_state.get(f"cmp{_ci+1}_custom_label")
                                if _custom_w:
                                    _c10_tks = list(_custom_w.keys())
                                    _c10_wts = list(_custom_w.values())
                                    if _custom_l:
                                        _c10_label = _custom_l
                            elif _csel in POPULAR_PORTFOLIOS and POPULAR_PORTFOLIOS[_csel]:
                                _c10_tks, _wmap = _resolve_preset(_csel)
                                if _c10_tks:
                                    # Decimal weights to match the rest of this branch
                                    _total = sum(_wmap.values()) or 1.0
                                    _c10_wts = [_wmap.get(t, 0.0) / _total for t in _c10_tks]
                            if _c10_tks and _c10_wts:
                                _c10_cp, _ = get_prices_with_proxies(
                                    tuple(_c10_tks), str(start_full), str(end_full))
                                _c10_cp = _c10_cp.ffill().dropna(how="all")
                                _c10_vcols = [t for t in _c10_tks if t in _c10_cp.columns]
                                if _c10_vcols:
                                    _c10_w = np.array([_c10_wts[_c10_tks.index(t)] for t in _c10_vcols])
                                    if _c10_w.sum() > 0:
                                        _c10_w = _c10_w / _c10_w.sum()
                                    _c10_r = _c10_cp[_c10_vcols].pct_change().dropna()
                                    _c10_r = _c10_r[(_c10_r.index >= pd.Timestamp(start_full)) &
                                                    (_c10_r.index <= pd.Timestamp(end_full))]
                                    if len(_c10_r) >= 5:
                                        _c10_cum = (1 + pd.Series(
                                            _c10_r.values @ _c10_w, index=_c10_r.index
                                        )).cumprod()
                                        _c10_pct = _c10_cum - 1
                                        fig_hist.add_trace(go.Scatter(
                                            x=_c10_pct.index, y=_c10_pct.values, mode="lines",
                                            name=_clean_label(_c10_label),
                                            line=dict(width=1.6, color=_ccol, dash="solid"),
                                            hovertemplate=(f"<b>{_clean_label(_c10_label)}</b><br>"
                                                           "%{x|%b %Y}<br>%{y:+.1%}<extra></extra>")
                                        ))
                        except Exception:
                            pass
    
                    # Benchmark removed — S&P 500 shown via comparison portfolios
    
                    # Extra strategy overlays on 10yr hist (fetch full 10yr data)
                    for _ei, _en in enumerate(_extra_strats):
                        if _en in results:
                            try:
                                _ex_r = results[_en]
                                _ex_fp, _src = get_prices_with_proxies(tuple(tickers), str(start_full), str(end_full))
                                _ex_vc = [t for t in tickers if t in _ex_fp.columns]
                                if _ex_vc:
                                    _ex_w = np.array(_ex_r["weights"][:len(_ex_vc)]); _ex_w = _ex_w/_ex_w.sum()
                                    _ex_ret = _ex_fp[_ex_vc].pct_change().dropna()
                                    _ex_cum = (1+pd.Series(_ex_ret.values@_ex_w, index=_ex_ret.index)).cumprod()
                                    _ecol = _extra_colors[_ei % len(_extra_colors)]
                                    _ex_pct = (_ex_cum - 1)
                                    _disp_e = _clean_label(_en)
                                    fig_hist.add_trace(go.Scatter(x=_ex_pct.index, y=_ex_pct.values, mode="lines",
                                        name=_disp_e, line=dict(width=1.5, color=_ecol, dash="solid"),
                                        hovertemplate=f"<b>{_disp_e}</b><br>%{{x|%b %Y}}<br>%{{y:+.1%}}<extra></extra>"))
                            except Exception: pass
    
                    if len(fig_hist.data) == 0:
                        st.warning("Could not load 10-year data. Check tickers and try again.")
                    else:
                        fig_hist.add_hline(y=0, line_color="#e5e7eb", line_width=1,
                                            annotation_text="0%", annotation_position="left",
                                            annotation_font=dict(size=10, color="#9ca3af"))
                        fig_hist.update_layout(
                            title=dict(text=f"{_hist_yrs}-Year Historical Performance vs {bm_name_hist} ({start_full.strftime('%Y') if hasattr(start_full,'strftime') else str(start_full)[:4]}–{end_full.year})", x=0, xanchor="left",
                                       font=dict(size=14, color="#111827", family="Inter")),
                            yaxis=dict(title_text="Total Return (%)", tickformat="+.0%",
                                       gridcolor="#f0f0f0", showgrid=True,
                                       title_font=dict(size=12, color="#374151"),
                                       tickfont=dict(size=11, color="#6b7280"),
                                       zeroline=True, zerolinecolor="#e5e7eb", zerolinewidth=1),
                            xaxis=dict(title_text=None, gridcolor="#f0f0f0",
                                       tickfont=dict(size=11, color="#6b7280")),
                            height=520, template="plotly_white",
                            paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                            font=dict(color="#374151", family="Inter"),
                            # Equalized margins so chart content is visually
                            # centered, and bottom legend now centered to match.
                            margin=dict(t=54, b=160, l=72, r=72),
                            showlegend=True,
                            legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e5e7eb",
                                        borderwidth=1, font=dict(size=11, color="#374151"),
                                        orientation="h", x=0.5, y=-0.22,
                                        xanchor="center", yanchor="top"),
                            hoverlabel=dict(bgcolor="#111827", font=dict(color="#f9fafb", size=12)),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, key=f"hist10_{label}",
                                        config={"displayModeBar": True, "displaylogo": False,
                                                "modeBarButtonsToRemove": ["lasso2d","select2d"]})
    
                    # ── FORWARD MONTE CARLO ─────────────────────────────
                    # Heading reflects the actual projection horizon set on
                    # the slider below (defaults to 10y). Was previously
                    # hardcoded "10-Year" which was misleading at 30y.
                    _mc_years_for_heading = st.session_state.get(
                        f"mc_yrs_{label}", 10
                    )
                    st.markdown(
                        f"#### 🔭 {_mc_years_for_heading}-Year Forward "
                        f"Projection (Monte Carlo)"
                    )

                    # ── Flexible Monte Carlo Parameters ─────────────────────
                    with st.expander("⚙️ Projection Settings", expanded=False):
                        mc_pcols = st.columns(4)
                        mc_years = mc_pcols[0].slider("Projection Years", 5, 30, 10, key=f"mc_yrs_{label}")
                        mc_sims  = mc_pcols[1].slider("Simulations", 100, 1000, 500, step=100, key=f"mc_sims_{label}")
                        mc_ret_adj  = mc_pcols[2].slider("Return Adjustment (%/yr)", -5, 5, 0, key=f"mc_radj_{label}",
                                                          help="Shift expected return up/down from historical")
                        mc_vol_adj  = mc_pcols[3].slider("Volatility Adjustment (%)", -5, 10, 0, key=f"mc_vadj_{label}",
                                                          help="Increase/decrease volatility assumption")

                        # ── Expected-return mode ─────────────────────────────
                        # The single most important methodology choice in any
                        # forward Monte Carlo. Default 'CMA-shrunk' applies
                        # an industry-standard shrinkage of the historical
                        # mean toward a long-run capital markets assumption
                        # (7%/yr equity-blended), which prevents the projection
                        # from extrapolating recent strong-decade returns into
                        # the next decade. Raw historical is preserved as an
                        # option but not the default — it produces unrealistic
                        # forward projections when the trailing window happens
                        # to be unusually strong (e.g. the last 10 years for
                        # US equities).
                        mc_ret_mode = st.radio(
                            "Expected return assumption",
                            options=[
                                "Conservative (CMA-shrunk to 7%/yr)",
                                "Balanced (50/50 historical and 8% CMA)",
                                "Historical (raw 10-year mean)",
                            ],
                            index=0,
                            key=f"mc_ret_mode_{label}",
                            help=(
                                "Forward Monte Carlo is highly sensitive to the input mean. "
                                "Using raw historical returns from a strong decade biases the "
                                "projection upward — the math compounds the optimism. "
                                "CMA-shrunk shrinks toward a 7%/yr long-run equity assumption, "
                                "which is closer to industry-standard capital markets expectations. "
                                "Volatility is preserved from the historical sample regardless of mode."
                            ),
                        )

                        # ── Input data diagnostic ─────────────────────────────
                        # Lives at the bottom of Projection Settings so all the
                        # controls + their effective state are in one place.
                        # Reads from session_state keys populated when the chart
                        # was last built (one rerun behind on first paint, but
                        # any subsequent setting change re-renders the chart
                        # which writes fresh values, so it stays in sync after
                        # the first interaction).
                        st.markdown("---")
                        st.markdown(
                            "**🔬 Input data diagnostic**  \n"
                            "<span style='color:#6B7E8A;font-size:0.78rem'>"
                            "What's actually feeding the simulation, after all "
                            "settings above are applied. Updates when you change "
                            "any setting.</span>",
                            unsafe_allow_html=True,
                        )
                        _diag_final = st.session_state.get(f"_mc_diag_final_{label}")
                        _diag_orig  = st.session_state.get(f"_mc_diag_orig_{label}")
                        _diag_src   = st.session_state.get(f"_mc_diag_src_{label}", "—")
                        if _diag_final and len(_diag_final) > 1:
                            import numpy as _np_d
                            _arr_d = _np_d.array(_diag_final, dtype=float)
                            _md = float(_arr_d.mean())
                            _sd = float(_arr_d.std())
                            _eff_ret = (1 + _md) ** 252 - 1
                            _eff_vol = _sd * (252 ** 0.5)
                            _yrs_d   = len(_arr_d) / 252.0
                            _dc1, _dc2, _dc3, _dc4 = st.columns(4)
                            _dc1.metric("Source",      _diag_src)
                            _dc2.metric("Sample size", f"{len(_arr_d)} days  (≈{_yrs_d:.1f}yr)")
                            _dc3.metric("Effective ann. return", f"{_eff_ret:+.1%}/yr",
                                        help="Final mean after CMA shrinkage AND any "
                                             "Return Adjustment slider value.")
                            _dc4.metric("Effective ann. vol",    f"{_eff_vol:.1%}/yr",
                                        help="Final vol after any Volatility "
                                             "Adjustment slider value.")
                            # Warn for raw-historical mode + high underlying return
                            if _diag_orig and len(_diag_orig) > 1:
                                _arr_o = _np_d.array(_diag_orig, dtype=float)
                                _orig_ann = (1 + float(_arr_o.mean())) ** 252 - 1
                                if _orig_ann > 0.18 and mc_ret_mode.startswith("Historical"):
                                    st.warning(
                                        f"⚠️ Raw historical mode is selected and the "
                                        f"trailing 10-year annualized return is "
                                        f"**{_orig_ann:+.1%}/yr** — unusually high. "
                                        f"The forward projection will inherit this. "
                                        f"Switch to 'Conservative (CMA-shrunk)' for "
                                        f"more realistic forward expectations."
                                    )
                        else:
                            st.caption(
                                "_Diagnostic populates after the chart renders. "
                                "Change any setting above and it'll appear._"
                            )

                    st.caption(f"{mc_sims} simulations · {mc_years}yr horizon · "
                               f"based on 10-year historical returns · "
                               f"expected return assumption applied per Projection Settings · "
                               f"not a guarantee of future returns.")
    
                    psel_c1, psel_c2 = st.columns([2, 2])
                    # Sort: My Portfolio first, then other strategies
                    _my_strats   = [n for n in results.keys() if n.startswith("⭐ ")]
                    _other_strats= [n for n in results.keys() if not n.startswith("⭐ ")]
                    all_proj_strats = _my_strats + _other_strats
                    my_proj_keys    = _my_strats  # keys starting with ⭐
                    default_proj    = my_proj_keys[0] if my_proj_keys else all_proj_strats[0] if all_proj_strats else None
    
                    if default_proj:
                        proj_strategy = psel_c1.selectbox(
                            "Strategy to project", all_proj_strats,
                            index=0,  # Always default to first = My Portfolio
                            key=f"proj_sel_{label}", label_visibility="collapsed")
    
                        # Build the comparison-portfolio list for this projection.
                        # Default: include all the same comparisons the
                        # cumulative/drawdown/Sharpe charts show — Step 2's
                        # client-current portfolio + advisor cmp_port1/2 +
                        # the toggled-on fixed market benchmarks. Advisor can
                        # tick any off via the multiselect if they want a
                        # cleaner projection chart.
                        _proj_opts_full = []
                        # Client's current portfolio — only if Step 2 picker is set
                        _curr_proj_set = st.session_state.get("client_current_portfolio")
                        if _curr_proj_set and _curr_proj_set.get("tickers"):
                            _proj_opts_full.append("👤 Client's Current")
                        # Advisor-set comparison portfolios from Step 2
                        for _ci, _ckey in enumerate(("cmp_port1", "cmp_port2"), start=1):
                            _csel = st.session_state.get(_ckey, "None")
                            if _csel and _csel != "None":
                                # Use a stable label that matches std_chart_comparisons
                                if _csel.startswith("📁 "):
                                    _proj_opts_full.append(_csel)
                                elif _csel.startswith("✏️ Custom"):
                                    _custom_label = st.session_state.get(f"cmp{_ci}_custom_label")
                                    if _custom_label:
                                        _proj_opts_full.append(_custom_label)
                                elif _csel in POPULAR_PORTFOLIOS:
                                    _proj_opts_full.append(f"🧩 {_csel[:14]}")
                        # Fixed market benchmarks per their toggles
                        if st.session_state.get("show_bench_bnd", True):
                            _proj_opts_full.append("🟦 100% Bonds (BND)")
                        if st.session_state.get("show_bench_6040", True):
                            _proj_opts_full.append("⚖️ 60/40 (SPY+AGG)")
                        if st.session_state.get("show_bench_spy", True):
                            _proj_opts_full.append("📈 S&P 500 (SPY)")
                        proj_cmp = psel_c2.multiselect(
                            "Add comparisons", _proj_opts_full,
                            default=_proj_opts_full,  # Default: all on
                            key=f"proj_cmp_{label}", label_visibility="collapsed",
                            placeholder="Add comparison portfolios...")

                        if proj_strategy in results:
                            proj_r = results[proj_strategy]
                            # ── Use 10-year returns for projection ─────────
                            # The forward Monte Carlo should always be based on
                            # the longest available history, NOT whatever period
                            # sub-tab the user is on. Feeding 1-year returns
                            # into a 10-year forward sim produces wildly
                            # inflated forecasts because a single-year window
                            # is too narrow a sample of return distribution
                            # behavior. Pull the 10-year backtest's returns for
                            # the same strategy when available; fall back to
                            # the period's own returns only if 10y data is
                            # missing (which happens for very short-history
                            # tickers).
                            _bt10_state = st.session_state.get("bt10")
                            _bt10_results = (
                                _bt10_state[0] if _bt10_state and isinstance(_bt10_state, (list, tuple))
                                else (_bt10_state or {})
                            )
                            _proj_returns_for_mc = proj_r.get("returns", [])
                            if _bt10_results and proj_strategy in _bt10_results:
                                _bt10_ret = _bt10_results[proj_strategy].get("returns", [])
                                if _bt10_ret and len(_bt10_ret) >= 60:
                                    _proj_returns_for_mc = _bt10_ret
                            bm_rets_proj = (st.session_state.get("bmark_returns_10 Years") or
                                            st.session_state.get("bmark_returns_5 Years") or
                                            st.session_state.get("bmark_returns_3 Years"))
    
                            # Build comparison list
                            comparison_list = []
                            # Add extra selected strategies from PCM selector
                            for _ei, _en in enumerate(_extra_strats):
                                if _en in results:
                                    try:
                                        _ex_r2 = results[_en]
                                        comparison_list.append((_en, _ex_r2["returns"]))
                                    except Exception: pass
                            # Pull the picked comparison portfolios from
                            # std_chart_comparisons (already built above with
                            # client-current + advisor cmps + fixed benchmarks
                            # all in one place). Fall back to a fresh fetch
                            # for fixed benchmarks if not in std_chart_comparisons
                            # for some reason.
                            cmp_map = {
                                "🟦 100% Bonds (BND)":  (["BND"],          [1.0]),
                                "⚖️ 60/40 (SPY+AGG)":   (["SPY", "AGG"],   [0.60, 0.40]),
                                "📈 S&P 500 (SPY)":     (["SPY"],          [1.0]),
                            }
                            for cmp_name in proj_cmp:
                                # Prefer reusing the already-fetched series
                                if cmp_name in std_chart_comparisons:
                                    try:
                                        _series = std_chart_comparisons[cmp_name]
                                        comparison_list.append((cmp_name, _series.tolist()))
                                        continue
                                    except Exception:
                                        pass
                                # Fallback: fixed benchmarks via cmp_map
                                if cmp_name in cmp_map:
                                    try:
                                        cmp_tks, cmp_wts = cmp_map[cmp_name]
                                        cend   = date.today(); cstart = cend - relativedelta(years=3)
                                        cp2, _src = get_prices_cached(tuple(cmp_tks), str(cstart), str(cend))
                                        cp2 = cp2.ffill().dropna(how="all")
                                        vcols_p = [t for t in cmp_tks if t in cp2.columns]
                                        warr_p  = np.array([cmp_wts[cmp_tks.index(t)] for t in vcols_p])
                                        warr_p  = warr_p / warr_p.sum()
                                        cr2     = cp2[vcols_p].pct_change().dropna().values @ warr_p
                                        comparison_list.append((cmp_name, cr2.tolist()))
                                    except Exception: pass
    
                            proj_lbl = proj_strategy.replace("⭐ ","")
                            bm_name_proj = st.session_state.get("benchmark_label","SPY")

                            # ─────────────────────────────────────────────
                            # CMA SHRINKAGE — applied BEFORE chart build so
                            # the chart, percentile cards, and benchmark line
                            # all use the same shrunk inputs. Without this,
                            # the chart was rendering the SPY benchmark with
                            # raw historical mean (~13%/yr the last decade)
                            # and projecting that 10y forward, producing the
                            # wildly optimistic +680% line you saw.
                            # ─────────────────────────────────────────────
                            import numpy as _np_shr

                            def _shrink_returns(rets_list, mode):
                                """Rescale a daily returns list to a target
                                annualized return per the selected mode.
                                Preserves the daily *deviations* from the mean
                                (vol/skew/kurt) — only shifts the mean."""
                                if not rets_list:
                                    return rets_list
                                arr = _np_shr.array(rets_list, dtype=float)
                                if len(arr) < 2:
                                    return rets_list
                                hist_mean_d = float(arr.mean())
                                hist_ann = (1.0 + hist_mean_d) ** 252 - 1.0
                                if mode.startswith("Conservative"):
                                    target_ann = 0.07
                                elif mode.startswith("Balanced"):
                                    target_ann = 0.5 * hist_ann + 0.5 * 0.08
                                else:
                                    target_ann = hist_ann  # raw historical
                                target_mean_d = (1.0 + target_ann) ** (1.0 / 252) - 1.0
                                shrunk = arr - hist_mean_d + target_mean_d
                                return shrunk.tolist()

                            # Use the longest-available history series
                            # (already 10-year per _proj_returns_for_mc fix)
                            _proj_rets_shrunk = _shrink_returns(
                                _proj_returns_for_mc, mc_ret_mode
                            )
                            # Same shrinkage applied to benchmark so its
                            # projected line is on the same methodological
                            # basis as the portfolio line.
                            _bm_rets_shrunk = (
                                _shrink_returns(list(bm_rets_proj), mc_ret_mode)
                                if bm_rets_proj else None
                            )
                            # And to each comparison series in the legend
                            _comp_shrunk = (
                                [(cn, _shrink_returns(list(cr), mc_ret_mode))
                                 for cn, cr in comparison_list]
                                if comparison_list else None
                            )

                            fig_proj = build_projection_chart(
                                strategy_name=proj_lbl,
                                returns=_proj_rets_shrunk,
                                benchmark_returns=_bm_rets_shrunk,
                                bm_label=bm_name_proj,
                                years=mc_years,
                                comparison_list=_comp_shrunk,
                            )
                            st.plotly_chart(fig_proj, use_container_width=True,
                                            # Key includes mc_ret_mode + mc_years so Streamlit
                                            # treats the chart as new whenever the user changes
                                            # those settings. Without this, Streamlit caches the
                                            # plotly figure by key and the chart visually doesn't
                                            # update even though the underlying figure object is
                                            # different — classic stale-cache bug.
                                            key=f"proj_{label}_{proj_strategy[:8]}_"
                                                f"{mc_ret_mode[:4]}_{mc_years}_"
                                                f"{mc_ret_adj}_{mc_vol_adj}",
                                            config={"displayModeBar": True, "displaylogo": False,
                                                    "modeBarButtonsToRemove": ["lasso2d","select2d"]})
    
                            # Apply flexible MC params from settings
                            # Same shrunk series feeds the percentile cards
                            # below — so chart and cards agree on the input.
                            _mc_rets = list(_proj_rets_shrunk)

                            # Apply user's Return Adjustment / Volatility Adjustment sliders
                            # (these stack on top of the CMA shrinkage so the advisor can
                            # nudge from the default Conservative target if they want).
                            if mc_ret_adj != 0 or mc_vol_adj != 0:
                                import numpy as _np
                                _r = _np.array(_mc_rets)
                                _r = _r + (mc_ret_adj / 100.0 / 252.0)
                                if mc_vol_adj != 0:
                                    _m = _r.mean()
                                    _r = _m + (_r - _m) * (1 + mc_vol_adj/100.0)
                                _mc_rets = _r.tolist()

                            # Stash the final input series + the unshrunk source for
                            # the diagnostic display that lives inside Projection Settings.
                            # We populate two session_state keys so the expander block
                            # above (which has already rendered) can pick them up on
                            # the next rerun via Streamlit's normal state propagation.
                            st.session_state[f"_mc_diag_final_{label}"] = list(_mc_rets)
                            st.session_state[f"_mc_diag_orig_{label}"]  = list(_proj_returns_for_mc)
                            st.session_state[f"_mc_diag_src_{label}"]   = (
                                "10-year backtest"
                                if (_bt10_results
                                    and proj_strategy in _bt10_results
                                    and _bt10_results[proj_strategy].get("returns")
                                    and len(_bt10_results[proj_strategy]["returns"]) >= 60)
                                else f"period-tab fallback ({label})"
                            )
                            days, p5, p10, p25, p50, p75, p90, p95, _, prob_loss = run_monte_carlo(
                                tuple(_mc_rets), years_forward=mc_years,
                                n_simulations=mc_sims)

                            # Helper: convert a cumulative multiple (e.g. 55.5x)
                            # into an annualized return so the numbers feel
                            # sane. A median +5,457% sounds insane until you
                            # realize that over 30 years it's just +14% annual.
                            def _ann(cum_multiple, yrs):
                                if yrs <= 0 or cum_multiple <= 0:
                                    return 0.0
                                return cum_multiple ** (1.0 / yrs) - 1.0

                            st.caption(
                                f"**Projection at year {mc_years}** — final "
                                f"portfolio value relative to today, expressed "
                                f"as cumulative return. Annualized rate shown "
                                f"in parens."
                            )
                            pc1,pc2,pc3,pc4,pc5 = st.columns(5)
                            pc1.metric("Median",
                                       f"{p50[-1]-1:+.1%}",
                                       delta=f"{_ann(p50[-1], mc_years):+.1%}/yr",
                                       delta_color="off")
                            pc2.metric("Best 25%",
                                       f"{p75[-1]-1:+.1%}",
                                       delta=f"{_ann(p75[-1], mc_years):+.1%}/yr",
                                       delta_color="off")
                            pc3.metric("Worst 25%",
                                       f"{p25[-1]-1:+.1%}",
                                       delta=f"{_ann(p25[-1], mc_years):+.1%}/yr",
                                       delta_color="off")
                            pc4.metric("Best 10%",
                                       f"{p90[-1]-1:+.1%}",
                                       delta=f"{_ann(p90[-1], mc_years):+.1%}/yr",
                                       delta_color="off")
                            pc5.metric("Worst 10%",
                                       f"{p10[-1]-1:+.1%}",
                                       delta=f"{_ann(p10[-1], mc_years):+.1%}/yr",
                                       delta_color="off")
                            # Probability of loss + worst-case (5th pct) row
                            if prob_loss:
                                qc1, qc2, qc3, qc4, qc5 = st.columns(5)
                                _yrs_avail = sorted([y for y in (1,3,5,10) if y in prob_loss])
                                # Labels intentionally short so they don't truncate
                                # in the narrow metric columns. Help tooltips give
                                # the full context.
                                qc1.metric("Worst 5%", f"{p5[-1]-1:+.1%}",
                                           help="Worst-case scenario — the 5th percentile "
                                                "final outcome. A true downside scenario.")
                                _qcs = [qc2, qc3, qc4, qc5]
                                for _i, _y in enumerate(_yrs_avail[:4]):
                                    _qcs[_i].metric(
                                        f"Loss Yr {_y}",
                                        f"{prob_loss[_y]*100:.1f}%",
                                        help=f"Probability of loss at year {_y} — "
                                             f"percent of simulations ending below "
                                             f"starting value at that horizon."
                                    )
    
                    # ── DRAWDOWN CHART ─────────────────────────────────────
                    st.markdown("#### Drawdown (%)")
                    fig_dd = go.Figure()
    
                    my_dd_keys = [k for k in results if k.startswith("⭐ ")] or (list(results.keys())[:1] if results else [])
                    for name in my_dd_keys:
                        r   = results[name]
                        s   = pd.Series(r["returns"], index=pd.to_datetime(r["index"]))
                        cum = (1+s).cumprod(); dd = (cum/cum.cummax()-1)*100
                        _loaded_src_dd = st.session_state.get("portfolio_source","")
                        port_lbl_dd = name.replace("⭐ ","")
                        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
                            name=port_lbl_dd, line=dict(width=2.5, color=MY_PORT_COLOR, dash="solid"),
                            hovertemplate=f"<b>{port_lbl_dd}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>"))
    
                    for cmp_name, cmp_series in std_chart_comparisons.items():
                        try:
                            cmp_cum = (1+cmp_series).cumprod(); cmp_dd = (cmp_cum/cmp_cum.cummax()-1)*100
                            c_ = COMP_COLORS.get(cmp_name, "#94a3b8")
                            _disp = _clean_label(cmp_name)
                            fig_dd.add_trace(go.Scatter(x=cmp_dd.index, y=cmp_dd.values, mode="lines",
                                name=_disp, line=dict(width=1.5, color=c_, dash="solid"),
                                hovertemplate=f"<b>{_disp}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>"))
                        except Exception: pass
    
                    # Extra strategy overlays on DD
                    for _ei, (_en, _es) in enumerate(_extra_chart_data.items()):
                        try:
                            _ec = (1+_es).cumprod(); _edd = (_ec/_ec.cummax()-1)*100
                            _ecol = _extra_colors[_ei % len(_extra_colors)]
                            _disp_e = _clean_label(_en)
                            fig_dd.add_trace(go.Scatter(x=_edd.index, y=_edd.values, mode="lines",
                                name=_disp_e, line=dict(width=1.5, color=_ecol, dash="solid"),
                                hovertemplate=f"<b>{_disp_e}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>"))
                        except Exception: pass
    
                    fig_dd.add_hline(y=0, line_color="#e5e7eb", line_width=1)
                    fig_dd.update_layout(
                        title=dict(text="Drawdown (%)", x=0, xanchor="left"),
                        yaxis_title="Drawdown (%)", xaxis_title=None, height=360,
                        showlegend=True,
                        # Bottom-centered legend, matching the projection chart's
                        # convention. Previously left-anchored at x=0 which
                        # looked unbalanced with the equalized margins.
                        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e5e7eb",
                                    borderwidth=1, font=dict(size=11, color="#374151"),
                                    orientation="h", x=0.5, y=-0.22,
                                    xanchor="center", yanchor="top"),
                        **PLOT_THEME)
                    fig_dd.update_xaxes(title_text=None)
                    st.plotly_chart(fig_dd, use_container_width=True, key=f"dd_{label}",
                                    config={"displayModeBar": False})
    
                    # ── ROLLING SHARPE (window adapts to available data) ──
                    # Default: 12-month (252d) for smoothness. But on short
                    # periods (1y/3y tabs after train/test split) there aren't
                    # enough observations, so we shrink the window to keep the
                    # line visible while still being representative.
                    st.markdown("#### Rolling Sharpe Ratio")
                    fig_rs = go.Figure()

                    def _pick_window(n):
                        # Use ~40% of available obs, clamped between 42 and 252
                        return max(42, min(252, int(n * 0.4)))

                    def _rolling_sharpe(series, window=None):
                        n = len(series)
                        if n < 45:
                            return pd.Series([], dtype=float)
                        w = window or _pick_window(n)
                        w = min(w, max(21, n - 5))  # don't demand more data than we have
                        roll = series.rolling(w).apply(
                            lambda x: (x.mean()*252)/(x.std()*np.sqrt(252)) if x.std()>0 else 0
                        ).dropna()
                        # Smooth the resulting curve a bit
                        smooth = 5 if w >= 120 else 3
                        if len(roll) > smooth:
                            roll = roll.rolling(smooth, min_periods=1).mean()
                        return roll

                    # Determine window from "My Portfolio" length (keeps traces aligned)
                    _rs_window = None
                    _mp_ref_key = next((n for n in results if n.startswith("⭐ ")), None)
                    if _mp_ref_key:
                        _rs_window = _pick_window(len(results[_mp_ref_key].get("returns", [])))

                    my_rs_keys = [n for n in results if n.startswith("⭐ ")] or (list(results.keys())[:1] if results else [])
                    for name in my_rs_keys:
                        r    = results[name]
                        s    = pd.Series(r["returns"], index=pd.to_datetime(r["index"]))
                        roll = _rolling_sharpe(s, window=_rs_window)
                        if len(roll) < 5: continue
                        _loaded_src_rs = st.session_state.get("portfolio_source","")
                        port_lbl_rs = name.replace("⭐ ","")
                        fig_rs.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines",
                            name=port_lbl_rs, line=dict(width=2.5, color=MY_PORT_COLOR, dash="solid"),
                            hovertemplate=f"<b>{port_lbl_rs}</b><br>%{{x|%b %Y}}<br>Sharpe: %{{y:.2f}}<extra></extra>"))

                    for cmp_name, cmp_series in std_chart_comparisons.items():
                        try:
                            roll_cmp = _rolling_sharpe(cmp_series, window=_rs_window)
                            if len(roll_cmp) < 5: continue
                            c_ = COMP_COLORS.get(cmp_name, "#94a3b8")
                            _disp = _clean_label(cmp_name)
                            fig_rs.add_trace(go.Scatter(x=roll_cmp.index, y=roll_cmp.values, mode="lines",
                                name=_disp, line=dict(width=1.5, color=c_, dash="solid"),
                                hovertemplate=f"<b>{_disp}</b><br>%{{x|%b %Y}}<br>Sharpe: %{{y:.2f}}<extra></extra>"))
                        except Exception: pass
    
                    # Extra strategy overlays on RS
                    for _ei, (_en, _es) in enumerate(_extra_chart_data.items()):
                        try:
                            _roll = _rolling_sharpe(_es, window=_rs_window)
                            _ecol = _extra_colors[_ei % len(_extra_colors)]
                            _disp_e = _clean_label(_en)
                            fig_rs.add_trace(go.Scatter(x=_roll.index, y=_roll.values, mode="lines",
                                name=_disp_e, line=dict(width=1.5, color=_ecol, dash="solid"),
                                hovertemplate=f"<b>{_disp_e}</b><br>%{{x|%b %Y}}<br>Sharpe: %{{y:.2f}}<extra></extra>"))
                        except Exception: pass
    
                    fig_rs.add_hline(y=0, line_color="#e5e7eb", line_width=1.5,
                                      annotation_text="0", annotation_position="right",
                                      annotation_font=dict(size=10, color="#9ca3af"))
                    fig_rs.add_hline(y=1, line_color="#bbf7d0", line_width=1,
                                      annotation_text="1.0", annotation_position="right",
                                      annotation_font=dict(size=10, color="#059669"))
                    _rs_months = round((_rs_window or 126) / 21)
                    fig_rs.update_layout(
                        title=dict(text=f"Rolling {_rs_months}-Month Sharpe Ratio", x=0, xanchor="left"),
                        yaxis_title="Sharpe Ratio", xaxis_title=None, height=360,
                        showlegend=True,
                        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e5e7eb",
                                    borderwidth=1, font=dict(size=11, color="#374151"),
                                    orientation="h", x=0.5, y=-0.22,
                                    xanchor="center", yanchor="top"),
                        **PLOT_THEME)
                    fig_rs.update_xaxes(title_text=None)
                    st.plotly_chart(fig_rs, use_container_width=True, key=f"rs_{label}",
                                    config={"displayModeBar": False})
    
            if mode in ("all", "results", "alloc_only"):
                # (Portfolio Weight Comparison section removed per UX feedback —
                #  pie charts below serve the same purpose more visually.)

                # Pie charts — only shown in alloc_only mode (Optimizer tab)
                if mode == "alloc_only":
                    try:
                        _tickers = st.session_state.get("tickers", [])
                        _my_keys = [k for k in results if k.startswith("⭐ ")]
                        if _my_keys and _tickers:
                            st.markdown(
                                '<h4 style="margin:6px 0 2px 0;font-size:1.05rem;'
                                'color:#111827;font-weight:700">🥧 Portfolio Allocations</h4>',
                                unsafe_allow_html=True,
                            )
                            # Drop Equal Weight from pie lineup
                            _pie_strats = _my_keys + [
                                k for k in results
                                if not k.startswith("⭐ ") and "Equal Weight" not in k
                            ][:5]
                            _ncols  = min(3, len(_pie_strats))
                            _pcols  = st.columns(_ncols)
                            for _pi, _psname in enumerate(_pie_strats[:6]):
                                _pr = results[_psname]
                                _pw = _pr.get("weights", [])
                                if not _pw:
                                    continue
                                _pshort = _psname.replace("⭐ ", "")[:22]
                                _fig_p  = make_allocation_pie(
                                    list(_tickers[:len(_pw)]),
                                    list(_pw),
                                    title=_pshort,
                                    height=420,
                                )
                                _pcols[_pi % _ncols].plotly_chart(
                                    _fig_p, use_container_width=True,
                                    key=f"opt_pie_{label}_{_pi}",
                                    config={"displayModeBar": False},
                                )
                    except Exception:
                        pass


        # Results stored in session state for Tab2/Tab3 display
        st.session_state["results_ready"] = True
        st.success("✅ Analysis complete — view results in the **Results** and **Charts** tabs above.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════
with main_tab2:
    st.session_state["optimizer_tab_active"] = False
    if not st.session_state.get("results_ready") or not st.session_state.get("bt1"):
        st.info("👆 Run your analysis in the **Analyzer** tab first.")
    else:
        # ── Fixed benchmark toggles ─────────────────────────────────────────
        # Three universal market-reference portfolios shown alongside the
        # analyzed portfolio across every period chart. Replaces the older
        # ad-hoc preset comparison set (60/40 + 90/10 + S&P + Conservative).
        # Each toggle is default-on, advisor can hide individually.
        if "show_bench_bnd" not in st.session_state:
            st.session_state.show_bench_bnd = True
        if "show_bench_6040" not in st.session_state:
            st.session_state.show_bench_6040 = True
        if "show_bench_spy" not in st.session_state:
            st.session_state.show_bench_spy = True

        with st.expander("📊 Fixed market benchmarks", expanded=False):
            st.caption(
                "Universal reference portfolios shown alongside your analyzed "
                "portfolio. Toggle off the ones that aren't useful for this client."
            )
            _b1, _b2, _b3 = st.columns(3)
            with _b1:
                st.checkbox("🟦 100% Bonds (BND)",
                            value=st.session_state.show_bench_bnd,
                            key="show_bench_bnd",
                            help="Vanguard Total Bond Market ETF — broad investment-grade bonds")
            with _b2:
                st.checkbox("⚖️ 60/40 (SPY + AGG)",
                            value=st.session_state.show_bench_6040,
                            key="show_bench_6040",
                            help="Classic balanced portfolio — 60% S&P 500, 40% aggregate bonds")
            with _b3:
                st.checkbox("📈 100% S&P 500 (SPY)",
                            value=st.session_state.show_bench_spy,
                            key="show_bench_spy",
                            help="SPDR S&P 500 ETF — broad US equity benchmark")

        # bt1/3/5/10 are stored as (results_dict, ef_points) tuples
        _bt1  = st.session_state.get("bt1")
        _bt3  = st.session_state.get("bt3")
        _bt5  = st.session_state.get("bt5")
        _bt10 = st.session_state.get("bt10")
        bt1   = _bt1[0]  if _bt1  and isinstance(_bt1, (list,tuple)) else (_bt1  or {})
        bt3   = _bt3[0]  if _bt3  and isinstance(_bt3, (list,tuple)) else (_bt3  or {})
        bt5   = _bt5[0]  if _bt5  and isinstance(_bt5, (list,tuple)) else (_bt5  or {})
        bt10  = _bt10[0] if _bt10 and isinstance(_bt10,(list,tuple)) else (_bt10 or {})
        ef1   = _bt1[1]  if _bt1  and isinstance(_bt1, (list,tuple)) else []
        bt3_ef  = _bt3[1]  if _bt3  and isinstance(_bt3, (list,tuple)) else []
        bt5_ef  = _bt5[1]  if _bt5  and isinstance(_bt5, (list,tuple)) else []
        bt10_ef = _bt10[1] if _bt10 and isinstance(_bt10,(list,tuple)) else []

        r2_1, r2_3, r2_5, r2_10 = st.tabs([
            "1 Year","3 Years","5 Years","10 Years"
        ])
        with r2_1:   render_tab(bt1,    ef1,     "1 Year",   mode="all")
        with r2_3:   render_tab(bt3,    bt3_ef,  "3 Years",  mode="all")
        with r2_5:   render_tab(bt5,    bt5_ef,  "5 Years",  mode="all")
        with r2_10:  render_tab(bt10,   bt10_ef, "10 Years", mode="all")

        # NOTE: The standalone "Save PDF Report" panel that previously lived
        # here was removed in May 2026. Reports are now generated through
        # the Optimizer → "Generate Proposal" flow (saved to Client Records
        # or Unassociated Reports depending on whether a client is selected
        # in Analyzer → Step 1). The underlying generate_pdf_report() helper
        # was also removed since it was the only consumer.

# ═══════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZER & PORTFOLIO RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
with main_tab3:
    # Signal to the sidebar that the Optimizer tab is active — sliders will show
    st.session_state["optimizer_tab_active"] = True

    if not st.session_state.get("results_ready") or not st.session_state.get("bt1"):
        st.info("👆 Run your analysis in the **Analyzer** tab first.")
    else:
        _bt10 = st.session_state.get("bt10") or st.session_state.get("bt1")
        bt10  = _bt10[0] if _bt10 and isinstance(_bt10,(list,tuple)) else (_bt10 or {})
        bt10_ef = _bt10[1] if _bt10 and isinstance(_bt10,(list,tuple)) else []
        ef1     = bt10_ef

        st.markdown("## Portfolio Optimizer & Recommendations")

        # Use 10yr data for optimizer (most accurate for recommendations)
        _opt_results = bt10 or st.session_state.get("bt1")

        # ═══════════════════════════════════════════════════════════════
        # ACTIVE CLIENT STATUS BAR (read-only)
        # The client picker now lives in the Analyzer tab (Step 1), so the
        # Optimizer is no longer where the advisor selects who they're
        # working with — we just display who's currently active and offer
        # a Reload button to force-regenerate recommendations.
        # ═══════════════════════════════════════════════════════════════
        _active_ck_top         = st.session_state.get("_active_client_key")
        _active_name_top       = st.session_state.get("_active_client_name", "—")
        _active_score_top      = st.session_state.get("_active_client_score")
        _active_priorities_top = st.session_state.get("_active_client_priorities", []) or []

        _status_cols = st.columns([3, 1, 1])
        with _status_cols[0]:
            if _active_ck_top:
                st.markdown(
                    f"<div style='padding:10px 14px;background:#EEF3F6;border:1px solid #E1E8EE;"
                    f"border-radius:10px;font-size:0.85rem'>"
                    f"<span style='color:#6B7E8A;font-size:0.7rem;letter-spacing:0.06em;"
                    f"text-transform:uppercase;font-weight:600'>Active client</span><br/>"
                    f"<b>{_active_name_top}</b></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No client selected — recommendations are saved to Unassociated Reports. "
                        "Pick a client in **Analyzer → Step 1** to associate this analysis with a client.")
        with _status_cols[1]:
            if _active_ck_top:
                st.metric("Risk Score", _active_score_top)
            else:
                st.metric("Risk Score", "—", help="Default score 50 (no client)")
        with _status_cols[2]:
            if st.button("Reload",
                         use_container_width=True,
                         help="Force regenerate recommendations from scratch",
                         key="reload_recs_top"):
                for k in list(st.session_state.keys()):
                    if k.startswith("proposals_working_") or k.startswith("proposals_basis_fp_"):
                        st.session_state.pop(k, None)
                st.rerun()

        # ═══════════════════════════════════════════════════════════════
        # AUTO-GENERATE PROPOSALS
        # If no working proposal exists for the effective client key, generate
        # one immediately on tab render. No button click required.
        #
        # Also auto-regenerate when the submitted portfolio changes (e.g. user
        # re-runs the Analyzer with different tickers/weights). We track this
        # via a fingerprint of (tickers + weights). Without this, the Balanced
        # tier would stay stuck on the original portfolio even after the user
        # submits a new one.
        # ═══════════════════════════════════════════════════════════════
        _eff_ck_top = _active_ck_top if _active_ck_top else UNASSOCIATED_CLIENT_KEY
        _eff_wkey   = f"proposals_working_{_eff_ck_top}"

        # Compute a fingerprint of the currently-submitted portfolio
        _sub_for_fp = dict(st.session_state.get("submitted_weights", {}) or {})
        _fp_now = tuple(sorted(
            (t.upper(), round(float(w or 0), 4))
            for t, w in _sub_for_fp.items() if float(w or 0) > 0
        ))
        # Include the active blend in the fingerprint so any Composite
        # Optimizer slider change triggers a proposal rebuild — Option 2
        # defaults to Step 1 verbatim when no sliders are set, but flips
        # to the blend output as soon as any slider goes above 0.
        _blend_fp = tuple(sorted(
            (str(k), int(v))
            for k, v in (st.session_state.get("blend_knobs", {}) or {}).items()
            if int(v or 0) > 0
        ))
        _fp_now_full = (_fp_now, _blend_fp)
        _fp_key = f"proposals_basis_fp_{_eff_ck_top}"
        _fp_prev = st.session_state.get(_fp_key)

        # Trigger regeneration if no working proposal OR fingerprint changed
        _need_regen = (not st.session_state.get(_eff_wkey)) or (
            _fp_prev is not None and _fp_prev != _fp_now_full and len(_fp_now) > 0
        )

        if _need_regen:
            # Build Balanced basis from submitted weights (no Step 1 adjustment)
            _auto_weights = dict(st.session_state.get("submitted_weights", {}) or {})
            # ── Determine the portfolio feeding the optimizer ─────────────
            # NEW DATA FLOW (post-redesign):
            #   • Step 1 (Analyzer Securities)   = the portfolio the optimizer
            #     uses as its base (what it builds Option 1/2/3 variants of).
            #   • Step 2 (Client's Current)      = comparison baseline only.
            #     Shown in PCM & on proposal cards as "Client's Current",
            #     but the optimizer no longer reads from it.
            #
            # Earlier the priority was reversed — Step 2 won and Step 1 was
            # the fallback. The user has flipped this so the analyzed
            # portfolio is always the optimizer's starting point. The
            # client_current_portfolio key still exists, just isn't consulted
            # here anymore.
            if not _auto_weights:
                _analyzed_tks = list(st.session_state.get("tickers", []) or [])
                if _analyzed_tks:
                    per = 100.0 / len(_analyzed_tks)
                    _auto_weights = {t: per for t in _analyzed_tks}
            _auto_user_tks = [t for t, w in _auto_weights.items() if float(w or 0) > 0]
            _auto_user_ws  = {t: float(w) for t, w in _auto_weights.items() if float(w or 0) > 0}

            if _auto_user_tks:
                _auto_score  = _active_score_top if _active_ck_top else 50
                _auto_priors = _active_priorities_top if _active_ck_top else []

                # If any Composite Optimizer slider is set above zero, use the
                # blend output as Option 2; otherwise use Step 1 verbatim.
                _blend_active = any(
                    int(v or 0) > 0
                    for v in (st.session_state.get("blend_knobs", {}) or {}).values()
                )
                _blend_basis = st.session_state.get("proposal_blend_basis")

                # Lazy compute: if blend is active but no basis is cached
                # (or it's stale because sliders moved after analysis ran),
                # rebuild it from the current blend_knobs + bt10's strategy
                # weights. This is what makes slider movement immediately
                # drive Option 2 without re-running the whole analysis.
                if _blend_active:
                    try:
                        _bt10_for_blend = st.session_state.get("bt10")
                        _bt10_dict = (
                            _bt10_for_blend[0]
                            if _bt10_for_blend and isinstance(_bt10_for_blend, (list, tuple))
                            else (_bt10_for_blend or {})
                        )
                        if _bt10_dict:
                            _knobs_now = {
                                k: int(v) for k, v in
                                (st.session_state.get("blend_knobs", {}) or {}).items()
                                if int(v or 0) > 0
                            }
                            # Pull strategy weights for the matching strategies
                            _avail_w = {}
                            for _sn, _kv in _knobs_now.items():
                                _sr = _bt10_dict.get(_sn)
                                if _sr and _sr.get("weights"):
                                    _avail_w[_sn] = list(_sr["weights"])
                            if _avail_w:
                                _blend_w_now = blend_strategies(
                                    {k: v for k, v in _knobs_now.items() if k in _avail_w},
                                    _avail_w,
                                )
                                if _blend_w_now is not None:
                                    _b_tks = list(_auto_user_tks)[:len(_blend_w_now)]
                                    _b_ws  = [float(w) for w in _blend_w_now[:len(_b_tks)]]
                                    _blend_basis = {
                                        "balanced_tickers": _b_tks,
                                        "balanced_weights": _b_ws,
                                    }
                                    st.session_state["proposal_blend_basis"] = _blend_basis
                    except Exception:
                        pass

                _has_blend    = bool(_blend_active and _blend_basis
                                     and _blend_basis.get("balanced_tickers"))

                if _has_blend:
                    # Blend mode: Option 2 = the blend output. Options 1 and 3
                    # are derived from the blend's allocation by ±15% bond/equity tilt.
                    st.session_state[_eff_wkey] = generate_three_tier_from_blend(
                        _blend_basis, _auto_score, priorities=_auto_priors,
                        mode="manual",
                    )
                else:
                    # Default mode: Option 2 = Step 1 holdings verbatim.
                    st.session_state[_eff_wkey] = generate_three_tier_proposals(
                        _auto_score, priorities=_auto_priors,
                        user_tickers=_auto_user_tks,
                        user_weights=_auto_user_ws if _auto_user_ws else None,
                    )
                # Stash full fingerprint (tickers + blend) so a slider change
                # forces a fresh regen on the next rerun.
                st.session_state[_fp_key] = _fp_now_full

        st.markdown("---")

        # (Composite Optimizer sliders live in the sidebar; the visible blender
        #  chart was removed per UX feedback. Blend computation happens inline
        #  below when the Load Recommendations section needs it.)

        # ── Use This Blend For Proposals ─────────────────────────────
        # Three proposal-generation actions moved here from Client Proposals:
        #   1. Auto-Generate from Risk Score — uses user's submitted tickers
        #   2. Generate from Blend — uses current composite blend as Balanced basis
        # ═══════════════════════════════════════════════════════════════
        # STEP 1 · REVIEW & EDIT PROPOSALS
        # Proposals were auto-generated at the top when this tab rendered (or
        # when a client was selected). This section renders the editable tier
        # tabs. Loading is implicit — no separate "Load Recommendations" button.
        # ═══════════════════════════════════════════════════════════════
        st.markdown(
            '<h3 style="margin:16px 0 4px 0;font-size:1.25rem;color:#111827;'
            'font-weight:700;letter-spacing:-0.01em">'
            'Step 1 · Review &amp; Edit Proposals</h3>',
            unsafe_allow_html=True,
        )
        _active_ck         = _active_ck_top
        _active_name       = _active_name_top
        _active_score      = _active_score_top
        _active_priorities = _active_priorities_top

        if _active_ck:
            st.caption(
                f"Proposals tailored for **{_active_name}** · score **{_active_score}**. "
                "Edit any tier below; changes save automatically to the working version."
            )
        else:
            st.caption(
                "_No client selected — proposals use a default score of 50 and save to "
                "Unassociated Reports. Pick a client above to retarget._"
            )

        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # CLIENT PROPOSALS — tier rendering + save flow
        # Client selection happens at the TOP of the Optimizer tab now; this
        # section just reads active-client state and renders the rest of the flow.
        # ═══════════════════════════════════════════════════════════════

        _all_profiles = _load_json_safe(CLIENT_PROFILES_FILE)
        _pickable_profiles = {k: p for k, p in _all_profiles.items()
                               if k != UNASSOCIATED_CLIENT_KEY
                               and not p.get("_is_unassociated", False)}

        # Resolve effective client from the top-level picker's stashed state
        _picked_ck = st.session_state.get("_active_client_key")

        if _picked_ck:
            _ck = _picked_ck
            _cp = _all_profiles.get(_ck, {})
            _cscore = int(_cp.get("overall_score", 50))
            _cname  = _cp.get("client_name", "Client")
            _cpriorities = _cp.get("priorities", []) or []

            # Show client priorities as chips so advisor sees what's tilting the proposal
            if _cpriorities:
                _PRIORITY_LBL_SHORT = {
                    "capital_preservation": "🛡️ Capital Preservation",
                    "insurance_planning":   "🔒 Insurance Planning",
                    "income_generation":    "💵 Income",
                    "capital_appreciation": "📈 Growth",
                    "diversification":      "🌐 Diversification",
                    "tax_efficiency":       "🧾 Tax Efficiency",
                    "liquidity":            "💧 Liquidity",
                    "social_impact":        "🌱 Social/Impact",
                    "legacy_planning":      "🏛 Legacy",
                }
                chips = "".join(
                    f"<span style='display:inline-block;background:#D8ECEC;color:#0E5C5E;"
                    f"padding:4px 10px;border-radius:14px;font-size:0.74rem;font-weight:600;"
                    f"margin-right:5px;margin-bottom:5px'>{_PRIORITY_LBL_SHORT.get(pk, pk)}</span>"
                    for pk in _cpriorities
                )
                st.markdown(
                    "<div style='margin-top:4px;margin-bottom:8px'>"
                    "<span style='font-size:0.78rem;color:#6b7280;font-weight:600;"
                    "margin-right:8px'>Client priorities:</span>" + chips + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            # No client picked — route to Unassociated Reports folder
            ensure_unassociated_profile()
            _ck = UNASSOCIATED_CLIENT_KEY
            _cscore      = 50
            _cname       = "📂 Unassociated Reports"
            _cpriorities = []

        # Everything from here down runs whether _ck is a real client or unassociated.

        # (The "Selected for proposal" summary strip that previously rendered
        # here was removed per advisor feedback — Step 2 already shows this
        # information, so it was redundant in the Optimizer view.)

        # ── Generate / regenerate proposals ────────────────
        _working_key = f"proposals_working_{_ck}"
        if _working_key not in st.session_state:
            st.session_state[_working_key] = None

        _blend_basis = st.session_state.get("proposal_blend_basis")
        _has_blend   = bool(_blend_basis and _blend_basis.get("balanced_tickers"))

        # (Auto-Generate / Generate from Blend / Reset buttons now live
        #  above in the Composite Optimizer "Generate Client Proposals" section.)

        # ── Show editable proposal cards ───────────────────
        _working = st.session_state[_working_key]

        # Always-prepare tier structure so Final Selection logic below works
        # whether proposals are generated or not
        if _working:
            # Display labels for the tier tabs. Internal keys
            # (conservative/balanced/aggressive/alternate) stay unchanged so
            # all save/load and PDF code continues working — only what the
            # advisor sees on screen has been re-ordered. New convention
            # (per advisor preference):
            #   Option 1 = Proposed (balanced — Step 1 holdings verbatim)
            #   Option 2 = Slightly more conservative (corridor min-vol)
            #   Option 3 = Slightly more aggressive (corridor max-Sharpe)
            # The Proposed option is presented first because that's the
            # default starting point the advisor will most often build from;
            # the conservative/aggressive variants are alternatives.
            _tier_tabs_cfg = [
                ("Option 1 (proposed)",                    "balanced"),
                ("Option 2 (slightly more conservative)",  "conservative"),
                ("Option 3 (slightly more aggressive)",    "aggressive"),
                ("Broad-ETF Alternate",                    "alternate"),
            ]
            _tier_tabs_cfg = [x for x in _tier_tabs_cfg if x[1] in _working]
            tier_keys = [t[1] for t in _tier_tabs_cfg]
        else:
            _tier_tabs_cfg = []
            tier_keys = []

        if _working:
            st.markdown(
                "<h4 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
                "margin:8px 0 6px 0'>Proposed Portfolios</h4>",
                unsafe_allow_html=True,
            )
            st.caption("Edit each tier → check 'Save to profile' to make it available in the final Option dropdowns below.")
            # ── Composite Optimizer (moved here from sidebar) ────────────────
            # The blend sliders that mix HRP / NCO / MaxDiv / etc. live here on
            # the Optimizer tab where they're actually relevant. Default state
            # is all-zero (= ignore the blend), so opening this tab without
            # touching the sliders produces no difference vs the prior
            # behavior. Touching any slider activates the blend in place of
            # the default Balanced strategy when proposals are generated.
            with st.expander("Composite Optimizer (blend strategies)", expanded=False):
                st.caption(
                    "0 = ignore · 10 = full weight. Mix multiple optimization "
                    "strategies into a custom blend. The active blend is used "
                    "as the basis for the Balanced tier when you generate proposals."
                )
                _opt_blend_knobs = {}
                for _grp_name, _strategies in STRATEGY_GROUPS.items():
                    with st.expander(_grp_name, expanded=False):
                        for _strat_name, _strat_desc in _strategies.items():
                            _val = st.slider(
                                _strat_name, min_value=0, max_value=10, value=0,
                                key=f"opt_blend_{_strat_name.replace(' ','_').replace('/','_')}",
                                help=_strat_desc,
                            )
                            _opt_blend_knobs[_strat_name] = _val
                # Active blend summary
                _opt_active = {k: v for k, v in _opt_blend_knobs.items() if v > 0}
                if _opt_active:
                    _opt_total = sum(_opt_active.values())
                    _opt_parts = [f"<b>{k}</b> {v/_opt_total:.0%}"
                                  for k, v in sorted(_opt_active.items(), key=lambda x: -x[1])]
                    st.markdown(
                        "<div style='background:#EEF3F6;border:1px solid #E1E8EE;"
                        "border-radius:10px;padding:10px 14px;margin-top:8px;"
                        "font-size:0.78rem;color:#0B1F2A;line-height:1.6'>"
                        "<div style='font-size:0.66rem;color:#6B7E8A;letter-spacing:0.06em;"
                        "text-transform:uppercase;font-weight:600;margin-bottom:4px'>Active blend</div>"
                        + " · ".join(_opt_parts) +
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("_No sliders set — proposals will use default Balanced strategy._")
                # Persist for proposal generation
                st.session_state["blend_knobs"] = _opt_blend_knobs
            st.markdown("")

            tier_tabs = st.tabs([t[0] for t in _tier_tabs_cfg])

            for _tab, _tk in zip(tier_tabs, tier_keys):
                with _tab:
                    prop = _working[_tk]
                    st.caption(prop["rationale"])

                    # ── Save-to-profile checkbox (per-tier flag) ─────
                    prop["save_to_profile"] = st.checkbox(
                        "💾 Save this tier as a recommended portfolio on the client's profile",
                        value=prop.get("save_to_profile", False),
                        key=f"save_flag_{_ck}_{_tk}",
                        help="When checked, this tier will appear in the Option #1/2/3 dropdowns below as a recommended portfolio.",
                    )

                    # ── Live tier: gauge + pie on left, editable key on right ─
                    _live_tks = prop.get("tickers", [])
                    _live_ws  = prop.get("weights", [])
                    if _live_tks and _live_ws and sum(float(w) for w in _live_ws) > 0:
                        # New scoring: per-holding score → weighted avg → correlation adj.
                        # Per-holding scores via security_risk_score (cached/quick).
                        _hold_scores = []
                        _hold_vols   = []
                        for _tt in _live_tks:
                            _r = security_risk_score(_tt)
                            if _r:
                                _hold_scores.append(_r["score"])
                                _hold_vols.append(_r.get("ann_vol", 0.15))
                            else:
                                _hold_scores.append(50)
                                _hold_vols.append(0.15)
                        # Compute REAL portfolio vol from price history so the
                        # correlation discount applies (matches the PCM exactly,
                        # avoiding the previous 2-point divergence between the
                        # optimizer gauge and the PCM Risk Score row).
                        # Cached on session state by (tickers, rounded weights)
                        # so we don't re-fetch prices when the same portfolio
                        # is rendered multiple times in the same session.
                        _portfolio_vol = _cached_portfolio_vol(
                            tuple(_live_tks),
                            tuple(round(float(w), 4) for w in _live_ws),
                        )
                        _tier_score = compute_portfolio_risk_score(
                            _live_tks, _live_ws,
                            holding_scores=_hold_scores,
                            holding_vols=_hold_vols,
                            portfolio_vol=_portfolio_vol,
                        )

                        # Layout: pie+gauge on the left, editable key on the right
                        _tier_pie_col, _tier_key_col = st.columns([2, 1])

                        with _tier_pie_col:
                            # Small gauge on top-left, then pie fills below
                            _tg_col, _tg_spacer = st.columns([1, 2])
                            with _tg_col:
                                st.plotly_chart(
                                    make_risk_gauge(_tier_score, height=160),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"prop_gauge_work_{_ck}_{_tk}",
                                )
                            _fig_t = make_allocation_pie(
                                _live_tks, _live_ws,
                                title=None,   # tier label is on the tab above
                                height=360,
                            )
                            # Hide Plotly legend — render editable key in right column
                            _fig_t.update_layout(showlegend=False)
                            _fig_t.update_traces(domain=dict(x=[0.05, 0.95], y=[0.0, 1.0]))
                            st.plotly_chart(
                                _fig_t,
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"prop_pie_work_{_ck}_{_tk}",
                            )

                        with _tier_key_col:
                            st.markdown(
                                "<div style='font-size:0.78rem;font-weight:700;color:#64748b;"
                                "letter-spacing:0.06em;text-transform:uppercase;"
                                "margin:10px 0 6px 0'>Holdings</div>",
                                unsafe_allow_html=True,
                            )
                            # Build palette matching the pie's sort+stride ordering
                            _pair_sorted_t = sorted(zip(_live_tks, _live_ws),
                                                    key=lambda x: -float(x[1] or 0))
                            _nn = len(_pair_sorted_t)
                            if _nn <= len(PIE_PALETTE):
                                _stride = max(1, len(PIE_PALETTE) // max(_nn, 1))
                                _tpal = [PIE_PALETTE[(i * _stride) % len(PIE_PALETTE)]
                                         for i in range(_nn)]
                            else:
                                _tpal = [PIE_PALETTE[i % len(PIE_PALETTE)]
                                         for i in range(_nn)]

                            # Build in-order palette for the underlying stored ticker order
                            # (pie sorts internally; key uses sorted order for visual match)
                            _edit_rows = []
                            _ttotal = 0.0
                            # We want to let the user delete/edit rows. Iterate by sorted
                            # order in the UI; index back to the live lists via ticker name.
                            for _ii, (_tkr_k, _wk) in enumerate(_pair_sorted_t):
                                _color = _tpal[_ii]
                                _sc, _tc, _ic, _rc = st.columns([0.25, 0.95, 1.1, 0.35])
                                _sc.markdown(
                                    f"<div style='width:12px;height:12px;background:{_color};"
                                    f"border-radius:3px;margin-top:9px'></div>",
                                    unsafe_allow_html=True,
                                )
                                _new_tkr = _tc.text_input(
                                    f"ticker_{_tk}_{_ii}",
                                    value=_tkr_k,
                                    key=f"prop_{_ck}_{_tk}_t_{_ii}",
                                    label_visibility="collapsed",
                                ).upper().strip()
                                _new_wt = _ic.number_input(
                                    f"w_{_tk}_{_ii}",
                                    min_value=0.0, max_value=100.0,
                                    value=float(_wk or 0),
                                    step=0.5, format="%.2f",
                                    key=f"prop_{_ck}_{_tk}_w_{_ii}",
                                    label_visibility="collapsed",
                                )
                                _rm = _rc.button(
                                    "🗑",
                                    key=f"prop_{_ck}_{_tk}_rm_{_ii}",
                                    help="Remove row",
                                )
                                if not _rm and _new_tkr:
                                    _edit_rows.append((_new_tkr, _new_wt))
                                    _ttotal += _new_wt

                            # Add-row control
                            _ac1, _ac2 = st.columns([1.6, 0.6])
                            _new_add_t = _ac1.text_input(
                                f"add_{_tk}",
                                key=f"prop_{_ck}_{_tk}_add",
                                placeholder="+ Add ticker",
                                label_visibility="collapsed",
                            ).upper().strip()
                            if _ac2.button("Add", key=f"prop_{_ck}_{_tk}_add_btn",
                                           use_container_width=True) and _new_add_t:
                                _edit_rows.append((_new_add_t, 0.0))

                            # Persist edits back into session state
                            st.session_state[_working_key][_tk]["tickers"] = [r[0] for r in _edit_rows]
                            st.session_state[_working_key][_tk]["weights"] = [r[1] for r in _edit_rows]

                            # Total line + normalize button
                            st.markdown(
                                f"<div style='border-top:1px solid #e5e7eb;margin-top:8px;"
                                f"padding-top:8px;font-size:0.8rem;color:#111827'>"
                                f"<b>Total: {_ttotal:.1f}%</b>"
                                + ("" if abs(_ttotal - 100.0) <= 0.1
                                   else f" <span style='color:#dc2626'>"
                                        f"({'over' if _ttotal > 100 else 'under'} 100%)</span>")
                                + "</div>",
                                unsafe_allow_html=True,
                            )
                            if abs(_ttotal - 100.0) > 0.1 and _ttotal > 0:
                                if st.button(
                                    "⚖ Normalize to 100%",
                                    key=f"prop_normalize_{_ck}_{_tk}",
                                    use_container_width=True,
                                    type="primary",
                                ):
                                    _scale = 100.0 / _ttotal
                                    st.session_state[_working_key][_tk]["weights"] = [
                                        round(w * _scale, 2) for w in
                                        st.session_state[_working_key][_tk]["weights"]
                                    ]
                                    st.rerun()
        else:
            st.info("No proposals yet — this means no portfolio has been analyzed. "
                    "Run an analysis in the **Analyzer** tab first, then return here. "
                    "Recommendations will auto-generate based on your submitted portfolio.")

        # ── FINAL SELECTION — always visible ──────────────────
        st.markdown("---")
        st.markdown(
            "<h4 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
            "margin:8px 0 6px 0'>Step 2 · Select Final Proposal for Report</h4>",
            unsafe_allow_html=True,
        )
        st.caption("Option #1 defaults to your **Subject Portfolio** (the holdings you submitted). "
                   "Options #2 and #3 default to recommended Conservative / Aggressive tiers. "
                   "All three are editable and can be set to any saved or preset portfolio.")

        # Build the union of available options
        _final_options = ["— none —", "Custom — Enter Your Own Tickers"]

        # ── Subject Portfolio: the user's submitted holdings ──
        # This is the actual portfolio they entered in the Analyzer. Promote
        # it to the top of the list so it's discoverable as a baseline option.
        _subject_label = "📊 Subject Portfolio (your submitted holdings)"
        _subject_weights = dict(st.session_state.get("submitted_weights", {}) or {})
        _subject_tickers = [t for t, w in _subject_weights.items() if float(w or 0) > 0]
        if _subject_tickers:
            _final_options.append(_subject_label)

        # Analyzer saved portfolios
        try:
            _final_options += [f"📁 {n}" for n in load_saved().keys()]
        except Exception: pass
        # Preset allocations
        _popular_keys = [
            k for k in POPULAR_PORTFOLIOS
            if not k.startswith("── ") and k != "Custom — Enter Your Own Tickers"
            and POPULAR_PORTFOLIOS.get(k)
        ]
        _final_options += [f"🧩 {k}" for k in _popular_keys]
        # Comparison portfolios if picked in Analyzer
        for _cmp_key in ("cmp_port1", "cmp_port2"):
            _sel = st.session_state.get(_cmp_key, "None")
            if _sel and _sel != "None" and _sel != "Custom — Enter Your Own Tickers":
                _tag = _sel if _sel.startswith("📁 ") else f"🧩 {_sel}"
                if _tag not in _final_options:
                    _final_options.append(_tag)

        # Recommended portfolios — ALL tiers from the working proposal show up
        # automatically so the user can pick them without needing to toggle
        # "Save to profile" first. The save-to-profile flag only matters for
        # saving the tier to the client's profile later.
        _recommended_options = {}   # maps "⭐ Recommended — {Label}" → tier_key
        if _working:
            for _tk in tier_keys:
                _t = _working[_tk]
                if _t.get("tickers"):
                    _label = f"⭐ Recommended — {_t.get('label', _tk.title())}"
                    _recommended_options[_label] = _tk
                    if _label not in _final_options:
                        _final_options.append(_label)

        # Default mapping (post-restructure):
        #   Option 1 → Subject Portfolio (Step 1 holdings verbatim)
        #              — falls back to Balanced tier if subject not available
        #   Option 2 → Recommended Conservative tier (corridor min-vol)
        #   Option 3 → Recommended Aggressive tier (corridor max-Sharpe)
        _slot_defaults_priority = {
            "option_1": [("subject", None), ("recommended", "balanced")],
            "option_2": [("recommended", "conservative")],
            "option_3": [("recommended", "aggressive")],
        }

        _opt_cols = st.columns(3)
        _final_picks = {}
        for _i, (_col, _slot) in enumerate(zip(_opt_cols, ("option_1", "option_2", "option_3"))):
            _wkey_final = f"final_{_ck}_{_slot}"
            # Compute a sensible default in priority order
            _default = "— none —"
            for _kind, _arg in _slot_defaults_priority.get(_slot, []):
                if _kind == "subject" and _subject_label in _final_options:
                    _default = _subject_label
                    break
                if _kind == "recommended":
                    _found = False
                    for _lbl, _tk_map in _recommended_options.items():
                        if _tk_map == _arg:
                            _default = _lbl
                            _found = True
                            break
                    if _found:
                        break
            # Use any previous user selection for this slot; otherwise use the computed default
            _prev = st.session_state.get(_wkey_final, _default)
            if _prev not in _final_options:
                _prev = _default
            _idx = _final_options.index(_prev) if _prev in _final_options else 0
            with _col:
                _pick = st.selectbox(
                    f"**Option #{_i+1}**",
                    options=_final_options,
                    index=_idx,
                    key=_wkey_final,
                )
                _final_picks[_slot] = _pick

        # ── Advisor notes + save (require _working to save) ─
        st.markdown("---")
        adv_notes = st.text_area(
            "Advisor notes for this proposal version *(optional)*",
            placeholder="e.g. 'Initial proposal for Q2 2026 portfolio review.'",
            key=f"prop_notes_{_ck}", height=80,
        )
        _can_save = bool(_working and tier_keys)
        if st.button("💾 Save Proposal Version", type="primary",
                     use_container_width=True, key=f"save_prop_{_ck}",
                     disabled=not _can_save,
                     help=(None if _can_save else
                           "Proposals auto-generate from your Analyzer submission. "
                           "Run an analysis in the Analyzer tab first if you don't "
                           "see any tiers.")):
            _all_ok = all(
                abs(sum(st.session_state[_working_key][tk]["weights"]) - 100.0) <= 0.1
                for tk in tier_keys
            )
            if not _all_ok:
                st.error(f"All {len(tier_keys)} tiers must have weights summing to 100%.")
            else:
                from datetime import datetime as _dt
                _vid = f"v{_dt.now().strftime('%Y%m%d_%H%M%S')}"
                # If saving to the unassociated folder, make sure it exists
                if _ck == UNASSOCIATED_CLIENT_KEY:
                    ensure_unassociated_profile()
                proposal_dict = {
                    "version_id":   _vid,
                    "created_at":   _dt.now().strftime("%Y-%m-%d %H:%M"),
                    "client_key":   _ck,
                    "client_name":  _cname,
                    "client_score": _cscore,
                    "advisor_notes": adv_notes.strip(),
                    "tiers": {
                        tk: dict(st.session_state[_working_key][tk])
                        for tk in tier_keys
                    },
                    "final_picks": _final_picks,
                    # Snapshot the Step 2 client-current portfolio at save
                    # time so the PDF can render it on the backtest chart
                    # as a distinct "Current Portfolio" line. The PDF doesn't
                    # have access to live session_state at render time —
                    # this is the only way to thread Step 2 through.
                    "client_current_portfolio":
                        st.session_state.get("client_current_portfolio"),
                }
                # ── Backtest each tier and attach returns to the proposal ─────
                # This is what makes the saved-proposal PDFs able to render
                # drawdown / rolling-Sharpe / Monte Carlo charts. Adds ~3-8s
                # of network I/O at save time but means PDFs render fast and
                # offline forever after. Falls back gracefully if any ticker
                # can't be priced — the PDF builder will skip charts for that
                # tier rather than crash.
                _enrich_ok = False
                _enrich_err = None
                with st.spinner("Backtesting tiers and attaching return data…"):
                    try:
                        attach_returns_to_proposal(proposal_dict, years=5)
                        # Count how many tiers ended up with usable return data
                        _tiers_with_data = [
                            tk for tk, tv in proposal_dict.get("tiers", {}).items()
                            if isinstance(tv, dict) and tv.get("returns")
                        ]
                        _enrich_ok = len(_tiers_with_data) > 0
                    except Exception as _enr_e:
                        _enrich_err = str(_enr_e)

                save_proposal(_ck, _vid, proposal_dict)

                if _enrich_ok:
                    st.success(
                        f"✅ Proposal {_vid} saved for {_cname}. "
                        f"Performance chart data attached for {len(_tiers_with_data)} "
                        f"tier(s) — drawdown, rolling Sharpe, and Monte Carlo charts "
                        f"are now available in the PDF Report Builder."
                    )
                elif _enrich_err:
                    st.warning(
                        f"Proposal {_vid} saved, but couldn't fetch backtest data "
                        f"for charts: {_enrich_err}. "
                        "PDF will still generate, just without performance charts. "
                        "Re-save once your network is back to enable them."
                    )
                else:
                    st.warning(
                        f"Proposal {_vid} saved, but no chart data was attached "
                        "(prices may have been unavailable for these tickers). "
                        "PDF will still generate without performance charts."
                    )
                st.session_state[_working_key] = None
                st.rerun()

        # ═══════════════════════════════════════════════════════════════
        # PORTFOLIO ALLOCATION PIE CHARTS (moved to bottom per UX feedback)
        # ═══════════════════════════════════════════════════════════════
        # Guarded: only runs if analysis results exist. _opt_results may not
        # be in scope on first render before any analysis has run.
        if st.session_state.get("results_ready") and st.session_state.get("bt1"):
            _bt_src  = st.session_state.get("bt10") or st.session_state.get("bt1")
            _opt_res = _bt_src[0] if _bt_src and isinstance(_bt_src,(list,tuple)) else (_bt_src or {})
            _opt_ef  = _bt_src[1] if _bt_src and isinstance(_bt_src,(list,tuple)) else []
            if _opt_res:
                st.markdown("---")
                render_tab(_opt_res, _opt_ef, "Optimizer_alloc", mode="alloc_only")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — CLIENT RECORDS
# ═══════════════════════════════════════════════════════════════
with main_tab4:
    st.session_state["optimizer_tab_active"] = False
    # Header removed — the tab label "👥 Client Records" already names the
    # section, so the duplicate H3 below it was redundant.
    st.caption("Risk assessments from the client portal appear here automatically.")

    if st.button("Refresh Records", key="refresh_records"):
        st.rerun()

    PROFILES_FILE          = "risk_profiles.json"
    CLIENT_HOLDINGS_FILE   = "client_holdings.json"
    CLIENT_WATCHLISTS_FILE = "client_watchlists.json"

    # Local helper that delegates to data_store so this tab reads from
    # the shared GitHub repo (where the portal writes), not from local
    # ephemeral disk. (The earlier ad-hoc os.path.exists/open() loader
    # silently bypassed the shared store and was the reason this tab
    # showed no client records even when profiles existed in the repo.)
    def _safe_load_json(path):
        val = _shared_load_json(path, default={})
        return val if isinstance(val, dict) else {}

    client_profiles      = _safe_load_json(PROFILES_FILE)
    all_client_holdings  = _safe_load_json(CLIENT_HOLDINGS_FILE)
    all_client_watchlist = _safe_load_json(CLIENT_WATCHLISTS_FILE)

    if not client_profiles:
        st.markdown("""
        <div style='background:#f9fafb;border:1.5px dashed #e5e7eb;border-radius:16px;
                    padding:48px;text-align:center;margin-top:16px'>
            <div style='font-size:2rem;margin-bottom:12px'>📭</div>
            <div style='font-weight:600;color:#111827'>No Client Records Yet</div>
            <div style='color:#9ca3af;font-size:0.875rem;margin-top:6px'>
                Records appear here when clients complete a risk assessment on the client portal.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Stats exclude the synthetic "Unassociated Reports" folder
        _real_profiles = {k: v for k, v in client_profiles.items()
                          if k != UNASSOCIATED_CLIENT_KEY
                          and not v.get("_is_unassociated")}
        total_c  = len(_real_profiles)
        avg_sc   = (sum(p.get("overall_score",0) for p in _real_profiles.values()) / total_c
                    if total_c else 0)
        complete = sum(1 for p in _real_profiles.values() if p.get("status") == "Complete")
        high_r   = sum(1 for p in _real_profiles.values() if p.get("overall_score",0) > 60)
        sm1,sm2,sm3,sm4 = st.columns(4)
        sm1.metric("Total Clients",   total_c)
        sm2.metric("Complete",         complete)
        sm3.metric("Avg Risk Score",   f"{avg_sc:.0f}")
        sm4.metric("High Risk (>60)",  high_r)
        st.markdown("---")

        # Sort: Unassociated Reports folder pinned to the top; real clients
        # sorted by most recent completion timestamp descending.
        def _sort_key(item):
            k, p = item
            if k == UNASSOCIATED_CLIENT_KEY or p.get("_is_unassociated"):
                return (0, "")  # always first
            return (1, -1 * len(p.get("completed_at", "")),
                    p.get("completed_at", ""))
        sorted_profiles = sorted(client_profiles.items(), key=_sort_key,
                                  reverse=False)
        # Actually simpler: split into two lists for clarity
        _unassoc_entries = [
            (k, p) for k, p in client_profiles.items()
            if k == UNASSOCIATED_CLIENT_KEY or p.get("_is_unassociated")
        ]
        _real_entries = sorted(
            [(k, p) for k, p in client_profiles.items()
             if k != UNASSOCIATED_CLIENT_KEY and not p.get("_is_unassociated")],
            key=lambda x: x[1].get("completed_at",""), reverse=True,
        )
        sorted_profiles = _unassoc_entries + _real_entries

        for profile_key, p in sorted_profiles:
            # ── UNASSOCIATED FOLDER — special rendering ─────────────
            if profile_key == UNASSOCIATED_CLIENT_KEY or p.get("_is_unassociated"):
                _u_props = load_all_proposals().get(profile_key, {})
                _count   = len(_u_props)
                with st.expander(
                    f"📂 Unassociated Reports  ·  {_count} "
                    f"proposal{'s' if _count != 1 else ''}",
                    expanded=(_count > 0),
                ):
                    st.caption(
                        "Proposals generated in the Optimizer without a client selected are "
                        "stored here. Use the **Move to Client** button on any proposal below "
                        "to associate it with a real client record."
                    )
                    if _count == 0:
                        st.info("No unassociated proposals yet.")
                    else:
                        # List all unassociated proposals with a move-to-client action
                        # Build client options for moving (exclude this folder itself)
                        _move_options = {
                            f"{cp.get('client_name','?')} — score {cp.get('overall_score','?')}": ck
                            for ck, cp in client_profiles.items()
                            if ck != UNASSOCIATED_CLIENT_KEY
                            and not cp.get("_is_unassociated")
                        }
                        for _uvid in sorted(_u_props.keys(), reverse=True):
                            _uprop = _u_props[_uvid]
                            _u1, _u2, _u3 = st.columns([3, 3, 2])
                            _u1.markdown(
                                f"**{_uvid}**<br>"
                                f"<span style='font-size:0.78rem;color:#6b7280'>"
                                f"Created {_uprop.get('created_at','—')}</span>",
                                unsafe_allow_html=True,
                            )
                            _tiers_preview = list(_uprop.get("tiers", {}).keys())
                            _u2.markdown(
                                f"<span style='font-size:0.82rem;color:#374151'>"
                                f"{len(_tiers_preview)} tiers: "
                                + ", ".join(t.title() for t in _tiers_preview)
                                + "</span>",
                                unsafe_allow_html=True,
                            )
                            with _u3:
                                if _move_options:
                                    _move_pick = st.selectbox(
                                        "Move to client",
                                        options=["— select —"] + list(_move_options.keys()),
                                        key=f"move_pick_{_uvid}",
                                        label_visibility="collapsed",
                                    )
                                    _mc1, _mc2 = st.columns(2)
                                    if _mc1.button(
                                        "🔀 Move", key=f"move_btn_{_uvid}",
                                        disabled=(_move_pick == "— select —"),
                                        use_container_width=True,
                                    ):
                                        _target_ck = _move_options[_move_pick]
                                        # Move the proposal
                                        _all = load_all_proposals()
                                        _all.setdefault(_target_ck, {})[_uvid] = _uprop
                                        # Update embedded client fields to match new owner
                                        _target_p = client_profiles.get(_target_ck, {})
                                        _all[_target_ck][_uvid]["client_key"]   = _target_ck
                                        _all[_target_ck][_uvid]["client_name"]  = _target_p.get("client_name","")
                                        _all[_target_ck][_uvid]["client_score"] = _target_p.get("overall_score","")
                                        # Remove from unassociated
                                        if _uvid in _all.get(profile_key, {}):
                                            del _all[profile_key][_uvid]
                                        _save_json_safe(CLIENT_PROPOSALS_FILE, _all)
                                        st.success(
                                            f"✅ Proposal {_uvid} moved to "
                                            f"{_target_p.get('client_name', _target_ck)}."
                                        )
                                        st.rerun()
                                    if _mc2.button(
                                        "🗑", key=f"del_un_{_uvid}",
                                        use_container_width=True,
                                        help="Delete this unassociated proposal permanently."
                                    ):
                                        delete_proposal(profile_key, _uvid)
                                        st.rerun()
                                else:
                                    st.caption("_No clients to move to yet._")

                            # Build PDF Report — unified inline expander
                            # (replaces the prior one-click button that used
                            # hardcoded section selections; the advisor now
                            # picks sections in the same UI used for saved
                            # client proposals).
                            _render_pdf_builder(
                                proposal=_uprop,
                                client_profile=None,  # unassociated → stub
                                key_prefix=f"unassoc_{_uvid}",
                                download_filename=f"unassociated_proposal_{_uvid}.pdf",
                            )
                            st.markdown("---")
                # Skip the normal client-record rendering for the unassociated folder
                continue

            # ── REAL CLIENT — existing rendering below ─────────────
            score = p.get("overall_score", 0)
            label_r = p.get("risk_label","—")
            sc = "#16a34a" if score<=15 else "#65a30d" if score<=30 else "#d97706" if score<=45 else "#ea580c" if score<=60 else "#dc2626"

            # Outer client expander — defaults closed. (Previously this
            # was kept open when the legacy "Report Builder" panel was
            # active for one of the client's proposals; that two-step
            # flow has been replaced by per-proposal inline expanders,
            # so the keep-open logic is no longer needed.)
            with st.expander(
                f"👤 {p.get('client_name', profile_key)}  ·  Score {score}  ·  {p.get('date', p.get('completed_at','')[:10])}",
                expanded=False,
            ):
                ci1,ci2,ci3,ci4,ci5 = st.columns(5)
                ci1.markdown(f"**Name**  \n{p.get('client_name','—')}")
                ci2.markdown(f"**Email**  \n{p.get('client_email','—') or '—'}")
                ci3.markdown(f"**Phone**  \n{p.get('client_phone','—') or '—'}")
                _age_val = p.get("client_age") or p.get("age") or "—"
                ci4.markdown(f"**Age**  \n{_age_val}")
                ci5.markdown(f"**Advisor Ref**  \n{p.get('advisor','—') or '—'}")
                st.markdown("---")
                rs1,rs2,rs3,rs4 = st.columns(4)
                for col, val, lbl in [
                    (rs1, str(score), "Overall Score"),
                    (rs2, str(p.get('tolerance_score','—')), "Tolerance"),
                    (rs3, str(p.get('capacity_score','—')), "Capacity"),
                    (rs4, label_r, "Risk Profile"),
                ]:
                    col.markdown(f"""<div style='text-align:center;background:#f9fafb;
                        border:1px solid #e5e7eb;border-radius:10px;padding:12px 8px'>
                        <div style='font-size:1.4rem;font-weight:700;color:{sc}'>{val}</div>
                        <div style='font-size:0.65rem;color:#9ca3af;text-transform:uppercase;
                                    font-weight:600'>{lbl}</div></div>""", unsafe_allow_html=True)

                # ── Client priorities (from Goals & Priorities multi-select) ──
                _priorities = p.get("priorities", []) or []
                if _priorities:
                    st.markdown("---")
                    st.markdown("**Client Priorities**")
                    _PRIORITY_LABELS = {
                        "capital_preservation": ("🛡️ Capital Preservation", "#dcfce7", "#166534"),
                        "insurance_planning":   ("🔒 Insurance Planning",   "#e0e7ff", "#3730a3"),
                        "income_generation":    ("💵 Income Generation",    "#fef3c7", "#92400e"),
                        "capital_appreciation": ("📈 Capital Appreciation", "#dbeafe", "#1e40af"),
                        "diversification":      ("🌐 Diversification",      "#f3e8ff", "#6b21a8"),
                        "tax_efficiency":       ("🧾 Tax Efficiency",       "#fce7f3", "#9f1239"),
                        "liquidity":            ("💧 Liquidity",            "#cffafe", "#155e75"),
                        "social_impact":        ("🌱 Social / Impact",      "#d1fae5", "#065f46"),
                        "legacy_planning":      ("🏛 Legacy Planning",       "#fef9c3", "#854d0e"),
                    }
                    chips_html = ""
                    for pk in _priorities:
                        label, bg, fg = _PRIORITY_LABELS.get(pk, (pk, "#f3f4f6", "#374151"))
                        chips_html += (
                            f"<span style='display:inline-block;background:{bg};color:{fg};"
                            f"padding:5px 11px;border-radius:16px;font-size:0.78rem;"
                            f"font-weight:600;margin-right:6px;margin-bottom:6px'>{label}</span>"
                        )
                    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

                if p.get("questions"):
                    st.markdown("---")
                    st.markdown("**Assessment Responses**")
                    qa = [{"Section":q.get("section",""),"Category":q.get("category",""),
                           "Question":q.get("text","")[:80],"Answer":q.get("answer","—")}
                          for q in p["questions"]]
                    st.dataframe(pd.DataFrame(qa), use_container_width=True, hide_index=True)

                # ── Client's personal holdings & watchlist ─────
                _email_key   = (p.get("client_email") or p.get("client_name") or "").strip().lower()
                _client_hold = all_client_holdings.get(_email_key, {})
                _client_wl   = all_client_watchlist.get(_email_key, [])

                st.markdown("---")
                cw1, cw2 = st.columns(2)
                with cw1:
                    st.markdown("**Client Holdings**")
                    if _client_hold:
                        _hrows = [{"Ticker": t,
                                   "Shares":   f"{float(h.get('shares', 0)):,.2f}",
                                   "Avg Cost": f"${float(h.get('avg_cost', 0)):,.2f}"}
                                  for t, h in _client_hold.items()]
                        st.dataframe(pd.DataFrame(_hrows),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.caption("_No holdings entered by client._")
                with cw2:
                    st.markdown("**Client Watchlist**")
                    if _client_wl:
                        st.write(", ".join(_client_wl))
                    else:
                        st.caption("_No tickers on client's watchlist._")

                # ── SAVED PROPOSALS ─────────────────────────────────
                _all_proposals = load_all_proposals()
                _client_proposals = _all_proposals.get(profile_key, {})

                st.markdown("---")
                st.markdown(f"**📋 Saved Proposals ({len(_client_proposals)})**")
                if not _client_proposals:
                    st.caption(
                        "_No proposals yet. Use the Optimizer tab → "
                        "Client Proposals section to generate and save portfolio options._"
                    )
                else:
                    # Sort newest first
                    _sorted_v = sorted(_client_proposals.items(),
                                       key=lambda kv: kv[1].get("created_at", ""),
                                       reverse=True)
                    for _vid, _prop in _sorted_v:
                        with st.container():
                            pv_col1, pv_col2 = st.columns([4, 1])
                            pv_col1.markdown(
                                f"**{_vid}** · {_prop.get('created_at','—')}"
                                + (f" · _{_prop['advisor_notes'][:60]}…_"
                                   if _prop.get('advisor_notes') else "")
                            )
                            if pv_col2.button("🗑️ Delete", key=f"del_prop_{profile_key}_{_vid}",
                                              use_container_width=True):
                                delete_proposal(profile_key, _vid)
                                st.rerun()

                            # ── Pie charts for the tiers (3 or 4 depending on vintage) ──
                            # Order matches the new tier-tab strip:
                            # Option 1 = balanced (proposed), Option 2 = conservative,
                            # Option 3 = aggressive. Internal keys unchanged.
                            _tier_cfg_full = [
                                ("balanced",     "Option 1 (proposed)"),
                                ("conservative", "Option 2"),
                                ("aggressive",   "Option 3"),
                                ("alternate",    "Broad-ETF Alternate"),
                            ]
                            _present = [(k, l) for k, l in _tier_cfg_full
                                        if k in _prop.get("tiers", {})]
                            if _present:
                                _pie_cols = st.columns(len(_present))
                                for _pc, (_ptk, _ptlbl) in zip(_pie_cols, _present):
                                    _tp = _prop.get("tiers", {}).get(_ptk, {})
                                    _ptks = _tp.get("tickers", [])
                                    _pws  = _tp.get("weights", [])
                                    if _ptks and _pws and sum(float(w) for w in _pws) > 0:
                                        # New unified scoring: per-holding → weighted avg → correlation adj
                                        _sh_scores = []
                                        _sh_vols   = []
                                        for _tt in _ptks:
                                            _rr = security_risk_score(_tt)
                                            if _rr:
                                                _sh_scores.append(_rr["score"])
                                                _sh_vols.append(_rr.get("ann_vol", 0.15))
                                            else:
                                                _sh_scores.append(50)
                                                _sh_vols.append(0.15)
                                        # Compute real portfolio vol so correlation discount
                                        # applies — matches the PCM Risk Score row exactly.
                                        # Cached per (tickers, weights) so saved-proposal
                                        # gauges don't re-fetch prices repeatedly.
                                        _saved_pvol = _cached_portfolio_vol(
                                            tuple(_ptks),
                                            tuple(round(float(w), 4) for w in _pws),
                                        )
                                        _saved_score = compute_portfolio_risk_score(
                                            _ptks, _pws,
                                            holding_scores=_sh_scores,
                                            holding_vols=_sh_vols,
                                            portfolio_vol=_saved_pvol,
                                        )
                                        with _pc:
                                            # Compact gauge above the pie
                                            st.plotly_chart(
                                                make_risk_gauge(_saved_score, height=140),
                                                use_container_width=True,
                                                config={"displayModeBar": False},
                                                key=f"saved_prop_gauge_{profile_key}_{_vid}_{_ptk}",
                                            )
                                            st.plotly_chart(
                                                make_allocation_pie(
                                                    _ptks, _pws,
                                                    title=_ptlbl,
                                                    height=340,
                                                ),
                                                use_container_width=True,
                                                config={"displayModeBar": False},
                                                key=f"saved_prop_pie_{profile_key}_{_vid}_{_ptk}",
                                            )

                            # Compact view of the tiers (summary cards, 3 or 4).
                            # Labels are kept short here because each tile is
                            # ≤25% of row width. Order matches the new tier-tab
                            # strip: Proposed=Option 1, Conservative=Option 2,
                            # Aggressive=Option 3.
                            _tile_cfg_full = [
                                ("balanced",     "⚖️ Option 1 (proposed)"),
                                ("conservative", "🛡️ Option 2"),
                                ("aggressive",   "🚀 Option 3"),
                                ("alternate",    "🧭 Broad-ETF Alt."),
                            ]
                            _tile_present = [(k, l) for k, l in _tile_cfg_full
                                             if k in _prop.get("tiers", {})]
                            if _tile_present:
                                _tile_cols = st.columns(len(_tile_present))
                                for _tc, (_tk, _tlabel) in zip(_tile_cols, _tile_present):
                                    tprop = _prop.get("tiers", {}).get(_tk, {})
                                    _tc.markdown(
                                        f"<div style='background:#f9fafb;border:1px solid #e5e7eb;"
                                        f"border-radius:10px;padding:12px'>"
                                        f"<div style='font-weight:600;font-size:0.78rem;"
                                        f"color:#6b7280;margin-bottom:6px'>{_tlabel} · "
                                        f"target {tprop.get('target_score','?')}</div>"
                                        f"<div style='font-size:0.72rem;color:#111827'>"
                                        f"{tprop.get('equity_pct',0):.0f}% eq / "
                                        f"{tprop.get('bond_pct',0):.0f}% bd / "
                                        f"{tprop.get('cash_pct',0):.0f}% cash</div>"
                                        f"<div style='font-size:0.7rem;color:#6b7280;margin-top:4px'>"
                                        f"{', '.join(tprop.get('tickers', [])[:5])}"
                                        + ("…" if len(tprop.get('tickers', [])) > 5 else "")
                                        + "</div></div>",
                                        unsafe_allow_html=True,
                                    )

                            # Action row for this version (2 columns now;
                            # the prior 3rd column "Build PDF Report" button
                            # was replaced by the inline expander below).
                            ra1, ra3 = st.columns(2)
                            _tok = make_proposal_token(profile_key, _vid)
                            _share_url = f"?view=proposal&token={_tok}"
                            ra1.markdown(
                                f"<a href='{_share_url}' target='_blank' "
                                f"style='display:inline-block;padding:8px 14px;background:#0E5C5E;"
                                f"color:#fff;border-radius:10px;font-weight:600;font-size:0.85rem;"
                                f"text-decoration:none;text-align:center;width:100%'>"
                                f"🔗 Open Client View</a>",
                                unsafe_allow_html=True,
                            )
                            if ra3.button("📋 Copy Link",
                                          key=f"cp_{profile_key}_{_vid}",
                                          use_container_width=True):
                                st.code(_share_url, language=None)

                            # Build PDF Report — unified inline expander.
                            # Replaces the prior 2-step "click button → open
                            # builder panel → pick sections → download" flow
                            # with a single expander that shows checkboxes
                            # immediately and downloads on click. Same code
                            # path as the unassociated-proposal PDF flow.
                            _safe_name = (p.get("client_name", "Client") or "Client").replace(" ", "_")
                            _render_pdf_builder(
                                proposal=_prop,
                                client_profile=p,
                                key_prefix=f"saved_{profile_key}_{_vid}",
                                download_filename=f"{_safe_name}_{_vid}.pdf",
                            )

                st.markdown("---")
                # Per-version PDF generation lives on each proposal row above
                # ("📄 Build PDF Report"). The bottom action bar only carries
                # the two record-level actions: open in Optimizer and delete.
                if not _client_proposals:
                    st.caption(
                        "_No saved proposals yet — switch to the Optimizer tab "
                        "(section 6, Client Proposals) to generate one for this client. "
                        "Saved proposals appear above each with their own "
                        "**📄 Build PDF Report** button._"
                    )
                ab1, ab2 = st.columns(2)
                if ab1.button("📊 Run in Optimizer", key=f"open_opt_{profile_key}",
                              use_container_width=True):
                    st.info(f"Switch to Optimizer tab and scroll to section 6 (Client Proposals) to generate portfolios for {p.get('client_name','this client')}.")
                if ab2.button("🗑️ Delete Record", key=f"del_rec_{profile_key}",
                              use_container_width=True):
                    del client_profiles[profile_key]
                    # Route through data_store so the delete syncs to the
                    # shared GitHub repo, not just local ephemeral disk.
                    _shared_save_json(PROFILES_FILE, client_profiles)
                    st.success("Deleted.")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS (firm branding, advisor info, logo + photo)
# ═══════════════════════════════════════════════════════════════
with main_tab5:
    st.session_state["optimizer_tab_active"] = False
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>Firm Branding &amp; Advisor Info</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "One-time firm setup. These details appear on every generated "
        "proposal PDF and on the client portal's Advisor tab — update "
        "once and changes flow through both surfaces automatically."
    )
    st.markdown("---")

    # ── FIRM BRANDING ─────────────────────────────────────────────
    # One-time setup: firm name, advisor name/title, and the two images
    # (firm logo + advisor photo) that get embedded on every generated
    # proposal PDF. Stored globally so all client PDFs share the same
    # branding. Pass B picks these up in the PDF builder.
    _fs = load_firm_settings()

    fb_l, fb_r = st.columns(2)
    with fb_l:
        firm_name = st.text_input(
            "Firm name",
            value=_fs.get("firm_name", ""),
            placeholder="Foresight Wealth Partners",
            key="fb_firm_name",
        )
        advisor_name = st.text_input(
            "Advisor name",
            value=_fs.get("advisor_name", ""),
            placeholder="Sarah Whitfield, CFP®",
            key="fb_advisor_name",
        )
        advisor_title = st.text_input(
            "Advisor title",
            value=_fs.get("advisor_title", ""),
            placeholder="Senior Financial Advisor",
            key="fb_advisor_title",
        )
    with fb_r:
        advisor_email = st.text_input(
            "Advisor email",
            value=_fs.get("advisor_email", ""),
            placeholder="sarah@foresightwealth.com",
            key="fb_advisor_email",
        )
        advisor_phone = st.text_input(
            "Advisor phone",
            value=_fs.get("advisor_phone", ""),
            placeholder="(612) 555-0142",
            key="fb_advisor_phone",
        )
        firm_website = st.text_input(
            "Firm website",
            value=_fs.get("firm_website", ""),
            placeholder="www.foresightwealth.com",
            key="fb_firm_website",
        )

    # ── Compliance: default advisory fee ────────────────────
    # The SEC Marketing Rule requires fee disclosure on any advertised
    # performance. The fee set here populates the "Net of Fees" disclosure
    # on every generated proposal PDF. Per-client overrides can be set on
    # individual client profiles (rare but supported); this is the firm-
    # wide default.
    st.caption("**Compliance — default advisory fee for proposal disclosures:**")
    default_advisory_fee_pct = st.number_input(
        "Default advisory fee (% per year)",
        min_value=0.0, max_value=10.0, step=0.05, format="%.2f",
        value=float(_fs.get("default_advisory_fee_pct", 1.00)),
        help=("Annual AUM fee that appears in the SEC Marketing Rule "
              "disclosure on every generated proposal PDF. Should match "
              "the fee schedule on your firm's Form ADV. Set to 0 if your "
              "firm does not charge an AUM fee. A per-client override "
              "can be set on individual client profiles if needed."),
        key="fb_default_advisory_fee_pct",
    )

    # ── Client-portal-only fields ─────────────────────────
    # These don't appear on the PDF (the PDF stays compact); they show
    # up on the client portal's Advisor tab so clients see the office
    # address and a short bio of who they're working with.
    st.caption("**Shown on the client portal Advisor tab:**")
    firm_address = st.text_input(
        "Firm address",
        value=_fs.get("firm_address", ""),
        placeholder="200 South Sixth Street, Suite 1200, Minneapolis, MN 55402",
        key="fb_firm_address",
    )
    advisor_bio = st.text_area(
        "Advisor bio (short paragraph)",
        value=_fs.get("advisor_bio", ""),
        placeholder=("Sarah has spent fifteen years helping families plan "
                     "for retirement, education, and legacy goals. She's a "
                     "Certified Financial Planner™ and a fiduciary."),
        key="fb_advisor_bio",
        height=80,
    )

    st.markdown("---")
    img_l, img_r = st.columns(2)
    with img_l:
        st.markdown("**Firm logo**")
        if os.path.exists(FIRM_LOGO_PATH):
            st.image(FIRM_LOGO_PATH, width=140)
            if st.button("Remove logo", key="fb_logo_remove"):
                try:
                    os.remove(FIRM_LOGO_PATH)
                    st.rerun()
                except OSError as _e:
                    st.error(f"Couldn't remove: {_e}")
        else:
            st.caption("_No logo uploaded._")
        _logo_up = st.file_uploader(
            "Upload firm logo",
            type=["png", "jpg", "jpeg"],
            key="fb_logo_uploader",
            label_visibility="collapsed",
        )
        if _logo_up is not None:
            with open(FIRM_LOGO_PATH, "wb") as _f:
                _f.write(_logo_up.getbuffer())
            st.success("Logo saved.")
            st.rerun()

    with img_r:
        st.markdown("**Advisor photo**")
        if os.path.exists(ADVISOR_PHOTO_PATH):
            st.image(ADVISOR_PHOTO_PATH, width=140)
            if st.button("Remove photo", key="fb_photo_remove"):
                try:
                    os.remove(ADVISOR_PHOTO_PATH)
                    st.rerun()
                except OSError as _e:
                    st.error(f"Couldn't remove: {_e}")
        else:
            st.caption("_No photo uploaded._")
        _photo_up = st.file_uploader(
            "Upload advisor photo",
            type=["png", "jpg", "jpeg"],
            key="fb_photo_uploader",
            label_visibility="collapsed",
        )
        if _photo_up is not None:
            with open(ADVISOR_PHOTO_PATH, "wb") as _f:
                _f.write(_photo_up.getbuffer())
            st.success("Photo saved.")
            st.rerun()

    st.markdown("---")
    if st.button("💾 Save firm details", type="primary", key="fb_save"):
        _payload = {
            "firm_name":     firm_name.strip(),
            "advisor_name":  advisor_name.strip(),
            "advisor_title": advisor_title.strip(),
            "advisor_email": advisor_email.strip(),
            "advisor_phone": advisor_phone.strip(),
            "firm_website":  firm_website.strip(),
            "firm_address":  firm_address.strip(),
            "advisor_bio":   advisor_bio.strip(),
            # SEC Marketing Rule disclosure — appears on every proposal PDF.
            # Rounded to 2 decimal places to match the disclosure formatter.
            "default_advisory_fee_pct": round(float(default_advisory_fee_pct), 2),
        }
        # Try the write, surface any error visibly, then re-read the
        # file to verify the data actually landed on disk. If save was
        # silently failing before this, the user would see a dead
        # button — now they get either a green confirmation that
        # includes the file path + field count, or a red error.
        try:
            save_firm_settings(_payload)
        except Exception as _save_err:
            import traceback as _tb
            st.error(f"Couldn't save firm settings: {_save_err}")
            with st.expander("Debug info"):
                st.code(_tb.format_exc())
                st.markdown(
                    f"**Tried to write to:** `{FIRM_SETTINGS_FILE}`  \n"
                    f"**App directory:** `{_APP_DIR}`"
                )
        else:
            # Verify by re-reading
            _verify = load_firm_settings()
            _populated = sum(
                1 for k, v in _payload.items()
                if isinstance(v, str) and v.strip() and _verify.get(k) == v
            )
            _total_filled = sum(
                1 for v in _payload.values()
                if isinstance(v, str) and v.strip()
            )
            if _populated == _total_filled and _total_filled > 0:
                st.success(
                    f"✅ Firm branding saved — {_populated} field(s) written to disk. "
                    "Updates apply to new PDFs and the client portal immediately. "
                    "Refresh the portal page to see changes."
                )
                st.caption(f"Saved to: `{FIRM_SETTINGS_FILE}`")
            elif _total_filled == 0:
                st.warning(
                    "All fields are blank — nothing was saved. "
                    "Fill in at least one field (firm name, advisor name, etc.) "
                    "and try again."
                )
            else:
                st.warning(
                    f"Save partially succeeded: {_populated} of {_total_filled} "
                    f"non-empty fields verified on re-read. The file may be "
                    f"locked or have permission issues."
                )
                st.caption(f"Wrote to: `{FIRM_SETTINGS_FILE}`")

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>⚙️ Optimization Settings</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Estimators and models used by the analyzer. These were previously in "
        "Section 3 of the Analyzer; they live here now to keep the analysis "
        "flow uncluttered. Defaults are sensible — most advisors won't need "
        "to touch these."
    )

    _opt_l, _opt_r = st.columns(2)
    with _opt_l:
        st.selectbox(
            "Covariance Estimator",
            ["Ledoit-Wolf", "Gerber", "Denoised", "Empirical"],
            index=["Ledoit-Wolf","Gerber","Denoised","Empirical"].index(
                st.session_state.get("opt_cov_estimator", "Ledoit-Wolf")
            ),
            key="opt_cov_estimator",
            help="Ledoit-Wolf reduces estimation error. Gerber is robust to outliers. "
                 "Denoised removes noise.",
        )
    with _opt_r:
        st.selectbox(
            "Returns Estimator",
            ["Shrunk", "EWM", "Empirical"],
            index=["Shrunk","EWM","Empirical"].index(
                st.session_state.get("opt_mu_estimator", "Shrunk")
            ),
            key="opt_mu_estimator",
            help="Shrunk reduces overfitting. EWM weights recent data more.",
        )

    st.markdown("**Strategies to include in analysis:**")
    _opt_s1, _opt_s2, _opt_s3, _opt_s4 = st.columns(4)
    with _opt_s1:
        st.checkbox("Include HRP",
                    value=st.session_state.get("opt_use_hrp", True),
                    key="opt_use_hrp",
                    help="Hierarchical Risk Parity")
    with _opt_s2:
        st.checkbox("Include NCO",
                    value=st.session_state.get("opt_use_nco", True),
                    key="opt_use_nco",
                    help="Nested Clusters Optimization")
    with _opt_s3:
        st.checkbox("Include Max Diversification",
                    value=st.session_state.get("opt_use_maxdiv", True),
                    key="opt_use_maxdiv")
    with _opt_s4:
        st.checkbox("Walk-Forward CV",
                    value=st.session_state.get("opt_use_wf", False),
                    key="opt_use_wf",
                    help="More realistic but slower backtesting")

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>🔑 Market Data API Keys</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Configure the Financial Modeling Prep and Alpha Vantage keys "
        "used for expense ratios, fund profiles, and price-data fallbacks. "
        "Status, manual entry, and live diagnostics for each provider."
    )

    # ── FMP API STATUS ──────────────────────────────────────────
    # Diagnostic block: shows whether the Financial Modeling Prep API key
    # is loading, lets the advisor paste a key into the session as a
    # fallback if secrets.toml isn't being read for some reason, and runs
    # a one-click test call to confirm the API is reachable.
    st.divider()
    with st.expander("🔑 FMP API Status", expanded=False):
        # 1. Detect where the key is coming from
        _key_from_secrets = None
        try:
            _key_from_secrets = st.secrets.get("FMP_API_KEY")
        except Exception:
            _key_from_secrets = None
        _key_from_env     = os.environ.get("FMP_API_KEY")
        _key_from_session = st.session_state.get("fmp_api_key_manual")

        _active_key  = _key_from_session or _key_from_secrets or _key_from_env
        _active_src  = ("manual entry" if _key_from_session else
                        "secrets.toml"  if _key_from_secrets else
                        "env variable"  if _key_from_env else
                        "(none)")

        if _active_key:
            _masked = _active_key[:4] + "•"*8 + _active_key[-4:] if len(_active_key) >= 8 else "•••"
            st.markdown(
                f"<div style='font-size:0.78rem;color:#cbd5e1'>"
                f"<b>Key detected:</b> <code>{_masked}</code><br/>"
                f"<b>Source:</b> {_active_src}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:#fca5a5'>"
                "<b>⚠ No FMP API key detected.</b><br/>"
                "ER lookups for non-cached funds will return —"
                "</div>", unsafe_allow_html=True,
            )

        # 2. Manual key entry (overrides secrets.toml for this session only)
        st.caption("Override or provide a key for this session:")
        _manual = st.text_input(
            "FMP API Key", type="password",
            value=_key_from_session or "",
            key="_fmp_key_input_field",
            label_visibility="collapsed",
            placeholder="paste key here",
        )
        _b1, _b2 = st.columns(2)
        if _b1.button("Save", key="_fmp_save_btn", use_container_width=True):
            if _manual.strip():
                st.session_state["fmp_api_key_manual"] = _manual.strip()
                # Clear caches so next ER lookup uses the new key
                _expense_ratio_for_ticker._session_cache = {}
                st.success("Key saved for this session.")
                st.rerun()
            else:
                st.warning("Empty key — nothing saved.")
        if _b2.button("Clear", key="_fmp_clear_btn", use_container_width=True):
            st.session_state.pop("fmp_api_key_manual", None)
            _expense_ratio_for_ticker._session_cache = {}
            st.success("Manual key cleared.")
            st.rerun()

        # 3. Live diagnostic — tests 6 FMP endpoints across data categories
        # so you can see exactly what your plan supports before integrating.
        if st.button("🧪 Run full endpoint diagnostic", key="_fmp_test_btn",
                     use_container_width=True,
                     help="Tests FMP across price, profile, quote, ratios, "
                          "news, and ETF endpoints to see what's available "
                          "on your current plan."):
            if not _active_key:
                st.error("No key configured — set one above first.")
            else:
                import requests as _req
                # Six endpoint families to test, ordered by likelihood of
                # being available on the free tier
                _diag_endpoints = [
                    ("📊 Real-time Quote",
                     "https://financialmodelingprep.com/api/v3/quote/AAPL",
                     {"apikey": _active_key},
                     "price"),
                    ("🏢 Company Profile",
                     "https://financialmodelingprep.com/api/v3/profile/AAPL",
                     {"apikey": _active_key},
                     "companyName"),
                    ("📈 Historical Prices",
                     "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL",
                     {"apikey": _active_key, "from": "2025-04-01", "to": "2025-04-25"},
                     "historical"),
                    ("📐 Financial Ratios",
                     "https://financialmodelingprep.com/api/v3/ratios/AAPL",
                     {"apikey": _active_key, "limit": "1"},
                     "priceEarningsRatio"),
                    ("📰 Stock News",
                     "https://financialmodelingprep.com/api/v3/stock_news",
                     {"apikey": _active_key, "tickers": "AAPL", "limit": "1"},
                     "title"),
                    ("💵 ETF Profile (paid?)",
                     "https://financialmodelingprep.com/api/v3/etf-info/VTI",
                     {"apikey": _active_key},
                     "expenseRatio"),
                    ("🌟 Stable ETF Info",
                     "https://financialmodelingprep.com/stable/etf-info",
                     {"symbol": "VTI", "apikey": _active_key},
                     "expenseRatio"),
                ]
                _results = []
                for _name, _url, _params, _key_field in _diag_endpoints:
                    try:
                        _r = _req.get(_url, params=_params, timeout=10)
                        _code = _r.status_code
                        if _code == 200:
                            try:
                                _data = _r.json()
                                if isinstance(_data, dict): _data = [_data]
                                if _data and isinstance(_data, list) and len(_data) > 0:
                                    _row = _data[0] if _data else {}
                                    if _key_field in _row and _row.get(_key_field) is not None:
                                        _val = str(_row.get(_key_field))[:50]
                                        _summary = f"✅ HTTP 200 — {_key_field}: {_val}"
                                        _ok = True
                                    else:
                                        _summary = f"⚠ HTTP 200 — but {_key_field} not in response"
                                        _ok = False
                                else:
                                    _summary = "⚠ HTTP 200 — empty response"
                                    _ok = False
                            except Exception:
                                _summary = "⚠ HTTP 200 — non-JSON response"
                                _ok = False
                        elif _code == 401:
                            _summary = "❌ HTTP 401 — key invalid"
                            _ok = False
                        elif _code == 403:
                            _summary = "❌ HTTP 403 — paid tier"
                            _ok = False
                        elif _code == 429:
                            _summary = "❌ HTTP 429 — rate limited"
                            _ok = False
                        else:
                            _summary = f"❌ HTTP {_code}"
                            _ok = False
                        _results.append((_name, _summary, _ok))
                    except Exception as _e:
                        _results.append((_name, f"❌ Network: {str(_e)[:50]}", False))

                # Render results table
                _passed = sum(1 for _, _, ok in _results if ok)
                _total  = len(_results)
                if _passed == _total:
                    st.success(f"✅ {_passed}/{_total} endpoints available")
                elif _passed > 0:
                    st.warning(f"⚠ {_passed}/{_total} endpoints available on your plan")
                else:
                    st.error(f"❌ {_passed}/{_total} endpoints available — key may be invalid")

                for _name, _summary, _ok in _results:
                    st.markdown(
                        f"<div style='font-size:0.78rem;font-family:monospace;"
                        f"padding:3px 0;color:{'#22c55e' if _ok else '#ef4444'}'>"
                        f"<b>{_name}</b><br>&nbsp;&nbsp;{_summary}</div>",
                        unsafe_allow_html=True,
                    )

                # Recommendations based on what worked
                _profile_works = any(name == "🏢 Company Profile" and ok
                                      for name, _, ok in _results)
                _etf_works = any("ETF" in name and ok
                                  for name, _, ok in _results)
                if _profile_works:
                    st.info(
                        "✅ Profile endpoint works — we can use FMP for "
                        "company names/sectors in the PDF (faster than yfinance)."
                    )
                if not _etf_works:
                    st.warning(
                        "❌ No ETF expense-ratio endpoint works on your plan. "
                        "ER lookups will continue using the hardcoded override "
                        "dict (~120 popular tickers). Funds outside the dict "
                        "will show '—'."
                    )

        # 4. Cache stats
        try:
            _disk_cache = _load_er_cache()
            _ses_cache = getattr(_expense_ratio_for_ticker, "_session_cache", {}) or {}
            st.caption(
                f"Cache: **{len(_disk_cache)}** tickers on disk · "
                f"**{len(_ses_cache)}** in this session"
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # ALPHA VANTAGE API STATUS
    # Used as a secondary expense-ratio source. ETF_PROFILE function on
    # the free tier returns net_expense_ratio for ETFs/MF (FMP free tier
    # doesn't). Free tier limit: 25 calls/day, 5/min — sufficient since
    # we cache results for 30 days.
    # ─────────────────────────────────────────────────────────
    with st.expander("🔑 Alpha Vantage API Status", expanded=False):
        # 1. Detect key source
        _av_from_secrets = None
        try:
            _av_from_secrets = st.secrets.get("ALPHA_VANTAGE_API_KEY")
        except Exception:
            _av_from_secrets = None
        _av_from_env     = os.environ.get("ALPHA_VANTAGE_API_KEY")
        _av_from_session = st.session_state.get("av_api_key_manual")

        _av_active_key = _av_from_session or _av_from_secrets or _av_from_env
        _av_active_src = ("manual entry" if _av_from_session else
                          "secrets.toml"  if _av_from_secrets else
                          "env variable"  if _av_from_env else
                          "(none)")

        if _av_active_key:
            _av_masked = (_av_active_key[:4] + "•"*8 + _av_active_key[-4:]
                          if len(_av_active_key) >= 8 else "•••")
            st.markdown(
                f"<div style='font-size:0.78rem;color:#cbd5e1'>"
                f"<b>Key detected:</b> <code>{_av_masked}</code><br/>"
                f"<b>Source:</b> {_av_active_src}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:#fca5a5'>"
                "<b>⚠ No Alpha Vantage API key detected.</b><br/>"
                "ER fallback for non-cached funds will only use FMP + override dict."
                "</div>", unsafe_allow_html=True,
            )

        # 2. Manual key entry
        st.caption("Override or provide a key for this session:")
        _av_manual = st.text_input(
            "Alpha Vantage API Key", type="password",
            value=_av_from_session or "",
            key="_av_key_input_field",
            label_visibility="collapsed",
            placeholder="paste key here",
        )
        _avb1, _avb2 = st.columns(2)
        if _avb1.button("Save", key="_av_save_btn", use_container_width=True):
            if _av_manual.strip():
                st.session_state["av_api_key_manual"] = _av_manual.strip()
                _expense_ratio_for_ticker._session_cache = {}
                st.success("Key saved for this session.")
                st.rerun()
            else:
                st.warning("Empty key — nothing saved.")
        if _avb2.button("Clear", key="_av_clear_btn", use_container_width=True):
            st.session_state.pop("av_api_key_manual", None)
            _expense_ratio_for_ticker._session_cache = {}
            st.success("Manual key cleared.")
            st.rerun()

        # 3. Live diagnostic — tests AV's ETF_PROFILE + OVERVIEW endpoints
        if st.button("🧪 Run Alpha Vantage diagnostic", key="_av_test_btn",
                     use_container_width=True,
                     help="Tests ETF_PROFILE (expense ratio) and OVERVIEW "
                          "(stock metadata) endpoints. Takes ~30 seconds — "
                          "free tier is 5 calls/minute so we space requests."):
            if not _av_active_key:
                st.error("No key configured — set one above first.")
            else:
                import requests as _req
                st.info("Running diagnostic — spacing calls 13 seconds apart "
                        "to stay under the 5/min free-tier limit. ~30s total.")
                _av_endpoints = [
                    ("💵 ETF_PROFILE (VTI)",
                     "https://www.alphavantage.co/query",
                     {"function": "ETF_PROFILE", "symbol": "VTI",
                      "apikey": _av_active_key},
                     "net_expense_ratio"),
                    ("🏢 OVERVIEW (AAPL)",
                     "https://www.alphavantage.co/query",
                     {"function": "OVERVIEW", "symbol": "AAPL",
                      "apikey": _av_active_key},
                     "Name"),
                    ("📊 GLOBAL_QUOTE (AAPL)",
                     "https://www.alphavantage.co/query",
                     {"function": "GLOBAL_QUOTE", "symbol": "AAPL",
                      "apikey": _av_active_key},
                     "Global Quote"),
                ]
                _av_results = []
                import time as _time
                for _i, (_name, _url, _params, _key_field) in enumerate(_av_endpoints):
                    # Rate-limit guard: 5 calls/min on free tier means ~13s
                    # spacing. Wait 13s between calls (skipping before first).
                    if _i > 0:
                        _time.sleep(13)
                    try:
                        _r = _req.get(_url, params=_params, timeout=10)
                        _code = _r.status_code
                        if _code == 200:
                            try:
                                _data = _r.json()
                                # AV returns errors inside HTTP 200. Both
                                # "Note" and "Information" usually mean
                                # rate-limit (25/day or 5/min hit), NOT
                                # actual premium-only blocking.
                                if "Error Message" in _data:
                                    _summary = f"❌ AV error: {_data['Error Message'][:60]}"
                                    _ok = False
                                elif "Note" in _data:
                                    _summary = f"❌ Rate limited (per-minute): {_data['Note'][:80]}"
                                    _ok = False
                                elif "Information" in _data:
                                    _msg = _data['Information']
                                    # Message-based detection: "premium" usually
                                    # appears in true premium-only responses
                                    if "premium" in _msg.lower():
                                        _summary = f"❌ Premium-only: {_msg[:80]}"
                                    else:
                                        _summary = f"❌ Rate limited (daily): {_msg[:80]}"
                                    _ok = False
                                elif _key_field in _data:
                                    _val = str(_data.get(_key_field))[:50]
                                    _summary = f"✅ HTTP 200 — {_key_field}: {_val}"
                                    _ok = True
                                else:
                                    _summary = f"⚠ HTTP 200 — {_key_field} not in response"
                                    _ok = False
                            except Exception as _e:
                                _summary = f"⚠ HTTP 200 — JSON parse error: {_e}"
                                _ok = False
                        elif _code == 429:
                            _summary = "❌ HTTP 429 — rate limited (25/day free)"
                            _ok = False
                        else:
                            _summary = f"❌ HTTP {_code}"
                            _ok = False
                        _av_results.append((_name, _summary, _ok))
                    except Exception as _e:
                        _av_results.append((_name, f"❌ Network: {str(_e)[:50]}", False))

                _av_passed = sum(1 for _, _, ok in _av_results if ok)
                _av_total  = len(_av_results)
                if _av_passed == _av_total:
                    st.success(f"✅ {_av_passed}/{_av_total} Alpha Vantage endpoints available")
                elif _av_passed > 0:
                    st.warning(f"⚠ {_av_passed}/{_av_total} endpoints available")
                else:
                    st.error(f"❌ {_av_passed}/{_av_total} endpoints — key may be invalid")

                for _name, _summary, _ok in _av_results:
                    st.markdown(
                        f"<div style='font-size:0.78rem;font-family:monospace;"
                        f"padding:3px 0;color:{'#22c55e' if _ok else '#ef4444'}'>"
                        f"<b>{_name}</b><br>&nbsp;&nbsp;{_summary}</div>",
                        unsafe_allow_html=True,
                    )

                _etf_profile_ok = any(name.startswith("💵") and ok
                                       for name, _, ok in _av_results)
                if _etf_profile_ok:
                    st.info(
                        "✅ ETF_PROFILE works — Alpha Vantage will be used "
                        "as a fallback for funds outside the override dict. "
                        "Cached for 30 days per ticker."
                    )

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>🔄 HubSpot CRM Sync</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Status of the HubSpot integration that pushes new client "
        "registrations into HubSpot as Contacts + follow-up Deals. "
        "Drop hubspot_sync.py and a hubspot_config.json into this app "
        "directory to enable sync; nothing breaks if either is missing."
    )
    try:
        import hubspot_sync as _hs_mod
        _hs_available = True
    except Exception as _hs_imp_err:
        _hs_available = False
        st.info(
            "`hubspot_sync.py` not found in this directory — HubSpot "
            "sync is disabled. Drop the module file alongside `app.py` "
            "to enable."
        )

    if _hs_available:
        # Status row
        _hs_configured = _hs_mod.is_configured()
        _hs_pending    = _hs_mod.pending_count()
        _hs_deadletter = _hs_mod.get_deadletter()

        _s1, _s2, _s3 = st.columns(3)
        with _s1:
            if _hs_configured:
                st.success("✅ Configured")
            else:
                st.warning("⚠ No token")
                st.caption("Set HUBSPOT_TOKEN env var or write a "
                           "`hubspot_config.json` file with `{\"token\": "
                           "\"pat-na1-...\"}` next to `app.py`.")
        with _s2:
            st.metric("Queued", _hs_pending,
                      help="Pending sync attempts waiting on the worker. "
                           "Should drop to 0 within 10s of a successful "
                           "network call.")
        with _s3:
            _dl_count = len(_hs_deadletter)
            st.metric("Failed", _dl_count,
                      help="Permanently failed syncs (after 5 retries). "
                           "Inspect below — usually indicates a bad email "
                           "format, missing required HubSpot field, or "
                           "invalid token.")

        # Deadletter contents
        if _hs_deadletter:
            with st.expander(f"❌ View {_dl_count} failed sync(s)", expanded=False):
                for _i, _entry in enumerate(_hs_deadletter, 1):
                    _email = (_entry.get("contact_props") or {}).get("email", "(unknown)")
                    _err   = _entry.get("_error") or _entry.get("_error_deal") or "(no error message)"
                    _when  = _entry.get("_failed_at", "")
                    st.markdown(
                        f"**{_i}. `{_email}`** — failed at `{_when}`"
                    )
                    st.code(str(_err), language="text")
                if st.button("🗑 Clear deadletter", key="hs_clear_dl"):
                    _hs_mod.clear_deadletter()
                    st.success("Deadletter cleared.")
                    st.rerun()

        # Manual init button (worker normally starts on first sync_contact
        # call but exposing it here is useful for confirming the thread
        # is alive and draining the queue).
        if _hs_configured and st.button(
            "🚀 Start/restart sync worker", key="hs_init_worker",
            help="Idempotent — safe to click anytime."):
            _hs_mod.init()
            st.success("Worker thread running. Check Queued count to "
                       "confirm it drains.")


# ═══════════════════════════════════════════════════════════════
# TAB 6 — FEE DRAG ANALYZER
# ═══════════════════════════════════════════════════════════════
# Educational/comparative tool that visualizes the long-term impact of
# expense ratio differences between two hypothetical portfolios. Useful
# in client meetings to demonstrate why a 25-50bps fee difference
# materially affects outcomes over a 10-year horizon. Independent of
# the main analysis pipeline — doesn't require a client portfolio loaded.
#
# Math model: at each compounding period, portfolio value is multiplied
# by (1 + gross_return / N) and then by (1 - expense_ratio / N) where
# N is the number of periods per year (12 / 4 / 2 / 1). This mirrors
# how funds typically accrue expenses by prorating the annual rate.
# Higher deduction frequency → marginally less drag because expenses
# come out of a slightly smaller balance each period; the difference
# is in the low single basis points per year and is shown but not
# overemphasized — the dominant effect is the expense ratio itself.

with main_tab6:
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>Fee Drag Analyzer</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Compare the long-term impact of expense ratios between two "
        "hypothetical portfolios. Configure each portfolio's assumed "
        "annual return, expense ratio, and how often expenses are "
        "deducted — the chart and table below show how the gap "
        "compounds over 10 years."
    )

    # ── Default values (sensible starting point for first render) ──
    _fda_defaults = {
        "fda_p1_return":   8.0,
        "fda_p1_er":       0.05,
        "fda_p1_freq":     "Quarterly",
        "fda_p2_return":   8.0,
        "fda_p2_er":       0.75,
        "fda_p2_freq":     "Quarterly",
        "fda_initial":     100000.0,
    }
    for _k, _v in _fda_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Initial investment (shared input — both portfolios start equal) ──
    _fda_top_l, _fda_top_r = st.columns([1, 2])
    with _fda_top_l:
        _initial = st.number_input(
            "Initial investment ($)",
            min_value=1000.0, max_value=100_000_000.0,
            value=float(st.session_state["fda_initial"]),
            step=1000.0, format="%.0f",
            key="fda_initial",
            help="Both portfolios start at this amount.",
        )

    st.markdown("---")

    # ── Two portfolio config columns ──
    _fda_c1, _fda_c2 = st.columns(2)

    _FREQ_PERIODS = {
        "Monthly": 12, "Quarterly": 4, "Semi-Annually": 2, "Annually": 1,
    }
    _freq_options = list(_FREQ_PERIODS.keys())

    def _portfolio_input_block(col, slot, default_label):
        """Render the input block for one portfolio. `slot` is 'p1' or 'p2'."""
        with col:
            st.markdown(f"#### Portfolio {slot[-1]}")
            _name = st.text_input(
                "Label",
                value=default_label,
                key=f"fda_{slot}_label",
                help="Name shown in the chart legend and table column.",
            )
            _ret = st.number_input(
                "Assumed annual return (%)",
                min_value=-20.0, max_value=50.0,
                value=float(st.session_state[f"fda_{slot}_return"]),
                step=0.25, format="%.2f",
                key=f"fda_{slot}_return",
                help="Gross annual return before expenses. Compounded "
                     "at the deduction frequency.",
            )
            _er = st.number_input(
                "Expense ratio (%)",
                min_value=0.0, max_value=5.0,
                value=float(st.session_state[f"fda_{slot}_er"]),
                step=0.01, format="%.2f",
                key=f"fda_{slot}_er",
                help="Annual expense ratio as a percent (e.g. 0.50 for 50bps).",
            )
            _freq = st.selectbox(
                "Expenses deducted",
                options=_freq_options,
                index=_freq_options.index(st.session_state[f"fda_{slot}_freq"]),
                key=f"fda_{slot}_freq",
                help="How often the fund accrues its expense ratio. Most "
                     "real-world ETFs/MFs accrue daily, but quarterly is "
                     "the common pedagogical proxy.",
            )
            return {
                "name":   _name,
                "return": float(_ret) / 100.0,    # decimal
                "er":     float(_er)  / 100.0,    # decimal
                "freq":   _freq,
                "n":      _FREQ_PERIODS[_freq],
            }

    p1 = _portfolio_input_block(_fda_c1, "p1", "Portfolio 1")
    p2 = _portfolio_input_block(_fda_c2, "p2", "Portfolio 2")

    # ── Run the simulation ──
    # For each portfolio, simulate balance at each YEAR boundary using
    # the per-period growth factor (1 + r/n)*(1 - er/n). At each year
    # mark we apply n compounding periods.
    YEARS = 10
    years = list(range(0, YEARS + 1))   # 0 through 10 inclusive

    def _simulate(initial, gross_return, er, n_per_year, years_max):
        """Return list of balances at each year boundary (length years_max+1).

        Per-period factor: (1 + r/n) for growth, then (1 - er/n) for fees.
        Result at year y is initial * factor^(n*y).
        """
        per_period = (1.0 + gross_return / n_per_year) * (1.0 - er / n_per_year)
        balances = []
        for y in range(years_max + 1):
            periods_so_far = n_per_year * y
            balances.append(initial * (per_period ** periods_so_far))
        return balances

    p1_bal = _simulate(_initial, p1["return"], p1["er"], p1["n"], YEARS)
    p2_bal = _simulate(_initial, p2["return"], p2["er"], p2["n"], YEARS)

    # Net annualized returns (geometric) for the metric tiles
    def _net_annualized(initial, final_bal, years_n):
        if initial <= 0 or final_bal <= 0 or years_n <= 0:
            return 0.0
        return (final_bal / initial) ** (1.0 / years_n) - 1.0

    p1_net_ann = _net_annualized(_initial, p1_bal[-1], YEARS)
    p2_net_ann = _net_annualized(_initial, p2_bal[-1], YEARS)

    # ── Summary metric tiles ──
    st.markdown("---")
    _m1, _m2, _m3, _m4 = st.columns(4)
    with _m1:
        st.metric(
            f"{p1['name']} — Final Balance",
            f"${p1_bal[-1]:,.0f}",
            delta=f"{p1_net_ann*100:.2f}% net/yr",
            delta_color="off",
        )
    with _m2:
        st.metric(
            f"{p2['name']} — Final Balance",
            f"${p2_bal[-1]:,.0f}",
            delta=f"{p2_net_ann*100:.2f}% net/yr",
            delta_color="off",
        )
    with _m3:
        _gap = p1_bal[-1] - p2_bal[-1]
        st.metric(
            "Difference at Year 10",
            f"${abs(_gap):,.0f}",
            delta=f"{p1['name']} ahead" if _gap > 0
                  else (f"{p2['name']} ahead" if _gap < 0 else "tied"),
            delta_color="off",
        )
    with _m4:
        # Total fees paid — back-calculate as gross balance minus net balance
        def _gross_balance(initial, gross_return, n_per_year, years_n):
            per_period = 1.0 + gross_return / n_per_year
            return initial * (per_period ** (n_per_year * years_n))
        p1_gross = _gross_balance(_initial, p1["return"], p1["n"], YEARS)
        p2_gross = _gross_balance(_initial, p2["return"], p2["n"], YEARS)
        p1_fees = p1_gross - p1_bal[-1]
        p2_fees = p2_gross - p2_bal[-1]
        st.metric(
            "Total Fees Paid (combined)",
            f"${p1_fees + p2_fees:,.0f}",
            delta=f"P1: ${p1_fees:,.0f} · P2: ${p2_fees:,.0f}",
            delta_color="off",
        )

    # ── Line chart: balances over time ──
    st.markdown(
        "<h4 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:8px 0 6px 0'>10-Year Growth Comparison</h4>",
        unsafe_allow_html=True,
    )
    import plotly.graph_objects as _go

    _fig_fda = _go.Figure()
    _fig_fda.add_trace(_go.Scatter(
        x=years, y=p1_bal,
        mode="lines+markers",
        name=p1["name"],
        line=dict(color="#0E5C5E", width=2.5),  # brand teal
        marker=dict(size=6),
        hovertemplate="Year %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    _fig_fda.add_trace(_go.Scatter(
        x=years, y=p2_bal,
        mode="lines+markers",
        name=p2["name"],
        line=dict(color="#D97706", width=2.5),  # orange — strong contrast vs teal
        marker=dict(size=6),
        hovertemplate="Year %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    # Shaded area between the two curves to visually emphasize the gap
    _fig_fda.add_trace(_go.Scatter(
        x=years + years[::-1],
        y=p1_bal + p2_bal[::-1],
        fill="toself",
        fillcolor="rgba(217, 119, 6, 0.08)",  # orange tint, very transparent
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="",
    ))
    _fig_fda.update_layout(
        height=480,
        margin=dict(l=70, r=70, t=30, b=80),
        xaxis_title="Year",
        yaxis_title="Account balance",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.22,
            xanchor="center",  x=0.5,
            bgcolor="rgba(255,255,255,0)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#0B1F2A"),
    )
    _fig_fda.update_xaxes(
        showgrid=False, zeroline=False, tickmode="linear", dtick=1,
        showline=True, linecolor="#E1E8EE",
    )
    _fig_fda.update_yaxes(
        showgrid=True, gridcolor="#F4F7F9", zeroline=False,
        showline=True, linecolor="#E1E8EE",
        tickformat="$,.0f",
    )
    st.plotly_chart(_fig_fda, use_container_width=True,
                    config={"displayModeBar": False})

    # ── Year-by-year table ──
    st.markdown(
        "<h4 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:8px 0 6px 0'>Year-by-Year Comparison</h4>",
        unsafe_allow_html=True,
    )
    import pandas as _pd_fda

    _table_df = _pd_fda.DataFrame({
        "Year":            years,
        p1["name"]:        [f"${v:,.0f}" for v in p1_bal],
        p2["name"]:        [f"${v:,.0f}" for v in p2_bal],
        "Difference ($)":  [f"${(p1_bal[i] - p2_bal[i]):+,.0f}"
                            for i in range(len(years))],
        "Difference (%)":  [
            (f"{((p1_bal[i] - p2_bal[i]) / p2_bal[i] * 100):+.2f}%"
             if p2_bal[i] != 0 else "—")
            for i in range(len(years))
        ],
    })
    st.dataframe(
        _table_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Honest note about deduction frequency ──
    with st.expander("ℹ️ A note on how this is modeled"):
        st.markdown(
            """
            **The math.** At each compounding period, portfolio value
            grows by `(1 + return / N)` and then loses `(expense_ratio / N)`
            to fees, where `N` is the number of periods per year (12, 4,
            2, or 1).

            **Deduction frequency has a small effect.** Quarterly vs
            annual deduction at the same expense ratio differs by only
            a few basis points per year — fund expenses come out of a
            slightly smaller balance with more frequent deductions, but
            the dominant effect is always the expense ratio itself.
            Don't expect the line for "quarterly" to look dramatically
            different from "annually" at the same ER.

            **Real-world funds accrue daily**, not on these clean
            calendar boundaries. This tool uses the periods you choose
            because it's a clearer pedagogical model — the year-end
            balances match what you'd get from any standard
            compound-interest formula at the chosen frequency.

            **No taxes, no flows, no rebalancing.** This is a clean
            lump-sum projection at constant return. It's designed to
            isolate the expense ratio effect, not predict actual
            outcomes. For a full backtest with real prices, use the
            Analyzer tab.
            """
        )
