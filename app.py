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
import data_store as _ds   # module handle for is_remote()/clear_cache() diagnostics

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


# LAZY SKFOLIO/SKLEARN LOADER (2026-07-15 performance pass) ──────────────
# skfolio drags in the sklearn+scipy import chain — several seconds of
# cold-start on Community Cloud — and is only needed on the optimizer /
# backtest / projection paths. _load_skfolio() imports on first use and
# injects the symbols into module globals so all existing call sites work
# unchanged. Call it at the top of any function that uses these symbols.
# (Dropped in this pass: RatioMeasure, BlackLitterman — zero usages.)
_SKFOLIO_LOADED = False

def _load_skfolio():
    global _SKFOLIO_LOADED
    if _SKFOLIO_LOADED:
        return
    global prices_to_returns, MeanRisk, RiskBudgeting, EqualWeighted
    global HierarchicalRiskParity, NestedClustersOptimization
    global MaximumDiversification, ObjectiveFunction
    global LedoitWolf, GerberCovariance, DenoiseCovariance, EWMu, ShrunkMu
    global EmpiricalPrior, WalkForward, cross_val_predict
    global RiskMeasure, train_test_split
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
    from skfolio.prior import EmpiricalPrior
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio import RiskMeasure
    from sklearn.model_selection import train_test_split
    _g = globals()
    _g.update({
        "prices_to_returns": prices_to_returns, "MeanRisk": MeanRisk,
        "RiskBudgeting": RiskBudgeting, "EqualWeighted": EqualWeighted,
        "HierarchicalRiskParity": HierarchicalRiskParity,
        "NestedClustersOptimization": NestedClustersOptimization,
        "MaximumDiversification": MaximumDiversification,
        "ObjectiveFunction": ObjectiveFunction, "LedoitWolf": LedoitWolf,
        "GerberCovariance": GerberCovariance,
        "DenoiseCovariance": DenoiseCovariance, "EWMu": EWMu,
        "ShrunkMu": ShrunkMu, "EmpiricalPrior": EmpiricalPrior,
        "WalkForward": WalkForward, "cross_val_predict": cross_val_predict,
        "RiskMeasure": RiskMeasure, "train_test_split": train_test_split,
    })
    _SKFOLIO_LOADED = True

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
COVER_WATERMARK_PATH   = _data_path("cover_watermark.png")  # three-helmet mark, baked ~8.5% navy alpha
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


# ── Brand image sync ──────────────────────────────────────────────
# The logo and advisor photo used to live ONLY as local PNGs (firm_logo.png
# / advisor_photo.png), so an upload here never reached the client portal
# and didn't survive a Cloud restart. These helpers also carry the images
# through the shared store: the uploaded bytes are downscaled and base64
# data-URI'd into firm_settings.json (firm.logo_data_uri /
# advisor.photo_data_uri) — the same shared file the text identity already
# syncs through. The portal reads those fields directly; this app keeps
# using the local PNG paths, re-materialized from the data URI on boot.
def _img_bytes_to_data_uri(raw, max_px: int = 400) -> str:
    """Downscale to <= max_px on the long edge and return a base64 PNG data
    URI, kept small so it doesn't bloat every firm_settings.json commit."""
    import base64 as _b64
    try:
        from PIL import Image as _Img
        im = _Img.open(BytesIO(raw))
        im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = BytesIO()
        im.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception:
        pass  # fall back to original bytes if PIL/resize is unavailable
    return "data:image/png;base64," + _b64.b64encode(raw).decode("ascii")


def _set_brand_image(kind: str, data_uri) -> None:
    """Write (data_uri set) or clear (data_uri falsy) a brand image in the
    shared firm_settings.json so it syncs to the portal."""
    section, field = (("firm", "logo_data_uri") if kind == "logo"
                      else ("advisor", "photo_data_uri"))
    def _mut(s):
        s.setdefault(section, {})
        if data_uri:
            s[section][field] = data_uri
        else:
            s[section].pop(field, None)
    _shared_update_json(FIRM_SETTINGS_FILE, _mut)


def _hydrate_brand_images() -> None:
    """On boot, if a local logo/photo PNG is missing but its data URI is in
    the shared firm_settings.json, decode it back to disk — so every existing
    local-path reader (PDF builder, previews, _circular_photo) keeps working
    unchanged after a fresh container start."""
    import base64 as _b64
    try:
        fs = load_firm_settings() or {}
    except Exception:
        return
    for path, sect, field in ((FIRM_LOGO_PATH, "firm", "logo_data_uri"),
                              (ADVISOR_PHOTO_PATH, "advisor", "photo_data_uri")):
        if os.path.exists(path):
            continue
        uri = (fs.get(sect, {}) or {}).get(field)
        if not uri or "," not in uri:
            continue
        try:
            with open(path, "wb") as _f:
                _f.write(_b64.b64decode(uri.split(",", 1)[1]))
        except Exception:
            pass


_hydrate_brand_images()


# ── PDF CONTENT (advisor-customizable closing sections) ───────────
# The closing sections of the client proposal PDF — Advisor Notes,
# Implementation Plan, How This Proposal Was Built, Key Definitions
# and Disclosures — are editable from the "PDF Content" tab.
# Customizations persist to pdf_content.json (its OWN file, kept
# separate from firm_settings.json so the Firm Branding save in
# Settings — which rewrites that whole file — can't clobber them).
# Any section the advisor hasn't customized falls back to
# DEFAULT_PDF_CONTENT below, so an untouched install renders exactly
# as it did before this tab existed.
PDF_CONTENT_FILE = _data_path("pdf_content.json")

DEFAULT_PDF_CONTENT = {
    # Generic advisor note — used when an individual proposal carries
    # no note of its own. Paragraphs are blank-line separated.
    "advisor_notes": (
        "Thank you for the opportunity to review your portfolio and "
        "prepare this proposal. The recommendations on the preceding "
        "pages reflect the risk profile and goals we discussed, and "
        "are intended as a starting point for our conversation rather "
        "than a final decision.\n\n"
        "I would welcome the chance to walk through any section in "
        "more detail and answer any questions before we move "
        "forward. Please don't hesitate to reach out."
    ),
    # Implementation Plan — rows of [Stage, Cadence, Action].
    "implementation_plan": [
        ["Initial Funding", "Day 0",
         "Fund account and execute initial allocation per selected "
         "option."],
        ["First Review", "30 days",
         "Confirm execution, verify holdings match proposal."],
        ["Rebalancing", "Quarterly",
         "Drift threshold 5% per position; tax-aware rebalancing "
         "where applicable."],
        ["Performance Review", "Semi-Annual",
         "Review against benchmarks; discuss changes in goals."],
        ["Full Re-Assessment", "Annual",
         "Update risk profile; refresh proposal if score or goals "
         "change."],
    ],
    # How This Proposal Was Built — blank-line-separated paragraphs.
    "methodology": (
        "Each recommended portfolio is constructed using "
        "institutional-grade optimization techniques. Allocations are "
        "informed by risk-score-targeted equity/bond/cash splits with "
        "priority-driven tilts applied on top.\n\n"
        "<b>Risk-targeted base allocation</b> - a mapping from the "
        "client's 1-99 risk score to a target equity / bond / cash "
        "split forms the starting point.\n\n"
        "<b>Priority tilts</b> - client-stated goals (e.g. capital "
        "preservation, income, social impact) adjust both the "
        "asset-class mix and the ticker universe.\n\n"
        "<b>Holdings selection</b> - where possible, proposals use "
        "the client's own submitted securities; gaps are filled with "
        "broadly-diversified index ETFs."
    ),
    # Key Definitions — rows of [Term, Definition].
    "key_definitions": [
        ["Risk Number",
         "Integer 1-99 summarizing combined risk tolerance "
         "(willingness) and capacity (ability)."],
        ["Equity / Bond / Cash",
         "High-level split across growth, stability, and liquidity "
         "objectives."],
        ["CAGR",
         "Compound annual growth rate \u2014 the constant yearly rate "
         "that would grow the starting value to the ending value over "
         "the measurement window."],
        ["Sharpe Ratio",
         "Return per unit of total volatility. Values above 1.0 "
         "indicate strong risk-adjusted performance."],
        ["Maximum Drawdown",
         "Largest peak-to-trough decline over the measurement "
         "period."],
        ["Priority Tilt",
         "Adjustment to base allocation based on client-stated "
         "goals."],
    ],
    # Disclosures — blank-line-separated paragraphs. The token
    # {advisory_fee} is replaced with the firm's resolved advisory
    # fee at render time.
    "disclosures": (
        "<b>Past performance is no guarantee of future results.</b> "
        "Investment return and principal value of an investment will "
        "fluctuate; therefore, you may have a gain or loss when you "
        "sell your shares. Current performance may be higher or lower "
        "than the performance data quoted.\n\n"
        "<b>Net of Fees Performance.</b> Performance figures shown in "
        "this report reflect the underlying funds' net expense ratios "
        "but are gross of advisory fees. Your actual return would be "
        "reduced by the firm's advisory fee of {advisory_fee} per "
        "year, as well as any brokerage commissions, custodial costs "
        "and other expenses.\n\n"
        "<b>Hypothetical and Backtested Data.</b> Where the analysis "
        "includes performance for portfolio combinations the client "
        "did not actually hold during the measurement period, those "
        "results are hypothetical and backtested. Such results are "
        "achieved by retroactively applying a model to historical "
        "data and do not represent actual trading. Forward-looking "
        "projections (e.g., Monte Carlo) are estimates, not "
        "guarantees.\n\n"
        "<b>Benchmark Comparisons.</b> Where a benchmark portfolio is "
        "shown for comparative purposes, it is illustrative only. "
        "Indexes are unmanaged; you cannot invest directly in an "
        "index. Benchmark performance reflects only the underlying "
        "index or model and not the deduction of advisory fees.\n\n"
        "<b>Limitations and Risk.</b> All investing involves risk, "
        "including possible loss of principal. Diversification and "
        "asset allocation do not guarantee a profit or protect "
        "against loss. This report is informational only and does "
        "not constitute tax, legal, or accounting advice."
    ),
}


def load_pdf_content() -> dict:
    """Return advisor-saved PDF section customizations from
    pdf_content.json. Routes through data_store (same as
    firm_settings.json). Returns {} if nothing is saved or on error."""
    try:
        val = _shared_load_json(PDF_CONTENT_FILE, default={})
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def save_pdf_content(content: dict) -> None:
    """Persist PDF section customizations to pdf_content.json via
    data_store (so the file lands in the shared GitHub repo)."""
    _shared_save_json(PDF_CONTENT_FILE, content)


def get_pdf_content() -> dict:
    """Effective PDF section content: advisor customizations layered
    over DEFAULT_PDF_CONTENT. A section the advisor hasn't customized
    — or has cleared back to empty — falls back to its default, so the
    PDF is unchanged until something is actually edited."""
    custom = load_pdf_content()
    merged = {}
    for key, default in DEFAULT_PDF_CONTENT.items():
        val = custom.get(key)
        if val is None or val == "" or val == []:
            merged[key] = default
        else:
            merged[key] = val
    return merged


# ── CLIENT PORTAL AGREEMENT (Terms & Privacy shown at registration) ─────
# Edited from the PDF Content tab and stored in its OWN shared file so the
# Firm Branding / PDF Content saves never clobber it. The client portal reads
# legal_content.json and shows the text in registration popups. Defaults below
# mirror the portal's placeholder text — replace via the editor with your
# CCO/counsel-approved language.
LEGAL_CONTENT_FILE = _data_path("legal_content.json")

DEFAULT_LEGAL_CONTENT = {
    "version": "2026-06-01",
    "terms": """**Placeholder — replace with your approved Terms & Conditions. Not legal advice.**

**1. Acceptance.** By creating an account and checking the agreement box, you agree to these Terms and to the Privacy Policy.

**2. Nature of the service.** This portal provides an educational risk-profile assessment and related information. It is not investment, legal, or tax advice and is not an offer to buy or sell any security. Advisory services are provided only under a separate written agreement with MRB Capital Group.

**3. No guarantees.** All investing involves risk, including possible loss of principal. Risk-profile results are estimates based on the information you provide and do not guarantee any outcome.

**4. Your information.** You agree to provide accurate information and to keep your contact details current. Your email is used to sign in.

**5. Electronic communications.** You consent to receive communications and disclosures electronically.

**6. Privacy.** Your information is handled as described in the Privacy Policy.

**7. Limitation of liability.** To the fullest extent permitted by law, MRB Capital Group is not liable for indirect or consequential damages arising from use of this portal. _[Confirm with counsel.]_

**8. Governing law.** These Terms are governed by the laws of [STATE]. _[Confirm with counsel.]_

**9. Changes.** These Terms may be updated; the version shown reflects the current terms.

**10. Contact.** Questions? Reach your advisor through the Advisor tab.""",
    "privacy": """**Placeholder — replace with your approved Privacy Policy. Not legal advice.**

**What we collect.** Name, email, phone, optional address/ZIP, age, and your risk-questionnaire answers.

**How we use it.** To generate your risk profile, let you sign in, and allow your advisor to follow up with you.

**Sharing.** Information may be shared with the service providers that operate this platform (for example, the firm's CRM) and is not sold. _[Confirm provider list with counsel.]_

**Security.** Reasonable safeguards are used to protect your information.

**Your choices.** You can request to update or delete your information by contacting your advisor.

**Contact.** Reach your advisor through the Advisor tab.""",
}

def load_legal_content() -> dict:
    """Advisor-saved client-portal Terms/Privacy text from legal_content.json.
    Routes through data_store (shared GitHub repo) so the portal reads it."""
    try:
        val = _shared_load_json(LEGAL_CONTENT_FILE, default={})
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}

def save_legal_content(content: dict) -> None:
    _shared_save_json(LEGAL_CONTENT_FILE, content)

def get_legal_content() -> dict:
    """Effective agreement content: advisor edits layered over defaults."""
    custom = load_legal_content()
    merged = {}
    for key, default in DEFAULT_LEGAL_CONTENT.items():
        val = custom.get(key)
        merged[key] = default if (val is None or val == "") else val
    return merged


def _render_pdf_prose(text, style, gap=5.76):
    """Split advisor-edited prose into ReportLab flowables.

    Paragraphs are separated by blank lines; a small Spacer (default
    5.76pt = 0.08in) is placed between them. Bare ampersands are
    escaped so ReportLab's mini-HTML parser doesn't choke, while
    <b>/<i> tags are left intact so the advisor can bold lead-ins.
    Single newlines within a paragraph become <br/> line breaks.

    Paragraph/Spacer are imported here, not at module scope: the PDF
    builder imports reportlab.platypus locally inside its own
    function, so this module-level helper can't rely on those names
    being in the global namespace.
    """
    from reportlab.platypus import Paragraph, Spacer
    flows = []
    for chunk in str(text).split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if flows:
            flows.append(Spacer(1, gap))
        safe = chunk.replace("&", "&amp;").replace("\n", "<br/>")
        flows.append(Paragraph(safe, style))
    return flows

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

# ── Zacks Investment Management SMA strategies (Q4 2025 GIPS factsheets) ──
# Built from each strategy's disclosed Top-10 holdings, equal-weighted. The
# factsheets publish only the top 10 of ~50–100 positions and give no
# individual weights, so these are a TOP-HOLDINGS PROXY of the strategy, not
# the full composite — the app's backtest/risk on these reflects the 10
# names shown, not Zacks' published composite returns. Adjust the holdings
# or weights in the Portfolio Manager once the full sleeve is known.
# Standard management fees per the sheets: 1.75%/yr for All-Cap Core,
# Dividend, Focus Growth, and Mid-Cap Core; 0.75%/yr for Small-Cap Equity.
_ZACKS_PORTFOLIOS = {
    "── Zacks Strategies ──": None,
    "Zacks All-Cap Core": {
        "tickers": ["NVDA", "AAPL", "GOOGL", "MSFT", "META",
                    "AMZN", "CAT", "JPM", "AXP", "WMT"],
        "weights": [10.0] * 10,
    },
    "Zacks Dividend": {
        "tickers": ["JPM", "PH", "XOM", "CAT", "CSCO",
                    "PM", "WMT", "JNJ", "BLK", "MSFT"],
        "weights": [10.0] * 10,
    },
    "Zacks Focus Growth": {
        "tickers": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
                    "META", "AVGO", "TSLA", "LLY", "APH"],
        "weights": [10.0] * 10,
    },
    "Zacks Mid-Cap Core": {
        "tickers": ["NTRS", "EXPE", "MCK", "GLW", "APH",
                    "HHH", "SOFI", "NFG", "APP", "EME"],
        "weights": [10.0] * 10,
    },
    "Zacks Small-Cap Equity": {
        "tickers": ["FN", "ALNT", "FIX", "SSRM", "RIGL",
                    "NXT", "CRDO", "TCBI", "AEIS", "KLIC"],
        "weights": [10.0] * 10,
    },
}
POPULAR_PORTFOLIOS.update(_ZACKS_PORTFOLIOS)

# ── WisdomTree model portfolios (trade notifications, May 2026) ──
# The full WisdomTree model lineup — 36 models across 12 delivery files,
# each with complete ETF holdings and target weights summing to 100%, so
# they load as faithful models (not proxies). Families: Strategic
# Allocations, Building Blocks, U.S. Growth, Endowment, Global Multi-Asset
# Income, Efficient Core, Multi-Factor, Select (PIMCO), Liquid Alternatives,
# Geopolitically Risk Aware, Global Dividend, and Siegel-WisdomTree.
_WISDOMTREE_PORTFOLIOS = {
    "── WisdomTree Models ──": None,
    'WisdomTree Efficient Core Moderate': {
        "tickers": ["BTAL", "DBMF", "GTR", "WTMF", "DDWM", "EES", "GDE", "NTSE", "NTSI", "NTSX", "QGRW", "VYM", "HYZD", "MTGP"],
        "weights": [4, 4, 4, 6, 3, 4, 7, 6, 14, 27, 7, 4, 2, 8],
    },
    'WisdomTree Efficient Core Aggressive': {
        "tickers": ["BTAL", "GTR", "WTMF", "DDWM", "DGRW", "DGS", "DLS", "DON", "EES", "GDE", "NTSE", "NTSI", "NTSX", "QGRW", "VYM"],
        "weights": [5, 3, 4, 5, 6, 4, 2.5, 4.5, 4, 10, 4, 12, 15, 15, 6],
    },
    'WisdomTree Endowment Conservative': {
        "tickers": ["BTAL", "EMLP", "GCC", "PPI", "WTMF", "WTPI", "DGRS", "DXJ", "EPS", "NTSX", "QGRW", "SPDW", "XSOE", "AGGY", "ELD", "MTGP", "QHY", "STOT"],
        "weights": [0.75, 0.5, 1, 1, 1, 0.75, 1.5, 0.75, 6, 3, 2.25, 2, 1.5, 48, 4, 14.4, 5.6, 6],
    },
    'WisdomTree Endowment Moderately Conservative': {
        "tickers": ["BTAL", "EMLP", "GCC", "PPI", "WTMF", "WTPI", "DGRS", "DXJ", "EPS", "NTSX", "QGRW", "SPDW", "XSOE", "AGGY", "ELD", "MTGP", "QHY", "STOT"],
        "weights": [1.5, 1, 2, 2, 2, 1.5, 2.5, 1.25, 8, 5, 3.75, 4, 2.5, 39.25, 3, 11.7, 4.55, 4.5],
    },
    'WisdomTree Endowment Moderate': {
        "tickers": ["BTAL", "EMLP", "GCC", "PPI", "WTMF", "WTPI", "DGRS", "DXJ", "EPS", "NTSX", "QGRW", "SPDW", "XSOE", "AGGY", "ELD", "MTGP", "QHY", "STOT"],
        "weights": [2.25, 1.5, 3, 3, 3, 2.25, 4.5, 2.25, 13.5, 9, 6.75, 7.5, 4.5, 24, 2, 7.2, 2.8, 1],
    },
    'WisdomTree Endowment Moderately Aggressive': {
        "tickers": ["BTAL", "EMLP", "GCC", "PPI", "WTMF", "WTPI", "DGRS", "DXJ", "EPS", "NTSX", "QGRW", "SPDW", "XSOE", "AGGY", "MTGP", "QHY"],
        "weights": [3, 2, 4, 4, 4, 3, 5, 2.5, 15, 10, 7.5, 8, 5, 19.5, 5.4, 2.1],
    },
    'WisdomTree Endowment Aggressive': {
        "tickers": ["BTAL", "EMLP", "GCC", "PPI", "WTMF", "WTPI", "DGRS", "DXJ", "EPS", "NTSX", "QGRW", "SPDW", "XSOE", "AGGY", "MTGP", "QHY"],
        "weights": [4.5, 3, 6, 6, 6, 4.5, 5, 2.5, 15, 10, 7.5, 8, 5, 12, 3.6, 1.4],
    },
    'WisdomTree Geopolitically Risk Aware': {
        "tickers": ["BTCW", "DGRW", "DXJ", "EPI", "EPOL", "EWW", "GCC", "HEDJ", "OPPJ", "SHAG", "USDU", "WCBR", "WCLD", "XSOE"],
        "weights": [3, 28.5, 7.5, 6, 3, 7, 5, 5, 5, 15, 2.5, 2.5, 5, 5],
    },
    'WisdomTree Global Dividend': {
        "tickers": ["DDWM", "DEM", "DES", "DGRW", "DGS", "DHS", "DON", "DTD", "VYM", "VYMI"],
        "weights": [10, 5, 5, 15, 5, 10, 7, 16, 17, 10],
    },
    'WisdomTree Global Multi-Asset Income Conservative': {
        "tickers": ["DDWM", "DEM", "DHS", "DON", "DTD", "EMLP", "QYLD", "VYM", "AGGY", "ELD", "MTGP", "QHY", "SHAG", "TLH", "VCIT"],
        "weights": [4.4, 2, 2, 3, 4.1, 1, 1, 4.5, 20, 8, 12, 20, 6, 4, 8],
    },
    'WisdomTree Global Multi-Asset Income Moderately Conservative': {
        "tickers": ["DDWM", "DEM", "DHS", "DON", "DTD", "EMLP", "IQDG", "QYLD", "VYM", "AGGY", "ELD", "MTGP", "QHY", "SHAG", "TLH", "VCIT"],
        "weights": [5.1, 3, 3, 4, 6.4, 1.5, 1, 1.5, 6.5, 17.5, 7, 10.5, 17.5, 5, 3.5, 7],
    },
    'WisdomTree Global Multi-Asset Income Moderate': {
        "tickers": ["DDWM", "DEM", "DES", "DHS", "DON", "DTD", "DXJ", "EMLP", "IQDG", "QYLD", "VYM", "AGGY", "ELD", "MTGP", "QHY", "SHAG", "TLH", "VCIT"],
        "weights": [7.2, 6, 3, 6, 5, 11.5, 3, 3, 3.6, 3, 11.7, 10, 4, 6, 10, 2, 1, 4],
    },
    'WisdomTree Global Multi-Asset Income Moderately Aggressive': {
        "tickers": ["DDWM", "DEM", "DES", "DHS", "DON", "DTD", "DXJ", "EMLP", "IQDG", "QYLD", "VYM", "AGGY", "ELD", "MTGP", "QHY", "SHAG", "VCIT"],
        "weights": [8.4, 7, 3.5, 7, 5.5, 13.6, 3.5, 3.5, 5, 3.5, 13.5, 6.25, 3, 4.5, 7.5, 1.75, 3],
    },
    'WisdomTree Global Multi-Asset Income Aggressive': {
        "tickers": ["DDWM", "DEM", "DES", "DHS", "DON", "DTD", "DXJ", "EMLP", "IQDG", "QYLD", "VYM", "AGGY", "ELD", "MTGP", "QHY", "VCIT"],
        "weights": [9.6, 8, 4, 8, 6, 16.4, 4, 4, 4.8, 4, 15.2, 4, 2, 3, 5, 2],
    },
    'WisdomTree Liquid Alternatives': {
        "tickers": ["BTAL", "DBMF", "USDU", "WTMF", "WTPI"],
        "weights": [15, 25, 10, 30, 20],
    },
    'WisdomTree US Factor': {
        "tickers": ["DGRS", "DGRW", "DON", "EPS", "MTUM", "SCHG", "USMF", "WTV"],
        "weights": [5, 20, 5, 20, 5, 30, 5, 10],
    },
    'WisdomTree Developed International Factor': {
        "tickers": ["DDWM", "DLS", "IMTM", "IQDG"],
        "weights": [50, 10, 25, 15],
    },
    'WisdomTree EM Factor': {
        "tickers": ["CXSE", "DGRE", "DGS", "EPI", "FNDE", "XSOE"],
        "weights": [10, 15, 10, 10, 15, 40],
    },
    'WisdomTree Select Conservative (PIMCO)': {
        "tickers": ["DDWM", "DON", "EPS", "QGRW", "XSOE", "BOND", "CORP", "HYS", "LDUR", "MINT", "ZROZ"],
        "weights": [5, 3, 8, 4, 2, 44, 8, 4, 12, 6, 4],
    },
    'WisdomTree Select Moderate (PIMCO)': {
        "tickers": ["DDWM", "DGRW", "DON", "EPI", "EPS", "QGRW", "USMF", "WTV", "XSOE", "BOND", "CORP", "HYS", "LDUR", "MINT", "ZROZ"],
        "weights": [15, 9, 6.8, 1.8, 6, 10, 6, 4.8, 3.6, 22, 4, 2, 6, 1, 2],
    },
    'WisdomTree Select Aggressive (PIMCO)': {
        "tickers": ["DDWM", "DGRW", "DON", "EPI", "EPS", "QGRW", "USMF", "WTV", "XSOE", "BOND", "CORP", "HYS", "LDUR", "ZROZ"],
        "weights": [20, 13, 8.4, 2.4, 8, 13, 8, 6.4, 4.8, 11, 2, 1, 1, 1],
    },
    'Siegel-WisdomTree Global Equity': {
        "tickers": ["BBEU", "DEM", "DGRS", "DGRW", "DTD", "DXJ", "IQDG", "IVV", "SPMD", "VYM", "VYMI"],
        "weights": [4, 8, 5, 20, 15, 4, 4, 15, 7, 8, 10],
    },
    'Siegel-WisdomTree Longevity': {
        "tickers": ["AGGY", "BBEU", "DEM", "DGRS", "DGRW", "DTD", "DXJ", "IQDG", "IVV", "QHY", "SPMD", "VYM", "VYMI", "WTMF"],
        "weights": [15, 4, 7, 4, 8, 13.5, 3, 4, 8, 8, 6.5, 6, 8, 5],
    },
    'Siegel-WisdomTree Moderate': {
        "tickers": ["BBEU", "DEM", "DGRS", "DGRW", "DTD", "DXJ", "IQDG", "IVV", "SPMD", "VYM", "VYMI", "AGGY", "ELD", "MTGP", "QHY", "TLH", "USFR"],
        "weights": [3, 4.8, 2.8, 14.5, 9, 2.4, 3, 8.1, 4, 4.8, 6.6, 16.3, 2, 9.2, 2.5, 5.6, 1.4],
    },
    'Siegel-WisdomTree Moderately Conservative': {
        "tickers": ["DEM", "DGRS", "DGRW", "DTD", "IQDG", "IVV", "VYM", "VYMI", "AGGY", "BIV", "ELD", "MTGP", "QHY", "TLH", "USFR", "VCSH"],
        "weights": [4, 4, 11, 7, 5, 7, 5, 5, 12, 6, 3, 12, 4, 6, 5, 4],
    },
    'Siegel-WisdomTree Conservative': {
        "tickers": ["DEM", "DGRW", "DTD", "IVV", "VYM", "VYMI", "AGGY", "BIV", "ELD", "MTGP", "QHY", "TLH", "USFR", "VCSH"],
        "weights": [3, 8, 6, 5, 4, 6, 14, 8, 4, 15, 5, 6, 10, 6],
    },
    'WisdomTree Conservative': {
        "tickers": ["DGRW", "DON", "EES", "QGRW", "SPDW", "XSOE", "AGGY", "BIV", "ELD", "MTGP", "QHY", "QSIG", "TLH", "USFR"],
        "weights": [6, 2, 2, 6, 4, 2, 24.8, 4, 4, 17.6, 4.8, 7.2, 11.2, 4.4],
    },
    'WisdomTree Moderate': {
        "tickers": ["DDWM", "DGRW", "EES", "EPI", "IJH", "JVAL", "QGRW", "SPDW", "WTV", "XSOE", "AGGY", "ELD", "MTGP", "QHY", "TLH", "USFR"],
        "weights": [7.8, 8.5, 2.8, 1.8, 3.4, 4.8, 16.5, 5.4, 9, 3, 16.3, 2, 9.2, 2.5, 5.6, 1.4],
    },
    'WisdomTree Aggressive': {
        "tickers": ["DDWM", "DGRW", "EES", "EPI", "IJH", "JVAL", "QGRW", "SPDW", "WTV", "XSOE", "AGGY", "MTGP", "QHY", "TLH"],
        "weights": [10, 11.3, 3.4, 2.5, 4.2, 6.4, 25, 6.8, 10.4, 4, 7.6, 4, 1.4, 3],
    },
    'WisdomTree Core Equity': {
        "tickers": ["DDWM", "DGRW", "EES", "EPI", "IJH", "JVAL", "QGRW", "SPDW", "WTV", "XSOE"],
        "weights": [13, 12, 5, 3, 5, 5, 28, 9, 15, 5],
    },
    'WisdomTree Fixed Income': {
        "tickers": ["AGGY", "BIV", "BLV", "ELD", "MTGP", "QHY", "QSIG", "TLH", "USFR"],
        "weights": [30, 5, 4, 5, 24, 6, 9, 9, 8],
    },
    'WisdomTree Short Duration Fixed Income': {
        "tickers": ["ELD", "HYZD", "MTGP", "QSIG", "SHAG", "USFR"],
        "weights": [5, 15, 20, 20, 25, 15],
    },
    'WisdomTree U.S. Conservative Growth': {
        "tickers": ["DGRS", "DGRW", "DON", "EPS", "MTUM", "SCHG", "USMF", "WTV", "AGGY", "BIV", "ELD", "MTGP", "QHY", "QSIG", "TLH", "USFR"],
        "weights": [2, 8, 2, 10, 2, 12, 2, 5, 18.6, 3, 3, 13.2, 3.6, 5.4, 8.4, 1.8],
    },
    'WisdomTree U.S. Moderate Growth': {
        "tickers": ["DGRS", "DGRW", "DON", "EPS", "MTUM", "SCHG", "USMF", "WTV", "AGGY", "BIV", "ELD", "MTGP", "QHY", "QSIG", "TLH", "USFR"],
        "weights": [3, 12, 3, 15, 3, 18, 3, 6, 11.4, 1, 2, 8.8, 2.4, 3.6, 5.6, 2.2],
    },
    'WisdomTree U.S. Growth': {
        "tickers": ["DGRS", "DGRW", "DON", "EPS", "MTUM", "SCHG", "USMF", "WTV", "AGGY", "BIV", "MTGP", "QHY", "QSIG", "TLH"],
        "weights": [3.5, 14, 3.5, 17.5, 3.5, 20, 5, 7, 9.2, 1.5, 6.6, 1.8, 2.7, 4.2],
    },
    'WisdomTree U.S. Aggressive Growth': {
        "tickers": ["DGRS", "DGRW", "DON", "EPS", "MTUM", "SCHG", "USMF", "WTV", "AGGY", "MTGP", "QHY", "QSIG", "TLH"],
        "weights": [4, 16, 4, 20, 4, 24, 4, 8, 5.8, 4.4, 1.2, 1.8, 2.8],
    },
}
POPULAR_PORTFOLIOS.update(_WISDOMTREE_PORTFOLIOS)


# ── Institution classification for portfolio presets ─────────────────────
# Powers the institution filter on the preset dropdowns. Provider is inferred
# from the label prefix; section headers ("── X ──") and Custom return None.
_INSTITUTION_ALL = "All institutions"


def _portfolio_institution(label):
    if not label or label.startswith("── ") or label.startswith("Custom"):
        return None
    low = label.lower()
    if low.startswith("schwab"):     return "Schwab"
    if low.startswith("zacks"):      return "Zacks"
    if "wisdomtree" in low:          return "WisdomTree"
    return "Other"


def _portfolio_institutions():
    """Ordered, de-duplicated list of institutions present in the presets."""
    seen = []
    for k, v in POPULAR_PORTFOLIOS.items():
        if v is None:
            continue
        inst = _portfolio_institution(k)
        if inst and inst not in seen:
            seen.append(inst)
    return seen


def _preset_labels_for(institution=None):
    """Preset labels (no separators, no Custom), optionally filtered to one
    institution. Pass None or _INSTITUTION_ALL for everything."""
    out = []
    for k, v in POPULAR_PORTFOLIOS.items():
        if v is None or k.startswith("Custom"):
            continue
        if institution and institution != _INSTITUTION_ALL:
            if _portfolio_institution(k) != institution:
                continue
        out.append(k)
    return out


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
    "SGOV": 0.0009, "BIL": 0.0014, "SHV": 0.0015, "ICSH": 0.0008,
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
    # ── Niche / specialty ETFs added 2026-05-28 ──
    # Tickers yfinance funds_data / AV / FMP don't reliably cover.
    # Verified against issuer fact sheets and SEC 485BPOS filings,
    # conservative round-up where multiple sources disagreed.
    "VGT":  0.0009,  # Vanguard Information Technology ETF
    "SPMO": 0.0013,  # Invesco S&P 500 Momentum ETF
    "AVUV": 0.0025,  # Avantis US Small Cap Value
    "AVEM": 0.0033,  # Avantis Emerging Markets Equity
    "FBTC": 0.0025,  # Fidelity Wise Origin Bitcoin Fund
    "UTES": 0.0049,  # Virtus Reaves Utilities ETF (active)
    "CTA":  0.0076,  # Simplify Managed Futures Strategy (gross)
    "MNA":  0.0077,  # NYLI Merger Arbitrage (formerly IQ Merger Arbitrage)
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

    SELF-HEALING: any cached entries tagged ``source: stock-fallback``
    are dropped at load time and the cleaned cache is persisted back.
    Those entries represent the "I don't know — defaulting to 0.0"
    branch for equity-classified tickers; once we added yfinance as a
    real lookup source above the API tier, any leftover 0.0s from the
    old fallback path were silently shadowing fresh yfinance data for
    30 days. Filtering on load makes existing bad entries expire
    immediately without manual cache-clear.

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
    # Strip self-healing-targeted entries and persist the cleaned cache
    # back so the next process doesn't see them either.
    _stale = [k for k, v in cache.items()
              if isinstance(v, dict) and v.get("source") == "stock-fallback"]
    if _stale:
        for _k in _stale:
            cache.pop(_k, None)
        try:
            _shared_save_json(EXPENSE_RATIO_CACHE_FILE, cache)
        except Exception:
            pass
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


def _yfinance_fetch_expense_ratio(ticker):
    """Fetch a fund's annual-report expense ratio via yfinance funds_data.

    Reads Ticker.funds_data.fund_operations — the same fund table Yahoo
    Finance displays. Free and unmetered (no API key, no daily quota),
    so this sits ABOVE the Alpha Vantage / FMP tiers in
    _expense_ratio_for_ticker. AV (25 calls/day, 12s throttle) and FMP
    (paywalled endpoints on free tier) become genuine fallbacks rather
    than load-bearing primary sources.

    Returns a decimal expense ratio (0.0003 = 0.03%) or None.

    SCALING ASSUMPTION: yfinance fund_operations is assumed to report
    the ratio as a raw decimal fraction (VTI ~ 0.0003), matching this
    app's internal representation — so no conversion is applied for
    normal values. The > 0.05 guard only catches a value clearly
    delivered as a percent figure. VERIFY after deploy: VTI must
    resolve to ~0.0003. If VTI comes back as 0.03, yfinance is handing
    back percent figures and the normalization needs to flip (drop the
    guard, divide everything by 100 instead).
    """
    if not ticker:
        return None
    try:
        import yfinance as _yf
        _fd = getattr(_yf.Ticker(ticker), "funds_data", None)
        if _fd is None:
            return None
        _ops = getattr(_fd, "fund_operations", None)
        if _ops is None or getattr(_ops, "empty", True):
            return None
        # Match the expense-ratio row by substring (label-change safe)
        _row_label = next(
            (ix for ix in _ops.index
             if "expense ratio" in str(ix).lower()), None)
        if _row_label is None:
            return None
        _row = _ops.loc[_row_label]
        # Prefer the fund's own column; fall back to the first column
        _col = next(
            (c for c in _ops.columns if str(c).upper() == ticker.upper()),
            _ops.columns[0] if len(_ops.columns) else None)
        if _col is None:
            return None
        _er = float(_row[_col])
        # NaN guard (pandas missing values are floats that != themselves)
        if _er != _er or _er <= 0:
            return None
        # Defensive: a value > 5% as a fraction is almost certainly a
        # percent figure that slipped through — normalize it down.
        if _er > 0.05:
            _er = _er / 100.0
        return _er
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


# ── AUTHORITATIVE ER OVERRIDES (beat the cache) ───────────────────────
# A small, issuer-verified set of expense ratios for tickers where the
# live/cached path is unreliable — typically proxy-stitched short-history
# funds (e.g. SGOV maps to BIL for *price* history, which can bleed into a
# wrong ER) or funds a prior run cached at a stale value. Unlike
# _EXPENSE_RATIO_OVERRIDES (a last-resort fallback consulted only when every
# live source fails), these are checked HIGH in the resolver — before the
# 30-day disk cache — so a stale or wrong cached value can't win. Keep this
# list tiny and verified against the issuer's prospectus; the date below is
# the last verification.
#   SGOV: iShares 0-3 Month Treasury Bond ETF — 0.09% (BlackRock, Jun 2026)
_ER_AUTHORITATIVE = {
    "SGOV": 0.0009,
}


def _expense_ratio_for_ticker(ticker):
    """Return the annual expense ratio (decimal) for a ticker.

    Decision tree (curated sources > yfinance > metered APIs > hardcoded):
      1. In-session cache (per-process, fastest)
      2. Schwab model portfolios JSON (authoritative for the ~30 tickers
         in the four Schwab series — Core ETF, Core Income, Enhanced
         Income, Passive-Active. Schwab publishes these alongside model
         rebalances, so they're always current and trump everything else.)
      3. Stocks (via _classify_ticker) → 0.0  (no ER applies)
      4. Crypto → 0.0
      5. Curated mutual fund table (_MUTUAL_FUND_TABLE) — covers Vanguard,
         Fidelity, etc. active funds whose ERs the live APIs don't
         reliably return.
      6. Shared cache via data_store (30-day TTL) — survives restarts
      7. yfinance funds_data.fund_operations → cache + return  (PRIMARY
         LIVE SOURCE; free and unmetered)
      8. Alpha Vantage ETF_PROFILE → cache + return  (fallback)
      9. FMP profile API → cache + return  (fallback)
     10. Hardcoded `_EXPENSE_RATIO_OVERRIDES` — last-resort emergency
     11. None for genuinely unknown tickers (rare)

    Why Schwab JSON sits above APIs: the four Schwab model series
    include active mutual funds (PRFDX, PDBZX, CPXIX, RPIFX, HFQIX,
    CSJIX, etc.) that Alpha Vantage's ETF_PROFILE endpoint doesn't
    consistently return data for — those would silently resolve to
    0.0% and understate portfolio cost. By promoting the JSON above
    the API tiers we get correct numbers for those funds without
    waiting for AV/FMP coverage to improve.

    Why yfinance was promoted above AV/FMP: AV is capped at 25 calls/day
    with 12s throttling on the free tier, so a cold-start 30-fund lookup
    blows past quota in seconds. yfinance is already the price backbone,
    has no rate limit, and carries the annual-report expense ratio for
    both ETFs and mutual funds.

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

    # 2b. Authoritative issuer-verified overrides — checked before the disk
    # cache so a stale/wrong cached value can't override a known-correct ER
    # (e.g. SGOV, which is price-proxied to BIL and otherwise drifts).
    if t in _ER_AUTHORITATIVE:
        sess[t] = float(_ER_AUTHORITATIVE[t])
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

    from datetime import datetime as _dt

    # 6. yfinance funds_data.fund_operations — PRIMARY LIVE SOURCE.
    # Free, unmetered, no API key required. Reads the same fund table
    # Yahoo Finance displays. Sits above AV/FMP so those become genuine
    # emergency fallbacks rather than load-bearing primaries.
    _yf_er = _yfinance_fetch_expense_ratio(t)
    if _yf_er is not None:
        disk_cache[t] = {
            "er": float(_yf_er),
            "fetched": _dt.now().isoformat(),
            "source": "yfinance",
        }
        _save_er_cache(disk_cache)
        sess[t] = float(_yf_er)
        return sess[t]

    # 7. Alpha Vantage ETF_PROFILE — fallback (was PRIMARY before
    # yfinance was added above). Free tier is 25 calls/day with a 12s
    # throttle, so a cold-start lookup of a 30-fund portfolio would
    # exceed quota in seconds — kept as fallback for the rare ticker
    # yfinance doesn't cover.
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
    # IMPORTANT: do NOT disk-cache this fallback. The "equity" class
    # also catches uncovered ETFs (e.g. niche thematic funds yfinance
    # hasn't indexed yet) — writing 0.0 to a 30-day disk cache would
    # shadow yfinance once coverage arrives. Session-cache only so
    # the next session retries fresh.
    if cls == "equity":
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
    #   • Option 3: maximize expected return, weights within ±50% of submitted
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
            f"{'minimize volatility' if 'conservative' in direction_text else 'maximize expected return'}."
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
                 else "Max-return re-optimization within ±50% corridor")
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
        user_tickers, user_weights, objective="max_return", corridor=0.5,
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
        "min_vol"     → minimize variance (Option 2, conservative)
        "max_return"  → maximize expected return (Option 3, aggressive)
        "max_sharpe"  → maximize Sharpe ratio (legacy; available but not
                        used by the standard 3-option proposal flow since
                        max-Sharpe can produce a less-aggressive portfolio
                        when the base is poorly diversified)
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
    _load_skfolio()
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
        elif objective == "max_return":
            # "More aggressive" means just that: tilt weights toward
            # holdings with the highest expected return, accepting more
            # volatility. Previously this branch labeled max-Sharpe as
            # "more aggressive," but max-Sharpe can land on a LESS
            # volatile portfolio if the base is already poorly diversified
            # — exactly what happened with Cole J's TSLA-heavy portfolio
            # (max-Sharpe trimmed concentration → lower vol → lower risk
            # score than the base). MAXIMIZE_RETURN biases toward the
            # higher-vol assets within the corridor instead.
            model = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
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

def _make_favicon():
    """Build the hexagon-pulse logo as a PIL Image for the browser-tab
    favicon.

    Mirrors _HEADER_LOGO_SVG (the in-app header mark) so the browser
    tab icon and the header logo are the same symbol — the advisor app
    previously showed a generic chart emoji (📈) in the tab.

    Streamlit's page_icon doesn't accept a raw SVG string, so the SVG
    path geometry is redrawn here with PIL primitives. The icon is
    rendered at 4× and downsampled with LANCZOS so the strokes get
    anti-aliased edges rather than the jagged ones PIL's line drawing
    produces natively. Pillow is a hard dependency of Streamlit, so
    it's always importable wherever this app runs.
    """
    from PIL import Image, ImageDraw
    _VB    = 24             # SVG viewBox is 24×24
    _OUT   = 64             # final favicon size in px
    _SS    = 4              # supersample factor for anti-aliasing
    _SZ    = _OUT * _SS     # 256px render canvas
    _scale = _SZ / _VB
    _teal  = (14, 92, 94, 255)   # #0E5C5E — same teal as _HEADER_LOGO_SVG
    _lw    = max(1, int(round(1.6 * _scale)))   # stroke width, scaled

    _img = Image.new("RGBA", (_SZ, _SZ), (0, 0, 0, 0))
    _d   = ImageDraw.Draw(_img)

    def _pts(seq):
        return [(x * _scale, y * _scale) for x, y in seq]

    # Hexagon outline — SVG path "M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z".
    # Drawn as a closed polyline (loop back to the first vertex) with
    # joint="curve" so the corners are rounded, matching the SVG's
    # stroke-linejoin="round".
    _hexagon = [(12, 2), (21, 7), (21, 17), (12, 22), (3, 17), (3, 7)]
    _hx = _pts(_hexagon)
    _d.line(_hx + [_hx[0]], fill=_teal, width=_lw, joint="curve")

    # Pulse line — SVG path "M6 12 L9 12 L10.5 9 L12 15 L13.5 11 L15
    # 13 L18 13". The heartbeat trace inside the hexagon.
    _pulse = [(6, 12), (9, 12), (10.5, 9), (12, 15),
              (13.5, 11), (15, 13), (18, 13)]
    _d.line(_pts(_pulse), fill=_teal, width=_lw, joint="curve")

    return _img.resize((_OUT, _OUT), Image.LANCZOS)


# Favicon — hexagon-pulse mark matching the in-app header logo. Falls
# back to the chart emoji if PIL rendering ever fails so the app still
# boots.
try:
    _FAVICON = _make_favicon()
except Exception:
    _FAVICON = "📈"

st.set_page_config(
    page_title="Foresight Portfolio Intelligence",
    page_icon=_FAVICON,
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
        padding-top: 0rem !important;
        padding-bottom: 5rem !important;
        max-width: 1360px !important;
    }
    /* Collapse Streamlit's top toolbar so the banner sits flush at the very
       top (the menu/deploy controls still float in the top-right corner). */
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
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
        background: #FFFFFF !important;
        border: 1px solid #E1E8EE !important;
        border-top: none !important;
        border-radius: 0 0 16px 16px !important;
        padding: 0 14px !important;
        gap: 4px !important;
        margin-top: -1rem !important;
        margin-bottom: 24px !important;
        box-shadow: 0 1px 3px rgba(11,31,42,0.05) !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: visible !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 0 !important;
        color: #6B7E8A !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 12px 6px !important;
        min-height: auto !important;
        min-width: 0 !important;
        flex: 1 1 0 !important;
        justify-content: center !important;
        text-align: center !important;
        white-space: nowrap !important;
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
    /* Kill the cursor that shows inside dropdowns. TWO different things
       can appear: (1) a blinking text CARET in BaseWeb's filter <input>,
       removed with caret-color:transparent; and (2) the I-beam MOUSE
       pointer when hovering that input (the input defaults to
       cursor:text), removed with cursor:pointer so the whole control
       reads as a clickable dropdown. caret-color alone does NOT change
       the mouse pointer — that needs cursor. Broad selectors because
       Streamlit's class/testid names drift between versions;
       role="combobox" is the stable hook on the BaseWeb input. */
    [data-baseweb="select"],
    [data-baseweb="select"] *,
    [data-baseweb="select"] input,
    [data-baseweb="select"] input:focus,
    .stSelectbox input,
    .stMultiSelect input,
    [data-testid="stSelectbox"] input,
    [data-testid="stMultiSelect"] input,
    [role="combobox"],
    input[role="combobox"],
    input[role="combobox"]:focus {
        caret-color: transparent !important;
        cursor: pointer !important;
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
        border-bottom: none;
        border-radius: 16px 16px 0 0;
        padding: 16px 36px 12px 36px;
        margin-bottom: 0;
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
        margin: 0 !important;
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
    _load_skfolio()
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


def _compute_period_price_series(portfolios, periods=None):
    """Compute cumulative-return TIME SERIES for each (name, tickers,
    weights) across each notable historical window.

    Counterpart to `_compute_period_returns` which returns just the
    final scalar. This one returns the full per-day path so the caller
    can draw line charts of how the portfolio moved through the event
    window, not just where it ended up.

    Args:
        portfolios: list of (name, tickers, weights) tuples. Weights are
            in percent (0-100); will be normalized.
        periods:   optional override list of (label, start, end, desc).
            Defaults to NOTABLE_PERIODS.

    Returns:
        dict of {portfolio_name: {period_label: list[(date_str, ret)]}}
        where each list is the cumulative-return curve indexed at the
        window's start (start = 0.0, e.g. -0.18 at trough = -18%).
        Missing data → None for that (portfolio, period) cell.

        Also includes a "_periods" key listing the period labels in
        order. Sized lists match daily trading days within the window.
    """
    import pandas as _pd

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

                # Per-ticker cumulative return path = price/price.iloc[0] - 1.
                # Then portfolio path = weighted sum of per-ticker paths.
                # Buy-and-hold approximation (no daily rebalancing) — same
                # convention as _compute_period_returns. For short event
                # windows the drift between this and rebal-daily is small.
                norm = prices.div(prices.iloc[0])
                weighted = norm.mul(w_eff, axis=1).sum(axis=1)
                series = (weighted - 1.0)
                # Tuple format: (ISO date string, decimal return)
                out[name][label] = [
                    (d.strftime("%Y-%m-%d"), float(v))
                    for d, v in series.items()
                ]
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


def _goal_fmt_money(v):
    """Compact USD for goal figures: $1.92M / $200K / $2,000 / —."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    av = abs(v)
    if av >= 1_000_000:
        return f"${v/1_000_000:.2f}M".replace(".00M", "M")
    if av >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"


def _project_savings(starting, monthly, accum_r, years, contrib_growth=0.0):
    """Future value at the target of a starting balance plus monthly
    contributions that step up by `contrib_growth` each year, all compounded
    monthly at the annual `accum_r`.

    The contribution is level within a year and increases by contrib_growth
    at each year boundary (models raising savings as income grows). The
    closed form reduces exactly to the level-contribution annuity when
    contrib_growth == 0, and is validated against a month-by-month build.
    """
    Y = int(years or 0)
    n = Y * 12
    if n <= 0:
        return float(starting or 0)
    rm = float(accum_r) / 12.0
    g = float(contrib_growth or 0.0)
    fv_start = float(starting or 0) * ((1.0 + rm) ** n)
    if rm > 0:
        af = (1.0 + rm) ** 12                 # one-year growth factor
        per_year = monthly * (af - 1.0) / rm  # FV of 12 monthly pmts, year end
        q = (1.0 + g) / af
        geom = Y if abs(q - 1.0) < 1e-9 else (1.0 - q ** Y) / (1.0 - q)
        contrib_fv = per_year * (af ** (Y - 1)) * geom
    else:
        # No return — just the escalating annual contribution stream.
        contrib_fv = (12.0 * monthly * Y if abs(g) < 1e-9
                      else 12.0 * monthly * (((1.0 + g) ** Y - 1.0) / g))
    return fv_start + contrib_fv


def _compute_goal_metrics(goal):
    """Derive every figure the page-1 Investment Goal section renders.

    Two modes:
      • Retirement — income-replacement chain. Current income × replacement
        ratio is the income need; household Social Security (today's dollars,
        assumed to carry COLAs ≈ inflation) is subtracted; the remaining gap
        is inflated to the retirement year and capitalized as a growing-
        annuity present value — discounted at the IN-RETIREMENT return,
        growing at inflation, over the retirement duration (life expectancy −
        retirement age, floored at 10 yrs). That nest egg is the target.
      • Other goals — the advisor's hand-entered target amount is the target.

    In both modes the savings plan (starting balance + monthly contributions
    compounded at the ACCUMULATION return over the years-to-target) is
    projected and compared to the target. Never raises — bad inputs resolve
    to None'd fields so callers render whatever is available.
    """
    g = dict(goal or {})

    def _f(key, default=0.0):
        try:
            return float(g.get(key) or 0)
        except (TypeError, ValueError):
            return default

    def _i(key):
        try:
            return int(g.get(key))
        except (TypeError, ValueError):
            return None

    is_retirement = (g.get("type") or "").strip().lower() == "retirement"
    target_age  = _i("target_age")
    current_age = _i("current_age")
    years_to_ret = None
    if target_age is not None and current_age is not None:
        years_to_ret = max(0, target_age - current_age)

    starting  = _f("starting_amount")
    monthly   = _f("monthly_contribution")
    accum_r   = _f("assumed_return_pct") / 100.0
    inflation = _f("inflation_pct") / 100.0
    in_ret_r  = _f("in_retirement_return_pct") / 100.0
    contrib_growth = _f("contribution_growth_pct") / 100.0

    out = {
        "is_retirement": is_retirement,
        "years_to_ret":  years_to_ret,
        "starting": starting, "monthly": monthly,
        "accum_return": accum_r, "inflation": inflation,
        "in_ret_return": in_ret_r, "contribution_growth": contrib_growth,
        "current_income": None, "replacement_pct": None,
        "income_need_today": None, "ss_today": None,
        "portfolio_gap_today": None, "gap_at_ret": None,
        "retirement_years": None, "nest_egg": None,
        "target": None, "projected": None,
        "funding_pct": None, "gap": None, "required_return_pct": None,
        "gap_monthly": None,
    }

    if is_retirement:
        income      = _f("current_income")
        replace_pct = _f("replacement_pct")
        ss_today    = _f("monthly_ss") * 12.0
        life_exp    = _i("life_expectancy")

        income_need_today = income * (replace_pct / 100.0)
        gap_today = max(0.0, income_need_today - ss_today)

        ret_years = None
        if life_exp is not None and target_age is not None:
            ret_years = max(10, life_exp - target_age)

        gap_at_ret = gap_today
        if years_to_ret:
            gap_at_ret = gap_today * ((1.0 + inflation) ** years_to_ret)

        nest_egg = None
        if ret_years and gap_at_ret > 0:
            r, gth = in_ret_r, inflation
            if abs(r - gth) < 1e-9:
                nest_egg = gap_at_ret * ret_years / (1.0 + r)
            else:
                nest_egg = (gap_at_ret / (r - gth)) * (
                    1.0 - ((1.0 + gth) / (1.0 + r)) ** ret_years
                )
            nest_egg = max(0.0, nest_egg)

        out.update({
            "current_income": income, "replacement_pct": replace_pct,
            "income_need_today": income_need_today, "ss_today": ss_today,
            "portfolio_gap_today": gap_today, "gap_at_ret": gap_at_ret,
            "retirement_years": ret_years, "nest_egg": nest_egg,
            "target": nest_egg,
        })
    else:
        out["target"] = _f("target_amount")

    # Savings projection (accumulation phase, with annual contribution
    # step-up).
    projected = _project_savings(starting, monthly, accum_r, years_to_ret,
                                 contrib_growth)
    out["projected"] = projected

    target = out["target"]
    if target and target > 0:
        out["funding_pct"] = projected / target * 100.0
        out["gap"] = target - projected
        if (years_to_ret or 0) > 0:
            def _fv_at(_r):
                return _project_savings(starting, monthly, _r, years_to_ret,
                                        contrib_growth)
            _lo, _hi = 0.0, 0.40
            if _fv_at(_lo) <= target <= _fv_at(_hi):
                for _ in range(60):
                    _mid = (_lo + _hi) / 2.0
                    if _fv_at(_mid) < target:
                        _lo = _mid
                    else:
                        _hi = _mid
                out["required_return_pct"] = (_lo + _hi) / 2.0 * 100.0

        # Additional *level* monthly contribution (no step-up) that would
        # close the shortfall to target, holding the assumed return fixed.
        # FV is linear in the monthly amount, so a single $1/mo probe gives
        # the per-dollar future-value factor; the gap divided by that factor
        # is the extra monthly needed. None when already funded.
        if (years_to_ret or 0) > 0:
            _k = _project_savings(0.0, 1.0, accum_r, years_to_ret, 0.0)
            _gap_amt = target - projected
            if _k > 0 and _gap_amt > 0:
                out["gap_monthly"] = _gap_amt / _k

    return out


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

        # Name shown on the proposal cover and page footers. Defaults to the
        # client's name but is editable, so a couple can be addressed jointly
        # (e.g. "Mr. & Mrs. Johnson"). The field is keyed per proposal builder,
        # so each client/version remembers its own override within the session;
        # it's applied at build time only and never changes the client record.
        _default_name = (((client_profile or {}).get("client_name") or "").strip()
                         or "Unassociated Proposal")
        _display_name = st.text_input(
            "Name shown on the proposal",
            value=_default_name,
            key=f"{key_prefix}_dispname",
            help='How the client name appears on the cover and page footers — '
                 'e.g. "Mr. & Mrs. Johnson" for a couple. Does not change the '
                 'client record.',
        )

        # Client's current total balance — drives the dollar amounts and the
        # balance shown on the Holdings pages, and seeds the RMD starting
        # balance. Applied to a copy of the proposal at build time
        # (portfolio_value), so the stored record is never changed. Pre-fills
        # from whichever balance field the client record carries.
        def _first_positive(*vals):
            for _v in vals:
                try:
                    _f = float(_v)
                    if _f > 0:
                        return _f
                except (TypeError, ValueError):
                    continue
            return 0.0
        _pf = proposal or {}
        _clp = client_profile or {}
        _cur_bal_default = _first_positive(
            _pf.get("portfolio_value"), _pf.get("account_value"),
            _pf.get("total_value"), _pf.get("balance"),
            _clp.get("portfolio_value"), _clp.get("account_value"),
            _clp.get("total_value"), _clp.get("balance"),
            _clp.get("total_assets"), _clp.get("investable_assets"),
        )
        _cur_balance_input = st.number_input(
            "Client's current balance ($)", min_value=0.0,
            value=_cur_bal_default, step=10000.0, format="%.0f",
            key=f"{key_prefix}_curbal",
            help="Total account value — drives the dollar amounts and the "
                 "TOTAL PORTFOLIO BALANCE shown on the Holdings pages, and "
                 "seeds the RMD starting balance. Required for those amounts "
                 "to be correct. Applies to this PDF only.",
        )

        rb1, rb2 = st.columns(2)
        # Always-available sections (no underlying data dependency)
        sec_cover     = rb1.checkbox("Cover page + client summary",  True, key=f"{key_prefix}_cov")
        sec_profile   = rb1.checkbox("Risk profile results",         True, key=f"{key_prefix}_prof")
        sec_proposals = rb1.checkbox("3-tier proposal summary",      True, key=f"{key_prefix}_prop")
        sec_hist      = rb1.checkbox("Historical performance table", True, key=f"{key_prefix}_hist")
        sec_fee_comp  = rb1.checkbox(
            "Fee comparison table",  True, key=f"{key_prefix}_feecmp",
            help="Shows the impact of different annual fee levels (0%, 0.25%, "
                 "0.5%, 0.75%, 1%, 1.5%, 2%, 2.5%) on a $100 starting balance over "
                 "1/3/5/7/10 years. Useful for clients comparing your firm's fee "
                 "against industry alternatives.",
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
        sec_goal    = rb2.checkbox(
            "Investment Goal (page 1)", False, key=f"{key_prefix}_goal",
            help="Adds a goal-focused block to page 1 — objective, a funding "
                 "projection toward a dollar target, time horizon, and focus "
                 "areas. Optional; especially useful for prospects with no "
                 "current portfolio. Fields appear below when enabled.",
        )
        sec_rmd     = rb2.checkbox(
            "RMD Projections", False, key=f"{key_prefix}_rmd",
            help="Adds a page after Proposed Holdings projecting required "
                 "minimum distributions year by year — beginning balance, IRS "
                 "Uniform Lifetime factor, the RMD, and ending balance through "
                 "a chosen age. Uses SECURE 2.0 start ages (73 / 75). Inputs "
                 "appear below when enabled.",
        )
        _rmd_payload = None

        # Goal inputs — shown only when the Investment Goal section is on.
        # Pre-filled from the client profile (retirement age, portfolio
        # value, age) so the advisor reviews rather than re-keys. Assembled
        # into _goal_payload and injected onto a copy of the profile at
        # build time as client_profile["goal"]; nothing is persisted, so
        # this stays a pure per-PDF presentation choice.
        _goal_payload = None
        if sec_goal:
            _gp = client_profile or {}
            _g_retire_default = 65
            for _k in ("retirement_age", "target_retirement_age",
                       "retire_age", "retirement_target_age"):
                if _gp.get(_k):
                    try:
                        _g_retire_default = int(_gp.get(_k))
                        break
                    except (TypeError, ValueError):
                        pass
            try:
                _g_start_default = float(_gp.get("portfolio_value") or 0.0)
            except (TypeError, ValueError):
                _g_start_default = 0.0
            try:
                _g_cur_age = int(_gp.get("client_age") or _gp.get("age"))
            except (TypeError, ValueError):
                _g_cur_age = None

            st.markdown("**Investment goal**")
            _gcol1, _gcol2, _gcol3 = st.columns(3)
            _g_type = _gcol1.selectbox(
                "Goal type",
                ["Retirement", "Home purchase", "Education",
                 "Wealth accumulation", "Custom"],
                key=f"{key_prefix}_goal_type",
            )
            _g_label = _gcol2.text_input(
                "Goal label", value=_g_type, key=f"{key_prefix}_goal_label",
            )
            _g_target_age = _gcol3.number_input(
                "Target age", min_value=1, max_value=120,
                value=_g_retire_default, step=1, key=f"{key_prefix}_goal_age",
            )
            # Shared accumulation inputs (both modes).
            _gs1, _gs2, _gs3, _gs4 = st.columns(4)
            _g_start = _gs1.number_input(
                "Starting amount ($)", min_value=0.0, value=_g_start_default,
                step=10000.0, format="%.0f", key=f"{key_prefix}_goal_start",
            )
            _g_monthly = _gs2.number_input(
                "Monthly contribution ($)", min_value=0.0, value=0.0,
                step=100.0, format="%.0f", key=f"{key_prefix}_goal_monthly",
            )
            _g_cgrowth = _gs3.number_input(
                "Contribution increase (%/yr)", min_value=0.0, max_value=25.0,
                value=0.0, step=0.5, format="%.1f",
                key=f"{key_prefix}_goal_cgrowth",
                help="Annual step-up to the monthly contribution — models "
                     "raising savings as income grows. 0% keeps it level.",
            )
            _g_return = _gs4.number_input(
                "Assumed return (%/yr)", min_value=0.0, max_value=30.0,
                value=7.0, step=0.1, format="%.1f",
                key=f"{key_prefix}_goal_return",
                help="Accumulation-phase return for the savings projection. "
                     "Set to the proposed portfolio's expected return (see the "
                     "Historical Backtest page) to reflect the recommendation.",
            )

            # Mode-specific inputs. Retirement derives its target from an
            # income-replacement chain; every other goal type takes a manual
            # dollar target.
            _g_target = 0.0
            _g_income = _g_replace = _g_infl = _g_ss = _g_inret = 0.0
            _g_life_exp = 90
            if _g_type == "Retirement":
                st.caption(
                    "Retirement target is computed from income replacement — "
                    "no dollar target needed. Social Security is treated as "
                    "today's-dollars household income carrying COLAs."
                )
                _gr1, _gr2, _gr3 = st.columns(3)
                _g_income = _gr1.number_input(
                    "Current annual income ($)", min_value=0.0, value=0.0,
                    step=5000.0, format="%.0f", key=f"{key_prefix}_goal_income",
                )
                _g_replace = _gr2.number_input(
                    "Income replacement (%)", min_value=0.0, max_value=200.0,
                    value=80.0, step=5.0, format="%.0f",
                    key=f"{key_prefix}_goal_replace",
                )
                _g_ss = _gr3.number_input(
                    "Monthly Social Security ($)", min_value=0.0, value=0.0,
                    step=100.0, format="%.0f", key=f"{key_prefix}_goal_ss",
                    help="Household monthly Social Security in today's dollars. "
                         "Subtracted from the income need; the portfolio funds "
                         "the remaining gap.",
                )
                _gr4, _gr5, _gr6 = st.columns(3)
                _g_infl = _gr4.number_input(
                    "Inflation (%/yr)", min_value=0.0, max_value=15.0,
                    value=3.0, step=0.1, format="%.1f",
                    key=f"{key_prefix}_goal_infl",
                )
                _g_inret = _gr5.number_input(
                    "In-retirement return (%/yr)", min_value=0.0, max_value=20.0,
                    value=5.0, step=0.1, format="%.1f",
                    key=f"{key_prefix}_goal_inret",
                    help="Return assumed during the drawdown phase — sizes the "
                         "nest egg. Usually lower than the accumulation return.",
                )
                _g_life_exp = _gr6.number_input(
                    "Life expectancy (age)", min_value=1, max_value=120,
                    value=90, step=1, key=f"{key_prefix}_goal_life",
                    help="Retirement duration = life expectancy − target age, "
                         "floored at 10 years.",
                )
            else:
                _g_target = st.number_input(
                    "Target amount ($)", min_value=0.0, value=0.0,
                    step=10000.0, format="%.0f", key=f"{key_prefix}_goal_target",
                )

            _g_narr = st.text_input(
                "Goal narrative (optional — auto-composed if left blank)",
                value="", key=f"{key_prefix}_goal_narr",
            )
            _goal_payload = {
                "type":                 _g_type,
                "label":                (_g_label or _g_type).strip(),
                "target_age":           int(_g_target_age),
                "current_age":          _g_cur_age,
                "target_amount":        float(_g_target or 0),
                "starting_amount":      float(_g_start or 0),
                "monthly_contribution": float(_g_monthly or 0),
                "assumed_return_pct":   float(_g_return or 0),
                "contribution_growth_pct": float(_g_cgrowth or 0),
                "current_income":       float(_g_income or 0),
                "replacement_pct":      float(_g_replace or 0),
                "inflation_pct":        float(_g_infl or 0),
                "monthly_ss":           float(_g_ss or 0),
                "in_retirement_return_pct": float(_g_inret or 0),
                "life_expectancy":      int(_g_life_exp or 0),
                "narrative":            (_g_narr or "").strip(),
            }

            # Live preview so the advisor sees the computed target + funding
            # before generating the PDF.
            try:
                _gm_prev = _compute_goal_metrics(_goal_payload)
                _tgt_prev = (_gm_prev.get("nest_egg")
                             if _g_type == "Retirement" else _gm_prev.get("target"))
                if _tgt_prev:
                    _lbl_prev = ("Nest egg target" if _g_type == "Retirement"
                                 else "Target")
                    _msg_prev = (f"{_lbl_prev} **{_goal_fmt_money(_tgt_prev)}** · "
                                 f"projected **{_goal_fmt_money(_gm_prev.get('projected'))}**")
                    if _gm_prev.get("funding_pct") is not None:
                        _msg_prev += f" ({_gm_prev['funding_pct']:.0f}% funded)"
                    st.info(_msg_prev)
            except Exception:
                pass

        # ── Per-PDF overrides: advisory fee + advisor notes ──────────
        # Both apply to a copy of the proposal at build time only (never
        # persisted), mirroring the display-name and goal overrides above.
        # The fee feeds _resolve_advisory_fee_pct → the Fee Comparison page
        # (★ row) and the SEC fee-impact disclosure; the note feeds the
        # Advisor Notes section.
        st.markdown("**Fees & advisor notes**")
        _fee_default = _resolve_advisory_fee_pct(
            proposal, client_profile, load_firm_settings())
        _adv_fee_input = st.number_input(
            "Advisory fee shown on PDF (%/yr)",
            min_value=0.0, max_value=10.0,
            value=float(_fee_default), step=0.05, format="%.2f",
            key=f"{key_prefix}_advfee",
            help="Sets the firm fee highlighted (★) on the Fee Comparison "
                 "page and used in the fee-impact disclosure. Applies to this "
                 "PDF only — does not change firm settings.",
        )
        _cur_fee_input = st.number_input(
            "Client's current advisory fee (%/yr · 0 = N/A)",
            min_value=0.0, max_value=10.0,
            value=float((proposal or {}).get("current_advisory_fee_pct") or 0.0),
            step=0.05, format="%.2f",
            key=f"{key_prefix}_curfee",
            help="If the client currently pays another advisor, enter that fee "
                 "to show it (●) alongside your proposed fee on the Fee "
                 "Comparison page. Leave at 0 if not applicable.",
        )
        _notes_default = ((proposal.get("advisor_notes") or "").strip()
                          or (get_pdf_content().get("advisor_notes") or "").strip())
        _adv_notes_input = st.text_area(
            "Advisor notes (this PDF)",
            value=_notes_default, height=120,
            key=f"{key_prefix}_advnotes",
            help="Appears in the Advisor Notes section of the PDF. Pre-filled "
                 "from the firm-wide default — edit it for this client. Applies "
                 "to this PDF only.",
        )

        # RMD projection inputs — shown only when the section is enabled.
        # Pre-filled from the client profile (tax-deferred balance defaults to
        # portfolio value; birth year derived from age). Assembled into
        # _rmd_payload and injected onto a copy of the profile at build time
        # as client_profile["rmd"]; nothing is persisted.
        if sec_rmd:
            _rp = client_profile or {}
            try:
                _rmd_bal_default = float(_cur_balance_input or 0.0)
            except (TypeError, ValueError):
                _rmd_bal_default = 0.0
            _this_year = date.today().year
            try:
                _rmd_age = int(_rp.get("client_age") or _rp.get("age"))
            except (TypeError, ValueError):
                _rmd_age = None
            _rmd_birth_default = (_this_year - _rmd_age) if _rmd_age else 1953

            st.markdown("**RMD projection**")
            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            _rmd_bal = _rc1.number_input(
                "Tax-deferred balance ($)", min_value=0.0,
                value=_rmd_bal_default, step=10000.0, format="%.0f",
                key=f"{key_prefix}_rmd_bal",
                help="Balance subject to RMDs — traditional IRA / 401(k) / SEP "
                     "/ SIMPLE. Roth IRAs are excluded. Defaults to the "
                     "portfolio value; override if only part is tax-deferred.",
            )
            _rmd_birth = _rc2.number_input(
                "Birth year", min_value=1900, max_value=2010,
                value=_rmd_birth_default, step=1,
                key=f"{key_prefix}_rmd_birth",
            )
            _rmd_growth = _rc3.number_input(
                "Growth assumption (%/yr)", min_value=0.0, max_value=20.0,
                value=5.0, step=0.5, format="%.1f",
                key=f"{key_prefix}_rmd_growth",
                help="Assumed annual growth on the balance remaining after each "
                     "year's withdrawal.",
            )
            _rmd_end_age = _rc4.number_input(
                "Project through age", min_value=73, max_value=110,
                value=95, step=1, key=f"{key_prefix}_rmd_end",
            )
            _rmd_payload = {
                "balance":     float(_rmd_bal),
                "birth_year":  int(_rmd_birth),
                "growth_rate": float(_rmd_growth),
                "end_age":     int(_rmd_end_age),
            }

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
                "goal": sec_goal,
                "rmd_projection": sec_rmd,
                # Legacy keys kept off — the rendering paths for these
                # were removed when Notable Market Periods replaced them.
                # Setting False so any saved-section dicts referencing
                # these old keys don't trigger removed code.
                "drawdown": False, "rolling": False, "forward": False,
            }
            try:
                with st.spinner("Building PDF…"):
                    # Apply the editable display name as a build-time override
                    # on a copy of the profile, so every "client_name" the PDF
                    # renders (cover, footers, snapshot title) picks it up
                    # without mutating the stored client record.
                    _cp = dict(client_profile or {"client_name": "Unassociated Proposal"})
                    _nm = (_display_name or "").strip()
                    if _nm:
                        _cp["client_name"] = _nm
                    # Thread the opt-in goal block through to the PDF on a
                    # copy of the profile — never mutates the stored record.
                    if _goal_payload:
                        _cp["goal"] = _goal_payload
                    if _rmd_payload:
                        _cp["rmd"] = _rmd_payload
                    # Per-PDF overrides onto a copy of the proposal so the
                    # stored proposal record is never mutated.
                    _prop = dict(proposal or {})
                    # Current balance drives the Holdings dollar amounts and
                    # the total shown on those pages (portfolio_value).
                    try:
                        if float(_cur_balance_input) > 0:
                            _prop["portfolio_value"] = float(_cur_balance_input)
                            _cp["portfolio_value"] = float(_cur_balance_input)
                    except (TypeError, ValueError):
                        pass
                    try:
                        _prop["advisory_fee_pct"] = float(_adv_fee_input)
                    except (TypeError, ValueError):
                        pass
                    try:
                        if float(_cur_fee_input) > 0:
                            _prop["current_advisory_fee_pct"] = float(_cur_fee_input)
                    except (TypeError, ValueError):
                        pass
                    _an = (_adv_notes_input or "").strip()
                    if _an:
                        _prop["advisor_notes"] = _an
                    pdf_bytes = build_client_proposal_pdf(
                        client_profile=_cp,
                        proposal=_prop,
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


# ── RMD projection engine (shared by the proposal PDF) ───────────────────────
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


def _rmd_projection(balance, birth_year, growth_rate, current_year, end_age):
    """Year-by-year RMD projection from the first required year through
    end_age. Convention: each year's RMD comes off the beginning (prior
    year-end) balance, then the remainder grows at growth_rate. If the first
    RMD year is in the future, the starting balance is first grown to that year.

    Returns (rows, summary). Each row is a dict with age, year, begin, factor,
    rmd, pct, end.
    """
    birth_year = int(birth_year)
    end_age = int(end_age)
    g = float(growth_rate) / 100.0
    start_age = _rmd_start_age(birth_year)
    cur_age = int(current_year) - birth_year

    first_year = current_year if cur_age >= start_age else birth_year + start_age
    pre_years = max(0, first_year - int(current_year))

    bal = float(balance) * ((1.0 + g) ** pre_years)  # grow to first RMD year
    rows = []
    cumulative = 0.0
    end_bal = bal
    for yr in range(first_year, birth_year + end_age + 1):
        age = yr - birth_year
        factor = _RMD_UNIFORM_LIFETIME.get(min(max(age, 72), 120))
        if factor is None:
            continue
        begin = end_bal
        rmd = begin / factor
        end_bal = max(0.0, begin - rmd) * (1.0 + g)
        cumulative += rmd
        rows.append({"age": age, "year": yr, "begin": begin, "factor": factor,
                     "rmd": rmd, "pct": 100.0 / factor, "end": end_bal})

    summary = {
        "first_year": first_year,
        "first_rmd": rows[0]["rmd"] if rows else 0.0,
        "total_rmd": cumulative,
        "end_balance": rows[-1]["end"] if rows else bal,
        "end_age": end_age,
        "pre_years": pre_years,
        "start_age": start_age,
    }
    return rows, summary


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
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, BaseDocTemplate,
                                    PageTemplate, Frame, NextPageTemplate,
                                    Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, PageBreak,
                                    KeepTogether, Flowable, Image)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import (Drawing, Wedge, Rect, String,
                                            Circle, Ellipse, Line, PolyLine,
                                            Path)
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics import renderPDF
    from datetime import datetime as _dt

    # Design system — colors and ticker palette come from firm_settings.json
    # via mrb_design. This is the Phase 1 PDF restyle: same layout, new brand
    # palette (MRB navy + gold + Schwab-inspired chart palette). Later phases
    # rebuild the layout itself. The constant names below are preserved so
    # the 2,000+ lines of platypus flowable code that reference them need
    # zero edits — only the underlying hex values change.
    from mrb_design import (load_settings, get_ticker_color,
                            lump_to_other, pick_alignment_tier)
    _SETTINGS = load_settings()
    # Show the pie/legend rollup bucket as a ticker-style "OTHER" (all caps),
    # not the wordy "Other holdings" that wrapped to two lines in narrow
    # legends. Overriding the setting here propagates to the pie wedge label,
    # _OTHER_LABEL below, and every legend's rollup-row lookup.
    try:
        _SETTINGS["chart_palette"]["other_bucket"]["other_label"] = "OTHER"
    except (KeyError, TypeError):
        pass
    _B = _SETTINGS["brand"]
    _CHART_TICKERS = _SETTINGS["chart_palette"]["tickers"]

    # ── CLIENT PROPOSAL PALETTE — sourced from firm_settings.json ─────
    # Constant names preserved from the legacy "mid-blue snapshot" palette
    # so downstream layout code keeps working unchanged. Mapping:
    #   NAVY/NAVY_DEEP/NAVY_MID  → MRB navy family
    #   ACCENT/ACCENT_SOFT       → MRB gold (was mid/sky blue — biggest
    #                              visual change; ACCENT is now the brand
    #                              accent everywhere it was previously
    #                              used as a structural blue)
    #   CHARCOAL/SLATE/GRAY/...  → text scale on the new palette
    #   BG_SOFT/BG_LIGHT/BORDER  → cream surfaces + soft borders
    NAVY        = colors.HexColor(_B["primary"]["navy"])         # #1a2b4a
    NAVY_DEEP   = colors.HexColor(_B["primary"]["navy_dark"])    # #0e1830
    NAVY_MID    = colors.HexColor(_B["primary"]["navy_light"])   # #2b3d5e
    ACCENT      = colors.HexColor(_B["accent"]["gold"])          # #b8943f
    ACCENT_SOFT = colors.HexColor(_B["accent"]["gold_light"])    # #d4b676
    CHARCOAL    = colors.HexColor(_B["text"]["primary"])         # #1a2030
    SLATE       = colors.HexColor(_B["text"]["secondary"])       # #4a4a4a
    GRAY        = colors.HexColor(_B["text"]["tertiary"])        # #6b6b6b
    GRAY_SOFT   = colors.HexColor(_B["text"]["muted"])           # #888888
    BORDER      = colors.HexColor(_B["border"]["medium"])        # #d8d8d8
    BORDER_SOFT = colors.HexColor(_B["border"]["light"])         # #e8e4dc
    BG_SOFT     = colors.HexColor(_B["surface"]["cream"])        # #fafaf6
    BG_LIGHT    = colors.HexColor(_B["surface"]["cream_warm"])   # #fbf8ef
    WHITE       = colors.white
    BLACK       = colors.black

    # Pie palette — 8 distinct slots per advisor revision. The previous
    # iterations had two collision problems:
    #   • 7-slot fallback had near-duplicate purples and cyans (two
    #     wedges in the same hue family that read as the same color).
    #   • 10-slot draft included orange and steel-blue slots that the
    #     advisor wanted removed.
    # The final 8 slots are maximally distinct on the cream BG_SOFT
    # background (#fafaf6) and avoid both NAVY (#1a2b4a — structural)
    # and ACCENT (#b8943f — brand gold), neither of which should
    # appear in a pie segment.
    #
    # Pairs with the 8-cap enforced by the local lump_to_other()
    # wrapper directly below: portfolios with up to 8 named holdings
    # get a distinct color per holding; anything beyond the 8th
    # collapses into the gray "Other" wedge (chart_palette.
    # other_bucket.other_color, #888888).
    #
    # IMPLEMENTATION NOTE: writes the new array directly back into
    # _CHART_TICKERS["_fallback_palette"], so every downstream call
    # to mrb_design.get_ticker_color(_, _SETTINGS) navigates through
    # the same dict and resolves hash-based assignments against the
    # new 8 slots. Explicit per-ticker overrides elsewhere in
    # chart_palette.tickers (e.g. SCHZ → teal, SCHD → berry) are
    # unaffected.
    _CHART_TICKERS["_fallback_palette"] = [
        "#5b3cb4",  # violet
        "#1890a8",  # teal
        "#1c6e3a",  # forest green
        "#8a9030",  # olive
        "#c5302b",  # crimson
        "#b03878",  # berry pink
        "#7a4a28",  # bronze
        "#2b5fb0",  # blue (replaces slate #516172 — slate read as a second
                    # grey wedge next to the gray OTHER bucket; blue is the
                    # palette's largest open hue gap, between teal and violet)
    ]
    _FALLBACK = _CHART_TICKERS["_fallback_palette"]
    PDF_PIE_PALETTE = [colors.HexColor(h) for h in _FALLBACK]

    # ── Local hard-cap wrapper around mrb_design.lump_to_other ─────
    # The 8-slot palette above must never be asked to color more than
    # 8 named wedges. mrb_design's upstream lump_to_other reads its
    # threshold from chart_palette.other_bucket (config key managed
    # in mrb_design / firm_settings.json); if that config doesn't
    # match the PDF palette size, the pie would ask for colors it
    # doesn't have and md5-hash would collide multiple holdings onto
    # the same slot — exactly the visual problem the palette redesign
    # set out to fix. This wrapper enforces the cap independently of
    # the upstream config: it calls the upstream function first, then
    # post-processes the result so at most 8 named holdings remain,
    # with the smallest extras folded into the existing 'Other' row
    # (creating one if upstream didn't return any).
    _orig_lump_to_other = lump_to_other
    _OTHER_LABEL = _SETTINGS["chart_palette"]["other_bucket"]["other_label"]

    def lump_to_other(tickers, weights, settings, _max_named=8,
                      _orig=_orig_lump_to_other,
                      _other_lbl=_OTHER_LABEL):
        ts, ws, has_other = _orig(tickers, weights, settings)
        # Recognize every other-bucket label variant. The display label has
        # changed over time ("Other", "Other holdings", "OTHER"), and the
        # upstream lump can emit one that no longer matches _other_lbl. If
        # such a stray slips through it gets treated as a named holding AND
        # is independently greyed by resolve_chart_colors — which is what
        # produced TWO adjacent grey wedges. Folding all variants into one
        # bucket here guarantees a single grey "Other".
        _aliases = {str(_other_lbl).strip().lower(), "other", "other holdings"}
        named = []
        other_w = 0.0
        for t, w in zip(ts, ws):
            if str(t).strip().lower() in _aliases:
                other_w += float(w or 0)
            else:
                named.append((t, float(w or 0)))
        # ALWAYS sort named largest-first, fold everything beyond the cap
        # into Other, and append Other LAST. This guarantees the pie draws
        # the largest holding at 12 o'clock and descends clockwise to the
        # smallest, with the single grey Other wedge at the end (just
        # before 12). Previously, when named <= cap this returned the
        # upstream order as-is, which was neither guaranteed sorted nor
        # guaranteed to place Other last.
        named.sort(key=lambda tw: -tw[1])
        keep = named[:_max_named]
        drop = named[_max_named:]
        for _, w in drop:
            other_w += w
        new_ts = [t for t, _ in keep]
        new_ws = [w for _, w in keep]
        if other_w > 0:
            new_ts.append(_other_lbl)
            new_ws.append(other_w)
        return new_ts, new_ws, other_w > 0

    def PDF_TICKER_COLOR(ticker):
        """Stable per-ticker color resolution for PDF pie segments.

        Wraps mrb_design.get_ticker_color() so the same SCHD always gets the
        same berry-pink in the PDF as on screen. Unknown tickers fall back
        to a deterministic md5-hashed slot in the fallback palette.

        NOTE: This is the per-ticker "preferred" lookup. For pie charts
        and their matching legends, call resolve_chart_colors(tickers)
        below instead — it enforces no-duplicates WITHIN a single chart
        while still giving each ticker its preferred color where the
        slot isn't already taken.
        """
        return colors.HexColor(get_ticker_color(ticker, _SETTINGS))

    def resolve_chart_colors(tickers):
        """Return a distinct color per ticker for a single chart.

        Tony's requirements, in priority order:
          1. NO two wedges in a single chart may share a color.
          2. Same ticker should get the same color across charts when
             possible (so SCHD's berry stays consistent between the
             cover donut, the current-portfolio donut on page 2, and
             the legend swatch in the proposed comparison cards).

        Algorithm — greedy first-fit on a fixed deterministic ordering:
          • Pass 1: walk the input list in order. For each ticker, take
            its "preferred" color from PDF_TICKER_COLOR (which resolves
            either an explicit anchor like SCHZ→teal or an md5-hashed
            palette slot for unknown tickers). If that color isn't
            already used in this chart, take it.
          • Pass 2: any tickers whose preferred color collided with an
            earlier ticker fall through to the next unused slot in the
            8-color palette, again in palette order. This is
            deterministic — same input list always produces the same
            output list — so the pie chart and its legend (which both
            call this function with the same ticker list) get matching
            colors.

        Goal #1 (no duplicates) is always satisfied as long as the
        ticker list has ≤ palette_size + 1 entries (the +1 being the
        reserved "Other" gray, which doesn't compete for palette slots).
        The local lump_to_other wrapper above enforces this cap.

        Goal #2 (cross-chart stability) is best-effort: explicit
        anchors are always honored on first occurrence; tickers that
        only differ between charts in their position-after-collision
        may shift slots between charts. Anchored tickers (SCHZ, SCHD,
        etc.) effectively never shift because their explicit hex is
        outside the collision-prone md5 path.
        """
        def _hex_of(c):
            return "#{:02x}{:02x}{:02x}".format(
                int(round(c.red   * 255)),
                int(round(c.green * 255)),
                int(round(c.blue  * 255)),
            )

        # "Other" color resolution — different versions of mrb_design's
        # chart_palette settings have placed this key at different
        # paths (chart_palette.other_color as a sibling of
        # other_bucket, vs. chart_palette.other_bucket.other_color
        # nested inside it). Try both, fall back to GRAY_SOFT
        # (#888888 — the brand muted-gray, which is what the original
        # PDF_TICKER_COLOR returned for "Other" via get_ticker_color).
        _other_color = GRAY_SOFT
        _cp = _SETTINGS.get("chart_palette", {}) if isinstance(_SETTINGS, dict) else {}
        _ob = _cp.get("other_bucket", {}) if isinstance(_cp, dict) else {}
        for _candidate in (_ob.get("other_color"), _cp.get("other_color")):
            if isinstance(_candidate, str) and _candidate.startswith("#"):
                try:
                    _other_color = colors.HexColor(_candidate)
                except (ValueError, TypeError):
                    pass
                break
        # "Other" label — known-good path, but be defensive anyway.
        _other_label = _ob.get("other_label", "Other") if isinstance(_ob, dict) else "Other"

        n = len(tickers)
        result = [None] * n
        used_hexes = set()
        # Palette slots still up for reassignment, in palette order so
        # the second pass is deterministic.
        available_palette = list(PDF_PIE_PALETTE)

        # Pass 1 — claim preferred color when free
        for i, t in enumerate(tickers):
            if t == _other_label or t == "Other":
                result[i] = _other_color
                # Other's gray sits OUTSIDE the 8-slot palette, so it
                # doesn't consume an available slot, but we do mark it
                # used so a downstream re-resolution doesn't collide.
                used_hexes.add(_hex_of(_other_color))
                continue
            preferred = colors.HexColor(get_ticker_color(t, _SETTINGS))
            ph = _hex_of(preferred)
            if ph not in used_hexes:
                result[i] = preferred
                used_hexes.add(ph)
                # If preferred came from the palette (anchor matches a
                # palette slot, or md5 fallback hit), remove that slot
                # from the pool so Pass 2 doesn't reassign it.
                available_palette = [
                    c for c in available_palette if _hex_of(c) != ph
                ]

        # Pass 2 — fill colliders with next available palette slot
        for i, t in enumerate(tickers):
            if result[i] is not None:
                continue
            if available_palette:
                chosen = available_palette.pop(0)
                result[i] = chosen
                used_hexes.add(_hex_of(chosen))
            else:
                # Out of slots — only reachable if the input violated
                # the lump_to_other cap. Return preferred color anyway
                # (may duplicate); preferable to crashing.
                result[i] = colors.HexColor(get_ticker_color(t, _SETTINGS))

        return result

    # Tier colors — remapped to the new palette. "conservative" stays navy,
    # "balanced" picks up the brand gold (was the structural blue), and
    # "aggressive" deepens to navy_dark for visual hierarchy.
    TIER_COLORS = {
        "conservative": NAVY,
        "balanced":     ACCENT,
        "aggressive":   NAVY_DEEP,
        "alternate":    GRAY,
    }

    buf = BytesIO()
    # BaseDocTemplate with three PageTemplates so we can switch
    # orientation per page via NextPageTemplate:
    #   - 'cover'     : portrait letter, used for page 1 only (uses
    #                   _on_first_page callback for the dotted top rule)
    #   - 'portrait'  : portrait letter, default for all later pages
    #   - 'landscape' : landscape letter, used for the Holdings (page 2)
    #                   and Recommendations pages
    # Margins are identical across templates so the body content area
    # stays predictable. The on-page callbacks read _doc.pagesize so the
    # nav stripe + footer scale automatically to whichever orientation
    # is active.
    _portrait_size  = letter            # (612, 792)
    _landscape_size = landscape(letter) # (792, 612)
    _margins = dict(
        leftMargin=0.55*inch, rightMargin=0.55*inch,
        topMargin=0.55*inch,  bottomMargin=0.65*inch,
    )

    def _make_frame(pagesize, name):
        _pw, _ph = pagesize
        return Frame(
            _margins['leftMargin'],
            _margins['bottomMargin'],
            _pw - _margins['leftMargin'] - _margins['rightMargin'],
            _ph - _margins['topMargin'] - _margins['bottomMargin'],
            id=name, showBoundary=0,
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0,
        )

    # ── PAGE CALLBACKS ─────────────────────────────────────────
    # Defined here (above the PageTemplate constructors) because Python
    # treats `def` as an assignment, so referencing these names from the
    # PageTemplate(...) calls below would otherwise raise UnboundLocalError
    # if the defs lived further down in the function.
    # Page background / footer (later pages)
    def _on_page(canvas, _doc):
        # Read page dimensions from the ACTIVE page template's pagesize
        # so the callback adapts when we switch between portrait and
        # landscape templates. Falling back to _doc.pagesize (the
        # BaseDocTemplate default) would draw the nav stripe at portrait
        # width on landscape pages, leaving a visible gap on the right.
        _tmpl = getattr(_doc, 'pageTemplate', None)
        _pw, _ph = (_tmpl.pagesize if _tmpl and _tmpl.pagesize
                     else _doc.pagesize)
        canvas.saveState()
        # Thin nav stripe at top
        canvas.setFillColor(NAVY)
        canvas.rect(0, _ph - 0.20*inch, _pw, 0.20*inch, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, _ph - 0.24*inch, _pw, 0.04*inch, fill=1, stroke=0)
        # Footer
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            0.55*inch, 0.35*inch,
            f"Confidential — {client_profile.get('client_name','Client')}  ·  "
            f"Prepared {_dt.now().strftime('%B %d, %Y')}"
        )
        canvas.drawRightString(
            _pw - 0.55*inch, 0.35*inch,
            f"Page {_doc.page}"
        )
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.55*inch, 0.50*inch, _pw - 0.55*inch, 0.50*inch)
        canvas.restoreState()

    def _on_first_page(canvas, _doc):
        _tmpl = getattr(_doc, 'pageTemplate', None)
        _pw, _ph = (_tmpl.pagesize if _tmpl and _tmpl.pagesize
                     else _doc.pagesize)
        # Cover page header — full-bleed solid navy band with a small
        # cream-filled logo box inset on the left, firm name to its
        # right, and advisor info right-aligned. Gold accent rule
        # beneath. Matches the thin navy + gold nav stripe drawn at
        # the top of every other page (see _on_page), just taller so
        # it can hold the branding content. The cover Frame (see
        # _make_cover_frame) starts below this band so flowable content
        # doesn't render inside the navy fill.

        # Resolve firm/advisor info from settings via closure — these
        # are bound in the outer function scope by the time this
        # callback fires during doc.build. Defensive try/except in
        # case the variables aren't populated for some edge case.
        try:
            _hdr_firm     = _firm_name or ""
            _hdr_adv_name = _adv_name or ""
            _hdr_adv_title= _adv_title or ""
            _hdr_email    = _adv_email or ""
            _hdr_phone    = _adv_phone or ""
            _hdr_website  = _firm_website or ""
            _hdr_logo_ok  = _has_logo
        except NameError:
            _hdr_firm = _hdr_adv_name = _hdr_adv_title = ""
            _hdr_email = _hdr_phone = _hdr_website = ""
            _hdr_logo_ok = False

        canvas.saveState()

        # ── Cover watermark: three-helmet mark, centered, drawn FIRST so
        # the navy band and all flowables render over it. Opacity is
        # baked into the PNG's alpha channel (ReportLab mask='auto'
        # honors it), so no canvas transparency state is needed.
        try:
            if os.path.exists(COVER_WATERMARK_PATH):
                _wm_w = 5.8 * inch
                _wm_h = _wm_w * 0.5457   # asset aspect ratio (h/w)
                canvas.drawImage(
                    COVER_WATERMARK_PATH,
                    (_pw - _wm_w) / 2.0,
                    (_ph - _wm_h) / 2.0 - 0.55 * inch,
                    _wm_w, _wm_h,
                    preserveAspectRatio=True,
                    mask='auto',
                )
        except Exception:
            pass

        # Wordmark color matches the body antique gold (ACCENT) so the
        # masthead's gold reads as the same color as the document's
        # other gold accents (PORTFOLIO & RISK PROFILE REVIEW title,
        # eyebrows, inner badge rings, etc.) rather than the lighter
        # champagne shade used previously. The accent stripe beneath
        # the navy band also uses ACCENT to match.

        # ── Header band: navy fill + gold accent rule below ──
        _band_h   = 1.15 * inch
        _accent_h = 0.05 * inch
        canvas.setFillColor(NAVY)
        canvas.rect(0, _ph - _band_h, _pw, _band_h, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, _ph - _band_h - _accent_h,
                    _pw, _accent_h, fill=1, stroke=0)

        # ── Cream logo box inset on the left ──
        _box_w = _box_h = 0.75 * inch
        _box_x = 0.30 * inch
        _box_y = _ph - _band_h + (_band_h - _box_h) / 2.0
        canvas.setFillColor(BG_SOFT)
        canvas.rect(_box_x, _box_y, _box_w, _box_h, fill=1, stroke=0)

        if _hdr_logo_ok:
            try:
                _img_pad = 3
                canvas.drawImage(
                    FIRM_LOGO_PATH,
                    _box_x + _img_pad,
                    _box_y + _img_pad,
                    _box_w - 2 * _img_pad,
                    _box_h - 2 * _img_pad,
                    preserveAspectRatio=True,
                    mask='auto',
                )
            except Exception:
                pass

        # ── Resolve wordmark font ──
        # Try to register Cormorant Garamond Medium from a few plausible
        # paths. Falls back to Times-Bold if the .ttf isn't found so the
        # PDF always builds. Drop CormorantGaramond-Medium.ttf into the
        # repo (root, ./fonts/, or alongside app.py) to enable it.
        _word_font = "Times-Bold"
        try:
            import os as _os
            from reportlab.pdfbase import pdfmetrics as _pdfm
            from reportlab.pdfbase.ttfonts import TTFont as _TTFont
            _font_name = "CormorantGaramond-Medium"
            try:
                _pdfm.getFont(_font_name)
                _word_font = _font_name
            except KeyError:
                _candidate_paths = [
                    "fonts/CormorantGaramond-Medium.ttf",
                    "CormorantGaramond-Medium.ttf",
                ]
                try:
                    _here = _os.path.dirname(_os.path.abspath(__file__))
                    _candidate_paths.extend([
                        _os.path.join(_here, "fonts",
                                       "CormorantGaramond-Medium.ttf"),
                        _os.path.join(_here,
                                       "CormorantGaramond-Medium.ttf"),
                    ])
                except (NameError, OSError):
                    pass
                for _p in _candidate_paths:
                    if _os.path.exists(_p):
                        try:
                            _pdfm.registerFont(_TTFont(_font_name, _p))
                            _word_font = _font_name
                            break
                        except Exception:
                            pass
        except Exception:
            pass

        # ── Wordmark + website: positioned LEFT, immediately to the
        # right of the logo box, instead of right-anchored. The wordmark
        # ("MRB CAPITAL GROUP") in Cormorant Garamond Medium (or
        # Times-Bold fallback) 22pt gold, with a gold 1.8pt underline
        # rule beneath it that spans the natural width of the text, and
        # the firm website rendered in gold italic (Helvetica-Oblique
        # 9pt) directly under the rule. Layout shifts the wordmark from
        # the right of the band to the left so the firm identity sits
        # as a unified anchor with the logo on the left half. ──
        _word_text = (_hdr_firm or "MRB CAPITAL GROUP").upper()
        _word_size = 22
        _word_x = _box_x + _box_w + 0.15 * inch
        _word_y = _ph - 0.50 * inch
        _word_natural_w = canvas.stringWidth(
            _word_text, _word_font, _word_size)
        canvas.setFillColor(ACCENT)
        canvas.setFont(_word_font, _word_size)
        canvas.drawString(_word_x, _word_y, _word_text)

        # Underline rule beneath the wordmark
        _rule_y = _word_y - 5
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.8)
        canvas.line(_word_x, _rule_y,
                    _word_x + _word_natural_w, _rule_y)

        # Website beneath the underline, italic gold Helvetica
        if _hdr_website:
            _site_y = _word_y - 20
            canvas.setFont("Helvetica-Oblique", 9)
            canvas.setFillColor(ACCENT)
            canvas.drawString(_word_x, _site_y, _hdr_website)

        # ── Advisor info stack: 3 lines right-anchored ──
        #   Line 1: "Name · Title" combined in cream Helvetica-Bold 11pt
        #   Line 2: email in cream Helvetica 8.5pt
        #   Line 3: phone in cream Helvetica 8.5pt
        # (Website intentionally omitted from the right stack — it now
        # lives under the wordmark on the left.)
        _right_x = _pw - 0.55 * inch
        canvas.setFillColor(colors.white)

        _adv_y = _ph - 0.42 * inch
        if _hdr_adv_name or _hdr_adv_title:
            if _hdr_adv_name and _hdr_adv_title:
                _line_adv = f"{_hdr_adv_name} · {_hdr_adv_title}"
            else:
                _line_adv = _hdr_adv_name or _hdr_adv_title
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawRightString(_right_x, _adv_y, _line_adv)

        if _hdr_email:
            canvas.setFont("Helvetica", 8.5)
            canvas.drawRightString(_right_x, _ph - 0.62 * inch, _hdr_email)
        if _hdr_phone:
            canvas.setFont("Helvetica", 8.5)
            canvas.drawRightString(_right_x, _ph - 0.78 * inch, _hdr_phone)

        # ── Footer ──
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            0.55*inch, 0.35*inch,
            f"Confidential — {client_profile.get('client_name','Client')}  ·  "
            f"Prepared {_dt.now().strftime('%B %d, %Y')}"
        )
        canvas.drawRightString(
            _pw - 0.55*inch, 0.35*inch,
            f"Page {_doc.page}"
        )
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.55*inch, 0.50*inch, _pw - 0.55*inch, 0.50*inch)
        canvas.restoreState()

    def _make_cover_frame(pagesize, name):
        """Frame for the cover page — top margin extended so flowable
        content starts BELOW the full-bleed navy header band drawn by
        _on_first_page. Band is 1.15" navy + 0.05" champagne accent;
        we add ~0.10" of breathing room before the first flowable."""
        _pw, _ph = pagesize
        _cover_top = 1.30 * inch
        return Frame(
            _margins['leftMargin'],
            _margins['bottomMargin'],
            _pw - _margins['leftMargin'] - _margins['rightMargin'],
            _ph - _cover_top - _margins['bottomMargin'],
            id=name, showBoundary=0,
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0,
        )

    _cover_tmpl = PageTemplate(
        id='cover',
        frames=[_make_cover_frame(_portrait_size, 'cover_frame')],
        pagesize=_portrait_size,
        onPage=_on_first_page,
    )
    _portrait_tmpl = PageTemplate(
        id='portrait',
        frames=[_make_frame(_portrait_size, 'portrait_frame')],
        pagesize=_portrait_size,
        onPage=_on_page,
    )
    _landscape_tmpl = PageTemplate(
        id='landscape',
        frames=[_make_frame(_landscape_size, 'landscape_frame')],
        pagesize=_landscape_size,
        onPage=_on_page,
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=_portrait_size,    # default; overridden per-template
        title=f"Portfolio Snapshot — {client_profile.get('client_name','Client')}",
        author="Portfolio Intelligence",
        **_margins,
    )
    doc.addPageTemplates([_cover_tmpl, _portrait_tmpl, _landscape_tmpl])

    # ── TYPOGRAPHY ─────────────────────────────────────────────
    # Phase 1: display headings move to Times-Roman serif for editorial
    # voice (cover title, section headers). Tracked-caps eyebrows stay in
    # Helvetica-Bold because serifs render poorly at <9pt with letter-
    # spacing. Body copy stays in Helvetica for readability — established
    # editorial pattern: serif headlines, sans body. Body bold/italic
    # variants stay Helvetica too.
    snapshot_title = ParagraphStyle(
        "snap_title", fontSize=26, leading=30, textColor=CHARCOAL,
        fontName="Times-Roman", alignment=TA_LEFT, spaceAfter=2,
    )
    h1 = ParagraphStyle(
        "h1", fontSize=17.6, leading=22, textColor=NAVY,
        fontName="Times-Roman", alignment=TA_LEFT,
        spaceBefore=8, spaceAfter=3,
    )
    h1_eyebrow = ParagraphStyle(
        "h1_eyebrow", fontSize=8, leading=10, textColor=ACCENT,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=2,
    )
    h2 = ParagraphStyle(
        "h2", fontSize=13, leading=16, textColor=NAVY,
        fontName="Times-Roman", alignment=TA_LEFT,
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
        """Crest medallion risk badge — gold concentric rings + serif numeral.

        Phase 2: replaces the legacy boxed "RISK N" badge with the crest
        design from the cover mockups. Two concentric rings (navy outer,
        gold inner) around a Times-Roman numeral, with an optional
        eyebrow above and "/ 99" below. Renders proportionally at any
        size — the same helper drives the 0.8" cover hero badge and the
        0.45" mini badges in tier comparison ribbons. `needle_color` is
        kept in the signature for backward compatibility with existing
        call sites but no longer affects the design (the crest is always
        navy + gold, since color is now a structural element of the
        brand, not a per-instance choice).

        Args:
            score: Integer risk score (1–99) or "—" / None for placeholder.
            label: Eyebrow text above the number. Defaults to "RISK".
                   Set to None/empty to omit the eyebrow (used at tiny sizes).
            size: Outer diameter in points. The whole badge scales
                  proportionally — fonts, ring thickness, padding all
                  derived from this single number.
            needle_color: Legacy param, ignored. Kept so existing call
                          sites don't need editing.

        Returns:
            ReportLab Drawing, sized (size × 1.05, size × 1.05) to give
            the outer ring a tiny bit of breathing room.
        """
        d = Drawing(size * 1.05, size * 1.05)
        cx, cy = size / 2, size / 2

        # Outer navy ring — thickness proportional to size so small
        # badges don't have hairline borders and large ones don't have
        # chunky ones. Cream fill inside the ring so the numeral sits
        # on a clean cream face rather than the page background
        # showing through.
        outer_stroke = max(0.8, size * 0.022)
        d.add(Circle(cx, cy, size / 2 - outer_stroke / 2,
                     strokeColor=NAVY, strokeWidth=outer_stroke,
                     fillColor=BG_SOFT))

        # Inner gold ring — sits inset from the outer edge. The gold
        # accent inside the navy frame mirrors the framed-plaque feel
        # used throughout the document's editorial chrome.
        inner_radius = size / 2 - (size * 0.08)
        d.add(Circle(cx, cy, inner_radius,
                     strokeColor=ACCENT, strokeWidth=0.8,
                     fillColor=None))

        # ── RISK GAUGE — quarter-arc gradient at top of badge ─────
        # A 90° arc spanning -45° to +45° (where 0° = top, clockwise
        # positive) gets a green→amber→red gradient overlaid on the
        # gold inner ring. A thin navy tick marks the score's position
        # on the gauge. The arc is rendered as N short line segments
        # with interpolated colors at each midpoint since ReportLab
        # graphics primitives don't natively gradient-fill an arc.
        # Gated on `show_chrome` so mini-badges (< 0.6") skip the
        # gauge to avoid visual clutter at small sizes.
        if size >= 0.6 * inch:
            _gauge_r       = inner_radius
            _gauge_stroke  = max(1.5, size * 0.035)
            _arc_start_deg = -45.0
            _arc_span_deg  = 90.0
            _stop_green    = colors.HexColor("#97C459")
            _stop_amber    = colors.HexColor("#FAC775")
            _stop_red      = colors.HexColor("#F09595")
            _n_seg = 60
            for _i in range(_n_seg):
                _t0 = _i / _n_seg
                _t1 = (_i + 1) / _n_seg
                _a0 = np.radians(_arc_start_deg + _t0 * _arc_span_deg)
                _a1 = np.radians(_arc_start_deg + _t1 * _arc_span_deg)
                _x0 = cx + _gauge_r * np.sin(_a0)
                _y0 = cy + _gauge_r * np.cos(_a0)
                _x1 = cx + _gauge_r * np.sin(_a1)
                _y1 = cy + _gauge_r * np.cos(_a1)
                _tmid = (_t0 + _t1) / 2
                if _tmid < 0.5:
                    _u = _tmid * 2
                    _ca, _cb = _stop_green, _stop_amber
                else:
                    _u = (_tmid - 0.5) * 2
                    _ca, _cb = _stop_amber, _stop_red
                _r = _ca.red   * (1 - _u) + _cb.red   * _u
                _g = _ca.green * (1 - _u) + _cb.green * _u
                _b = _ca.blue  * (1 - _u) + _cb.blue  * _u
                _seg = Line(_x0, _y0, _x1, _y1)
                _seg.strokeColor = colors.Color(_r, _g, _b)
                _seg.strokeWidth = _gauge_stroke
                _seg.strokeLineCap = 1   # round caps so segments blend
                d.add(_seg)

            # Navy tick at score position. Maps score 1..99 to angle
            # -45° → +45° linearly. Tick is a radial line from just
            # inside the gauge to just outside, perpendicular to the
            # arc at the score point. Skipped when score isn't a
            # parseable int (e.g. "—" placeholder).
            try:
                _sc_int = (
                    int(score) if score not in ("—", None, "") else None
                )
            except (ValueError, TypeError):
                _sc_int = None
            if _sc_int is not None:
                _sc_int = max(1, min(99, _sc_int))
                _frac = _sc_int / 99.0
                _t_ang = np.radians(
                    _arc_start_deg + _frac * _arc_span_deg)
                _tick_in  = _gauge_r - max(2.5, size * 0.04)
                _tick_out = _gauge_r + max(2.5, size * 0.04)
                _txi = cx + _tick_in  * np.sin(_t_ang)
                _tyi = cy + _tick_in  * np.cos(_t_ang)
                _txo = cx + _tick_out * np.sin(_t_ang)
                _tyo = cy + _tick_out * np.cos(_t_ang)
                _tick = Line(_txi, _tyi, _txo, _tyo)
                _tick.strokeColor = NAVY
                _tick.strokeWidth = 1.5
                _tick.strokeLineCap = 1
                d.add(_tick)

        # Resolve the score for display
        try:
            n = str(int(score)) if score not in ("—", None, "") else "—"
        except (ValueError, TypeError):
            n = "—"

        # Decide whether to show chrome (eyebrow text + "/ 99" footer)
        # based on size. At small sizes (mini badges) the chrome competes
        # with the numeral, so we show only the number. Threshold: ~0.6"
        # outer diameter. The two chrome elements are gated independently:
        # the eyebrow only renders when a label is provided, but the
        # "/ 99" footer renders whenever the badge is big enough — this
        # lets caller render bare-label side badges (e.g. TOLERANCE /
        # CAPACITY) that still show "/ 99" so the score reads as a ratio
        # consistent with the eyebrow'd center badge.
        show_chrome = size >= 0.6 * inch

        if show_chrome:
            # Eyebrow text — only when a label is provided. Tiny tracked
            # caps in navy, positioned above the numeral.
            if label:
                eyebrow_pt = max(5.5, size * 0.075)
                eyebrow_y = cy + size * 0.22
                d.add(String(cx, eyebrow_y, label.upper(),
                             fontName="Helvetica-Bold", fontSize=eyebrow_pt,
                             fillColor=NAVY, textAnchor="middle"))
            # Big serif numeral — centered. When the eyebrow is shown
            # (center badges with label="PROFILE"), the numeral sits at
            # ~36% of badge diameter so it fits between the eyebrow and
            # the "/ 99" footer. When there's NO eyebrow (side badges
            # like TOLERANCE / CAPACITY), the numeral can grow to fill
            # the freed space at the top of the circle — bumped to
            # ~46% of diameter so the badge doesn't look half-empty.
            num_pt = size * 0.46 if not label else size * 0.36
            num_y = cy - num_pt * 0.32   # vertical centering tweak
            d.add(String(cx, num_y, n,
                         fontName="Times-Roman", fontSize=num_pt,
                         fillColor=NAVY, textAnchor="middle"))
            # "/ 99" below the number — always shown at chrome sizes
            of_pt = max(5.5, size * 0.085)
            of_y = cy - size * 0.28
            d.add(String(cx, of_y, "/ 99",
                         fontName="Helvetica", fontSize=of_pt,
                         fillColor=GRAY_SOFT, textAnchor="middle"))
        else:
            # Compact form — just the numeral, centered in the badge
            num_pt = size * 0.48
            num_y = cy - num_pt * 0.32
            d.add(String(cx, num_y, n,
                         fontName="Times-Roman", fontSize=num_pt,
                         fillColor=NAVY, textAnchor="middle"))

        return d

    def portfolio_badge(score, label="PORTFOLIO", size=0.55*inch,
                        chrome=None, filled=False):
        """Square badge for PORTFOLIO risk scores.

        Counterpart to risk_badge() (the client crest). The shape rule
        across the document:
            circle = client (a person's risk profile)
            square = portfolio (a basket of securities)

        Two visual variants:
          • Outlined (default, filled=False): cream fill, navy outer
            edge, gold inner ring inset, navy numeral. Used for the
            current portfolio + side comparison cards.
          • Nested-frame (filled=True): the "crest" hero treatment —
            thick gold outer frame, cream gap, thick navy inner frame,
            cream interior holding a navy eyebrow + serif numeral +
            gold "/ 99". Used for the PROPOSED portfolio anywhere it
            needs to anchor as the headline option. The `filled` name
            is preserved for caller compatibility even though the new
            variant is no longer a solid navy fill.

        Chrome (eyebrow text + "/ 99" footer):
          • chrome=True:  always render chrome
          • chrome=False: never render chrome (just the numeral)
          • chrome=None (default): size-aware — chrome at ≥0.6 inch,
            bare numeral below. Matches risk_badge's threshold for
            cross-badge consistency.

        Args:
            score: Integer 1-99 or "—" / None for placeholder.
            label: Eyebrow text. Default "PORTFOLIO". Has no effect
                   if chrome is False (or auto-disabled at small size).
            size: Badge edge length.
            chrome: Tri-state override for chrome rendering — see above.
            filled: True for the nested-frame hero variant (PROPOSED
                    portfolio crest). Default False for the simple
                    outlined cream-fill variant.

        Returns:
            ReportLab Drawing, sized (size × 1.05, size × 1.05) to give
            the outer edge a tiny bit of breathing room.
        """
        W = size * 1.05
        H = size * 1.05
        d = Drawing(W, H)
        cx, cy = W / 2, H / 2

        bx = (W - size) / 2
        by = (H - size) / 2

        if filled:
            # NESTED-FRAME hero variant — concentric squares around a
            # cream interior. Both frames are stroke-only; the cream
            # fill comes from a backdrop rect that fills the whole
            # badge so the gap between frames also reads as cream.
            face_fill   = BG_SOFT      # carried for text-color logic
            num_color   = NAVY         # navy serif numeral
            eyebrow_col = NAVY         # navy bold eyebrow
            of_color    = ACCENT       # gold "/ 99"

            # Cream backdrop (full badge area)
            d.add(Rect(bx, by, size, size,
                       strokeColor=None, strokeWidth=0,
                       fillColor=BG_SOFT))

            # Outer GOLD frame — thick stroke, inset by half-stroke so
            # the stroke sits on the badge edge instead of bleeding
            # outside (ReportLab centers strokes on the path).
            gold_stroke = max(2.0, size * 0.055)
            d.add(Rect(bx + gold_stroke/2, by + gold_stroke/2,
                       size - gold_stroke, size - gold_stroke,
                       strokeColor=ACCENT, strokeWidth=gold_stroke,
                       fillColor=None))

            # Inner NAVY frame — inset from the gold frame by `gap`.
            # The cream backdrop shows through both the gap and the
            # navy frame's interior.
            gap = size * 0.10
            navy_stroke = max(2.0, size * 0.055)
            d.add(Rect(bx + gap + navy_stroke/2,
                       by + gap + navy_stroke/2,
                       size - 2*gap - navy_stroke,
                       size - 2*gap - navy_stroke,
                       strokeColor=NAVY, strokeWidth=navy_stroke,
                       fillColor=None))
        else:
            # OUTLINED variant — original cream face, navy outer edge,
            # gold inner ring inset. Top edge of the inner inset now
            # carries a green→amber→red gradient gauge (matching the
            # quarter-arc gauge on the round risk_badge counterpart)
            # with a navy tick at the score position. The bottom, left,
            # and right edges remain solid gold so the inset still
            # reads as a complete square frame.
            face_fill   = BG_SOFT
            num_color   = NAVY
            eyebrow_col = NAVY
            of_color    = GRAY_SOFT

            outer_stroke = max(0.8, size * 0.022)
            d.add(Rect(bx, by, size, size,
                       strokeColor=NAVY, strokeWidth=outer_stroke,
                       fillColor=face_fill))
            inset = size * 0.08

            # Inset rect bounds (top-left, top-right, bot-left, bot-right
            # in ReportLab y-up coords: by+inset is BOTTOM, by+size-inset
            # is TOP).
            _ix0 = bx + inset
            _ix1 = bx + size - inset
            _iy0 = by + inset
            _iy1 = by + size - inset
            _inset_w = _ix1 - _ix0
            # Gauge owns middle 80% of top edge; gold endcaps cover the
            # outer 10% on each side. Tick can only ride within the
            # gauge zone — it never crosses the gold endcaps.
            _ec_w = _inset_w * 0.10
            _gauge_x0 = _ix0 + _ec_w
            _gauge_x1 = _ix1 - _ec_w
            _gauge_w  = _gauge_x1 - _gauge_x0

            # Bottom / left / right edges — solid gold lines (always).
            d.add(Line(_ix0, _iy0, _ix1, _iy0,
                       strokeColor=ACCENT, strokeWidth=0.8))
            d.add(Line(_ix0, _iy0, _ix0, _iy1,
                       strokeColor=ACCENT, strokeWidth=0.8))
            d.add(Line(_ix1, _iy0, _ix1, _iy1,
                       strokeColor=ACCENT, strokeWidth=0.8))

            # Top edge — branches by size:
            #   • size >= 0.6": gold ENDCAPS on the outer 10% each
            #     side, with the gauge gradient strip filling the
            #     middle 80%. Tick rides within the gauge zone.
            #   • size < 0.6": ONE CONTINUOUS gold line across the
            #     entire top edge. The gauge gradient would be too
            #     small to read at compact sizes, and the prior
            #     endcaps-only treatment left a visible gap across
            #     the middle of the top edge — making the inset
            #     frame's gold look broken on small badges (the
            #     comparison-#2 and #3 cards). Per advisor.
            if size >= 0.6 * inch:
                d.add(Line(_ix0, _iy1, _gauge_x0, _iy1,
                           strokeColor=ACCENT, strokeWidth=0.8))
                d.add(Line(_gauge_x1, _iy1, _ix1, _iy1,
                           strokeColor=ACCENT, strokeWidth=0.8))
            else:
                d.add(Line(_ix0, _iy1, _ix1, _iy1,
                           strokeColor=ACCENT, strokeWidth=0.8))

            # Gauge gradient strip on middle 80% of top edge. Same
            # green→amber→red palette as the risk_badge gauge and
            # the Risk Spectrum bar. Rendered as N small rectangles
            # since ReportLab doesn't natively gradient-fill rects.
            if size >= 0.6 * inch:
                _stop_green = colors.HexColor("#97C459")
                _stop_amber = colors.HexColor("#FAC775")
                _stop_red   = colors.HexColor("#F09595")
                _gauge_h = max(2.2, size * 0.05)
                _gauge_y = _iy1 - _gauge_h / 2
                _n_slices = 50
                for _i in range(_n_slices):
                    _t = (_i + 0.5) / _n_slices
                    if _t < 0.5:
                        _u = _t * 2
                        _ca, _cb = _stop_green, _stop_amber
                    else:
                        _u = (_t - 0.5) * 2
                        _ca, _cb = _stop_amber, _stop_red
                    _r = _ca.red   * (1 - _u) + _cb.red   * _u
                    _g = _ca.green * (1 - _u) + _cb.green * _u
                    _b = _ca.blue  * (1 - _u) + _cb.blue  * _u
                    _slice_x = _gauge_x0 + _gauge_w * (_i / _n_slices)
                    _slice_w = _gauge_w / _n_slices + 0.4
                    d.add(Rect(_slice_x, _gauge_y,
                               _slice_w, _gauge_h,
                               fillColor=colors.Color(_r, _g, _b),
                               strokeColor=None,
                               strokeWidth=0))

                # Navy tick at score position. Tick crosses the gauge
                # strip vertically; ride confined to the middle 80%.
                try:
                    _sc_int = (
                        int(score) if score not in ("—", None, "") else None
                    )
                except (ValueError, TypeError):
                    _sc_int = None
                if _sc_int is not None:
                    _sc_int = max(1, min(99, _sc_int))
                    _frac = _sc_int / 99.0
                    _tx = _gauge_x0 + _frac * _gauge_w
                    _tick_half = _gauge_h / 2 + max(2.0, size * 0.035)
                    _tick = Line(_tx, _iy1 - _tick_half,
                                 _tx, _iy1 + _tick_half)
                    _tick.strokeColor = NAVY
                    _tick.strokeWidth = 1.5
                    _tick.strokeLineCap = 1
                    d.add(_tick)

        # Resolve the score for display
        try:
            n = str(int(score)) if score not in ("—", None, "") else "—"
        except (ValueError, TypeError):
            n = "—"

        # Chrome decision: explicit override wins, else size-aware.
        if chrome is None:
            show_chrome = size >= 0.6 * inch
        else:
            show_chrome = bool(chrome)

        if show_chrome:
            # Eyebrow — only when caller passed a label. Tighter
            # vertical offset on the filled (nested-frame) variant
            # because the inner navy frame eats ~10% on each side, so
            # the usable interior is ~80% of the outer badge.
            _eye_off = size * 0.20 if filled else size * 0.22
            _of_off  = size * 0.24 if filled else size * 0.28
            if label:
                eyebrow_pt = max(5.5, size * 0.075)
                d.add(String(cx, cy + _eye_off, label.upper(),
                             fontName="Helvetica-Bold", fontSize=eyebrow_pt,
                             fillColor=eyebrow_col, textAnchor="middle"))
            # Big serif numeral
            num_pt = size * 0.36
            d.add(String(cx, cy - num_pt * 0.32, n,
                         fontName="Times-Roman", fontSize=num_pt,
                         fillColor=num_color, textAnchor="middle"))
            # "/ 99" footer
            of_pt = max(5.5, size * 0.085)
            d.add(String(cx, cy - _of_off, "/ 99",
                         fontName="Helvetica", fontSize=of_pt,
                         fillColor=of_color, textAnchor="middle"))
        else:
            # Compact form — just the numeral, centered. Numeral fills
            # more of the face since there's no eyebrow/footer to make
            # room for. Scaled down slightly on the nested-frame variant
            # so the numeral doesn't crowd the inner navy frame.
            num_pt = size * (0.40 if filled else 0.48)
            d.add(String(cx, cy - num_pt * 0.32, n,
                         fontName="Times-Roman", fontSize=num_pt,
                         fillColor=num_color, textAnchor="middle"))

        return d

    def portfolio_badge_horizontal(score, width=1.00*inch, height=0.45*inch):
        """Horizontal pill variant of portfolio_badge — used in the
        three-card comparison row so the PROPOSED card's PORTFOLIO
        badge fits in the SAME vertical space as the side cards'
        compact score boxes. Without this, the much taller square
        chromed badge (0.85") inflated the header row height and
        pushed the proposed card's allocation bar / donut / legend
        ~32pt below the side cards' content, breaking horizontal
        alignment across the three cards.

        Layout (wider than tall):
            ┌──────────────────────────┐
            │ ▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  ← gauge stripe + tick
            │                          │
            │ PORTFOLIO         28     │  ← eyebrow + numeral
            └──────────────────────────┘

        Returns a Drawing sized (width × height) with no extra
        breathing-room padding (caller controls outer spacing via
        the surrounding Table).
        """
        W = width
        H = height
        d = Drawing(W, H)

        # Outer frame — cream fill, thin navy outline matching the
        # square badge's outlined variant.
        outer_stroke = max(0.6, H * 0.025)
        d.add(Rect(0, 0, W, H,
                   strokeColor=NAVY, strokeWidth=outer_stroke,
                   fillColor=BG_SOFT))

        # Gauge stripe across the top — same green→amber→red gradient
        # as the square chromed badge and the Risk Spectrum bar.
        # Inset from edges so it doesn't touch the outline.
        inset = max(1.5, H * 0.06)
        gauge_h = max(2.2, H * 0.10)
        gauge_y0 = H - inset - gauge_h          # bottom edge of stripe
        gauge_x0 = inset
        gauge_x1 = W - inset
        gauge_w  = gauge_x1 - gauge_x0

        _stop_green = colors.HexColor("#97C459")
        _stop_amber = colors.HexColor("#FAC775")
        _stop_red   = colors.HexColor("#F09595")
        _n_slices = 50
        for _i in range(_n_slices):
            _t = (_i + 0.5) / _n_slices
            if _t < 0.5:
                _u = _t * 2
                _ca, _cb = _stop_green, _stop_amber
            else:
                _u = (_t - 0.5) * 2
                _ca, _cb = _stop_amber, _stop_red
            _r = _ca.red   * (1 - _u) + _cb.red   * _u
            _g = _ca.green * (1 - _u) + _cb.green * _u
            _b = _ca.blue  * (1 - _u) + _cb.blue  * _u
            _slice_x = gauge_x0 + gauge_w * (_i / _n_slices)
            _slice_w = gauge_w / _n_slices + 0.4
            d.add(Rect(_slice_x, gauge_y0, _slice_w, gauge_h,
                       fillColor=colors.Color(_r, _g, _b),
                       strokeColor=None, strokeWidth=0))

        # Navy tick at score position. Tick rides the full gauge
        # span (1..99 maps across the inset-to-inset width) and
        # extends slightly above/below the stripe for visibility.
        try:
            _sc_int = (
                int(score) if score not in ("—", None, "") else None
            )
        except (ValueError, TypeError):
            _sc_int = None
        if _sc_int is not None:
            _sc_int = max(1, min(99, _sc_int))
            _frac = _sc_int / 99.0
            _tx = gauge_x0 + _frac * gauge_w
            _gauge_cy = gauge_y0 + gauge_h / 2
            _tick_half = gauge_h / 2 + max(1.5, H * 0.06)
            _tick = Line(_tx, _gauge_cy - _tick_half,
                         _tx, _gauge_cy + _tick_half)
            _tick.strokeColor = NAVY
            _tick.strokeWidth = 1.4
            _tick.strokeLineCap = 1
            d.add(_tick)

        # Resolve display string for the score.
        try:
            n = str(int(score)) if score not in ("—", None, "") else "—"
        except (ValueError, TypeError):
            n = "—"

        # Text row sits in the area below the gauge stripe. Vertically
        # center the baseline of both the eyebrow and the numeral on
        # the midpoint of that area (numeral baseline gets the usual
        # -fontSize*0.32 nudge so its visual center aligns; eyebrow
        # gets a smaller nudge proportional to its own size).
        text_area_top    = gauge_y0 - max(0.5, H * 0.02)
        text_area_bottom = inset
        text_cy = (text_area_top + text_area_bottom) / 2

        # "PORTFOLIO" eyebrow on the left
        eye_pt = max(5.0, H * 0.17)
        d.add(String(inset + 2, text_cy - eye_pt * 0.32, "PORTFOLIO",
                     fontName="Helvetica-Bold", fontSize=eye_pt,
                     fillColor=NAVY, textAnchor="start"))

        # Numeral on the right (Times-Roman serif, same as square
        # badge for cross-badge consistency).
        num_pt = max(11.0, H * 0.50)
        d.add(String(W - inset - 2, text_cy - num_pt * 0.32, n,
                     fontName="Times-Roman", fontSize=num_pt,
                     fillColor=NAVY, textAnchor="end"))

        return d

    def cover_spectrum_band(profile, current_score, total_width=7.4*inch,
                             align_pct=None, align_color=None):
        """Compact two-marker spectrum band for the cover page.

        Per advisor revision pass: numbers and marker symbols are all
        SIZED UP from the previous compact version, and the current-
        portfolio dot is now an EMPTY navy outline circle (no fill) so
        the gold profile tick visually "wins" as the target while the
        current-portfolio circle reads as the marker to move.

        Layout: PROFILE numeral sits ABOVE the band (gold, Times-Bold
        14pt); PORTFOLIO numeral sits BELOW (navy). The two captions on
        opposite sides of the band makes their identity unambiguous and
        lets them sit close together horizontally without colliding.

        A light cream-gray panel sits behind the gradient band to give
        the spectrum a clear container distinct from the surrounding
        page background.

        Shows the client's profile target (gold tick) and the current
        portfolio's score (empty navy circle) on a green→amber→red
        gradient.

        Args:
            profile: Client profile score (1–99), int or None.
            current_score: Current portfolio score (1–99), int or None.
            total_width: Usable width in points.

        Returns:
            ReportLab Drawing.
        """
        W = total_width
        # 60pt compact spectrum height. NOTE: this was briefly bumped
        # to 68pt to give the numerals more clearance, but the extra
        # 8pt pushed the "Your Current Portfolio" block off the bottom
        # of page 1 (the cover page has no spare vertical room), so
        # it's back to 60pt. The content-page Risk Spectrum
        # (spectrum_band) keeps the taller, roomier layout — only this
        # cover band must stay compact to hold page 1. Layout top to
        # bottom: legend row (eyebrow on left, legend keys on right) ·
        # profile numeral · gradient bar · portfolio numeral ·
        # endpoint labels.
        H = 60
        d = Drawing(W, H)

        # Background panel — pale cream surface (BG_SOFT). Reads as
        # "lifted card on the cover" rather than a contrasting fill;
        # the wrapping navy box around the drawing already supplies the
        # structural containment.
        _panel_fill = BG_SOFT
        d.add(Rect(0, 0, W, H,
                   fillColor=_panel_fill, strokeColor=None))

        # Gradient stops — saturated Tailwind 100-level tints rather
        # than the semantic_bg tokens.
        stop_green = colors.HexColor("#97C459")   # green-200
        stop_amber = colors.HexColor("#FAC775")   # amber-100
        stop_red   = colors.HexColor("#F09595")   # red-200

        # Caption row at the top — 10pt down from the drawing top so
        # the eyebrow has clear breathing room. Eyebrow color is NAVY
        # (was ACCENT) to match the new navy treatment of endpoint
        # labels below — anchors the spectrum chrome in the same color
        # family as the body navy text.
        _top_y = H - 10
        d.add(String(W * 0.03, _top_y, "RISK SPECTRUM",
                     fontName="Helvetica-Bold", fontSize=8,
                     fillColor=NAVY, textAnchor="start"))
        if align_pct is not None:
            _eyebrow_w = 4.6 * len("RISK SPECTRUM")
            _align_x = W * 0.03 + _eyebrow_w + 6
            _align_col = (colors.HexColor(align_color)
                          if align_color else NAVY)
            d.add(String(_align_x, _top_y, "·",
                         fontName="Helvetica", fontSize=8,
                         fillColor=GRAY, textAnchor="start"))
            d.add(String(_align_x + 8, _top_y,
                         f"{align_pct:.0f}% aligned",
                         fontName="Helvetica-Bold", fontSize=8,
                         fillColor=_align_col, textAnchor="start"))

        # Right-side legend — walks right→left so labels can stack next
        # to their corresponding symbols. Symbol-to-text gap is 3pt.
        legend_y = _top_y
        legend_x = W * 0.97
        # 1) Portfolio (rightmost): empty navy outline circle + label
        d.add(String(legend_x, legend_y, "Your portfolio",
                     fontName="Helvetica", fontSize=8,
                     fillColor=CHARCOAL, textAnchor="end"))
        _portfolio_text_w = 60
        _portfolio_dot_x = legend_x - _portfolio_text_w - 3
        d.add(Circle(_portfolio_dot_x, legend_y + 3, 3.5,
                     fillColor=None,
                     strokeColor=NAVY, strokeWidth=1.2))
        # 2) Profile: gold tick + "Your profile" label (to the left).
        _profile_text_right = _portfolio_dot_x - 12
        d.add(String(_profile_text_right, legend_y, "Your profile",
                     fontName="Helvetica", fontSize=8,
                     fillColor=CHARCOAL, textAnchor="end"))
        _profile_text_w = 50
        _profile_tick_x = _profile_text_right - _profile_text_w - 3
        d.add(Line(_profile_tick_x, legend_y - 2,
                   _profile_tick_x, legend_y + 7,
                   strokeColor=ACCENT, strokeWidth=2.0))

        # Gradient band — 10pt thick ribbon, centered vertically.
        band_left  = W * 0.05
        band_right = W * 0.95
        band_h     = 10
        band_y     = (H - band_h) / 2
        band_len   = band_right - band_left
        n_slices   = 80
        for i in range(n_slices):
            t = i / (n_slices - 1)
            if t < 0.5:
                u = t * 2
                a, b = stop_green, stop_amber
            else:
                u = (t - 0.5) * 2
                a, b = stop_amber, stop_red
            r = a.red   * (1 - u) + b.red   * u
            g = a.green * (1 - u) + b.green * u
            bl = a.blue * (1 - u) + b.blue * u
            slice_x = band_left + band_len * t
            slice_w = band_len / n_slices + 0.5
            d.add(Rect(slice_x, band_y, slice_w, band_h,
                       fillColor=colors.Color(r, g, bl),
                       strokeColor=None))

        # Profile tick — gold vertical line extending 4pt above and
        # below the band. Numeral above the tick in 11pt Times-Bold
        # gold (bumped from 9pt per advisor revision pass so the
        # profile score reads at the same weight class as the bigger
        # portfolio marker below).
        if profile is not None:
            try:
                pv = int(profile)
                pf_frac = min(99, max(1, pv)) / 99.0
                pf_x = band_left + band_len * pf_frac
                d.add(Line(pf_x, band_y - 4, pf_x, band_y + band_h + 4,
                           strokeColor=ACCENT, strokeWidth=2.5))
                d.add(String(pf_x, band_y + band_h + 6, str(pv),
                             fontName="Times-Bold", fontSize=11,
                             fillColor=ACCENT, textAnchor="middle"))
            except (ValueError, TypeError):
                pass

        # Current portfolio dot — EMPTY navy outline circle (no fill)
        # so the underlying gradient color shows through the marker.
        # Per advisor revision pass: radius bumped 5→7pt and stroke
        # 1.5→1.8pt so the marker scales with the bigger Risk Summary
        # circles above. Numeral 11pt Times-Bold (was 9pt) below band.
        if current_score is not None:
            try:
                cv = int(current_score)
                cf_frac = min(99, max(1, cv)) / 99.0
                cd_x = band_left + band_len * cf_frac
                d.add(Circle(cd_x, band_y + band_h / 2, 7,
                             fillColor=None,
                             strokeColor=NAVY, strokeWidth=1.8))
                d.add(String(cd_x, band_y - 12, str(cv),
                             fontName="Times-Bold", fontSize=11,
                             fillColor=NAVY, textAnchor="middle"))
            except (ValueError, TypeError):
                pass

        # Endpoint labels at the bottom — bold NAVY at 8.5pt (was
        # CHARCOAL 9.5pt). Navy ties the endpoints to the new navy
        # treatment of the RISK SPECTRUM eyebrow above, anchoring the
        # spectrum chrome in a single color family.
        d.add(String(band_left, 3, "More Conservative",
                     fontName="Helvetica-Bold", fontSize=8.5,
                     fillColor=NAVY, textAnchor="start"))
        d.add(String(band_right, 3, "More Aggressive",
                     fontName="Helvetica-Bold", fontSize=8.5,
                     fillColor=NAVY, textAnchor="end"))

        return d

    def speedometer_gauge(score, size_inches=1.0):
        """Risk scale indicator for the recommendation cards.

        Phase 3 (revised): the half-circle speedometer was too heavy.
        Replaced with a lighter "Option B" treatment: a thin horizontal
        track with two markers — a small gold vertical tick at the
        client's profile score, and a navy dot at this option's score.
        The reader can see at a glance both where the option sits on the
        1-99 scale AND how it compares to the client's target profile.
        The score numeral is NOT repeated here because the ribbon above
        the card already shows it prominently.

        Function name kept as speedometer_gauge for backward compatibility
        with existing call sites — only the visual treatment changed.

        Args:
            score: Integer 1-99 or "—" / None for placeholder.
            size_inches: Outer width of the drawing in inches.

        Returns:
            ReportLab Drawing. Short and wide.
        """
        W = size_inches * inch
        H = 28
        d = Drawing(W, H)

        # Resolve the score safely
        try:
            v = int(score) if score not in ("—", None, "") else None
        except (ValueError, TypeError):
            v = None

        # Client profile score — for the gold tick on the track. Read
        # from outer scope (client_score) so the tick reflects the actual
        # client this proposal is being built for. Falls back gracefully
        # if client_score is missing or non-numeric.
        try:
            profile = (int(client_score)
                       if client_score not in ("—", None, "") else None)
        except (ValueError, TypeError):
            profile = None

        # Caption above the track, on the left, in gold tracked caps
        d.add(String(W * 0.10, H - 8, "POSITION ON SCALE",
                     fontName="Helvetica-Bold", fontSize=6.5,
                     fillColor=ACCENT, textAnchor="start"))

        # Track — thin pill with rounded ends, sits below the caption
        track_y = H * 0.30
        track_h = 3
        track_left = W * 0.10
        track_right = W * 0.90
        track_len = track_right - track_left
        d.add(Rect(track_left, track_y, track_len, track_h,
                   fillColor=BORDER_SOFT, strokeColor=None,
                   rx=track_h/2, ry=track_h/2))

        # Profile tick — small gold vertical mark at the client's profile
        # score. Sits BEHIND the score dot so a perfectly-aligned option
        # (delta ≤ 1) still reads as "dot on the gold mark".
        if profile is not None:
            pf_frac = min(99, max(0, profile)) / 99.0
            pf_x = track_left + track_len * pf_frac
            d.add(Line(pf_x, track_y - 4, pf_x, track_y + track_h + 4,
                       strokeColor=ACCENT, strokeWidth=1.8))

        # Score dot — navy circle marker at the option's score
        if v is not None:
            frac = min(99, max(0, v)) / 99.0
            marker_x = track_left + track_len * frac
            d.add(Circle(marker_x, track_y + track_h/2, 4.5,
                         fillColor=NAVY, strokeColor=WHITE, strokeWidth=1.2))

        # Endpoint labels: muted "1" at left, "99" at right
        d.add(String(track_left, track_y - 9, "1",
                     fontName="Helvetica", fontSize=6.5,
                     fillColor=GRAY_SOFT, textAnchor="start"))
        d.add(String(track_right, track_y - 9, "99",
                     fontName="Helvetica", fontSize=6.5,
                     fillColor=GRAY_SOFT, textAnchor="end"))

        # Profile reference label on the right side of the caption row
        if profile is not None:
            d.add(String(W * 0.90, H - 8,
                         f"Your profile: {profile}",
                         fontName="Helvetica", fontSize=6.5,
                         fillColor=GRAY, textAnchor="end"))

        return d

    def pie_drawing(tickers, weights, size=2.2*inch):
        """Render a vector pie chart with the same styling as on-screen.

        Calls lump_to_other() first so portfolios with more than max_distinct
        holdings (configured in chart_palette.other_bucket, default 10) get
        a clean gray 'Other' wedge instead of an unreadable rainbow of
        similar colors. The Other slice is positioned at the END of the
        pie (smallest emphasis), and every other slice uses its stable
        per-ticker color via PDF_TICKER_COLOR().

        SCHD is always berry, GLDM always amber, etc. — the PDF and the
        on-screen client portal use the same visual key.
        """
        # Apply lump-to-Other rule. Returns sorted (desc) tickers/weights
        # with 'Other' at the end if any holdings were collapsed.
        ts, ws, has_other = lump_to_other(tickers, weights, _SETTINGS)
        if not ts:
            d = Drawing(size, size)
            d.add(String(size/2, size/2, "No data",
                         fontName="Helvetica", fontSize=9,
                         fillColor=GRAY, textAnchor="middle"))
            return d

        # Per-chart distinct color resolution — see resolve_chart_colors().
        # Guarantees no two wedges in this pie share a color while still
        # giving each ticker its preferred color where available.
        colors_list = resolve_chart_colors(ts)

        d = Drawing(size, size)
        p = Pie()
        p.x = 0
        p.y = 0
        p.width = size
        p.height = size
        p.data = ws
        p.labels = None              # labels rendered in separate legend
        p.slices.strokeColor = WHITE
        p.slices.strokeWidth = 1.6
        p.startAngle = 90
        p.direction = "clockwise"
        for i, c in enumerate(colors_list):
            p.slices[i].fillColor = c
        # Donut hole — 32% of diameter so the ring is thinner and closer
        # to the Schwab-style annular chart aesthetic (was 18%).
        d.add(p)
        cx, cy = size/2, size/2
        d.add(Circle(cx, cy, size * 0.32, fillColor=WHITE,
                     strokeColor=None, strokeWidth=0))
        return d

    def pie_legend_table(tickers, weights, max_rows_per_col=8):
        """Static legend rendered as a ReportLab Table: color swatch +
        ticker + weight. Mirrors the on-screen key column.

        Calls lump_to_other() so the legend exactly matches the pie. Auto-
        splits into 2 columns when > max_rows_per_col rows. The Other row
        (if present) sits at the end and renders in italic muted text with
        the label "OTHER" instead of the ticker symbol.
        """
        ts, ws, has_other = lump_to_other(tickers, weights, _SETTINGS)
        if not ts:
            return Paragraph("—", body_small)

        # Per-chart distinct color resolution — same call as pie_drawing
        # above with the same ticker list, so the legend's swatches
        # match the pie's wedges exactly.
        row_colors = resolve_chart_colors(ts)
        n = len(ts)

        # Styles for the Other row — italic + muted gray.
        body_other = ParagraphStyle(
            "body_other", parent=body_small,
            fontName="Helvetica-Oblique", textColor=GRAY,
        )
        body_pct_muted = ParagraphStyle(
            "body_pct_muted", parent=body_small, textColor=GRAY,
        )
        # The rollup bucket's ticker is the configured other_label
        # (commonly "Other holdings"); some code paths use the literal
        # "Other" instead. _row matches either and always displays the
        # row as "OTHER".
        try:
            _other_lbl = (_SETTINGS.get("chart_palette", {})
                          .get("other_bucket", {})
                          .get("other_label", "Other"))
        except AttributeError:
            _other_lbl = "Other"

        def _row(color, t, w):
            swatch = Drawing(10, 10)
            swatch.add(Rect(0, 0, 10, 10, fillColor=color,
                            strokeColor=None, strokeWidth=0))
            if t == "Other" or t == _other_lbl:
                # Rollup row label — "OTHER" in all caps (advisor
                # request) so it reads as a category tag rather than
                # the descriptive phrase "Other holdings".
                return [swatch,
                        Paragraph("OTHER", body_other),
                        Paragraph(f"{w:.2f}%", body_pct_muted)]
            return [swatch,
                    Paragraph(f"<b>{t}</b>", body_small),
                    Paragraph(f"{w:.2f}%", body_small)]

        # Build (ticker, weight) pairs for row construction
        data = list(zip(ts, ws))

        # One column
        if n <= max_rows_per_col:
            rows = [_row(row_colors[i], t, w) for i, (t, w) in enumerate(data)]
            tbl = Table(rows, colWidths=[0.22*inch, 0.95*inch, 0.6*inch])
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
        tbl = Table(merged, colWidths=[0.22*inch, 0.85*inch, 0.55*inch,
                                        0.15*inch,
                                        0.22*inch, 0.85*inch, 0.55*inch])
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
        """Section header treatment: serif title + thin navy rule.
        Eyebrow ("SECTION N") is no longer rendered — advisor removed
        section numbering from all page headers. The argument is kept
        in the signature for backward compatibility with existing
        callers (pass any string, it won't appear in the output)."""
        return KeepTogether([
            Paragraph(title, h1),
            thin_rule(NAVY, 0.6),
        ])

    # Shared style for section intro/description paragraphs. Used on
    # pages 2-3 (holdings detail), page 4 (recommendations), page 7
    # (historical backtest), page 8 (fee comparison), and any other
    # section that wants a small italic caption between the section
    # header rule and the section body. Helvetica-Oblique 8.5pt
    # charcoal, left-aligned. Tight spacing so the paragraph hugs the
    # rule above it.
    _intro_desc_style = ParagraphStyle(
        "intro_desc",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=CHARCOAL,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
    )

    # ── CORNER BADGE FLOWABLE ─────────────────────────────────────
    # A zero-height flowable that draws a portfolio_badge in the
    # upper-right corner of the current page via the canvas, without
    # affecting the text flow below. Lets the section header + rule +
    # description on pages 2/3 use the same identical spacing pattern
    # as page 4 (where there's no badge), instead of having the badge
    # stretch the title row's height and create a gap between the rule
    # and the description that the cell-embedded approach couldn't
    # fully eliminate.
    class _CornerBadge(Flowable):
        def __init__(self, score, label="PORTFOLIO", size_inches=0.71):
            Flowable.__init__(self)
            self.score = score
            self.label = label
            self.size_inches = size_inches

        def wrap(self, availWidth, availHeight):
            return (0, 0)

        def draw(self):
            from reportlab.graphics import renderPDF as _renderPDF
            canv = self.canv
            badge = portfolio_badge(
                score=self.score,
                label=self.label,
                size=self.size_inches * inch,
                chrome=True,
                filled=False,
            )
            page_w, page_h = canv._pagesize
            margin = 0.55 * inch
            badge_w = self.size_inches * inch
            badge_total_h = self.size_inches * inch * 1.05  # chrome ~5%
            # Target position: upper-right corner of page (page margins)
            target_x = page_w - margin - badge_w
            target_y = page_h - margin - badge_total_h
            # The canvas has been translated to the flowable's location
            # in the document flow; convert target page coordinates into
            # the current canvas coordinate system before drawing.
            abs_x, abs_y = canv.absolutePosition(0, 0)
            _renderPDF.draw(
                badge, canv,
                target_x - abs_x, target_y - abs_y,
            )

    def _section_header_with_badge(eyebrow, title, score,
                                    label="PORTFOLIO", intro_text=None):
        """Section header for pages 2/3 (Current/Proposed Holdings).

        Returns a LIST of flowables — callers must `story.extend(...)`
        rather than `story.append(...)`. Structure:

          [0] _CornerBadge — zero-size, draws the portfolio_badge in
              the upper-right corner of the page via the canvas.
          [1] KeepTogether([title, width-constrained rule]) — title in
              standard h1 with default spacing; the rule sits in a Table
              constrained to 6.55" so it stops short of the badge area.
              Spacing matches page 4's section_header exactly.
          [2] (if intro_text) description Table with spaceBefore=-4 —
              identical to page 4's intro paragraph treatment so the
              rule-to-description gap reads the same on both pages.

        Eyebrow argument is kept for backward compatibility but no
        longer rendered."""
        flowables = []
        # Float the badge in the upper-right corner via canvas — zero
        # text-flow impact.
        flowables.append(_CornerBadge(score, label, 0.71))

        # Title + rule, identical to section_header() spacing but with
        # a width-constrained rule that stops short of the badge area.
        # NOTE: the Table wrapper swallows the inner HRFlowable's own
        # spaceBefore/spaceAfter — those properties only apply when an
        # HRFlowable is a top-level flowable, not when nested in a cell.
        # So we re-apply them on the Table itself to reproduce the
        # 2pt-above / 6pt-below spacing that section_header() gets
        # for free from its bare HRFlowable.
        _hdr_rule = Table(
            [[thin_rule(NAVY, 0.6)]],
            colWidths=[6.55*inch],
            hAlign="LEFT",
        )
        _hdr_rule.setStyle(TableStyle([
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        _hdr_rule.spaceBefore = 2
        _hdr_rule.spaceAfter  = 6
        flowables.append(KeepTogether([
            Paragraph(title, h1),
            _hdr_rule,
        ]))

        # Description: same pattern as page 4 (Recommendations) so the
        # rule→description spacing matches exactly. Width-constrained
        # to 6.38" (matching the rule's reach minus the 12pt visual
        # gap that the rule otherwise has to the badge area).
        if intro_text:
            _desc_tbl = Table(
                [[Paragraph(intro_text, _intro_desc_style)]],
                colWidths=[6.38*inch],
                hAlign="LEFT",
            )
            _desc_tbl.setStyle(TableStyle([
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))
            _desc_tbl.spaceBefore = -4
            flowables.append(_desc_tbl)

        return flowables

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
                         "Max-return re-optimization within ±50% corridor"),
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

    # ── CURRENT-PORTFOLIO PRESENCE FLAG ───────────────────────────
    # Single source of truth for "does the client actually have a
    # current portfolio to show?" Keyed on the Step 2 snapshot only
    # (the same signal the Backtest and Notable Periods sections use):
    # a current portfolio exists iff the advisor selected one via the
    # "Client's current portfolio" picker AND it carries at least one
    # positive-weight holding. When this is False the proposal reflows
    # to a proposed-only document — every current-portfolio element is
    # suppressed:
    #   • page-1 "Your Current Portfolio" card (donut + holdings split)
    #   • page-1 Risk Alignment row + cover spectrum band (alignment is
    #     profile-vs-current; with no current there is nothing to align)
    #   • the dedicated "Current Holdings" detail page
    # This also gates out the legacy fall-back that silently rendered
    # the balanced tier (i.e. PROPOSED Option 1) masquerading as the
    # client's "current" holdings whenever Step 2 was left unset.
    _has_current_portfolio = bool(
        _curr_snap_tks_pg1
        and isinstance(_curr_snap_w_pg1, dict)
        and any(
            float(_curr_snap_w_pg1.get(_t, 0) or 0) > 0
            for _t in _curr_snap_tks_pg1
        )
    )

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

    # Effective text/tables for the customizable closing sections
    # (Advisor Notes, Implementation Plan, How This Proposal Was Built,
    # Key Definitions, Disclosures). Advisor edits from the PDF Content
    # tab layered over DEFAULT_PDF_CONTENT; untouched sections use the
    # standard wording.
    _pdf_content = get_pdf_content()

    _has_logo = os.path.exists(FIRM_LOGO_PATH)
    # v2.5 schema: nested under firm.* and advisor.*. Use _SETTINGS which is
    # the validated/loaded design system (from mrb_design.load_settings()
    # at the top of this function), since that's guaranteed to have the
    # right shape. Fall back to _firm_settings (raw JSON via legacy loader)
    # in case _SETTINGS keys are missing in some deployment.
    def _fs(*keys, default=""):
        """Chained .get() across nested settings dicts. Returns first
        truthy string found, or default. Tries _SETTINGS then _firm_settings
        so we get the validated value if available, otherwise raw."""
        for source in (_SETTINGS, _firm_settings):
            cur = source
            for k in keys:
                if not isinstance(cur, dict):
                    cur = None; break
                cur = cur.get(k)
            if cur and isinstance(cur, str) and cur.strip():
                return cur.strip()
        return default

    _firm_name     = _fs("firm",    "name")
    _adv_name      = _fs("advisor", "name")
    _adv_title     = _fs("advisor", "title")
    _adv_email     = _fs("advisor", "email")
    _adv_phone     = _fs("advisor", "phone")
    _firm_website  = _fs("firm",    "website")
    _has_any_firm_text = any([_firm_name, _adv_name, _adv_title,
                              _adv_email, _adv_phone, _firm_website])

    # ── COVER HEADER ─────────────────────────────────────────────
    # The full-bleed navy header band (with cream logo box, firm name,
    # and advisor info) is drawn directly by the _on_first_page canvas
    # callback so it can extend edge-to-edge of the paper. The cover
    # Frame's top is inset (see _make_cover_frame) to clear the band,
    # so the first flowable below lands cleanly beneath the gold
    # accent rule. _firm_name / _adv_* / _has_logo are still resolved
    # above so the callback can read them via closure.

    # Header row: big "Risk Alignment Summary" title on left, client vs current
    # portfolio badges on the right
    def _mini_badge_cell(score, label_top, label_bot, color, shape="circle"):
        """Big badge + label stack for the cover ribbon.

        shape parameter routes the visual:
            'circle' → risk_badge (client crest) — for client profile score
            'square' → portfolio_badge — for any portfolio score

        Renders the badge at 0.85" so the full chrome appears (eyebrow
        cap, big serif numeral, "/ 99" footer). The badge IS the
        primary visual; the side labels (PROFILE/PORTFOLIO + client
        identity) are supporting type."""
        badge = (portfolio_badge(score, label="PORTFOLIO", size=0.85*inch)
                 if shape == "square"
                 else risk_badge(score, label="PROFILE", size=0.85*inch,
                                  needle_color=color))
        return Table(
            [[badge,
              Table([[Paragraph(f"<font color='{GRAY.hexval()}' size='7'><b>"
                                f"{label_top.upper()}</b></font>", body_small)],
                     [Paragraph(f"<font color='{CHARCOAL.hexval()}' size='10'><b>"
                                f"{label_bot}</b></font>", body_small)]],
                    colWidths=[1.4*inch])]],
            colWidths=[0.95*inch, 1.4*inch],
        )
    # Orphaned legacy: _mini_badge_cell1 / _mini_badge_cell2 used to
    # appear in the cover's title ribbon as small client/portfolio
    # badges next to "Risk Alignment Summary". The new cover layout
    # uses a three-badge risk summary (Tolerance / Overall / Capacity)
    # instead and doesn't reference these. Kept defined for backward
    # compat in case downstream code or future revisions want them.
    _mini_badge_cell1 = _mini_badge_cell(client_score, "YOUR RISK",
                                          "Client Profile", NAVY,
                                          shape="circle")
    _mini_badge_cell2 = _mini_badge_cell(current_score, "CURRENT",
                                          "Your Portfolio", ACCENT,
                                          shape="square")

    # ── Risk Alignment (computed for downstream copy, no badge or card) ──
    # The previous design rendered both a third badge ("ALIGNMENT · +31")
    # in the header ribbon AND a tinted callout card with a left rule.
    # Both have been retired in favor of the cover_spectrum_band, which
    # visualizes the same gap (client tick + portfolio dot on the gradient)
    # without competing visual elements. We still compute the alignment
    # values here in case downstream sections want to reference them.
    _align_status    = None
    _align_detail    = None
    _align_signed_delta = None

    try:
        _cs = float(client_score)  if client_score  not in ("—", None, "") else None
        _ps = float(current_score) if current_score not in ("—", None, "") else None
    except (TypeError, ValueError):
        _cs = _ps = None

    if _cs is not None and _ps is not None:
        # pick_alignment_tier still gives us the status label, detail
        # sentence, and signed delta for any downstream consumer.
        _align_status, _align_detail, _color_key, _align_signed_delta = \
            pick_alignment_tier(int(_cs), int(_ps), _SETTINGS)

    # ── REPORT TITLE + PREPARED DATE ───────────────────────────────
    # Eyebrow title + prepared date, sitting above the client identity
    # block. Moved here from the firm branding band so all "who/what/
    # when this report is for" elements live in one continuous
    # letterhead-style group on the cover.
    # Title bumped from 14pt to 20pt per advisor request — reads as a
    # proper headline above the date and name, more presence on the
    # cover.
    _title_para = Paragraph(
        f"<font face='Helvetica-Bold' size='20' color='{ACCENT.hexval()}'>"
        f"PORTFOLIO &amp; RISK PROFILE REVIEW</font>",
        ParagraphStyle("hdr_title", fontSize=20, leading=24,
                       alignment=TA_LEFT, spaceBefore=2, spaceAfter=4),
    )
    _date_para = Paragraph(
        f"<font face='Helvetica' size='9.5' color='{GRAY.hexval()}'>"
        f"Prepared {_dt.now().strftime('%B %d, %Y')}</font>",
        ParagraphStyle("hdr_date", fontSize=9.5, leading=12,
                       alignment=TA_LEFT, spaceBefore=0, spaceAfter=2),
    )
    story.append(_title_para)
    story.append(_date_para)
    story.append(thin_rule(BORDER, 0.6))

    # ── CLIENT NAME ────────────────────────────────────────────────
    # Letterhead-style: client name in big serif. Age was previously
    # appended after the name with a middle-dot separator; advisor
    # asked to drop the age from the cover header (still available
    # in the underlying client_profile if needed downstream, just not
    # displayed). Keeps the cover line cleaner and more name-focused.
    _client_name = client_profile.get("client_name", "—") or "—"
    _name_html = (
        f"<font face='Times-Roman' size='20' color='{NAVY.hexval()}'>"
        f"{_client_name}</font>"
    )
    _name_para = Paragraph(
        _name_html,
        ParagraphStyle("cover_client_name", fontSize=20, leading=24,
                       textColor=NAVY, fontName="Times-Roman",
                       alignment=TA_LEFT, spaceBefore=4, spaceAfter=2),
    )
    story.append(Spacer(1, 0.05*inch))
    story.append(_name_para)

    # ── RETIREMENT AGE + HORIZON (under name+age, above priorities) ───
    # Per advisor request: always show the client's target retirement
    # age and, in parentheses, their retirement horizon (retirement age
    # minus current age). When the horizon is zero or negative — i.e.
    # the client is at or past their target retirement age — show
    # "Currently retired" instead of the forward-looking "Age X (Y
    # years)" line. Falls back to age 65 as a placeholder when no
    # retirement field is populated in the client_profile, so the row
    # always renders and the advisor can see at-a-glance that the
    # client profile needs the retirement target wired in.
    _retire_age = (
        client_profile.get("retirement_age")
        or client_profile.get("target_retirement_age")
        or client_profile.get("retire_age")
        or client_profile.get("retirement_target_age")
        or 65   # default — update client_profile.retirement_age to override
    )
    try:
        _ra_int = int(_retire_age)
        _horizon = None
        if _age_val not in (None, "", "—"):
            try:
                _horizon = _ra_int - int(_age_val)
            except (ValueError, TypeError):
                _horizon = None
        _ret_html = (
            f"<font color='{ACCENT.hexval()}' size='8'>"
            f"<b>RETIREMENT</b></font>&nbsp;&nbsp;&nbsp;"
            f"<font color='{CHARCOAL.hexval()}' size='10'>"
        )
        if _horizon is not None and _horizon <= 0:
            # Currently retired — horizon is zero or negative
            _ret_html += "Currently retired"
        else:
            _ret_html += f"Age {_ra_int}"
            if _horizon is not None and _horizon > 0:
                _years = "year" if _horizon == 1 else "years"
                _ret_html += (
                    f"&nbsp;&nbsp;"
                    f"<font color='{GRAY.hexval()}'>"
                    f"({_horizon} {_years})</font>"
                )
        _ret_html += "</font>"
        _ret_para = Paragraph(
            _ret_html,
            ParagraphStyle("retire_line", fontSize=10, leading=13,
                           textColor=CHARCOAL, fontName="Helvetica",
                           alignment=TA_LEFT, spaceBefore=0,
                           spaceAfter=2),
        )
        story.append(_ret_para)
    except (ValueError, TypeError):
        pass

    # ── PRIORITIES (right under name) ──────────────────────────────
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
    if _pcs:
        _chips_html = "&nbsp;&nbsp;&middot;&nbsp;&nbsp;".join(
            _PRIORITY_LBL.get(p, p) for p in _pcs
        )
        _priorities_para = Paragraph(
            f"<font color='{ACCENT.hexval()}' size='8'><b>PRIORITIES</b></font>"
            f"&nbsp;&nbsp;&nbsp;"
            f"<font color='{CHARCOAL.hexval()}' size='10'>"
            f"{_chips_html}</font>",
            ParagraphStyle("prio_under_name", fontSize=10, leading=14,
                           textColor=CHARCOAL, fontName="Helvetica",
                           alignment=TA_LEFT, spaceBefore=2, spaceAfter=4),
        )
        story.append(_priorities_para)
    story.append(thin_rule(BORDER, 0.6))

    # ── RISK SUMMARY ──────────────────────────────────────────────
    # Three bare badges (no in-circle labels — labels live below).
    # Three equal-width columns so the gaps between badges are
    # visually identical. The center badge is BIGGER than the side
    # badges but sits in a column of the SAME width, so its bigger
    # radius doesn't push the centers off-axis.
    _section_title_para = Paragraph(
        "Risk Summary",
        ParagraphStyle("risk_section_title", fontSize=16, leading=20,
                       textColor=CHARCOAL, fontName="Times-Roman",
                       alignment=TA_LEFT, spaceBefore=4, spaceAfter=2),
    )
    story.append(_section_title_para)

    _tol_score = client_profile.get("tolerance_score")
    _cap_score = client_profile.get("capacity_score")
    _ovr_score = client_profile.get("overall_score")
    # Ordering: LOWER score on the LEFT, HIGHER on the RIGHT
    try:
        _tol_int = int(_tol_score) if _tol_score not in (None, "—", "") else 0
        _cap_int = int(_cap_score) if _cap_score not in (None, "—", "") else 0
    except (ValueError, TypeError):
        _tol_int = _cap_int = 0
    if _tol_int <= _cap_int:
        _left_score, _left_label   = _tol_score, "TOLERANCE"
        _right_score, _right_label = _cap_score, "CAPACITY"
    else:
        _left_score, _left_label   = _cap_score, "CAPACITY"
        _right_score, _right_label = _tol_score, "TOLERANCE"

    # SHARED GRID: both Risk Summary and Alignment Summary use the
    # same 3-column structure with equal widths. Each column is now
    # 1.85" wide (totaling 5.55"), narrower than the previous 2.45"
    # so the side badges (TOLERANCE/CAPACITY in Risk Summary, PROFILE
    # round and PORTFOLIO square in Alignment Summary) pull inward
    # toward the center. The row's total 5.55" centers within the
    # 7.4" content area via the Table's hAlign='CENTER' attribute,
    # leaving 0.925" of margin on each side of the badge row.
    _GRID_COL_W = 1.85 * inch

    def _risk_badge_cell(score, is_center=False):
        """Build just the badge Drawing for a Risk Summary cell.

        Sized so the numerals read at a glance from across a desk.
        Center OVERALL SCORE badge is 1.25" with "PROFILE" eyebrow
        (was 1.10"); side TOLERANCE/CAPACITY badges are 0.95" bare
        numeral with "/ 99" footer (was 0.85"). Chrome gating in
        risk_badge handles eyebrow visibility based on size.
        """
        size_inches = 1.25 if is_center else 0.95
        _in_circle_label = "PROFILE" if is_center else ""
        return risk_badge(score, label=_in_circle_label,
                           size=size_inches*inch)

    def _risk_caption_cell(label, is_center=False):
        """Build just the caption Paragraph for a Risk Summary cell."""
        _cap_size = 10 if is_center else 8.5
        _cap_color = CHARCOAL if is_center else GRAY
        return Paragraph(
            f"<font size='{_cap_size}' color='{_cap_color.hexval()}'>"
            f"<b>{label}</b></font>",
            ParagraphStyle(f"badge_cap_{label}", fontSize=_cap_size,
                           leading=_cap_size + 2, textColor=_cap_color,
                           fontName="Helvetica-Bold",
                           alignment=TA_CENTER),
        )

    # Two-row layout: row 0 holds the badges (MIDDLE-aligned so their
    # visual CENTERS align regardless of size difference), row 1 holds
    # the captions (TOP-aligned so they share a baseline). This is the
    # only structure that gives BOTH badge-center alignment AND caption
    # baseline alignment when the badges are different sizes — the
    # previous single-row BOTTOM-aligned layout had captions aligned
    # but the smaller side badges visually "hung low" relative to the
    # bigger center badge, which the mockup explicitly shows centered.
    _risk_row = Table(
        [
            [_risk_badge_cell(_left_score),
             _risk_badge_cell(_ovr_score, is_center=True),
             _risk_badge_cell(_right_score)],
            [_risk_caption_cell(_left_label),
             _risk_caption_cell("OVERALL SCORE", is_center=True),
             _risk_caption_cell(_right_label)],
        ],
        colWidths=[_GRID_COL_W, _GRID_COL_W, _GRID_COL_W],
        hAlign="CENTER",
    )
    _risk_row.setStyle(TableStyle([
        # Row 0 (badges): MIDDLE align so the small/large badges
        # share a common visual centerline.
        ("VALIGN",        (0,0), (-1,0), "MIDDLE"),
        # Row 1 (captions): TOP align so the caption text starts at
        # the same y, putting all three baselines on one line.
        ("VALIGN",        (0,1), (-1,1), "TOP"),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,0), 2),
        ("BOTTOMPADDING", (0,0), (-1,0), 4),
        # Caption row: small top padding gives the badge → caption
        # gap (since the badge row has BOTTOMPADDING=4, this adds up
        # to ~10pt of separation, same as the previous sub-Table
        # design used).
        ("TOPPADDING",    (0,1), (-1,1), 6),
        ("BOTTOMPADDING", (0,1), (-1,1), 4),
    ]))
    story.append(_risk_row)
    story.append(Spacer(1, 0.04*inch))
    story.append(thin_rule(BORDER, 0.6))

    # ── ALIGNMENT SUMMARY ─────────────────────────────────────────
    # Uses the SAME 3-column grid as Risk Summary above so PROFILE
    # sits directly under TOLERANCE, the alignment % sits directly
    # under OVERALL SCORE, and PORTFOLIO sits directly under
    # CAPACITY. The vertical registration creates the visual story:
    # tolerance/capacity (your traits) → profile/portfolio (the
    # measurements that derived from them); overall score → the
    # alignment % that measures portfolio vs profile.
    _section_title_para_align = Paragraph(
        "Alignment Summary",
        ParagraphStyle("align_section_title", fontSize=16, leading=20,
                       textColor=CHARCOAL, fontName="Times-Roman",
                       alignment=TA_LEFT, spaceBefore=6, spaceAfter=4),
    )
    # Suppress the Alignment Summary header too when there is no current
    # portfolio — otherwise it strands above the (already-suppressed)
    # alignment row and spectrum band, leaving a lone title over blank space.
    if _has_current_portfolio:
        story.append(_section_title_para_align)

    # ── Compute alignment percentage ──
    # Formula: min(portfolio, profile) / max(portfolio, profile) × 100
    # Symmetric — treats over-risked and under-risked the same way.
    # Thresholds:
    #   ≥90% aligned → green  (saturated green-400 #639922)
    #   80–90%       → gold   (brand accent #C9A961)
    #   <80%         → red    (saturated red-400 #E24B4A)
    # These thresholds give the gold color a comfortable band where it
    # reads as "acceptable, not great" — most clients will land here.
    _align_pct = None
    _align_color = GRAY.hexval()
    _ALIGN_GREEN = "#639922"
    _ALIGN_GOLD  = ACCENT.hexval()
    _ALIGN_RED   = "#E24B4A"
    if _cs is not None and _ps is not None:
        try:
            _cs_i = int(_cs)
            _ps_i = int(_ps)
            if _cs_i > 0 and _ps_i > 0:
                _align_pct = (min(_cs_i, _ps_i) / max(_cs_i, _ps_i)) * 100.0
                if _align_pct >= 90:
                    _align_color = _ALIGN_GREEN
                elif _align_pct >= 80:
                    _align_color = _ALIGN_GOLD
                else:
                    _align_color = _ALIGN_RED
        except (ValueError, TypeError):
            _align_pct = None

    # Build the three cells of the alignment row.
    #
    # Column widths: the alignment row uses a narrow / wide / narrow
    # structure (not the 3-equal grid the Risk Summary row uses) so the
    # connector line in the center cell can visually meet the edges of
    # the PROFILE and PORTFOLIO badges. Total row width stays at
    # 3 × _GRID_COL_W (5.55") so the row stays centered within the
    # page's content area like the Risk Summary row above. The L/R
    # cells are widened in this pass to hold the larger 1.25"/1.15"
    # badges introduced in the advisor revision (was 1.00"/0.92"
    # holding 0.94"/0.85" badges).
    _ROW_W           = 3 * _GRID_COL_W              # 5.55"
    _ALIGN_LEFT_W    = 1.12 * inch                  # PROFILE badge cell
    _ALIGN_RIGHT_W   = 0.92 * inch                  # PORTFOLIO badge cell
    _ALIGN_CENTER_W  = _ROW_W - _ALIGN_LEFT_W - _ALIGN_RIGHT_W

    # LEFT cell (under TOLERANCE column): PROFILE badge.
    # Bumped 10% from the original 0.94" to 1.04" so the alignment
    # summary's PROFILE badge has a touch more weight; the PORTFOLIO
    # square next to it stays at 0.85" by design — only the profile
    # got the bump per the advisor's revision pass.
    _profile_badge_drawing = risk_badge(
        _cs if _cs is not None else "—",
        label="PROFILE", size=1.04*inch,
    )
    _align_left_cell = Table(
        [[_profile_badge_drawing]],
        colWidths=[_ALIGN_LEFT_W],
    )
    _align_left_cell.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))

    # CENTER cell — circular progress ring (Drawing) showing the
    # alignment percentage as a partial-arc fill. Replaces the earlier
    # text-only "81% ALIGNMENT" callout. Visual treatment:
    #   - Track ring: pale navy tint (#F0F2F6), full circumference
    #   - Progress arc: threshold color, sweeps clockwise from 12 o'clock
    #   - Centered numeral: "81%" in Times-Roman navy
    #   - Below numeral: "ALIGNED" small caps in threshold color
    def _alignment_ring(pct, color_hex, size=0.95*inch):
        """Circular progress ring for the alignment percentage.

        Args:
            pct: Alignment percentage (0-100), or None for placeholder.
            color_hex: Hex string for the progress arc + label color.
            size: Outer diameter in points.
        """
        S = size
        d = Drawing(S, S)
        cx, cy = S / 2, S / 2
        outer_r = S / 2 - 1.5
        # Ring thickness: 6% of the badge diameter (was 10%) — thinner
        # band reads as a more delicate progress indicator and gives
        # the centered percentage numeral more visual room.
        ring_w = max(3.5, S * 0.06)
        # Track ring
        d.add(Circle(cx, cy, outer_r - ring_w / 2,
                     fillColor=None,
                     strokeColor=colors.HexColor("#F0F2F6"),
                     strokeWidth=ring_w))
        # Progress arc — built as an annular Wedge (outer radius =
        # outer_r, inner radius = outer_r - ring_w). `annular=True` is
        # required for the wedge to render as a ring band rather than
        # a filled pie slice. ReportLab's Wedge sweeps anticlockwise
        # from start to end angles; to sweep CLOCKWISE from 12 o'clock
        # we pass start=90 and end = 90 - sweep.
        if pct is not None:
            sweep_deg = max(0.1, min(360.0, float(pct) * 3.6))
            _start = 90.0
            _end   = 90.0 - sweep_deg
            d.add(Wedge(cx, cy, outer_r,
                        _end, _start,
                        radius1=outer_r - ring_w,
                        annular=True,
                        fillColor=colors.HexColor(color_hex),
                        strokeColor=None))
        # Centered percentage numeral
        if pct is not None:
            _pct_str = f"{pct:.0f}%"
            _pt = S * 0.28
            d.add(String(cx, cy + S * 0.01, _pct_str,
                         fontName="Times-Bold", fontSize=_pt,
                         fillColor=NAVY, textAnchor="middle"))
            # "ALIGNED" label in threshold color below the number
            d.add(String(cx, cy - S * 0.21, "ALIGNED",
                         fontName="Helvetica-Bold", fontSize=max(5.5, S * 0.085),
                         fillColor=colors.HexColor(color_hex),
                         textAnchor="middle"))
        else:
            d.add(String(cx, cy, "—",
                         fontName="Times-Roman", fontSize=S * 0.30,
                         fillColor=GRAY, textAnchor="middle"))
        return d

    # ── CENTER cell: alignment-as-connector ─────────────────────────
    # Replaces the previous standalone progress ring. The new treatment
    # makes the middle slot a visual bridge BETWEEN the PROFILE and
    # PORTFOLIO badges rather than a third standalone badge:
    #     [gold eyebrow:  N-POINT GAP]
    #     ─── ⃝ ───   (thin navy line with an open circle interrupt)
    #     [navy caption: NN% aligned with profile]
    # The badges remain the visual anchors; this connector quantifies the
    # relationship without competing for attention. _align_pct and
    # _align_color are still computed above so the spectrum band can
    # downstream-consume the same alignment values.

    # Gap in raw score points (e.g. PROFILE 37 vs PORTFOLIO 30 → "7-POINT GAP")
    _align_gap = None
    try:
        if _cs is not None and _ps is not None:
            _align_gap = abs(int(_cs) - int(_ps))
    except (ValueError, TypeError):
        _align_gap = None

    if _align_gap is not None:
        _gap_label = f"{_align_gap}-POINT GAP"
    else:
        _gap_label = "ALIGNMENT"

    _gap_eyebrow_para = Paragraph(
        f"<font face='Helvetica-Bold' size='13.5' color='{ACCENT.hexval()}'>"
        f"{_gap_label}</font>",
        ParagraphStyle("align_gap_eyebrow", fontSize=13.5, leading=16,
                       alignment=TA_CENTER, spaceBefore=0, spaceAfter=3),
    )

    if _align_pct is not None:
        _caption_html = (
            f"<font face='Helvetica' size='10' color='{NAVY.hexval()}'>"
            f"<b>{_align_pct:.0f}%</b> aligned with profile</font>"
        )
    else:
        _caption_html = (
            f"<font face='Helvetica' size='10' color='{GRAY.hexval()}'>"
            f"Alignment not available</font>"
        )
    _align_caption_para = Paragraph(
        _caption_html,
        ParagraphStyle("align_caption", fontSize=10, leading=12,
                       alignment=TA_CENTER, spaceBefore=2, spaceAfter=0),
    )

    def _alignment_line(width):
        """Draws the horizontal connector line interrupted by a small
        open circle at center — the visual bridge between the PROFILE
        and PORTFOLIO badges."""
        line_h = 12.0
        d = Drawing(width, line_h)
        cy = line_h / 2.0
        cx = width / 2.0
        circle_r = 3.5
        gap_around = 5.0   # blank space on each side of the open circle
        d.add(Line(0, cy, cx - circle_r - gap_around, cy,
                   strokeColor=NAVY, strokeWidth=1.0))
        d.add(Line(cx + circle_r + gap_around, cy, width, cy,
                   strokeColor=NAVY, strokeWidth=1.0))
        d.add(Circle(cx, cy, circle_r,
                     fillColor=BG_SOFT, strokeColor=NAVY,
                     strokeWidth=1.0))
        return d

    _align_center_cell = Table(
        [[_gap_eyebrow_para],
         [_alignment_line(_ALIGN_CENTER_W)],
         [_align_caption_para]],
        colWidths=[_ALIGN_CENTER_W],
    )
    _align_center_cell.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))

    # RIGHT cell (under CAPACITY column): PORTFOLIO badge
    # Bumped from 0.85" to 1.15" in the advisor revision pass to match
    # the larger Risk Summary side badges above (1.15") — the visual
    # weight of the PORTFOLIO score should equal that of TOLERANCE /
    # CAPACITY since it sits in the same column under CAPACITY.
    _portfolio_badge_drawing = portfolio_badge(
        _ps if _ps is not None else "—",
        label="PORTFOLIO", size=0.85*inch,
    )
    _align_right_cell = Table(
        [[_portfolio_badge_drawing]],
        colWidths=[_ALIGN_RIGHT_W],
    )
    _align_right_cell.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))

    _align_row = Table(
        [[_align_left_cell, _align_center_cell, _align_right_cell]],
        colWidths=[_ALIGN_LEFT_W, _ALIGN_CENTER_W, _ALIGN_RIGHT_W],
        hAlign="CENTER",
    )
    _align_row.setStyle(TableStyle([
        # MIDDLE so the ALIGNMENT % text in the center column shares
        # a horizontal centerline with the PROFILE (circle) and
        # PORTFOLIO (square) badges on either side. The previous
        # BOTTOM align baseline-aligned the badges but pushed the
        # alignment % text low, breaking the visual symmetry the
        # advisor wanted (badges and % at the same vertical level).
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ]))
    # Alignment is a profile-vs-current comparison; suppress the whole
    # row when there is no current portfolio (otherwise it degrades to a
    # "—" PORTFOLIO badge and an "Alignment not available" caption).
    if _has_current_portfolio:
        story.append(_align_row)
        story.append(Spacer(1, 0.10*inch))

    # ── COVER SPECTRUM BAND ──────────────────────────────────────
    # Wrapped in a navy-boxed Table that matches the Risk Spectrum
    # panel on page 3 — gives the spectrum a contained visual frame
    # rather than letting it float on the page. The thin separator
    # rule that previously sat between the alignment row and the
    # spectrum is no longer needed; the navy box does the separation.
    if _has_current_portfolio and (_cs is not None or _ps is not None):
        _cover_band_drawing = cover_spectrum_band(
            profile=int(_cs) if _cs is not None else None,
            current_score=int(_ps) if _ps is not None else None,
            total_width=7.4*inch,
            align_pct=_align_pct,
            align_color=_align_color,
        )
        _spec_wrapper = Table([[_cover_band_drawing]], colWidths=[7.4*inch])
        _spec_wrapper.setStyle(TableStyle([
            # No BACKGROUND override here — the Drawing's own panel fill
            # (cover_spectrum_band paints a #F0F2F6 rect across its full
            # canvas) provides the shaded background. A WHITE background
            # here would show as slivers between the panel and the navy
            # box if the drawing didn't perfectly match the cell.
            ("BOX",           (0,0), (-1,-1), 1.0, NAVY),
            # Zero padding so the Drawing's panel reaches the navy box
            # edges. Previously 8pt L/R padding left visible white
            # slivers between the cream panel and the navy frame.
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(_spec_wrapper)
        story.append(Spacer(1, 0.06*inch))

    # ═══════════════════════════════════════════════════════════
    # INVESTMENT GOAL SECTION (opt-in — sections["goal"])
    # ═══════════════════════════════════════════════════════════
    # Advisor-toggled block that frames the proposal around a funding
    # goal. Sits in the page-1 "alignment slot": when there is no current
    # portfolio it fills the space the alignment row vacated; when there
    # is one it renders just beneath the alignment summary. Independent of
    # _has_current_portfolio — inclusion is purely the advisor's choice.
    # Goal data arrives via client_profile["goal"] (injected by the PDF
    # builder UI). The whole block is wrapped in try/except so a bad input
    # degrades to a one-line note rather than taking down the PDF.
    if sections.get("goal") and (client_profile or {}).get("goal"):
        try:
            _goal = dict(client_profile.get("goal") or {})
            _gm = _compute_goal_metrics(_goal)

            _g_label    = (_goal.get("label") or _goal.get("type") or "your goal").strip()
            _g_is_ret   = bool(_gm.get("is_retirement"))
            _g_years    = _gm.get("years_to_ret")
            _g_start    = _gm.get("starting") or 0.0
            _g_monthly  = _gm.get("monthly") or 0.0
            _g_ret      = _gm.get("accum_return") or 0.0
            _g_target   = _gm.get("target") or 0.0
            _g_fv       = _gm.get("projected") or 0.0
            _g_fund_pct = _gm.get("funding_pct")
            _g_gap      = _gm.get("gap")
            _g_req_pct  = _gm.get("required_return_pct")
            _g_tage     = _goal.get("target_age")
            _g_concept3 = bool(_g_is_ret and _g_target)

            # ── Section header ──
            story.append(Paragraph(
                "Your Retirement Goal" if _g_is_ret else "Your Investment Objective",
                ParagraphStyle("goal_section_title", fontSize=16, leading=20,
                               textColor=CHARCOAL, fontName="Times-Roman",
                               alignment=TA_LEFT, spaceBefore=6, spaceAfter=4),
            ))
            story.append(thin_rule(BORDER, 0.6))
            story.append(Spacer(1, 0.06*inch))

            # ── Navy callout: advisor narrative or auto-composed sentence ──
            _g_narr = (_goal.get("narrative") or "").strip()
            if not _g_narr:
                if _g_is_ret:
                    _g_narr = (f"Replacing {_gm.get('replacement_pct') or 0:.0f}% of "
                               f"{_goal_fmt_money(_gm.get('current_income'))} income")
                    if _g_tage is not None:
                        _g_narr += f" in retirement at {int(_g_tage)}"
                    if _gm.get("gap_at_ret") is not None:
                        _g_narr += (f" means about {_goal_fmt_money(_gm.get('gap_at_ret'))}/yr "
                                    f"from the portfolio once inflation-adjusted")
                    if _g_target:
                        _g_narr += f" — a {_goal_fmt_money(_g_target)} nest egg"
                        if _gm.get("retirement_years"):
                            _g_narr += f" to fund {int(_gm['retirement_years'])} years"
                    _g_narr += "."
                else:
                    _g_narr = f"Building toward {_goal_fmt_money(_g_target)} for {_g_label.lower()}"
                    if _g_tage is not None:
                        _g_narr += f" at age {int(_g_tage)}"
                    if _g_years:
                        _g_narr += f" — a {_g_years}-year horizon"
                    _g_narr += "."
                if _g_fund_pct is not None:
                    _g_narr += (f" On the proposed plan, projected to reach "
                                f"~{_goal_fmt_money(_g_fv)} ({_g_fund_pct:.0f}% of target)")
                    _g_narr += (f", about {_goal_fmt_money(_g_gap)} short."
                                if (_g_gap is not None and _g_gap > 0) else ".")
            _g_callout = Table(
                [[Paragraph(
                    f"<font color='{WHITE.hexval()}' size='11'>{_g_narr}</font>",
                    ParagraphStyle("goal_callout", fontSize=11, leading=15,
                                   textColor=WHITE, fontName="Helvetica",
                                   alignment=TA_LEFT),
                )]],
                colWidths=[7.4*inch],
            )
            _g_callout.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), NAVY),
                ("LEFTPADDING",   (0,0), (-1,-1), 12),
                ("RIGHTPADDING",  (0,0), (-1,-1), 12),
                ("TOPPADDING",    (0,0), (-1,-1), 9),
                ("BOTTOMPADDING", (0,0), (-1,-1), 9),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ]))
            if not _g_concept3:
                story.append(_g_callout)
                story.append(Spacer(1, 0.10*inch))

            if _g_is_ret and _g_target:
                # ═══ CONCEPT 3 — retirement "runway to goal" ═══
                _g_surplus = (_g_gap is not None and _g_gap <= 0)
                _RED    = colors.HexColor("#a23b2e")
                _HATCH  = colors.HexColor("#d3c7ac")
                _GAPTXT = colors.HexColor("#7a6a3f")

                # Headline row: Projected | shortfall/% | Nest egg target.
                def _hl_cell(_lbl, _val, _vcolor, _al, _sub=None, _vsize=20, _subsize=8):
                    _a = {"L": TA_LEFT, "C": TA_CENTER, "R": TA_RIGHT}[_al]
                    _stack = [
                        Paragraph(f"<font color='{GRAY.hexval()}' size='7'>{_lbl}</font>",
                                  ParagraphStyle("hl_l", fontSize=7, leading=10,
                                                 fontName="Helvetica", alignment=_a)),
                        Paragraph(f"<font color='{_vcolor.hexval()}' size='{_vsize}'><b>{_val}</b></font>",
                                  ParagraphStyle("hl_v", fontSize=_vsize, leading=_vsize + 3,
                                                 fontName="Helvetica", alignment=_a)),
                    ]
                    if _sub is not None:
                        _stack.append(Paragraph(
                            f"<font color='{GRAY.hexval()}' size='{_subsize}'>{_sub}</font>",
                            ParagraphStyle("hl_s", fontSize=_subsize, leading=_subsize + 3,
                                           fontName="Helvetica", alignment=_a)))
                    _c = Table([[_p] for _p in _stack], colWidths=[2.46*inch])
                    _c.setStyle(TableStyle([
                        ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
                        ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),1),
                    ]))
                    return _c

                _pct_txt = (f"{_g_fund_pct:.0f}%"
                            if _g_fund_pct is not None else "\u2014")
                _hl_row = Table([[
                    _hl_cell("PROJECTED WITH PLAN", _goal_fmt_money(_g_fv), NAVY, "L"),
                    _hl_cell("FUNDED", _pct_txt, NAVY, "C", _vsize=26),
                    _hl_cell("PROJECTED TARGET", _goal_fmt_money(_g_target), ACCENT, "R"),
                ]], colWidths=[2.46*inch, 2.48*inch, 2.46*inch])
                _hl_row.setStyle(TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),6),
                ]))
                story.append(_hl_row)

                # Runway bar: navy funded portion + diagonally-hatched gap.
                _rw_w = 7.4 * inch
                _rw_h = 22
                _rw = Drawing(_rw_w, 28)
                _frac = max(0.0, min(1.0, _g_fv / _g_target)) if _g_target else 0.0
                _rw.add(Rect(0, 3, _rw_w, _rw_h, rx=6, ry=6,
                             fillColor=BORDER_SOFT, strokeColor=None))
                _gap_x0 = _rw_w * _frac
                if _gap_x0 < _rw_w - 1:
                    _hx = _gap_x0 + 6
                    while _hx < _rw_w:
                        _rw.add(Line(max(_gap_x0, _hx - _rw_h), 3,
                                     min(_hx, _rw_w), 3 + _rw_h,
                                     strokeColor=_HATCH, strokeWidth=1.2))
                        _hx += 7
                if _frac > 0:
                    _rw.add(Rect(0, 3, _rw_w * _frac, _rw_h, rx=6, ry=6,
                                 fillColor=NAVY, strokeColor=None))
                if _frac > 0.22:
                    _rw.add(String(12, 3 + _rw_h / 2.0 - 4, "Saved + projected",
                                   fontName="Helvetica", fontSize=9,
                                   fillColor=WHITE, textAnchor="start"))
                if (not _g_surplus) and (1.0 - _frac) > 0.18:
                    _rw.add(String(_rw_w - 12, 3 + _rw_h / 2.0 - 4, "gap to fund",
                                   fontName="Helvetica", fontSize=8,
                                   fillColor=_GAPTXT, textAnchor="end"))
                story.append(_rw)

                # Sub-labels under the runway.
                def _rw_sub(_txt, _al):
                    _a = {"L": TA_LEFT, "C": TA_CENTER, "R": TA_RIGHT}[_al]
                    return Paragraph(f"<font color='{GRAY.hexval()}' size='11'>{_txt}</font>",
                                     ParagraphStyle("rw_s", fontSize=11, leading=14,
                                                    fontName="Helvetica", alignment=_a))
                _mid_txt = ""
                if _g_years:
                    _mid_txt = f"{_g_years} yrs"
                    if _g_ret > 0:
                        _mid_txt += f" &middot; {_g_ret*100:.1f}%/yr assumed"
                    _cg = _gm.get("contribution_growth") or 0
                    if _cg > 0:
                        _mid_txt += f" &middot; +{_cg*100:.1f}%/yr"
                _right_txt = (f"Reaches target at ~{_g_req_pct:.1f}%/yr"
                              if _g_req_pct is not None else "")
                # Width-aware split: size the middle column to its rendered
                # text (+pad) so it stays on one line, then give the left and
                # right columns equal remaining width — which keeps the middle
                # label centered under the bar.
                from reportlab.pdfbase.pdfmetrics import stringWidth as _sw11
                _mid_plain = (_mid_txt.replace("&middot;", "\u00b7")
                                      .replace("&nbsp;", " ")
                                      .replace("&minus;", "\u2212"))
                _mid_w = min(max(_sw11(_mid_plain, "Helvetica", 11) + 14,
                                 2.2 * inch), 3.5 * inch)
                _side_w = (7.4 * inch - _mid_w) / 2.0
                _rw_subs = Table([[
                    _rw_sub(f"Today &middot; {_goal_fmt_money(_g_start)} saved", "L"),
                    _rw_sub(_mid_txt, "C"),
                    _rw_sub(_right_txt, "R"),
                ]], colWidths=[_side_w, _mid_w, _side_w])
                _rw_subs.setStyle(TableStyle([
                    ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),0),
                ]))
                story.append(_rw_subs)

                # Savings-lever summary: the current plan's monthly
                # contribution alongside the extra monthly saving that would
                # close the shortfall to target (companion to the required-
                # return figure at the right of the sub-labels above).
                _g_gap_mo = _gm.get("gap_monthly")
                _sv_parts = []
                if _g_monthly and _g_monthly > 0:
                    _now_mo = round(_g_monthly / 10.0) * 10
                    _sv_parts.append(
                        f"<font color='{GRAY.hexval()}'>Current savings</font> "
                        f"<b>${_now_mo:,.0f}/mo</b>")
                if _g_gap_mo and _g_gap_mo > 0:
                    _gap_mo_amt = round(_g_gap_mo / 10.0) * 10
                    _sv_parts.append(
                        f"<font color='{GRAY.hexval()}'>monthly shortfall</font> "
                        f"<b>${_gap_mo_amt:,.0f}/mo</b>")
                if _sv_parts:
                    story.append(Spacer(1, 0.05 * inch))
                    story.append(Paragraph(
                        f"<font color='{NAVY.hexval()}' size='10'>"
                        f"{' &nbsp;&middot;&nbsp; '.join(_sv_parts)}</font>",
                        ParagraphStyle("sv_mo", fontSize=10, leading=13,
                                       fontName="Helvetica", textColor=NAVY,
                                       alignment=TA_CENTER)))
                story.append(Spacer(1, 0.12*inch))

                # ── Income-need derivation table ──────────────────
                # Shows how the retirement income need is built, in TODAY's
                # dollars next to the same figures grown to retirement by
                # inflation. The chain is: income need (replacement % of
                # current income), less the Social Security estimate, leaving
                # the gap the portfolio must fund. This replaced the old
                # horizontal stat strip so each line reads label → amount(s)
                # and the inflation impact is visible side by side.
                def _full_money(_v):
                    try:
                        return f"${round(float(_v)/100.0)*100:,.0f}"
                    except (TypeError, ValueError):
                        return "\u2014"

                _g_income  = _gm.get("current_income") or 0
                _g_repl    = _gm.get("replacement_pct") or 0
                _g_need_t  = _gm.get("income_need_today") or 0
                _g_ss_t    = _gm.get("ss_today") or 0
                _g_gap_t   = _gm.get("portfolio_gap_today") or 0
                _g_gap_r   = _gm.get("gap_at_ret") or 0
                _g_infl    = _gm.get("inflation") or 0
                _g_yrs     = _gm.get("years_to_ret") or 0
                _g_retyrs  = _gm.get("retirement_years")
                _infl_fac  = (1.0 + _g_infl) ** _g_yrs if _g_yrs else 1.0
                _g_need_r  = _g_need_t * _infl_fac
                _g_ss_r    = _g_ss_t * _infl_fac

                _IN_LBL  = ParagraphStyle("in_lbl",  fontSize=10.5, leading=13,
                                          fontName="Helvetica", textColor=SLATE,
                                          alignment=TA_LEFT)
                _IN_LBLB = ParagraphStyle("in_lblb", fontSize=10.5, leading=13,
                                          fontName="Helvetica-Bold", textColor=NAVY,
                                          alignment=TA_LEFT)
                _IN_COLH = ParagraphStyle("in_colh", fontSize=8, leading=10,
                                          fontName="Helvetica", textColor=GRAY,
                                          alignment=TA_RIGHT)
                _IN_NUM  = ParagraphStyle("in_num",  fontSize=11.5, leading=14,
                                          fontName="Helvetica", textColor=NAVY,
                                          alignment=TA_RIGHT)
                _IN_NUMB = ParagraphStyle("in_numb", fontSize=12.5, leading=15,
                                          fontName="Helvetica-Bold", textColor=NAVY,
                                          alignment=TA_RIGHT)

                def _sub_lbl(_main, _sub):
                    return Paragraph(
                        f"{_main}<br/><font size='8' color='{GRAY.hexval()}'>{_sub}</font>",
                        _IN_LBL)

                _ss_t_disp = (f"\u2212{_full_money(_g_ss_t)}" if _g_ss_t > 0 else "$0")
                _ss_r_disp = (f"\u2212{_full_money(_g_ss_r)}" if _g_ss_t > 0 else "$0")

                _in_rows = [
                    [Paragraph("", _IN_LBL),
                     Paragraph("TODAY", _IN_COLH),
                     Paragraph("AT RETIREMENT", _IN_COLH)],
                    [_sub_lbl("Retirement income need",
                              f"{_g_repl:.0f}% of {_goal_fmt_money(_g_income)} income"),
                     Paragraph(_full_money(_g_need_t), _IN_NUM),
                     Paragraph(_full_money(_g_need_r), _IN_NUM)],
                    [_sub_lbl("Less: Social Security",
                              "estimated benefit (today's $)"),
                     Paragraph(_ss_t_disp, _IN_NUM),
                     Paragraph(_ss_r_disp, _IN_NUM)],
                    [Paragraph("Gap funded by portfolio", _IN_LBLB),
                     Paragraph(_full_money(_g_gap_t), _IN_NUMB),
                     Paragraph(_full_money(_g_gap_r), _IN_NUMB)],
                ]
                _in_tbl = Table(_in_rows,
                                colWidths=[3.30 * inch, 2.05 * inch, 2.05 * inch])
                _in_tbl.setStyle(TableStyle([
                    ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING",   (0, 0), (0, 0), 0),
                    ("BOTTOMPADDING",(0, 0), (-1, 0), 5),
                    ("TOPPADDING",   (0, 1), (-1, -2), 7),
                    ("BOTTOMPADDING",(0, 1), (-1, -2), 7),
                    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                    ("LINEBELOW",    (0, 0), (-1, 0), 0.5, BORDER),    # under headers
                    ("LINEBELOW",    (0, -2), (-1, -2), 0.5, BORDER),  # above the gap
                    ("BACKGROUND",   (0, -1), (-1, -1), BG_SOFT),      # shade gap row
                    ("TOPPADDING",   (0, -1), (-1, -1), 8),
                    ("BOTTOMPADDING",(0, -1), (-1, -1), 8),
                    ("LEFTPADDING",  (0, -1), (0, -1), 8),
                    ("RIGHTPADDING", (-1, -1), (-1, -1), 8),
                ]))
                story.append(_in_tbl)

                # Assumption + retirement-duration caption (explains the
                # inflation rate used and what drives the nest-egg target).
                _cap_bits = []
                if _g_infl > 0 and _g_yrs:
                    _cap_bits.append(
                        f"At-retirement figures grown at <b>{_g_infl*100:.1f}%/yr</b> "
                        f"inflation over {int(_g_yrs)} yrs.")
                if _g_retyrs:
                    _cap_bits.append(
                        f"Portfolio must cover this gap for ~<b>{int(_g_retyrs)}</b> yrs "
                        f"in retirement &rarr; target {_goal_fmt_money(_g_target)}.")
                if _cap_bits:
                    story.append(Spacer(1, 0.06 * inch))
                    story.append(Paragraph(
                        f"<font color='{GRAY.hexval()}' size='8.5'>{' '.join(_cap_bits)}</font>",
                        ParagraphStyle("in_cap", fontSize=8.5, leading=11,
                                       fontName="Helvetica", textColor=GRAY,
                                       alignment=TA_LEFT)))
                story.append(Spacer(1, 0.12*inch))
            else:
                # ── Non-retirement: compact stat cells ──
                def _g_stat_cell(_lbl, _val):
                    return Table(
                        [[Paragraph(
                            f"<font color='{GRAY.hexval()}' size='7'>{_lbl.upper()}</font>",
                            ParagraphStyle("gs_lbl", fontSize=7, leading=9,
                                           textColor=GRAY, fontName="Helvetica",
                                           alignment=TA_LEFT))],
                         [Paragraph(
                            f"<font color='{NAVY.hexval()}' size='14'><b>{_val}</b></font>",
                            ParagraphStyle("gs_val", fontSize=14, leading=17,
                                           textColor=NAVY, fontName="Helvetica",
                                           alignment=TA_LEFT))]],
                        colWidths=[1.6*inch],
                    )
                _g_stat_row = Table(
                    [[
                        _g_stat_cell("Target", _goal_fmt_money(_g_target) if _g_target else "—"),
                        _g_stat_cell("Starting", _goal_fmt_money(_g_start)),
                        _g_stat_cell("Contribution",
                                     f"{_goal_fmt_money(_g_monthly)}/mo" if _g_monthly else "—"),
                        _g_stat_cell("Projected",
                                     _goal_fmt_money(_g_fv) if (_g_target or _g_fv) else "—"),
                    ]],
                    colWidths=[1.85*inch]*4, hAlign="LEFT",
                )
                _g_stat_row.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), BG_SOFT),
                    ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
                    ("INNERGRID",     (0,0), (-1,-1), 0.5, BORDER),
                    ("LEFTPADDING",   (0,0), (-1,-1), 9),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 9),
                    ("TOPPADDING",    (0,0), (-1,-1), 7),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ]))
                story.append(_g_stat_row)
                story.append(Spacer(1, 0.12*inch))

            # ── Funding bar: today → projected → target (non-retirement) ──
            if (not _g_concept3) and _g_target and _g_target > 0:
                _gb_w = 7.4 * inch
                _gb = Drawing(_gb_w, 30)
                _bar_y, _bar_h = 14, 11
                _gb.add(Rect(0, _bar_y, _gb_w, _bar_h, rx=5, ry=5,
                             fillColor=BORDER_SOFT, strokeColor=None))
                _fill_frac = max(0.0, min(1.0, _g_fv / _g_target))
                if _fill_frac > 0:
                    _gb.add(Rect(0, _bar_y, _gb_w * _fill_frac, _bar_h, rx=5, ry=5,
                                 fillColor=NAVY, strokeColor=None))
                _gb.add(String(0, 2, f"Saved today · {_goal_fmt_money(_g_start)}",
                               fontName="Helvetica", fontSize=8,
                               fillColor=NAVY, textAnchor="start"))
                _g_end_lbl = "Nest egg" if _g_is_ret else "Target"
                _gb.add(String(_gb_w, 2, f"{_g_end_lbl} · {_goal_fmt_money(_g_target)}",
                               fontName="Helvetica", fontSize=8,
                               fillColor=ACCENT, textAnchor="end"))
                if _g_years:
                    _mid_lbl = f"{_g_years} yrs"
                    if _g_ret > 0:
                        _mid_lbl += f" · {_g_ret*100:.1f}%/yr"
                    _gb.add(String(_gb_w / 2.0, 2, _mid_lbl,
                                   fontName="Helvetica", fontSize=8,
                                   fillColor=GRAY, textAnchor="middle"))
                story.append(_gb)
                _cap = ""
                if _g_fund_pct is not None:
                    _cap = (f"Projected funding: <font color='{NAVY.hexval()}'><b>"
                            f"{_g_fund_pct:.0f}% of target</b></font>")
                    if _g_req_pct is not None:
                        _cap += f" &nbsp;&middot;&nbsp; reaches target at ~{_g_req_pct:.1f}%/yr"
                    _g_cg = _gm.get("contribution_growth") or 0
                    if _g_cg > 0:
                        _cap += (f" &nbsp;&middot;&nbsp; contributions step up "
                                 f"{_g_cg*100:.1f}%/yr")
                if _cap:
                    story.append(Paragraph(
                        _cap,
                        ParagraphStyle("goal_fund_cap", fontSize=9, leading=12,
                                       textColor=GRAY, fontName="Helvetica",
                                       alignment=TA_LEFT, spaceBefore=2),
                    ))
                story.append(Spacer(1, 0.10*inch))

            # ── Focus areas row removed — client priorities already
            #    appear at the top of the page, so the duplicate strip
            #    here was redundant. (The 0.10" spacer above provides
            #    the trailing breathing room before the next section.)
        except Exception as _goal_err:
            story.append(Paragraph(
                f"<i>Investment goal section unavailable: {_goal_err}</i>",
                ParagraphStyle("goal_err", fontSize=9, leading=12,
                               textColor=GRAY, fontName="Helvetica-Oblique"),
            ))

    # ── Section header for the current portfolio block ──────────
    # Renamed from "How your portfolio sits today" → "Your Current
    # Portfolio" per advisor request. Sized and styled to match the
    # "Risk Summary" / "Alignment Summary" section headers above
    # (Times-Roman 16pt charcoal, left-aligned) so the page reads as
    # three clearly delineated sections sharing the same typography.
    #
    # The entire Current Portfolio section (header + rule + content
    # row) is collected into `_curr_portfolio_block` and appended as a
    # single KeepTogether at the bottom of the section assembly. That
    # keeps the section from splitting across page 1 and page 2 — if
    # there isn't room for all of it on page 1, ReportLab will move
    # the whole section to a fresh page rather than orphaning the
    # header with the content on the following page.
    # Section header — single inline line: "Your Current Portfolio · N
    # total holdings". Previously the title and count rendered as a
    # 2-column row (title left, count right); advisor merged them into
    # one continuous phrase for a cleaner header. Title at 17.6pt and
    # count at 15.4pt (both bumped +10% per advisor request) with a
    # 13pt middle-dot separator so the title remains the visual
    # emphasis and the count reads as a supporting caption.
    _intro_title = Paragraph(
        f"<font face='Times-Roman' size='17.6' color='{CHARCOAL.hexval()}'>"
        f"Your Current Portfolio</font>"
        f"<font face='Times-Roman' size='13' color='{NAVY.hexval()}'>"
        f"&nbsp;&nbsp;&middot;&nbsp;&nbsp;</font>"
        f"<font face='Times-Roman' size='15.4' color='{NAVY.hexval()}'>"
        f"{len(_cur_tickers)} total holdings</font>",
        ParagraphStyle("intro_title", fontSize=17.6, leading=22,
                       textColor=CHARCOAL, fontName="Times-Roman",
                       alignment=TA_LEFT, spaceBefore=4, spaceAfter=2),
    )
    _curr_portfolio_block = [_intro_title, thin_rule(BORDER, 0.6)]

    # NOTE: the CLIENT PROFILE block used to render HERE — between the
    # cover title and the current portfolio card. As of the cover
    # restructure, the same content (demographics + scores + priorities)
    # is rendered ABOVE the title ribbon, so the report leads with WHO
    # before WHAT. The `sections.get("profile", True)` toggle still
    # controls whether to include it; honor that here by emitting a
    # no-op when profile is disabled. (If profile is False the cover
    # info block above is rendered unconditionally — that's intentional;
    # at minimum the reader needs to know who the proposal is for.
    # The toggle is preserved for backward compat with downstream code
    # that may check sections["profile"].)

    # ── CURRENT PORTFOLIO CARD ────────────────────────────
    # Shows what the client ACTUALLY HOLDS TODAY (Step 2 snapshot), not
    # the proposed/balanced tier. The current_score variable was already
    # computed correctly above from the Step 2 snapshot; here we mirror
    # that for the holdings list, pie chart, and equity/bond/cash split.
    # Falls back to the balanced tier (the legacy behavior) only when
    # Step 2 wasn't set when the proposal was saved.
    _curr_portfolio_block.append(Spacer(1, 0.05*inch))
    # Eyebrow "CURRENT PORTFOLIO" REMOVED per advisor request — the
    # section header above ("Your Current Portfolio") already names the
    # section, so the eyebrow was redundant chrome.
    # The portfolio badge for this page is composited into the donut
    # drawing further down (medium 0.55" inset-double-border variant,
    # chrome suppressed — just the numeral, no eyebrow, no "/ 99").

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

    # Layout per advisor revision:
    #   LEFT half:   "N current holdings" navy serif header
    #                 equity/bond/cash proportional bar
    #                 allocation list (tickers + percentages, 2 cols)
    #   RIGHT half:  donut pie chart with the portfolio badge composited
    #                over its upper-left (built later in this section)
    # The visual reads as a header/data block on the left paired with
    # a graphic on the right, rather than the previous centered-stack
    # arrangement that left the holdings count floating below the pie.

    # "N current holdings" header — Times-Roman sentence-case at 14pt
    # navy, matching the page-1 section header treatment ("Your Current
    # Portfolio", "Risk Summary", "Alignment Summary"). Replaces the
    # earlier Helvetica-Bold ALL-CAPS treatment that read as a
    # competing eyebrow.
    _holdings_count_para = Paragraph(
        f"{len(_cur_tickers)} total holdings",
        ParagraphStyle("holdings_count", fontSize=14, leading=18,
                       textColor=NAVY, fontName="Times-Roman",
                       alignment=TA_LEFT, spaceBefore=2,
                       spaceAfter=6),
    )

    # Equity / Bonds / Cash split — proportional stacked horizontal bar
    # with inline labels, replacing the earlier dot-separated text run
    # ("32% Equity · 64% Bonds · 4% Cash"). Bar reads in a glance: navy
    # for equities, gold for bonds, pale cream for cash. Labels sit
    # INSIDE each segment where the segment is wide enough; small
    # segments (typically cash) get their label suppressed and rely on
    # the segment color to communicate the share.
    def _cover_ebc_bar(eq_pct, bd_pct, cs_pct, width=3.4*inch):
        """Proportional eq/bd/cs strip for the cover.

        Inline equivalent of the page-3 _asset_class_bar but expressed
        as a single Drawing with embedded text — narrower segments
        drop their inline label when there's no room for it.
        """
        total = max(0.001, float(eq_pct) + float(bd_pct) + float(cs_pct))
        W = width
        H = 22  # bar height + breathing room
        bar_h = 16
        bar_y = (H - bar_h) / 2
        d = Drawing(W, H)
        _segments = [
            (float(eq_pct), NAVY,                       BG_SOFT, "Equity"),
            (float(bd_pct), ACCENT,                     NAVY,    "Bonds"),
            (float(cs_pct), colors.HexColor("#E8E0CC"), NAVY,    "Cash"),
        ]
        x = 0.0
        # Light cream backdrop behind the whole bar so any rounding
        # gap between segments doesn't show as the page background.
        d.add(Rect(0, bar_y, W, bar_h,
                   fillColor=colors.HexColor("#F4EFE0"),
                   strokeColor=None))
        for pct, fill_col, text_col, lbl in _segments:
            seg_w = (pct / total) * W
            if seg_w <= 0:
                continue
            d.add(Rect(x, bar_y, seg_w, bar_h,
                       fillColor=fill_col, strokeColor=None))
            # Inline label — only when the segment is wide enough to
            # comfortably hold "NN% Label" at 8.5pt (~6.5pt avg char
            # width, so a 6-char label needs ~42pt). Below that, the
            # color alone carries the meaning.
            _label_str = f"{pct:.0f}% {lbl}"
            _needed = 6.0 * len(_label_str)
            if seg_w >= _needed:
                d.add(String(x + seg_w / 2, bar_y + bar_h / 2 - 3,
                             _label_str,
                             fontName="Helvetica-Bold", fontSize=8.5,
                             fillColor=text_col, textAnchor="middle"))
            else:
                # Tight: just "NN%" with no label word
                _short = f"{pct:.0f}%"
                _need_short = 6.0 * len(_short)
                if seg_w >= _need_short:
                    d.add(String(x + seg_w / 2, bar_y + bar_h / 2 - 3,
                                 _short,
                                 fontName="Helvetica-Bold", fontSize=8.5,
                                 fillColor=text_col, textAnchor="middle"))
            x += seg_w
        # Thin navy outline around the whole bar to frame it.
        d.add(Rect(0, bar_y, W, bar_h,
                   fillColor=None,
                   strokeColor=NAVY, strokeWidth=0.5))
        return d

    _ebc_para = _cover_ebc_bar(_cur_eq, _cur_bd, _cur_cs, width=3.4*inch)

    # Build the allocation legend (tickers + percentages) for the
    # left column. Run the SAME lump_to_other pass as pie_drawing()
    # so the legend rows correspond 1:1 to the donut wedges — same
    # tickers, same sort order, same "Other" rollup if applicable.
    # Previously the legend sorted-by-weight but did NOT lump, while
    # the donut DID lump, so when the portfolio had more than 10
    # holdings the legend rows past row N didn't match any visible
    # wedge and the swatch colors appeared mismatched.
    _cv_ts, _cv_ws, _cv_has_other = lump_to_other(
        _cur_tickers, _cur_weights, _SETTINGS,
    )
    _cv_total = sum(_cv_ws) or 1.0
    _cv_pairs = [(t, (w / _cv_total) * 100.0) for t, w in zip(_cv_ts, _cv_ws)]

    # Build a chart-wide ticker→color map ONCE here so each ticker's
    # swatch in the legend matches its wedge in the donut, with no two
    # tickers sharing a color (see resolve_chart_colors). The donut
    # itself uses the same resolver with the same _cv_ts list a few
    # lines above, so the orderings line up.
    _cv_color_map = dict(zip(_cv_ts, resolve_chart_colors(_cv_ts)))

    _legend_t_style = ParagraphStyle("cv_lg_t", fontName="Helvetica-Bold",
                                      fontSize=9.5, leading=12,
                                      textColor=NAVY)
    _legend_p_style = ParagraphStyle("cv_lg_p", fontName="Helvetica",
                                      fontSize=9.5, leading=12,
                                      textColor=CHARCOAL,
                                      alignment=TA_LEFT)

    def _cv_row(tkr, pct):
        c = _cv_color_map.get(tkr, PDF_TICKER_COLOR(tkr))
        sw = Drawing(8, 9)
        sw.add(Rect(0, 0, 8, 8, fillColor=c, strokeColor=None))
        return [sw,
                Paragraph(tkr, _legend_t_style),
                Paragraph(f"{pct:.1f}%", _legend_p_style)]

    def _empty_cells():
        return [Drawing(8, 9),
                Paragraph("", _legend_t_style),
                Paragraph("", _legend_p_style)]

    _n = len(_cv_pairs)
    _half = (_n + 1) // 2
    _left_pairs  = _cv_pairs[:_half]
    _right_pairs = _cv_pairs[_half:]
    _max_rows = max(len(_left_pairs), len(_right_pairs), 1)

    _legend_grid = []
    for i in range(_max_rows):
        _row = []
        if i < len(_left_pairs):
            _row.extend(_cv_row(*_left_pairs[i]))
        else:
            _row.extend(_empty_cells())
        if i < len(_right_pairs):
            _row.extend(_cv_row(*_right_pairs[i]))
        else:
            _row.extend(_empty_cells())
        _legend_grid.append(_row)

    # Allocation list — 6-col table per row:
    #   [swatch, ticker, pct,  swatch, ticker, pct]
    # Swatches are 8pt color squares matching the pie wedge colors so
    # the legend visually maps to the donut. Centered horizontally
    # within the left zone via hAlign="CENTER" so the table sits
    # directly under the centered EBC bar above it. A 14pt left
    # padding on the second swatch column (col 3) creates the visual
    # gap between the two paired columns.
    # Ticker columns widened from 0.50" → 0.62" so 5-char mutual-fund
    # symbols (VWEHX, PDBZX, PHYZX, etc.) sit on a single line at
    # 9.5pt Helvetica-Bold instead of wrapping mid-name.
    _alloc_tbl = Table(
        _legend_grid,
        colWidths=[
            0.16*inch, 0.62*inch, 0.50*inch,
            0.30*inch, 0.62*inch, 0.50*inch,
        ],
        hAlign="CENTER",
    )
    _alloc_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 2),
        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
        ("TOPPADDING",    (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
        # No left padding on swatches so they hug the start of the cell
        ("LEFTPADDING",   (0,0), (0,-1), 0),
        ("LEFTPADDING",   (3,0), (3,-1), 14),  # gap between paired groups
        # Right-align percent columns so decimals align across rows
        ("ALIGN",         (2,0), (2,-1), "RIGHT"),
        ("ALIGN",         (5,0), (5,-1), "RIGHT"),
    ]))

    # Left cell: vertical stack of EBC bar → allocation list.
    # Holdings count moved to the section header row above per advisor
    # request, so this cell now just stacks the meter over the legend.
    # Narrowed from 4.9" to 3.7" so the content row sits on the LEFT
    # half of the page (the outer row uses hAlign="LEFT" to keep the
    # whole pie + table block aligned to the left edge of the content
    # area, with whitespace on the right balancing the section header
    # callout on the upper-right).
    _left_cell = Table(
        [[_ebc_para],
         [_alloc_tbl]],
        colWidths=[3.7*inch],
    )
    _left_cell.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        # EBC bar and allocation list both centered so they sit
        # visually stacked under one another in the middle of the
        # left cell.
        ("ALIGN",         (0,0), (0,-1), "CENTER"),
    ]))

    # Right cell: just the pie chart now — the duplicate portfolio
    # score badge that previously sat in the upper-right of this cell
    # has been removed because the same score already renders in the
    # Alignment Summary block above (the square PORTFOLIO badge with
    # its own gauge). With the badge gone, the right cell narrows and
    # the pie shifts right — gives the left zone more horizontal room
    # for the centered EBC bar + allocations list. Donut hole stays
    # at 76% of pie radius so the ring reads as a thin annular band.
    _PIE_PT = 2.0 * inch
    # Right cell is just wide enough for the pie + a small right
    # margin. Pie sits flush at x=0 of the cell so its left edge
    # nearly butts up against the left cell's content but its right
    # edge sits near the page's right margin.
    _RC_W = 2.5 * inch
    _RC_H = _PIE_PT
    _right_compose = Drawing(_RC_W, _RC_H)

    # Build the pie at its target x/y. This calls into pie_drawing()
    # to get the styling (lump_to_other rule, per-ticker colors, donut
    # hole), but we need the wedges placed at non-zero coordinates so
    # we rebuild here using the same machinery as pie_drawing().
    if _cur_tickers and _cur_weights:
        _ts, _ws, _has_other = lump_to_other(_cur_tickers, _cur_weights,
                                              _SETTINGS)
        if _ts:
            _colors_list = resolve_chart_colors(_ts)
            _p = Pie()
            # Center the pie horizontally within the right cell so it
            # has equal margins on each side. Pie is 2.0" and cell is
            # 2.5" → 0.25" margin on each side.
            _p.x = (_RC_W - _PIE_PT) / 2.0
            _p.y = 0
            _p.width = _PIE_PT
            _p.height = _PIE_PT
            _p.data = _ws
            _p.labels = None
            _p.slices.strokeColor = WHITE
            _p.slices.strokeWidth = 1.6
            _p.startAngle = 90
            _p.direction = "clockwise"
            for _i, _c in enumerate(_colors_list):
                _p.slices[_i].fillColor = _c
            _right_compose.add(_p)
            # Donut hole — wider than pie_drawing's default 0.32 so this
            # cover-page rendering reads as a thin ring. The standalone
            # pie_drawing() function used elsewhere keeps its own 32%
            # hole; only this composited cover-page instance is thinner.
            _pie_cx = _p.x + _PIE_PT / 2.0
            _pie_cy = _PIE_PT / 2.0
            _right_compose.add(Circle(_pie_cx, _pie_cy, _PIE_PT * 0.38,
                                       fillColor=WHITE, strokeColor=None,
                                       strokeWidth=0))

    # Outer 2-column row placing PIE + TABLE side by side. Both blocks
    # consolidate as one consolidated content row centered within the
    # 7.4" content area (advisor wanted equal left/right margins
    # instead of the previous left-aligned treatment). Pie sits in the
    # first column, EBC bar + alloc grid in the second. Total row
    # width 6.2" (pie 2.5" + table 3.7") → 0.6" of whitespace on each
    # side of the content area. hAlign="CENTER" pins it to the
    # horizontal midline of the page.
    _cur_row = Table(
        [[_right_compose, _left_cell]],
        colWidths=[2.5*inch, 3.7*inch],
        hAlign="CENTER",
    )
    _cur_row.setStyle(TableStyle([
        ("VALIGN",       (0,0), (0,0),   "TOP"),
        ("VALIGN",       (1,0), (1,0),   "MIDDLE"),
        ("ALIGN",        (1,0), (1,0),   "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    _curr_portfolio_block.append(_cur_row)
    # Drop the entire "Your Current Portfolio" card when the client has no
    # current portfolio — the block above is still assembled (cheap) but
    # never reaches the story, so page 1 reflows up cleanly.
    if _has_current_portfolio:
        story.append(KeepTogether(_curr_portfolio_block))

    # ── HOLDINGS DETAIL TABLE (dedicated page) ─────────────
    # Per-holding breakdown matching the reference design: small RISK badge,
    # ticker + name, amount, % of portfolio, 6-month historical range.
    # On its own page so the page-1 cover stays uncluttered and the holdings
    # always have room to render in full without splitting.
    # ── HOLDINGS DETAIL PAGE BUILDER ─────────────────────────────
    # Renders a portrait page with a section header (with risk badge
    # in the upper-right) + intro paragraph + a detailed per-holding
    # table. Used twice: once for the client's CURRENT portfolio
    # (Section 1 · Current Holdings) and once for the PROPOSED
    # comparison's allocation (Section 2 · Proposed Holdings) so the
    # reader can compare them side-by-side in the same format.
    def _render_holdings_page(tickers, weights, score, eyebrow,
                              title, intro_text, options_data=None):
        # Holdings page is PORTRAIT per advisor revision pass (was
        # landscape). The wider landscape page made the table breathe
        # but broke the visual rhythm of the rest of the report — every
        # other content page is portrait, so flipping to landscape just
        # for one table felt jarring. Column widths below have been
        # shrunk to fit the 7.4" portrait content area.
        #
        # MODES:
        #   options_data=None (default) — Current Holdings layout:
        #     7 columns: RISK | HOLDING | AMOUNT | % OF PORTFOLIO |
        #     SEC YIELD | EXPENSE RATIO | 6-MO RANGE
        #     Iterates over the (tickers, weights) pair passed in.
        #
        #   options_data=[(label, tickers, weights), ...] — Proposed
        #     Holdings comparison layout for page 3:
        #     8 columns: RISK | HOLDING | OPT 1 % | OPT 2 % | OPT 3 % |
        #     SEC YIELD | EXPENSE RATIO | 6-MO RANGE
        #     AMOUNT column is dropped (a per-option dollar amount is
        #     ambiguous when the same ticker is held at different
        #     weights across the three options). Rows are the UNION of
        #     tickers across all three options: Option #1 holdings
        #     first (sorted by risk score desc, today's behavior),
        #     then tickers held only by Option #2 or #3 below (also
        #     by risk score desc). Per row, the highest weight across
        #     the three options is rendered bold; the other two are
        #     muted. "—" where an option does not hold that ticker.
        #     The (tickers, weights, score) parameters still represent
        #     Option #1 — they drive the page's corner portfolio_badge
        #     and the page-level $ total context.
        story.append(NextPageTemplate('portrait'))
        story.append(PageBreak())
        # _section_header_with_badge now returns a LIST of flowables:
        #   [0] _CornerBadge (zero-size, draws badge via canvas in
        #       upper-right corner — doesn't affect text flow)
        #   [1] title + width-constrained rule (KeepTogether)
        #   [2] description (Table with spaceBefore=-4)
        # This matches page 4's section_header + intro spacing exactly,
        # with the badge floating independently of the text flow.
        story.extend(_section_header_with_badge(
            eyebrow, title, score, intro_text=intro_text))
        story.append(Spacer(1, 0.08*inch))
        try:
            import yfinance as _yf
            import pandas as _pd
            import numpy as _np
            from datetime import timedelta as _td2

            # Pull the account total from the proposal copy (set by the
            # "Client's current balance" field in the PDF builder). No
            # phantom default — when no balance is provided the total is 0
            # and the band/amounts simply don't render a fake figure.
            _holding_total_usd = float(
                proposal.get("portfolio_value")
                or client_profile.get("portfolio_value") or 0.0)

            # Total balance band — shown on the single-portfolio Holdings
            # pages (Current / Proposed), where a dollar AMOUNT column is
            # present. Sits just under the section header so the client sees
            # the account total before the line-by-line breakdown.
            if options_data is None and _holding_total_usd > 0:
                _tot_para = Paragraph(
                    f'<font face="Helvetica-Bold" size="7.5" color="#b8943f">'
                    f'TOTAL PORTFOLIO BALANCE</font>&nbsp;&nbsp;&nbsp;'
                    f'<font face="Helvetica-Bold" size="14" color="#1a2b4a">'
                    f'${_holding_total_usd:,.0f}</font>',
                    ParagraphStyle("hold_total", fontSize=14, leading=17,
                                   alignment=TA_LEFT))
                _tot_tbl = Table([[_tot_para]], colWidths=[7.4 * inch])
                _tot_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), BG_SOFT),
                    ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
                    ("LINEBELOW",     (0, 0), (-1, -1), 2.0, ACCENT),
                    ("TOPPADDING",    (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 13),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story.append(_tot_tbl)
                story.append(Spacer(1, 0.12 * inch))

            # ── Comparison-mode setup ──
            # When options_data is provided (Proposed Holdings page),
            # the table rows are the UNION of tickers across all three
            # options. We preserve Option #1's ticker order for the
            # primary group, then append tickers held only by Option
            # #2 or #3 in the order they're first seen. Each option's
            # weights are stored as upper-case-keyed dicts for O(1)
            # lookup during row construction.
            #
            # `_opt_weight_maps` is a list of 3 dicts (one per option):
            #   [{TICKER: weight_pct}, {TICKER: weight_pct}, {...}]
            # An absent ticker → option does not hold it → "—" in cell.
            _opt_weight_maps = []
            _opt1_tickers_upper = []
            if options_data is not None:
                for _i, (_lbl, _o_tks, _o_wts) in enumerate(options_data):
                    _m = {}
                    for _t, _w in zip(_o_tks or [], _o_wts or []):
                        if _t:
                            _m[str(_t).upper()] = float(_w or 0)
                    _opt_weight_maps.append(_m)
                    if _i == 0:
                        _opt1_tickers_upper = [str(t).upper()
                                                for t in (_o_tks or [])]
                # Build the union ticker list: Option #1 holdings first
                # (preserving order), then unique tickers from Option #2,
                # then unique tickers from Option #3.
                _seen = set()
                _union = []
                for _src in (options_data[0][1] if options_data else [],
                             options_data[1][1] if len(options_data) > 1 else [],
                             options_data[2][1] if len(options_data) > 2 else []):
                    for _t in (_src or []):
                        _tu = str(_t).upper()
                        if _tu not in _seen:
                            _seen.add(_tu)
                            _union.append(_t)
                _iter_tickers = _union
            else:
                _iter_tickers = tickers

            # Fetch 6-month price history for each ticker (and get info)
            _end2  = _dt.now()
            _start_6m = _end2 - _td2(days=200)
            _upper_tks = [t.upper() for t in _iter_tickers]
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
                # Fallback to yfinance. NOTE on units (yfinance >= 0.2.5x):
                #   .info["yield"]                        -> DECIMAL (0.033 = 3.3%)
                #   .info["dividendYield"]                -> PERCENT  (3.3  = 3.3%)
                #   .info["trailingAnnualDividendYield"]  -> DECIMAL
                # We normalize everything to DECIMAL here; the display
                # layer multiplies by 100 unconditionally.
                try:
                    _yt = _yf.Ticker(_tk_u)
                    try:
                        _inf = _yt.info or {}
                    except Exception:
                        _inf = {}
                    _y = _inf.get("yield")
                    if _y is None and _inf.get("dividendYield") is not None:
                        try:
                            _y = float(_inf["dividendYield"]) / 100.0
                        except (TypeError, ValueError):
                            _y = None
                    if _y is None:
                        _y = _inf.get("trailingAnnualDividendYield") or None
                    if not _y:
                        # Last resort: trailing-12mo dividends / last close.
                        # Uses the history endpoint, which typically still
                        # works when .info is blocked/429'd on cloud hosts.
                        try:
                            _dv = _yt.dividends
                            if _dv is not None and len(_dv):
                                _cut = _dv.index.max() - pd.Timedelta(days=370)
                                _ttm = float(_dv[_dv.index >= _cut].sum())
                                _px_hist = _yt.history(period="5d")["Close"]
                                _px = float(_px_hist.iloc[-1]) if len(_px_hist) else 0.0
                                if _px > 0 and _ttm > 0:
                                    _y = _ttm / _px
                        except Exception:
                            pass
                    _info_cache[_tk_u] = {
                        "name":      _inf.get("longName") or _inf.get("shortName") or _tk_u,
                        "sec_yield": _y,
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

            # Holdings badge: standard document format (cream fill, navy
            # outer, gold inner ring) with the numeral always in navy.
            # Per advisor feedback, the previous tier-tinted numeral
            # (forest / tan / oxblood) was dropped — the risk score
            # column already encodes the magnitude, and uniform navy
            # numerals read as a cleaner, more editorial set of matched
            # badges across the table.
            def _small_risk_badge(score, size=0.35*inch):
                s = size
                d = Drawing(s, s)
                try:
                    sv = int(score)
                except (ValueError, TypeError):
                    sv = 50
                numeral_color = NAVY
                # Outer square: cream fill, navy edge — matches the cover
                # PORTFOLIO badge and the recommendation-card mini badges
                d.add(Rect(0, 0, s, s,
                           strokeColor=NAVY, strokeWidth=0.9,
                           fillColor=BG_SOFT))
                # Inner gold ring — sits inset from the edge
                inset = s * 0.10
                d.add(Rect(inset, inset, s - 2*inset, s - 2*inset,
                           strokeColor=ACCENT, strokeWidth=0.55,
                           fillColor=None))
                # Numeral — navy serif, slightly smaller so the inner
                # ring has clearance
                num_pt = s * 0.42
                d.add(String(s/2, s/2 - num_pt * 0.32, str(sv),
                             fontName="Times-Roman", fontSize=num_pt,
                             fillColor=numeral_color, textAnchor="middle"))
                return d

            # Build table rows
            #
            # Comparison mode (options_data is set) uses a TWO-ROW
            # header so the three OPT % sub-columns sit visually under
            # a single "% OF PORTFOLIO" span. ReportLab SPAN styles
            # (applied below in the TableStyle) handle the vertical
            # merge on the outer columns and the horizontal merge on
            # the spanned % header.
            if options_data is None:
                holdings_hdr = [
                    Paragraph("<b>RISK</b>", eyebrow_cap),
                    Paragraph("<b>HOLDING</b>", eyebrow_cap),
                    Paragraph("<b>AMOUNT</b>", eyebrow_cap),
                    Paragraph("<b>% OF<br/>PORTFOLIO</b>", eyebrow_cap),
                    Paragraph("<b>SEC<br/>YIELD</b>", eyebrow_cap),
                    Paragraph("<b>EXPENSE<br/>RATIO</b>", eyebrow_cap),
                    Paragraph("<b>6-MO RANGE</b>", eyebrow_cap),
                ]
                holdings_rows = [holdings_hdr]
            else:
                # Centered sub-headers (OPT N) — TableStyle below applies
                # ALIGN=CENTER to columns 2-4 so the labels sit under
                # the spanned "% OF PORTFOLIO" label cleanly.
                _ctr_eyebrow = ParagraphStyle(
                    "ctr_eyebrow", parent=eyebrow_cap,
                    alignment=TA_CENTER,
                )
                holdings_hdr_row1 = [
                    Paragraph("<b>RISK</b>", eyebrow_cap),
                    Paragraph("<b>HOLDING</b>", eyebrow_cap),
                    Paragraph("<b>% OF PORTFOLIO</b>", _ctr_eyebrow),
                    "",  # spanned
                    "",  # spanned
                    Paragraph("<b>SEC<br/>YIELD</b>", eyebrow_cap),
                    Paragraph("<b>EXPENSE<br/>RATIO</b>", eyebrow_cap),
                    Paragraph("<b>6-MO RANGE</b>", eyebrow_cap),
                ]
                holdings_hdr_row2 = [
                    "",  # spanned from row 0
                    "",  # spanned from row 0
                    Paragraph("<b>OPT 1</b>", _ctr_eyebrow),
                    Paragraph("<b>OPT 2</b>", _ctr_eyebrow),
                    Paragraph("<b>OPT 3</b>", _ctr_eyebrow),
                    "",  # spanned from row 0
                    "",  # spanned from row 0
                    "",  # spanned from row 0
                ]
                holdings_rows = [holdings_hdr_row1, holdings_hdr_row2]

            # ── PRE-COMPUTE per-holding scores + global range bounds ──
            # Two-pass so we can: (1) sort by risk score descending, and
            # (2) build comparable range bars where each holding's bar
            # is drawn on the same x-scale across the whole table. The
            # bar's visual length encodes the magnitude of the range,
            # which is exactly the kind of "drawdown at a glance" signal
            # the advisor wants instead of color-coded numbers.
            #
            # In comparison mode the meta tuple carries the per-option
            # weight list and a group flag (0 = Option #1 holding,
            # 1 = held only by Opt #2 / #3). Sort below uses (group,
            # -score) so Group A always sits above Group B.
            _holding_meta = []  # default: (tkr, wt, score, lo, hi)
            _global_lo = 0.0
            _global_hi = 0.0
            if options_data is None:
                _meta_iter = zip(tickers, weights)
            else:
                _meta_iter = [(_t, None) for _t in _iter_tickers]
            for _tkr, _wt in _meta_iter:
                _tk_u = _tkr.upper()
                _lo_pct, _hi_pct = _range_cache.get(_tk_u, (None, None))
                _dd = abs(_lo_pct / 100.0) if _lo_pct is not None else 0
                _score = _score_from_vol_and_ticker(
                    _tk_u, _ticker_vol.get(_tk_u), _dd,
                )
                if options_data is None:
                    _holding_meta.append(
                        (_tkr, _wt, _score, _lo_pct, _hi_pct))
                else:
                    _opt_wts = [
                        _opt_weight_maps[_i].get(_tk_u)
                        if _i < len(_opt_weight_maps) else None
                        for _i in range(3)
                    ]
                    _group = 0 if _tk_u in _opt1_tickers_upper else 1
                    _holding_meta.append(
                        (_tkr, _opt_wts, _score, _lo_pct, _hi_pct, _group))
                if _lo_pct is not None:
                    _global_lo = min(_global_lo, _lo_pct)
                if _hi_pct is not None:
                    _global_hi = max(_global_hi, _hi_pct)
            # Pad the axis so bars don't touch the cell edges
            _axis_span = max(abs(_global_lo), abs(_global_hi), 5.0)
            _axis_min = -_axis_span * 1.05
            _axis_max =  _axis_span * 1.05

            # Sort by WEIGHT descending — largest position at the top down
            # to the smallest. (Comparison mode keeps its group + risk-score
            # ordering so Option #1 holdings sit above Opt #2/#3-only rows.)
            if options_data is None:
                _holding_meta.sort(key=lambda m: -float(m[1] or 0))
            else:
                _holding_meta.sort(key=lambda m: (m[5], -int(m[2])))

            # ── Range bar drawer ──
            # Renders a single 6-month range as a horizontal bar inside
            # a fixed-width cell. Axis spans _axis_min .. _axis_max,
            # consistent across all rows so visual length is comparable.
            # Elements:
            #   • Faint gray track behind the bar (the full axis range)
            #   • Thin navy "0%" tick line
            #   • Solid band from low → high in muted navy
            #   • Small caption with the actual numbers, charcoal (no color)
            def _range_bar(lo, hi, width=1.55*inch):
                W = width
                H = 26
                d = Drawing(W, H)
                # Reserve top half for the bar, bottom half for the
                # numeric caption. Bar y is centered in the top region.
                bar_y = H - 12
                bar_h = 6
                axis_w = W - 4   # leave 2pt margin each side
                axis_x0 = 2

                if lo is None or hi is None:
                    d.add(String(W/2, bar_y - 1, "—",
                                 fontName="Helvetica", fontSize=8,
                                 fillColor=GRAY, textAnchor="middle"))
                    return d

                # Faint full-width gray track (shows the axis extent)
                d.add(Rect(axis_x0, bar_y, axis_w, bar_h,
                           fillColor=colors.HexColor("#efece3"),
                           strokeColor=None))

                # Position helpers
                def _x_at(pct):
                    # Map pct in [_axis_min .. _axis_max] → [axis_x0 .. axis_x0+axis_w]
                    span = _axis_max - _axis_min
                    if span <= 0:
                        return axis_x0 + axis_w / 2
                    return axis_x0 + (pct - _axis_min) / span * axis_w

                lo_x = _x_at(lo)
                hi_x = _x_at(hi)
                zero_x = _x_at(0)

                # Split the band into two colored segments at zero.
                # Negative portion (lo_x → zero_x) renders in GOLD,
                # positive portion (zero_x → hi_x) renders in NAVY.
                # This matches the caption colors below so the reader
                # connects bar color to value sign without effort.
                #
                # Edge cases:
                #   - If both lo and hi are negative (all-loss range),
                #     only the gold segment renders.
                #   - If both are positive (all-gain range), only the
                #     navy segment renders.
                #   - If lo < 0 < hi (mixed), both segments render.
                if lo < 0:
                    # Gold segment: from lo_x to min(zero_x, hi_x)
                    _gold_end = min(zero_x, hi_x)
                    _gold_w = max(0, _gold_end - lo_x)
                    if _gold_w > 0:
                        d.add(Rect(lo_x, bar_y, _gold_w, bar_h,
                                   fillColor=ACCENT, strokeColor=None))
                if hi > 0:
                    # Navy segment: from max(zero_x, lo_x) to hi_x
                    _navy_start = max(zero_x, lo_x)
                    _navy_w = max(0, hi_x - _navy_start)
                    if _navy_w > 0:
                        d.add(Rect(_navy_start, bar_y, _navy_w, bar_h,
                                   fillColor=NAVY, strokeColor=None))

                # Zero tick — small navy notch above and below the bar.
                # Per advisor feedback, the previous gold tick clashed
                # with the gold negative segment; navy provides a clean
                # vertical accent that helps the eye find zero without
                # competing with the bar colors on either side.
                d.add(Line(zero_x, bar_y - 2, zero_x, bar_y + bar_h + 2,
                           strokeColor=NAVY, strokeWidth=1.2))

                # Numeric caption underneath — split into three colored
                # parts: negative number in GOLD (matches bar), " / "
                # separator in gray, positive number in NAVY (matches
                # bar). Reportlab's String only accepts one fillColor
                # per element, so we render three separate Strings
                # positioned manually relative to the centerline.
                # Approximate character widths: at 7.5pt Helvetica,
                # ~4.0pt per digit/sign char, ~3.0pt for "%".
                _lo_str  = f"{lo:+.2f}%"
                _sep_str = " / "
                _hi_str  = f"{hi:+.2f}%"
                # Measure widths so we can right-align the lo string
                # immediately before the separator, and left-align
                # the hi string immediately after it.
                from reportlab.pdfbase.pdfmetrics import stringWidth
                _lo_w  = stringWidth(_lo_str,  "Helvetica", 7.5)
                _sep_w = stringWidth(_sep_str, "Helvetica", 7.5)
                _hi_w  = stringWidth(_hi_str,  "Helvetica", 7.5)
                _total_w = _lo_w + _sep_w + _hi_w
                _start_x = W / 2 - _total_w / 2
                d.add(String(_start_x, 1, _lo_str,
                             fontName="Helvetica", fontSize=7.5,
                             fillColor=ACCENT, textAnchor="start"))
                d.add(String(_start_x + _lo_w, 1, _sep_str,
                             fontName="Helvetica", fontSize=7.5,
                             fillColor=GRAY, textAnchor="start"))
                d.add(String(_start_x + _lo_w + _sep_w, 1, _hi_str,
                             fontName="Helvetica", fontSize=7.5,
                             fillColor=NAVY, textAnchor="start"))
                return d

            for _meta_tuple in _holding_meta:
                # Mode-aware unpack: comparison meta carries an extra
                # per-option weight list and a group flag.
                if options_data is None:
                    _tkr, _wt, _score, _lo, _hi = _meta_tuple
                    _opt_wts = None
                else:
                    _tkr, _opt_wts, _score, _lo, _hi, _group = _meta_tuple
                    # `_wt` is unused in comparison mode (no AMOUNT
                    # column and per-option weights handled below).
                    _wt = None

                _tk_u = _tkr.upper()
                _info = _info_cache.get(_tk_u, {})
                _name = _info.get("name") or _tkr
                _name_short = _name if len(_name) <= 42 else _name[:40] + "…"
                _type = _info.get("type") or ""
                _sec_y = _info.get("sec_yield")
                # sec_yield is stored as a DECIMAL everywhere (0.033 = 3.3%)
                # — all fetch paths normalize (AV returns decimals natively;
                # the yfinance fallback converts). Display x100 always; show
                # an em-dash for None/zero (non-payers).
                _sec_str = (f"{_sec_y*100:.2f}%"
                            if isinstance(_sec_y, (int, float)) and _sec_y
                            else "—")

                # Look up expense ratio via the existing helper. Returns
                # a decimal (0.0005 = 0.05%) or None when no source has
                # data. Format as e.g. "0.05%"; show "—" when unknown.
                try:
                    _er = _expense_ratio_for_ticker(_tkr)
                except Exception:
                    _er = None
                if isinstance(_er, (int, float)) and _er >= 0:
                    _er_str = f"{_er * 100:.2f}%"
                else:
                    _er_str = "—"

                if options_data is None:
                    _amount_usd = float(_wt or 0) / 100.0 * _holding_total_usd
                    holdings_rows.append([
                        _small_risk_badge(_score, size=0.35*inch),
                        Paragraph(
                            f"<b>{_tkr}</b> · {_name_short}<br/>"
                            f"<font color='{GRAY.hexval()}' size='7'>{_type}</font>",
                            body_small,
                        ),
                        Paragraph(f"${_amount_usd:,.0f}", body_small),
                        Paragraph(f"{float(_wt or 0):.1f}%", body_small),
                        Paragraph(_sec_str, body_small),
                        Paragraph(_er_str, body_small),
                        _range_bar(_lo, _hi, width=1.45*inch),
                    ])
                else:
                    # Bold the highest-weight option for this row; the
                    # other two render in regular weight. If no option
                    # holds this ticker (shouldn't happen — it's in the
                    # union — but defensive), bold nothing.
                    _present = [(_i, _w) for _i, _w in enumerate(_opt_wts)
                                if _w is not None]
                    _bold_idx = (max(_present, key=lambda x: x[1])[0]
                                 if _present else -1)
                    _opt_cells = []
                    for _i, _w in enumerate(_opt_wts):
                        if _w is None:
                            _opt_cells.append(
                                Paragraph(
                                    f"<font color='{GRAY.hexval()}'>—</font>",
                                    body_small,
                                )
                            )
                        else:
                            _w_str = f"{float(_w):.1f}%"
                            if _i == _bold_idx:
                                _opt_cells.append(
                                    Paragraph(f"<b>{_w_str}</b>", body_small)
                                )
                            else:
                                _opt_cells.append(
                                    Paragraph(_w_str, body_small)
                                )
                    holdings_rows.append([
                        _small_risk_badge(_score, size=0.35*inch),
                        Paragraph(
                            f"<b>{_tkr}</b> · {_name_short}<br/>"
                            f"<font color='{GRAY.hexval()}' size='7'>{_type}</font>",
                            body_small,
                        ),
                        _opt_cells[0],
                        _opt_cells[1],
                        _opt_cells[2],
                        Paragraph(_sec_str, body_small),
                        Paragraph(_er_str, body_small),
                        _range_bar(_lo, _hi, width=1.45*inch),
                    ])

            # Portrait page: 8.5" wide minus 0.55" margins on each side
            # = 7.4" usable.
            #
            # Current Holdings (options_data=None) — 7 columns:
            #   RISK 0.42 + HOLDING 2.45 + AMOUNT 0.80 + % 0.75
            #   + SEC 0.68 + EXPENSE 0.85 + RANGE 1.45 = 7.40
            #
            # Proposed Holdings (options_data set) — 8 columns:
            #   RISK 0.42 + HOLDING 2.50 + OPT1 0.50 + OPT2 0.50
            #   + OPT3 0.50 + SEC 0.68 + EXPENSE 0.85 + RANGE 1.45
            #   = 7.40. AMOUNT dropped; HOLDING gains 0.05 back over
            #   today's width (no truncation regression).
            if options_data is None:
                holdings_tbl = Table(
                    holdings_rows,
                    colWidths=[0.42*inch, 2.45*inch, 0.80*inch, 0.75*inch,
                               0.68*inch, 0.85*inch, 1.45*inch],
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
            else:
                holdings_tbl = Table(
                    holdings_rows,
                    colWidths=[0.42*inch, 2.50*inch,
                               0.50*inch, 0.50*inch, 0.50*inch,
                               0.68*inch, 0.85*inch, 1.45*inch],
                )
                holdings_tbl.setStyle(TableStyle([
                    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                    # ── Header structure ──
                    # Row 0: RISK | HOLDING | %_OF_PORTFOLIO_______ |
                    #        SEC | EXPENSE | RANGE
                    # Row 1: ____ | _______ | OPT1 | OPT2 | OPT3 |
                    #        ___ | _______ | _____
                    # Outer columns vertically span both header rows;
                    # the % header horizontally spans the three OPT
                    # sub-columns.
                    ("SPAN",          (0,0), (0,1)),   # RISK vertical
                    ("SPAN",          (1,0), (1,1)),   # HOLDING vertical
                    ("SPAN",          (2,0), (4,0)),   # % OF PORTFOLIO horizontal
                    ("SPAN",          (5,0), (5,1)),   # SEC YIELD vertical
                    ("SPAN",          (6,0), (6,1)),   # EXPENSE vertical
                    ("SPAN",          (7,0), (7,1)),   # 6-MO RANGE vertical
                    # ── Alignment ──
                    ("ALIGN",         (0,0), (0,-1),  "CENTER"),  # risk badges
                    ("ALIGN",         (2,0), (4,1),   "CENTER"),  # OPT headers
                    ("ALIGN",         (2,2), (4,-1),  "RIGHT"),   # OPT body cells
                    ("ALIGN",         (5,0), (6,1),   "LEFT"),    # SEC/EXP headers
                    ("ALIGN",         (5,2), (6,-1),  "RIGHT"),   # SEC/EXP body
                    # ── Padding ──
                    ("LEFTPADDING",   (0,0), (-1,-1), 4),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                    ("TOPPADDING",    (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    # ── Rules ──
                    # Navy under the full header (now 2 rows tall).
                    # Faint dividers between data rows, skipping the
                    # last row so the table doesn't draw its own
                    # bottom edge.
                    ("LINEBELOW",     (0,1), (-1,1),  1.0, NAVY),
                    ("LINEBELOW",     (0,2), (-1,-2), 0.25, BORDER_SOFT),
                ]))
            story.append(holdings_tbl)
            # Blended (allocation-weighted) fund expense ratio for this
            # portfolio. Rendered in single-portfolio mode only, so it
            # appears beneath both the Current Holdings and Proposed
            # Holdings tables — the reader sees the portfolio's total fund
            # cost, not just the per-holding figures in the column above.
            if options_data is None:
                try:
                    _bw_er, _bw_cov = weighted_expense_ratio(tickers, weights)
                    _bw_txt = (
                        f"Blended fund expense ratio: "
                        f"<b>{(_bw_er or 0.0) * 100:.2f}%</b> per year "
                        f"(weighted by allocation)"
                    )
                    if _bw_cov is not None and _bw_cov < 99.5:
                        _bw_txt += (f" — based on {_bw_cov:.0f}% of holdings "
                                    f"with available expense data")
                    story.append(Spacer(1, 0.05 * inch))
                    story.append(KeepTogether([
                        HRFlowable(width="100%", thickness=1.2, color=NAVY,
                                   spaceBefore=0, spaceAfter=6),
                        Paragraph(f"<i>{_bw_txt}</i>", caption),
                    ]))
                except Exception:
                    pass
        except Exception as _he:
            # Graceful fallback: just skip the detailed table
            story.append(Paragraph(
                f"<i>Detailed holdings breakdown unavailable: {_he}</i>",
                caption,
            ))

    # End of _render_holdings_page function definition. Now actually
    # render the two holdings pages — first the client's CURRENT
    # portfolio, then the PROPOSED allocation. Proposed is sourced
    # from the Step 2 · Select Final Proposal for Report dropdowns:
    # Option #1 drives the primary holdings list (and the corner
    # badge / page metadata); Options #2 and #3 contribute their
    # allocations to the side-by-side % OF PORTFOLIO sub-columns so
    # the advisor can see how each candidate option weights every
    # ticker at a glance.
    #
    # Previously the proposed page was filtered out of `picks` by
    # tk_key == "balanced" — which silently dropped the page whenever
    # the advisor's Option #1 was a 🧩 preset, a 📁 saved portfolio,
    # or a ⭐ recommended tier other than balanced. Resolving Option
    # #1 directly via the same helper the rest of the proposal uses
    # makes the page always render whenever Option #1 has resolvable
    # holdings.
    if _has_current_portfolio and _cur_tickers and _cur_weights:
        _render_holdings_page(
            tickers=_cur_tickers,
            weights=_cur_weights,
            score=current_score,
            eyebrow="Section 1",
            title="Current Holdings",
            intro_text=(
                "The full breakdown of the client's current portfolio: "
                "position size, weight, recent risk profile, and 6-month "
                "price range."
            ),
        )

    # ── Proposed Holdings rendering moved BELOW the Proposed
    #    Portfolios cards block (page swap per advisor revision —
    #    cards page now comes first as page 3, then the detailed
    #    OPT 1/2/3 holdings table as page 4). The actual render
    #    call lives further down, immediately after the cards
    #    section's `if picks:` block closes.

    story.append(HRFlowable(width="100%", thickness=2, color=NAVY,
                             spaceBefore=6, spaceAfter=10))

    # ═══════════════════════════════════════════════════════════
    # PROPOSED PORTFOLIOS (max 3, from Step 4 final_picks)
    # New side-by-side layout matching the design mockup: a gradient
    # spectrum band across the top shows how the three options sit
    # relative to the client's profile target, then three cards in a
    # 3-column row underneath. Each card has a colored ribbon header,
    # a compact donut, and a top-5 legend with an "Other" rollup.
    # ═══════════════════════════════════════════════════════════
    if picks:
        # Recommendations + Notable Periods (page 3 + page 4) render
        # LANDSCAPE — the 3-card comparison row needs the wider page
        # to hold each card's donut + legend without compressing.
        story.append(NextPageTemplate('landscape'))
        story.append(PageBreak())   # all 3 on a dedicated page
        story.append(section_header("Recommendations",
                                    f"Proposed Portfolios ({len(picks)})"))
        # Intro paragraph styled to match pages 2/3 — small italic
        # Helvetica-Oblique sitting tight beneath the header rule.
        _rec_desc_tbl = Table(
            [[Paragraph(
                "The recommendations below have been selected by your "
                "advisor. Each allocation is aligned to your risk profile "
                "and goals.",
                _intro_desc_style)]],
            colWidths=[9.85*inch],
            hAlign="LEFT",
        )
        _rec_desc_tbl.setStyle(TableStyle([
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        _rec_desc_tbl.spaceBefore = -4
        story.append(_rec_desc_tbl)
        story.append(Spacer(1, 0.04*inch))

        # ── HELPERS for the new side-by-side layout ──────────────

        # Resolve client profile score for the spectrum tick. Read once,
        # used by spectrum_band below. Falls back to None for graceful
        # rendering when client_score is missing or non-numeric.
        try:
            _spec_profile = (int(client_score)
                             if client_score not in ("—", None, "") else None)
        except (ValueError, TypeError):
            _spec_profile = None

        # ── Spectrum band ────────────────────────────────────────
        # Full-width gradient strip showing all three options + profile
        # on a 1-99 scale. The gradient runs green (conservative bg) →
        # amber (warning bg) → red (danger bg) to encode tilt direction.
        # Profile target shows as a gold vertical tick; each option is
        # a navy/gold/navy-dark dot using the same TIER_COLORS as ribbons.
        def spectrum_band(picks, profile, total_width=7.4*inch,
                           current_score=None):
            """Three-stop gradient strip with profile tick, option dots,
            and (optionally) the current portfolio's score.

            Args:
                picks: list of tuples (lbl, sub, tk, ptks, pws, pscore)
                profile: client's profile score (int) or None
                current_score: current portfolio score (int) or None.
                    When provided, an empty navy outline circle is drawn
                    on the bar at that score (matching the cover treatment)
                    and "Your Portfolio" is added to the legend.
                total_width: usable horizontal space in points
            """
            W = total_width
            # 72pt band height. Bumped from 64pt per advisor: the
            # profile and proposed score numerals each get ~4pt more
            # clearance from the gradient band, which needs the extra
            # panel height so the lower numeral doesn't collide with
            # the endpoint labels at the panel floor.
            H = 72
            d = Drawing(W, H)

            # Background panel — pale cream (BG_SOFT) with a thin
            # navy frame around the spectrum drawing. The frame
            # restores a visual container around the risk meter
            # (the prior outer wrapper navy box that contained both
            # cards and spectrum was dropped per advisor; this
            # smaller frame is around the spectrum alone). Inset
            # the stroke by 0.5pt so the 1pt navy line stays
            # fully inside the drawing's bounding box rather than
            # bleeding outside.
            _panel_fill = BG_SOFT
            d.add(Rect(0.5, 0.5, W - 1, H - 1,
                       fillColor=_panel_fill,
                       strokeColor=NAVY, strokeWidth=1))

            # Saturated gradient stops — green-200 → amber-100 → red-200.
            stop_green = colors.HexColor("#97C459")
            stop_amber = colors.HexColor("#FAC775")
            stop_red   = colors.HexColor("#F09595")

            # Caption row: eyebrow on left, legend keys on right.
            _top_y = H - 10
            d.add(String(W * 0.03, _top_y, "RISK SPECTRUM",
                         fontName="Helvetica-Bold", fontSize=8,
                         fillColor=ACCENT, textAnchor="start"))

            # Right-side legend: walking right→left.
            # Per advisor redesign:
            #   • "Your portfolio" entry dropped — the current portfolio
            #     dot on the bar was removed, so the legend item goes
            #     with it.
            #   • "Comparison options" (3-dot treatment) replaced with
            #     a single "Proposed range" entry showing a mini
            #     "(--O--)" symbol that mirrors the new on-band visual:
            #     a paren-bracketed dotted line spanning from the
            #     lowest option's score to the highest, with an empty
            #     navy O at the proposed (Option #1) score.
            #   • "Your profile" tick stays gold and stays in place.
            legend_y = _top_y
            legend_x = W * 0.97

            # 1) Profile tick + caption (rightmost). Gold, 3pt symbol→text gap.
            d.add(String(legend_x, legend_y, "Your profile",
                         fontName="Helvetica", fontSize=8,
                         fillColor=CHARCOAL, textAnchor="end"))
            _profile_text_w = 50
            _tick_x = legend_x - _profile_text_w - 3
            d.add(Line(_tick_x, legend_y - 2, _tick_x, legend_y + 7,
                       strokeColor=ACCENT, strokeWidth=2.2))

            # 2) "Proposed range" + mini symbol (leftmost).
            # The mini symbol compresses the on-band visual to its
            # essential silhouette: short navy dashes flanking a small
            # empty circle. Curved parens and dotted detail compress
            # to flat solid line at this scale and aren't worth the
            # rendering complexity for a tiny legend swatch.
            _range_caption_right = _tick_x - 12
            d.add(String(_range_caption_right, legend_y,
                         "Proposed range",
                         fontName="Helvetica", fontSize=8,
                         fillColor=CHARCOAL, textAnchor="end"))
            _range_text_w = 70
            _mini_right = _range_caption_right - _range_text_w - 3
            _mini_w     = 18
            _mini_left  = _mini_right - _mini_w
            _mini_y     = legend_y + 3
            # Left dash (flat segment in lieu of a curved paren at scale)
            d.add(Line(_mini_left, _mini_y, _mini_left + 5, _mini_y,
                       strokeColor=NAVY, strokeWidth=1.2))
            # Empty O at midpoint — navy, mirroring the on-band
            # treatment where the central circle is navy to match the
            # paren-bracketed dotted range drawn around it.
            d.add(Circle(_mini_left + _mini_w/2, _mini_y, 2.2,
                         fillColor=None, strokeColor=NAVY,
                         strokeWidth=1.0))
            # Right dash
            d.add(Line(_mini_left + _mini_w - 5, _mini_y,
                       _mini_left + _mini_w, _mini_y,
                       strokeColor=NAVY, strokeWidth=1.2))

            # Gradient band — 10pt thick (was 14pt) so the spectrum
            # reads as a more delicate ribbon. Centered vertically.
            band_left  = W * 0.04
            band_right = W * 0.96
            band_h     = 10
            band_y     = (H - band_h) / 2
            band_len   = band_right - band_left
            n_slices   = 80
            for i in range(n_slices):
                t = i / (n_slices - 1)
                if t < 0.5:
                    u = t * 2
                    a, b = stop_green, stop_amber
                else:
                    u = (t - 0.5) * 2
                    a, b = stop_amber, stop_red
                r = a.red   * (1 - u) + b.red   * u
                g = a.green * (1 - u) + b.green * u
                bl = a.blue * (1 - u) + b.blue * u
                slice_x = band_left + band_len * t
                slice_w = band_len / n_slices + 0.5
                d.add(Rect(slice_x, band_y, slice_w, band_h,
                           fillColor=colors.Color(r, g, bl),
                           strokeColor=None))

            # Track caption positions to avoid score-label collisions.
            placed_xs = []

            def _x_at(score):
                frac = min(99, max(1, int(score))) / 99.0
                return band_left + band_len * frac

            # Profile target tick (gold vertical line + 11pt Times-Bold
            # numeral above the band) — matches the cover spectrum.
            if profile is not None:
                pf_x = _x_at(profile)
                d.add(Line(pf_x, band_y - 4, pf_x, band_y + band_h + 4,
                           strokeColor=ACCENT, strokeWidth=2.5))
                d.add(String(pf_x, band_y + band_h + 10, str(int(profile)),
                             fontName="Times-Bold", fontSize=11,
                             fillColor=ACCENT, textAnchor="middle"))
                placed_xs.append((pf_x, "above"))

            # ── Proposed range visualization ─────────────────────
            # Per advisor redesign — replaces the prior treatment of
            # three separate per-option dots (cons / proposed / agg)
            # plus a "Your portfolio" current-score outline circle.
            #
            # The new symbol is a single navy paren-bracketed dotted
            # line spanning from the lowest option's score to the
            # highest, with an empty navy circle floating at the
            # proposed (Option #1) score in the middle of the range:
            #
            #     (..........O..........)
            #
            # All navy. The gold profile tick above stays as the only
            # gold element on the band — it visually contrasts as the
            # client's TARGET against the navy treatment of the
            # ADVISOR'S RECOMMENDATION span.
            _pick_scores = []
            _proposed_score = None
            for idx, (_, _, _, _, _, _pscore) in enumerate(picks):
                if _pscore is None:
                    continue
                try:
                    _sv = int(_pscore)
                except (ValueError, TypeError):
                    continue
                _pick_scores.append(_sv)
                if idx == 0:
                    # Option #1 = proposed (per advisor confirmation:
                    # whatever Option #1 dropdown resolves to drives
                    # the proposed score and the central O position).
                    _proposed_score = _sv

            if _proposed_score is not None:
                _center_y = band_y + band_h / 2
                _circle_r = 6.5
                _proposed_x = _x_at(_proposed_score)

                # Only render the paren-bracketed range when there's
                # a meaningful span across the picks (2+ distinct
                # scores spaced more than the circle diameter). A
                # single pick — or a degenerate case where all scores
                # collapse to ~one point — falls back to just the
                # empty O at the proposed score.
                _range_min = min(_pick_scores)
                _range_max = max(_pick_scores)
                _left_x  = _x_at(_range_min)
                _right_x = _x_at(_range_max)
                if (len(_pick_scores) >= 2
                        and (_right_x - _left_x) > 2 * _circle_r + 4):
                    # Curved paren brackets at each end — drawn as
                    # cubic Beziers (Path doesn't support quadratic
                    # curves natively; the cubic control-point
                    # placement at 2/3 of the way to the quadratic
                    # control point reproduces the same curve).
                    _paren_half_h = 7.5
                    _paren_bulge  = 4.5
                    _ctrl_y_top    = _center_y + _paren_half_h / 3.0
                    _ctrl_y_bottom = _center_y - _paren_half_h / 3.0
                    _ctrl_x_left   = _left_x  - _paren_bulge * 2.0 / 3.0
                    _ctrl_x_right  = _right_x + _paren_bulge * 2.0 / 3.0

                    _left_paren = Path(
                        strokeColor=NAVY, strokeWidth=1.8,
                        fillColor=None)
                    _left_paren.moveTo(
                        _left_x, _center_y + _paren_half_h)
                    _left_paren.curveTo(
                        _ctrl_x_left, _ctrl_y_top,
                        _ctrl_x_left, _ctrl_y_bottom,
                        _left_x, _center_y - _paren_half_h,
                    )
                    d.add(_left_paren)

                    _right_paren = Path(
                        strokeColor=NAVY, strokeWidth=1.8,
                        fillColor=None)
                    _right_paren.moveTo(
                        _right_x, _center_y + _paren_half_h)
                    _right_paren.curveTo(
                        _ctrl_x_right, _ctrl_y_top,
                        _ctrl_x_right, _ctrl_y_bottom,
                        _right_x, _center_y - _paren_half_h,
                    )
                    d.add(_right_paren)

                    # Dotted connector — two segments, with a gap
                    # around the central O so the circle reads as
                    # floating on the range rather than sitting on a
                    # solid line through it.
                    _line_gap = _circle_r + 2
                    _l1_x1 = _left_x + 3
                    _l1_x2 = _proposed_x - _line_gap
                    if _l1_x2 > _l1_x1:
                        d.add(Line(_l1_x1, _center_y,
                                   _l1_x2, _center_y,
                                   strokeColor=NAVY, strokeWidth=1.2,
                                   strokeDashArray=[1, 2.5]))
                    _l2_x1 = _proposed_x + _line_gap
                    _l2_x2 = _right_x - 3
                    if _l2_x2 > _l2_x1:
                        d.add(Line(_l2_x1, _center_y,
                                   _l2_x2, _center_y,
                                   strokeColor=NAVY, strokeWidth=1.2,
                                   strokeDashArray=[1, 2.5]))

                # Empty navy O at the proposed score — always drawn
                # (this is the single-element fallback for the
                # degenerate cases above, and the centerpiece of the
                # range otherwise). Per advisor revision pass: the
                # central hollow circle is navy, matching the
                # surrounding paren-bracketed dotted range so the
                # whole "advisor's proposed range" symbol reads as one
                # navy unit. The gold "Your profile" tick above the
                # band stays the lone gold element, so the client's
                # target still contrasts against the navy proposal.
                d.add(Circle(_proposed_x, _center_y, _circle_r,
                             fillColor=None,
                             strokeColor=NAVY, strokeWidth=1.8))

                # Proposed-score caption — ALWAYS below the band.
                # The gold profile-tick caption sits above the band,
                # so the proposed numeral takes the bottom slot
                # regardless of how close the two scores are. Per
                # advisor: portfolio risk score should be below the
                # line.
                _proposed_caption_y = band_y - 18
                d.add(String(_proposed_x, _proposed_caption_y,
                             str(_proposed_score),
                             fontName="Times-Bold", fontSize=11,
                             fillColor=NAVY, textAnchor="middle"))

            # Endpoint labels at the very bottom — NAVY at 8.5pt to
            # match the cover spectrum (was charcoal 8pt).
            d.add(String(band_left, 3, "More Conservative",
                         fontName="Helvetica-Bold", fontSize=8.5,
                         fillColor=NAVY, textAnchor="start"))
            d.add(String(band_right, 3, "More Aggressive",
                         fontName="Helvetica-Bold", fontSize=8.5,
                         fillColor=NAVY, textAnchor="end"))

            return d

        # ── Compact pie for narrow side-by-side cards ────────────
        def compact_pie(tickers, weights, size=1.4*inch):
            """Donut pie sized for the narrow 2.3" card columns. Uses the
            same lump_to_other rule as the main pie_drawing, but with
            wider donut hole and slightly thinner stroke so the chart
            stays readable at the smaller scale."""
            ts, ws, _ = lump_to_other(tickers, weights, _SETTINGS)
            if not ts:
                d = Drawing(size, size)
                d.add(String(size/2, size/2, "No data",
                             fontName="Helvetica", fontSize=8,
                             fillColor=GRAY, textAnchor="middle"))
                return d
            colors_list = resolve_chart_colors(ts)
            d = Drawing(size, size)
            p = Pie()
            p.x = 0; p.y = 0
            p.width = size; p.height = size
            p.data = ws; p.labels = None
            p.slices.strokeColor = WHITE
            p.slices.strokeWidth = 1.4
            p.startAngle = 90
            p.direction = "clockwise"
            for i, c in enumerate(colors_list):
                p.slices[i].fillColor = c
            d.add(p)
            # Wider donut hole (38% of diameter, vs 32% on the main pie)
            # so the narrow chart doesn't read as a heavy disc
            d.add(Circle(size/2, size/2, size * 0.38,
                         fillColor=BG_LIGHT, strokeColor=None))
            return d

        # ── Compact legend: top-5 + Other rollup ─────────────────
        def compact_legend(tickers, weights, top_n=5):
            """Single-column legend showing the top N holdings, with
            everything else rolled into a single italic 'Other (count)'
            row at the end. Designed for the narrow card width.

            Why we don't just call pie_legend_table: that function
            renders top-10 (or 10 + Other) which is too wide for a 2.3"
            card. This compact version shows the 5 largest positions
            (the ones an advisor would actually discuss) and rolls the
            rest into a transparent "+ N more" indicator."""
            ts, ws, _ = lump_to_other(tickers, weights, _SETTINGS)
            if not ts:
                return Paragraph("—", body_small)

            # Sort by weight (lump_to_other already does this, but be safe)
            pairs = list(zip(ts, ws))

            # The lump_to_other already produced an "Other" slot if there
            # were >10 holdings. For the narrow legend we want top-5 then
            # collapse everything else (including any pre-existing Other)
            # into a single bottom row.
            visible = pairs[:top_n]
            rest = pairs[top_n:]
            rest_weight = sum(w for _, w in rest)
            rest_count = sum(1 for t, _ in rest if t != "Other") + (
                # if rest contains an "Other" entry, count it as the rolled
                # tickers it represents (we don't have that count handy,
                # so just label it as one bucket)
                0
            )

            sm_style = ParagraphStyle("sm", fontName="Helvetica",
                                       fontSize=7.5, leading=10,
                                       textColor=CHARCOAL)
            sm_pct = ParagraphStyle("smpct", fontName="Helvetica",
                                     fontSize=7.5, leading=10,
                                     textColor=CHARCOAL, alignment=TA_RIGHT)
            sm_other = ParagraphStyle("smo", fontName="Helvetica-Oblique",
                                       fontSize=7.5, leading=10,
                                       textColor=GRAY)
            sm_other_pct = ParagraphStyle("smopct", fontName="Helvetica",
                                           fontSize=7.5, leading=10,
                                           textColor=GRAY, alignment=TA_RIGHT)

            # Build the ticker→color map ONCE for this chart so each
            # row picks up the chart-wide distinct color (see
            # resolve_chart_colors above). Map covers the full ts list
            # — only the top-N visible rows render with named swatches
            # here, but resolving across all ts keeps colors stable
            # with the matching pie/donut that uses the same ts.
            _cmap = dict(zip(ts, resolve_chart_colors(ts)))

            rows = []
            for t, w in visible:
                c = _cmap.get(t, PDF_TICKER_COLOR(t))
                sw = Drawing(7, 7)
                sw.add(Rect(0, 0, 7, 7, fillColor=c, strokeColor=None))
                if t == "Other":
                    # Renders as "OTHER" in all caps (advisor request)
                    # so the rollup row reads visually as a category
                    # tag rather than as a descriptive phrase like
                    # "Other holdings". Style stays italic + muted-
                    # gray to keep the row clearly subordinate to the
                    # named-ticker rows above it.
                    rows.append([
                        sw,
                        Paragraph("OTHER", sm_other),
                        Paragraph(f"{w:.0f}%", sm_other_pct),
                    ])
                else:
                    rows.append([
                        sw,
                        Paragraph(f"<b>{t}</b>", sm_style),
                        Paragraph(f"{w:.0f}%", sm_pct),
                    ])

            # Rollup row for everything past top_n
            if rest:
                rest_n = len(rest)
                sw = Drawing(7, 7)
                sw.add(Rect(0, 0, 7, 7, fillColor=colors.HexColor("#a8a8a8"),
                            strokeColor=None))
                rows.append([
                    sw,
                    Paragraph(f"<i>+ {rest_n} more</i>", sm_other),
                    Paragraph(f"{rest_weight:.0f}%", sm_other_pct),
                ])

            tbl = Table(rows, colWidths=[0.13*inch, 0.85*inch, 0.42*inch])
            tbl.setStyle(TableStyle([
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                ("TOPPADDING",    (0,0), (-1,-1), 1.5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 1.5),
            ]))
            return tbl

        # ── Spectrum band: render and append ─────────────────────
        # Pass the current portfolio score for backward-compat; the
        # spectrum_band function no longer draws a per-portfolio dot
        # on the band (the "Your portfolio" marker was dropped per
        # advisor redesign), but the parameter is preserved so older
        # callers don't break.
        try:
            _spec_current = (int(current_score)
                              if current_score not in ("—", None, "") else None)
        except (ValueError, TypeError):
            _spec_current = None
        spec_d = spectrum_band(picks, _spec_profile, total_width=9.85*inch,
                                current_score=_spec_current)
        # spec_d joins the SOLID OUTER TABLE assembled below — that
        # outer table frames the 3-card row and the spectrum band as
        # a single unified navy-bordered block (spectrum now sits
        # BELOW the cards rather than above; the previous comparison
        # metrics table was removed).

        # ── Helpers for the redesigned card layout ───────────────

        def _asset_class_bar(tickers, weights, width=2.0*inch):
            """Stacked horizontal bar showing the eq/bd/cs split.

            Built from the same _classify_ticker / _balanced_split
            machinery used on the cover. Renders ONLY the colored bar.
            Captions go in a separate Paragraph below the bar — earlier
            attempts to position captions over each segment caused
            overlaps when small segments (cash, bonds) sat adjacent.

            Args:
                tickers / weights: parallel lists. Weights in percent or
                                   decimal — we normalize to fractions.
                width: drawing width in points.

            Returns:
                (drawing, caption_html) tuple. The caller stacks them.
            """
            # Compute the three buckets
            total_w = sum(float(w or 0) for w in weights) or 1.0
            eq = bd = cs = 0.0
            for t, w in zip(tickers, weights):
                frac = float(w or 0) / total_w * 100.0
                try:
                    cls, _ = _classify_ticker(t.upper())
                except Exception:
                    cls = "equity"
                if cls == "cash":
                    cs += frac
                elif cls == "bond":
                    bd += frac
                elif cls == "balanced":
                    es, bs = _balanced_split(t)
                    eq += frac * es
                    bd += frac * bs
                else:
                    eq += frac

            W = width
            H = 8  # just the bar, no caption space
            bar_h = 7
            d = Drawing(W, H)
            # Colors: navy for equities, gold for bonds, light cream-ish
            # gray for cash. Matches the palette used elsewhere in the
            # document so the bar doesn't introduce new color semantics.
            x_cursor = 0
            segments = [
                (eq, NAVY,                       "eq"),
                (bd, ACCENT,                     "bd"),
                (cs, colors.HexColor("#d8d6cb"), "cs"),
            ]
            for pct, col, _name in segments:
                seg_w = (pct / 100.0) * W
                if seg_w > 0:
                    d.add(Rect(x_cursor, 0, seg_w, bar_h,
                               fillColor=col, strokeColor=None))
                x_cursor += seg_w

            # Caption HTML — three swatches + percentages in a single
            # inline row. Each swatch is a small inline square using
            # ReportLab's color hex. The bar's three colors map 1:1 to
            # the captions so the reader maps bar→text without effort.
            #
            # Two layout decisions to prevent the caption from wrapping
            # in narrow side cards:
            #   1. Use &nbsp; (non-breaking space) between the percentage
            #      and the unit label ("58%&nbsp;eq") so they can't split.
            #   2. Use single-space separators instead of " &middot; ",
            #      which compresses the line by ~12pt and keeps the row
            #      on one line at 2.05" card width with 7.5pt text.
            _NAVY_HEX = NAVY.hexval()
            _ACCENT_HEX = ACCENT.hexval()
            _CASH_HEX = "#d8d6cb"
            caption_html = (
                f"<font color='{_NAVY_HEX}'>&#9632;</font>"
                f"<font color='{CHARCOAL.hexval()}' size='7.5'>"
                f" <b>{eq:.0f}%</b>&nbsp;eq</font> &nbsp; "
                f"<font color='{_ACCENT_HEX}'>&#9632;</font>"
                f"<font color='{CHARCOAL.hexval()}' size='7.5'>"
                f" <b>{bd:.0f}%</b>&nbsp;bd</font> &nbsp; "
                f"<font color='{_CASH_HEX}'>&#9632;</font>"
                f"<font color='{CHARCOAL.hexval()}' size='7.5'>"
                f" <b>{cs:.0f}%</b>&nbsp;cs</font>"
            )
            return d, caption_html

        def _full_legend(tickers, weights, ncols=2, font_size=7.5,
                          total_width=None, canonical_order=None,
                          max_holdings=12):
            """Full holdings legend rendered in N columns.

            Replaces compact_legend (top-5 + 'more' rollup) — page 3 now
            shows EVERY holding so the reader can see the complete book.
            Use ncols=2 for the wider PROPOSED middle card; ncols=2 also
            works for the side cards but at smaller width.

            total_width: if set, scales column widths to fit. Default
            assumes narrow side-card sizing.

            canonical_order: optional list of ticker symbols defining
            the desired row order across all three comparison cards.
            When provided, this pick's tickers are sorted by their
            position in canonical_order so the same ticker appears in
            the same legend slot across cards. Tickers in
            canonical_order but missing from this pick are skipped.
            Tickers in this pick but missing from canonical_order
            (rare — non-overlapping picks) are appended at the end in
            weight-desc order.

            max_holdings: if the legend would have more than this many
            rows, the top max_holdings-1 are shown and the remainder
            roll up into a single "+M more (X.X%)" row at the bottom
            with a gray swatch. Default 12 — fits the solid-table
            recommendations block on one landscape page even when the
            proposed allocation has 17+ positions. Pass None to show
            every holding (legacy behavior, may push the solid block
            across pages for very long allocations).
            """
            if not tickers:
                return Paragraph("—", body_small)
            # Normalize to %, build a {ticker_upper: pct} map
            _tot_raw = sum(float(w or 0) for w in weights) or 1.0
            _wmap = {}
            for _t, _w in zip(tickers, weights):
                _wmap[_t.upper()] = (float(_w or 0) / _tot_raw) * 100.0

            # Decide SHOW vs ROLL-UP strictly by weight, so "+N more"
            # always captures this portfolio's SMALLEST allocations
            # (advisor request). Canonical order is then applied only to
            # the kept rows, for cross-card row alignment — it no longer
            # controls which holdings get rolled up.
            _by_weight = sorted(_wmap.items(), key=lambda x: -x[1])
            _rollup_count = 0
            _hidden_sum = 0.0
            if max_holdings is not None and len(_by_weight) > max_holdings:
                _kept_set = _by_weight[:max_holdings - 1]
                _hidden   = _by_weight[max_holdings - 1:]
                _hidden_sum = sum(p for _, p in _hidden)
                _rollup_count = len(_hidden)
            else:
                _kept_set = _by_weight
            _kept_tickers = {t for t, _ in _kept_set}

            if canonical_order:
                # Order the kept rows by canonical position so the same
                # ticker lands in the same slot across cards; kept tickers
                # absent from canonical_order are appended by weight desc.
                pairs = []
                _seen = set()
                for _tk in canonical_order:
                    _tk_u = _tk.upper()
                    if _tk_u in _kept_tickers and _tk_u not in _seen:
                        pairs.append((_tk_u, _wmap[_tk_u]))
                        _seen.add(_tk_u)
                _extras = sorted(
                    [(t, p) for t, p in _kept_set if t not in _seen],
                    key=lambda x: -x[1],
                )
                pairs.extend(_extras)
            else:
                pairs = list(_kept_set)   # already weight desc

            # Append the rollup row (smallest holdings) at the bottom.
            if _rollup_count > 0:
                pairs.append((f"_rollup:{_rollup_count}", _hidden_sum))

            sm_style = ParagraphStyle("legend_t", fontName="Helvetica",
                                       fontSize=font_size,
                                       leading=font_size + 2.5,
                                       textColor=CHARCOAL)
            # Percent: LEFT-aligned (was TA_RIGHT) so the value sits
            # tight against the ticker text rather than floating at
            # the far right of its column. Combined with a tightened
            # ticker column width this produces the compact
            # "SCHX 45.7%" pairing seen in the reference mockup.
            sm_pct = ParagraphStyle("legend_p", fontName="Helvetica",
                                     fontSize=font_size,
                                     leading=font_size + 2.5,
                                     textColor=CHARCOAL,
                                     alignment=TA_LEFT)

            # Build a chart-wide distinct ticker→color map for this
            # legend's display order. The rollup sentinel "_rollup:N"
            # is included here for index alignment but never read
            # (the rollup branch in _row_for renders its own gray
            # swatch). resolve_chart_colors enforces no duplicates
            # across the named ticker rows.
            _legend_color_map = dict(zip(
                [t for t, _ in pairs],
                resolve_chart_colors([t for t, _ in pairs]),
            ))

            def _row_for(tkr, pct):
                # Rollup sentinel — render as gray swatch + italic
                # "+N more" caption + the aggregate weight. The sentinel
                # encodes the hidden count as "_rollup:N" so we can
                # extract N without a separate parameter.
                if isinstance(tkr, str) and tkr.startswith("_rollup:"):
                    try:
                        _n_hidden = int(tkr.split(":", 1)[1])
                    except (ValueError, IndexError):
                        _n_hidden = 0
                    sw = Drawing(7, 7)
                    sw.add(Rect(0, 0, 7, 7,
                                fillColor=GRAY_SOFT, strokeColor=None))
                    return [sw,
                            Paragraph(
                                f"<i>+{_n_hidden} more</i>",
                                ParagraphStyle(
                                    "legend_more",
                                    parent=sm_style,
                                    textColor=GRAY,
                                ),
                            ),
                            Paragraph(
                                f"{pct:.1f}%",
                                ParagraphStyle(
                                    "legend_more_p",
                                    parent=sm_pct,
                                    textColor=GRAY,
                                ),
                            )]
                # Normal row — colored swatch + bold ticker + percent
                c = _legend_color_map.get(tkr, PDF_TICKER_COLOR(tkr))
                sw = Drawing(7, 7)
                sw.add(Rect(0, 0, 7, 7, fillColor=c, strokeColor=None))
                return [sw,
                        Paragraph(f"<b>{tkr}</b>", sm_style),
                        Paragraph(f"{pct:.1f}%", sm_pct)]

            if ncols == 1:
                rows = [_row_for(t, p) for t, p in pairs]
                tbl = Table(rows, colWidths=[0.14*inch, 0.7*inch, 0.55*inch])
                tbl.setStyle(TableStyle([
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                    ("LEFTPADDING",   (0,0), (-1,-1), 0),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                    ("TOPPADDING",    (0,0), (-1,-1), 1.2),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 1.2),
                ]))
                return tbl

            # 2-column layout — split pairs into left/right halves so the
            # left column fills first (ceiling-half count on the left)
            n = len(pairs)
            half = (n + 1) // 2
            left_pairs = pairs[:half]
            right_pairs = pairs[half:]

            # Build a single Table with 6 cols: swatch L | ticker L | pct L |
            # swatch R | ticker R | pct R. Empty cells fill the right side
            # if the column ran short.
            max_rows = max(len(left_pairs), len(right_pairs))
            grid = []
            for i in range(max_rows):
                row = []
                if i < len(left_pairs):
                    row.extend(_row_for(*left_pairs[i]))
                else:
                    row.extend(["", "", ""])
                if i < len(right_pairs):
                    row.extend(_row_for(*right_pairs[i]))
                else:
                    row.extend(["", "", ""])
                grid.append(row)

            # Column width strategy:
            # - Side cards (narrow): use compact layout that fits 1.77"
            #   (side card 2.05" minus 10pt padding each side)
            # - Middle PROPOSED card (wider): scale columns proportionally
            # Both keep the right-half swatch at 0.22" minimum for the
            # padding clearance the 8pt swatch needs. Ticker columns
            # widened to 0.42" / 0.40" to safely hold 5-char mutual-fund
            # symbols (PDBZX, PFORX, PHYZX, etc.) without wrapping —
            # the previous 0.35" / 0.34" was just barely too narrow for
            # 5 chars at 7.5pt Helvetica-Bold.
            if total_width is not None and total_width > 2.0 * inch:
                _swatch_w = 0.15 * inch
                _swatch_w_r = 0.25 * inch  # padding clearance
                _ticker_w = 0.42 * inch
                _pct_w    = 0.32 * inch
            else:
                _swatch_w = 0.12 * inch
                _swatch_w_r = 0.22 * inch  # padding clearance (8pt swatch)
                _ticker_w = 0.40 * inch
                _pct_w    = 0.30 * inch

            tbl = Table(
                grid,
                colWidths=[_swatch_w, _ticker_w, _pct_w,
                           _swatch_w_r, _ticker_w, _pct_w],
            )
            tbl.setStyle(TableStyle([
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 1),
                ("TOPPADDING",    (0,0), (-1,-1), 1.2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 1.2),
                # Small gap between the two columns
                ("LEFTPADDING",   (3,0), (3,-1), 6),
            ]))
            return tbl

        def _compute_pick_3yr_stats(picks_in):
            """Compute (total_return, vol, max_dd) per pick over the last
            3 years. Used by the bottom comparison table on page 3.

            Returns a dict {pick_idx: (tot, vol, dd)} keyed by ordered_picks
            index (0/1/2). Missing or short data → (None, None, None).
            """
            import yfinance as _yf
            import pandas as _pd
            import numpy as _np
            from datetime import timedelta as _td_stats

            out = {}
            _all = set()
            for _, _, _, tks, _, _ in picks_in:
                for t in tks or []:
                    if t:
                        _all.add(t.upper())
            if not _all:
                return {i: (None, None, None) for i in range(len(picks_in))}

            _end_s = _dt.now()
            _start_s = _end_s - _td_stats(days=365*3 + 30)
            try:
                _px = _yf.download(
                    list(_all), start=_start_s, end=_end_s,
                    auto_adjust=True, progress=False, threads=True,
                )["Close"]
                if isinstance(_px, _pd.Series):
                    _px = _px.to_frame()
                _px = _px.dropna(how="all")
                _r = _px.pct_change().dropna(how="all").fillna(0)
            except Exception:
                return {i: (None, None, None) for i in range(len(picks_in))}

            for i, (_, _, _, tks, wts, _) in enumerate(picks_in):
                try:
                    cols = [t.upper() for t in tks
                            if t.upper() in _r.columns]
                    if not cols or not wts:
                        out[i] = (None, None, None)
                        continue
                    aligned = _np.array([
                        float(wts[j] or 0) for j, t in enumerate(tks)
                        if t.upper() in _r.columns
                    ])
                    if aligned.sum() <= 0:
                        out[i] = (None, None, None)
                        continue
                    aligned = aligned / aligned.sum()
                    port_r = (_r[cols] * aligned).sum(axis=1)
                    if len(port_r) < 30:
                        out[i] = (None, None, None)
                        continue
                    tot = float((1 + port_r).prod() - 1)
                    vol = float(port_r.std() * _np.sqrt(252))
                    eq  = (1 + port_r).cumprod()
                    dd  = float(((eq / eq.cummax()) - 1).min())
                    out[i] = (tot, vol, dd)
                except Exception:
                    out[i] = (None, None, None)
            return out

        # ── Build each card ──────────────────────────────────────
        # Three cards in a 3-column row. The PROPOSED card (middle) is
        # WIDER than the side cards (matches mockup) and uses a gold
        # outline ONLY (no gold ribbon fill) — the previous design's
        # gold ribbon header competed with the cream PORTFOLIO badge for
        # attention. With the outline-only treatment, the badge sits on
        # the card's normal cream interior and reads cleanly.
        #
        # ORDERING RULE: conservative left, proposed middle, aggressive
        # right. Same as before.

        # Column widths — sized so the three cards + inter-card gaps
        # together span the landscape page's ~9.85" usable width.
        # cards_row has LEFTPADDING=3 + RIGHTPADDING=3 per cell × 2 gaps
        # ≈ 12pt = 0.17" of inter-cell padding, so cards sum to ~9.68".
        # Layout: 3.00 + 3.68 + 3.00 = 9.68". The middle card stays the
        # widest so PROPOSED visually anchors the row.
        _SIDE_W   = 3.00 * inch
        _MIDDLE_W = 3.68 * inch

        def _pick_sort_key(pick):
            """Sort picks: conservative (or lowest-score) first, balanced
            (PROPOSED) in the middle, aggressive (or highest-score) last."""
            lbl, sub, tk, ptks, pws, pscore = pick
            tier_order = {"conservative": 0, "balanced": 1, "aggressive": 2}
            primary = tier_order.get(tk, 3)
            try:
                secondary = int(pscore) if pscore else 50
            except (ValueError, TypeError):
                secondary = 50
            return (primary, secondary)

        ordered_picks = sorted(picks, key=_pick_sort_key)
        card_cells = []

        # Canonical ticker order across all 3 comparison cards. Sort
        # tickers by their weight in the PROPOSED option first (so the
        # most material proposed holdings sit at the top of every
        # card's legend); tickers that appear only in non-proposed
        # picks get appended by their max weight across those picks.
        # Result: the same ticker occupies the same row across cards
        # so the reader can scan a single ticker's weight horizontally
        # across all three options.
        _proposed_for_order = next(
            (p for p in ordered_picks if p[2] == "balanced"), None
        )
        _proposed_weights = {}
        if _proposed_for_order:
            _, _, _, _ptks_canon, _pws_canon, _ = _proposed_for_order
            _tot_canon = sum(float(w or 0) for w in _pws_canon) or 1.0
            for _t, _w in zip(_ptks_canon, _pws_canon):
                _proposed_weights[_t.upper()] = (
                    float(_w or 0) / _tot_canon * 100.0
                )
        _other_weights = {}
        for _pp in ordered_picks:
            if _pp is _proposed_for_order:
                continue
            _, _, _, _ptks_o, _pws_o, _ = _pp
            _tot_o = sum(float(w or 0) for w in _pws_o) or 1.0
            for _t, _w in zip(_ptks_o, _pws_o):
                _t_u = _t.upper()
                _other_weights[_t_u] = max(
                    _other_weights.get(_t_u, 0.0),
                    float(_w or 0) / _tot_o * 100.0,
                )
        _canonical_order = sorted(
            _proposed_weights.keys(),
            key=lambda t: -_proposed_weights[t],
        )
        for _t in sorted(_other_weights.keys(),
                         key=lambda t: -_other_weights[t]):
            if _t not in _proposed_weights:
                _canonical_order.append(_t)

        # Card header typography styles — same pattern as the mockup
        _opt_eyebrow = ParagraphStyle(
            "opt_eyebrow", fontSize=7.5, leading=10, textColor=GRAY,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        )
        _opt_eyebrow_proposed = ParagraphStyle(
            "opt_eyebrow_p", fontSize=10, leading=12, textColor=ACCENT,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        )
        _opt_title = ParagraphStyle(
            "opt_title", fontSize=12, leading=14, textColor=NAVY,
            fontName="Times-Roman", alignment=TA_LEFT, spaceAfter=1,
        )
        _opt_subtitle = ParagraphStyle(
            "opt_subtitle", fontSize=8, leading=11, textColor=GRAY,
            fontName="Helvetica-Oblique", alignment=TA_LEFT,
        )

        # Gutter between cards. cards_row applies 8pt LEFTPADDING +
        # 8pt RIGHTPADDING to every cell, which is meant to create
        # 16pt of cream space between adjacent cards. For that to
        # actually work, each card's width must equal its cell's
        # CONTENT area (cell width − 16pt), not the full cell width —
        # otherwise the card overflows its cell's padding on both
        # sides and the borders of adjacent cards collide at the
        # same X coordinate. The earlier code sized cards to the
        # full cell width, which caused the proposed card's 4pt
        # gold border to be painted directly on top of the side
        # cards' 1pt navy borders (gold "interrupted" by navy,
        # no visible cream gap between cards).
        _CARD_GUTTER_TOTAL = 16  # 8pt L + 8pt R per cards_row cell

        for idx, (lbl, sub, tk, ptks, pws, pscore) in enumerate(ordered_picks):
            is_proposed = (tk == "balanced") or (idx == 1)
            this_w = (_MIDDLE_W if is_proposed else _SIDE_W) - _CARD_GUTTER_TOTAL

            # Eyebrow construction — per advisor redesign:
            # Two-line layout with the descriptor (PROPOSED / MORE
            # CONSERVATIVE / MORE AGGRESSIVE) on the TOP line, and
            # COMPARISON #N on the BOTTOM line. The descriptor is the
            # headline; the comparison number is the subordinate
            # identifier. Sizes bumped across the board ("enlarge
            # all slightly") with the proposed card's descriptor
            # given the biggest uplift so it visually dominates.
            if tk == "balanced":
                descriptor = "PROPOSED"
            elif tk == "conservative":
                descriptor = "MORE CONSERVATIVE"
            elif tk == "aggressive":
                descriptor = "MORE AGGRESSIVE"
            else:
                _u = lbl.upper()
                if "CONSERVATIVE" in _u:
                    descriptor = "MORE CONSERVATIVE"
                elif "AGGRESSIVE" in _u:
                    descriptor = "MORE AGGRESSIVE"
                elif "PROPOSED" in _u or "RECOMMENDED" in _u:
                    descriptor = "PROPOSED"
                else:
                    descriptor = _u
            # Label number is determined by DESCRIPTOR, not visual
            # position. Per advisor preference:
            #   PROPOSED        → #1  (the recommended option)
            #   MORE CONSERVATIVE → #2
            #   MORE AGGRESSIVE   → #3
            # The visual order on the page stays conservative → proposed
            # → aggressive (left to right), so the labels run #2, #1, #3
            # across the row. Counter-intuitive but matches the advisor's
            # mental hierarchy (proposed is the headline option).
            if descriptor == "PROPOSED":
                _card_label_num = 1
            elif descriptor == "MORE CONSERVATIVE":
                _card_label_num = 2
            elif descriptor == "MORE AGGRESSIVE":
                _card_label_num = 3
            else:
                # Unknown descriptor — fall back to visual order
                _card_label_num = idx + 1
            # Build the two-line eyebrow content as a list of
            # Paragraphs (Table cells accept lists; they stack
            # vertically). Sizing diverges sharply between the
            # PROPOSED card (12pt gold descriptor) and the side
            # cards (9pt gray descriptor) so the proposed reads
            # as the headline option without ambiguity.
            if is_proposed:
                _eye_desc_style = ParagraphStyle(
                    "eye_desc_p", fontSize=12, leading=14,
                    textColor=ACCENT, fontName="Helvetica-Bold",
                    alignment=TA_LEFT, spaceAfter=0,
                )
                _eye_cmp_style = ParagraphStyle(
                    "eye_cmp_p", fontSize=8, leading=10,
                    textColor=GRAY, fontName="Helvetica",
                    alignment=TA_LEFT, spaceAfter=0,
                )
            else:
                _eye_desc_style = ParagraphStyle(
                    "eye_desc_s", fontSize=9, leading=11,
                    textColor=GRAY, fontName="Helvetica-Bold",
                    alignment=TA_LEFT, spaceAfter=0,
                )
                _eye_cmp_style = ParagraphStyle(
                    "eye_cmp_s", fontSize=7, leading=9,
                    textColor=GRAY, fontName="Helvetica",
                    alignment=TA_LEFT, spaceAfter=0,
                )
            _eyebrow_content = [
                Paragraph(descriptor, _eye_desc_style),
                Paragraph(f"COMPARISON #{_card_label_num}",
                          _eye_cmp_style),
            ]

            # Build the card header: eyebrow + cream-badge with score on
            # the right side of the same row. The badge replaces the old
            # gold ribbon — sits inline with the eyebrow, score visible.
            # Header row width must respect the card's 10pt L/R padding,
            # so total colWidths = this_w - 20pt (≈ 0.28"). Previously
            # the row spanned this_w - 0.10", which pushed the badge
            # past the card's right padding boundary and made it appear
            # clipped against the card outline.
            # Badge — visual variant differs by tier:
            #   PROPOSED (middle): HORIZONTAL pill variant — wider than
            #     tall (1.00" × 0.45"), preserves the PORTFOLIO eyebrow
            #     + serif numeral + gauge tick chrome, but at a height
            #     close to the side cards' 0.40" score boxes. The old
            #     0.85" square chromed badge inflated the header row
            #     height and pushed the proposed card's allocation bar,
            #     donut, and legend ~32pt below the side cards' content
            #     (breaking horizontal alignment across the three cards).
            #     With the pill, all three cards' header rows share the
            #     same height so their body content sits on the same
            #     baseline.
            #   SIDES: outlined cream variant at 0.40" with no chrome —
            #     just the numeral. Unchanged.
            if is_proposed:
                _hdr_badge = portfolio_badge_horizontal(
                    pscore if pscore else "—",
                    width=1.00 * inch, height=0.45 * inch,
                )
                _badge_col_w = 1.06 * inch
            else:
                _badge_size = 0.40 * inch
                _hdr_badge = portfolio_badge(
                    pscore if pscore else "—", label="",
                    size=_badge_size, filled=False,
                )
                _badge_col_w = 0.45 * inch
            _hdr_inner_w = this_w - 0.28*inch  # accounts for L/R padding
            header_row = Table(
                [[_eyebrow_content, _hdr_badge]],
                colWidths=[_hdr_inner_w - _badge_col_w, _badge_col_w],
            )
            header_row.setStyle(TableStyle([
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("ALIGN",         (1,0), (1,0),   "RIGHT"),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
                # Badge column gets explicit right padding so the
                # right-aligned badge doesn't sit flush against the
                # card's gold border (which would read as overlap
                # once the gold border is thickened).
                ("RIGHTPADDING",  (1,0), (1,0),   4),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))

            # Sub-headline + tiny caption REMOVED per advisor request.
            # The card eyebrow row already shows "COMPARISON #N · MORE
            # CONSERVATIVE / PROPOSED / MORE AGGRESSIVE" which carries
            # the tier semantics; the additional "Min-vol tilt /
            # ±50% corridor" descriptors were redundant and added
            # vertical noise without giving the reader new info.

            # Body content
            body_rows = []

            # Asset class bar
            if ptks and pws and sum(float(w or 0) for w in pws) > 0:
                _bar_w = this_w - 0.4*inch
                _bar_draw, _bar_caption_html = _asset_class_bar(
                    ptks, pws, width=_bar_w,
                )
                body_rows.append([Spacer(1, 0.06*inch)])
                body_rows.append([_bar_draw])
                # Caption row immediately below the bar — swatches +
                # percentages on a single line, CENTER-aligned so the
                # "70% eq · 24% bd · 6% cs" caption sits visually
                # centered under the bar (was left-aligned which made
                # the caption pull to the left while the bar itself
                # fills the full width of the cell).
                body_rows.append([Paragraph(
                    _bar_caption_html,
                    ParagraphStyle("acbar_cap", fontName="Helvetica",
                                   fontSize=8, leading=11,
                                   textColor=CHARCOAL,
                                   alignment=TA_CENTER, spaceBefore=2),
                )])

                # Pie chart — sized uniformly across all three comparison
                # cards (1.55") so the donut, legend, and overall card
                # height align between the proposed and side cards. The
                # earlier "bumped" 1.75" treatment on the PROPOSED-only
                # card made it ~14pt taller than the sides, causing the
                # proposed card to hang ~17pt below the side cards'
                # bottoms (combined with the legend font difference).
                # Proposed card emphasis now comes from: the 4pt gold
                # border, the wider column (3.68" vs 3.00"), the
                # PORTFOLIO pill badge, and the gold "PROPOSED"
                # descriptor — donut size no longer carries that load.
                pie_size = 1.55*inch
                body_rows.append([Spacer(1, 0.04*inch)])
                body_rows.append([compact_pie(ptks, pws, size=pie_size)])

                # Full holdings legend in 2 columns. Pass the card width
                # so the legend's columns scale: middle (PROPOSED) card
                # is wider so columns can be wider; font size is uniform
                # (7.0pt) so legend heights match across all three cards.
                body_rows.append([Spacer(1, 0.06*inch)])
                body_rows.append([_full_legend(
                    ptks, pws, ncols=2,
                    font_size=7.0,
                    total_width=this_w,
                    canonical_order=_canonical_order,
                )])
            else:
                body_rows.append([Spacer(1, 0.06*inch)])
                body_rows.append([Paragraph(
                    "<i>External portfolio — holdings not embedded in "
                    "this proposal.</i>",
                    ParagraphStyle("ext", fontName="Helvetica-Oblique",
                                   fontSize=7.5, leading=10,
                                   textColor=GRAY),
                )])

            body_tbl = Table(body_rows, colWidths=[this_w - 0.28*inch])
            body_tbl.setStyle(TableStyle([
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))

            # Combined card: header on top (with eyebrow + badge), body
            # underneath. Cards have a thin border by default; PROPOSED
            # gets a thicker gold outline (the only color treatment on
            # an otherwise uniform set of cards).
            card = Table(
                [[header_row], [body_tbl]],
                colWidths=[this_w],
            )
            card_style = [
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ("TOPPADDING",    (0,0), (0,0),   10),
                ("BOTTOMPADDING", (0,0), (0,0),   2),
                ("TOPPADDING",    (0,1), (-1,1),  2),
                ("BOTTOMPADDING", (0,1), (-1,1),  12),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ("BACKGROUND",    (0,0), (-1,-1), BG_SOFT),
            ]
            if is_proposed:
                # Thicker gold outline (4pt — bumped from 2.5pt) to
                # firmly anchor the proposed card as the headline.
                # The cards-row's surrounding navy box was removed in
                # an earlier pass, so the 4pt gold reads clean on all
                # four edges without competing with an outer frame.
                card_style.append(("BOX", (0,0), (-1,-1), 4, ACCENT))
            else:
                # Side cards get a navy outline (was a faint gray BORDER
                # which read as nearly absent on the cream background).
                # Navy at 1.0pt matches the visual weight of the gold
                # outline on the PROPOSED card while staying clearly
                # secondary — the gold is thicker (2.5 vs 1.0) and warmer,
                # so the proposed card still reads as the headline.
                card_style.append(("BOX", (0,0), (-1,-1), 1.0, NAVY))
            card.setStyle(TableStyle(card_style))
            card_cells.append(card)

        # Pad to 3 columns if fewer picks (rare edge case)
        while len(card_cells) < 3:
            card_cells.append(Spacer(1, 1))

        # Assemble side-by-side row. Widths follow the side / middle /
        # side pattern; if ordering somehow produces a non-middle
        # PROPOSED we still want the middle column wider, so always
        # use [_SIDE_W, _MIDDLE_W, _SIDE_W].
        cards_row = Table(
            [card_cells],
            colWidths=[_SIDE_W, _MIDDLE_W, _SIDE_W],
        )
        cards_row.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            # Gap between cards bumped from 3pt to 8pt per side
            # (16pt total cream space between adjacent cards). With
            # the middle card's 4pt gold border and the side cards'
            # 1pt navy borders, the tighter 3pt padding was visually
            # compressing the borders against each other — the right
            # gold edge of the proposed card read as fainter than the
            # left because it was nearly touching the right card's
            # navy edge. 8pt opens enough clean cream space on each
            # side for all four edges of the gold border to read at
            # the same visual weight.
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        # cards_row is no longer appended directly to story — it gets
        # combined with spec_d (the Risk Spectrum band) into a single
        # solid outer table below, so the cards row and the spectrum
        # read as one unified framed block.

        # ── Layout: cards row + spectrum (separated) ─────────────
        # Per advisor redesign — second iteration:
        #   • The outer navy box that previously wrapped the cards
        #     row and the spectrum together has been REMOVED. The
        #     middle (PROPOSED) card's 2.5pt gold border was clipped
        #     by the outer frame on the top and bottom edges; the
        #     wrap also made the spectrum read as part of the same
        #     block rather than as a separate summary.
        #   • Risk spectrum now sits BELOW the cards row with a
        #     vertical gap, rendering standalone (no surrounding
        #     box), so it reads as a related-but-separate panel.
        #   • Each card retains its own border (gold on the middle
        #     PROPOSED card, navy on the sides), now visible on all
        #     four edges without competing with an outer frame.
        story.append(Spacer(1, 0.10*inch))
        story.append(KeepTogether([
            cards_row,
            Spacer(1, 0.18*inch),
            spec_d,
        ]))

    # ── Proposed Holdings (page 4 — moved here from above the
    #    Proposed Portfolios block per advisor revision). Renders
    #    in PORTRAIT (handled internally by _render_holdings_page,
    #    which calls NextPageTemplate('portrait')+PageBreak at its
    #    start, so it correctly flips back to portrait from the
    #    landscape Proposed Portfolios page that precedes it).
    _opt1_resolved = _resolve_option("option_1")
    _opt2_resolved = _resolve_option("option_2")
    _opt3_resolved = _resolve_option("option_3")
    if _opt1_resolved and _opt1_resolved[3] and _opt1_resolved[4]:
        (_o1_lbl, _o1_sub, _o1_tk,
         _prop_tickers, _prop_weights, _prop_score) = _opt1_resolved
        # Proposed Holdings shows ONLY the proposed (Option #1) portfolio —
        # single-portfolio layout (RISK | HOLDING | AMOUNT | % OF PORTFOLIO |
        # SEC YIELD | EXPENSE RATIO | 6-MO RANGE), no three-option comparison
        # columns. A blended fund expense ratio is summarized beneath the
        # table (see _render_holdings_page).
        _render_holdings_page(
            tickers=_prop_tickers,
            weights=_prop_weights,
            score=_prop_score,
            eyebrow="Section 2",
            title="Proposed Holdings",
            intro_text=(
                "The full breakdown of the proposed allocation: each "
                "holding's weight, recent risk profile, yield, expense "
                "ratio, and 6-month price range."
                + (
                    " Use this alongside the Current Holdings section "
                    "above to compare positions."
                    if _has_current_portfolio else ""
                )
            ),
        )

    # ── RMD Projections (opt-in — sections["rmd_projection"]) ─────────
    # Renders right after Proposed Holdings, in PORTRAIT (Proposed Holdings
    # left us in portrait). Projects required minimum distributions year by
    # year using the IRS Uniform Lifetime Table and SECURE 2.0 start ages.
    _rmd_cfg = (client_profile or {}).get("rmd") or {}
    if sections.get("rmd_projection") and float(_rmd_cfg.get("balance") or 0) > 0:
        _rmd_rows, _rmd_sum = _rmd_projection(
            balance=float(_rmd_cfg.get("balance") or 0.0),
            birth_year=int(_rmd_cfg.get("birth_year") or 1953),
            growth_rate=float(_rmd_cfg.get("growth_rate") or 5.0),
            current_year=date.today().year,
            end_age=int(_rmd_cfg.get("end_age") or 95),
        )
        story.append(PageBreak())
        story.append(section_header("Section 2", "Required Minimum Distributions"))

        if not _rmd_rows:
            story.append(Paragraph(
                "No required minimum distributions fall within the projection "
                "window for the values provided.", _intro_desc_style))
        else:
            def _usd(v): return "${:,.0f}".format(v)
            _g_pct = float(_rmd_cfg.get("growth_rate") or 5.0)
            _start_bal = _rmd_rows[0]["begin"]

            _intro = (
                f"Projected required minimum distributions from the tax-deferred "
                f"balance, age {_rmd_rows[0]['age']} through {_rmd_sum['end_age']}. "
                f"Each year's RMD is the prior year-end balance divided by the IRS "
                f"Uniform Lifetime Table factor; the remaining balance grows "
                f"{_g_pct:.1f}% per year."
            )
            if _rmd_sum["pre_years"]:
                _intro += (
                    f" The balance is grown {_g_pct:.1f}% per year from today to "
                    f"the first RMD year ({_rmd_sum['first_year']})."
                )
            story.append(Spacer(1, 0.02 * inch))
            story.append(Paragraph(_intro, _intro_desc_style))
            story.append(Spacer(1, 0.12 * inch))

            # ── Summary band: starting balance, return rate, outcomes ──
            _band = ParagraphStyle("rmd_band", fontName="Helvetica",
                                   fontSize=8, leading=16, textColor=NAVY,
                                   alignment=TA_LEFT)
            def _band_cell(label, value):
                return Paragraph(
                    f'<font size="6.6" color="#6b6b6b">{label}</font><br/>'
                    f'<font size="12.5"><b>{value}</b></font>', _band)
            _sum_tbl = Table([[
                _band_cell("STARTING BALANCE", _usd(_start_bal)),
                _band_cell("RETURN RATE", f"{_g_pct:.1f}%"),
                _band_cell(f"FIRST RMD · {_rmd_sum['first_year']}",
                           _usd(_rmd_sum["first_rmd"])),
                _band_cell("TOTAL RMDs", _usd(_rmd_sum["total_rmd"])),
                _band_cell(f"BALANCE · AGE {_rmd_sum['end_age']}",
                           _usd(_rmd_sum["end_balance"])),
            ]], colWidths=[1.38 * inch] * 5)
            _sum_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), BG_SOFT),
                ("BOX",          (0, 0), (-1, -1), 1.0, NAVY),
                ("LINEAFTER",    (0, 0), (-2, -1), 0.5, BORDER),
                ("TOPPADDING",   (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 9),
                ("LEFTPADDING",  (0, 0), (-1, -1), 9),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(_sum_tbl)
            story.append(Spacer(1, 0.14 * inch))

            # ── Balance trajectory (projected ending balance over time) ──
            _cw, _chh = 6.9 * inch, 1.25 * inch
            _d = Drawing(_cw, _chh)
            _series = [_rmd_rows[0]["begin"]] + [r["end"] for r in _rmd_rows]
            _vmax = max(_series) or 1.0
            _mN = len(_series)
            _pl, _pr, _pt, _pb = 4, 4, 12, 16
            _pw = _cw - _pl - _pr
            _ph = _chh - _pt - _pb
            _d.add(Line(_pl, _pb + _ph, _pl + _pw, _pb + _ph,
                        strokeColor=BORDER_SOFT, strokeWidth=0.4))
            _d.add(Line(_pl, _pb, _pl + _pw, _pb,
                        strokeColor=BORDER, strokeWidth=0.5))
            _pts = []
            for _i, _v in enumerate(_series):
                _x = _pl + ((_pw * _i / (_mN - 1)) if _mN > 1 else 0)
                _y = _pb + (_ph * (_v / _vmax))
                _pts.extend([_x, _y])
            _d.add(PolyLine(points=_pts, strokeColor=NAVY, strokeWidth=1.6))
            _d.add(Circle(_pts[-2], _pts[-1], 2.4, fillColor=ACCENT,
                          strokeColor=None))
            _d.add(String(_pl, _pb + _ph + 3, "Projected balance",
                          fontName="Helvetica-Bold", fontSize=7.5,
                          fillColor=NAVY))
            _d.add(String(_pl + _pw, _pb + _ph + 3, _usd(_vmax),
                          fontName="Helvetica", fontSize=7, fillColor=GRAY,
                          textAnchor="end"))
            _d.add(String(_pl, _pb - 11, f"Age {_rmd_rows[0]['age']}",
                          fontName="Helvetica", fontSize=7, fillColor=GRAY))
            _d.add(String(_pl + _pw, _pb - 11, f"Age {_rmd_rows[-1]['age']}",
                          fontName="Helvetica", fontSize=7, fillColor=GRAY,
                          textAnchor="end"))
            story.append(_d)
            story.append(Spacer(1, 0.14 * inch))

            # ── Year-by-year projection table (two columns side-by-side so
            #    the full schedule fits one portrait page) ──
            _tbl_style = TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("TEXTCOLOR",     (0, 1), (-1, -1), CHARCOAL),
                ("ALIGN",         (0, 0), (-1, -1), "RIGHT"),
                ("ALIGN",         (0, 0), (1, -1), "CENTER"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 5),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
                ("TOPPADDING",    (0, 0), (-1, -1), 3.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, BG_SOFT]),
                ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
                ("LINEBELOW",     (0, 0), (-1, 0), 1.2, ACCENT),
            ])
            def _mk_block(rows_slice):
                _data = [["Age", "Year", "Factor", "RMD", "Balance"]]
                for r in rows_slice:
                    _data.append([str(r["age"]), str(r["year"]),
                                  f"{r['factor']:.1f}", _usd(r["rmd"]),
                                  _usd(r["end"])])
                _t = Table(_data, colWidths=[0.4 * inch, 0.52 * inch,
                                             0.48 * inch, 0.9 * inch, 1.0 * inch])
                _t.setStyle(_tbl_style)
                _t.repeatRows = 1
                return _t
            _half = (len(_rmd_rows) + 1) // 2
            _left_blk = _mk_block(_rmd_rows[:_half])
            _right_blk = (_mk_block(_rmd_rows[_half:]) if _rmd_rows[_half:]
                          else "")
            _two_up = Table([[_left_blk, _right_blk]],
                            colWidths=[3.45 * inch, 3.45 * inch])
            _two_up.setStyle(TableStyle([
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",  (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 10),
                ("LEFTPADDING",  (1, 0), (1, -1), 10),
                ("RIGHTPADDING", (1, 0), (-1, -1), 0),
                ("TOPPADDING",   (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
            ]))
            story.append(_two_up)
            story.append(Spacer(1, 0.12 * inch))

            story.append(Paragraph(
                "Estimates only — not tax advice. Figures use the IRS Uniform "
                "Lifetime Table (Pub. 590-B) and SECURE 2.0 starting ages (73 if "
                "born 1951–1959, 75 if 1960 or later). RMDs apply to traditional "
                "IRA / SEP / SIMPLE and most employer plans; Roth IRAs have no "
                "lifetime RMD for the original owner. The projection assumes a "
                "constant growth rate; actual balances, returns, and tax law will "
                "vary. The spouse-more-than-10-years-younger (Joint Life) case is "
                "not reflected.", _intro_desc_style))

    # Section 4 (Notable Market Periods) is LANDSCAPE — same as
    # Section 3 (Recommendations). The wider page gives each event
    # row room for the chart plus the per-portfolio mini-table on the
    # left without compressing either. After this section we flip
    # back to portrait for everything that follows.
    story.append(NextPageTemplate('landscape'))

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
        # Verbose 3-sentence intro removed per advisor request — the
        # section title plus per-event headers + descriptions carry the
        # context now. Bottom-of-section italic disclaimer covers the
        # legal language about hypothetical / past performance.
        story.append(Spacer(1, 0.06*inch))

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

        # Name list for the chart-rendering loops below. MUST be derived
        # from _np_portfolios — NOT from the page-4 backtest section's
        # _portfolio_names (which uses different names like "Current
        # Portfolio" / "Comparison #1/2/3"). Mixing the two caused every
        # period chart to fall through to "data unavailable" because the
        # _series_data dict was keyed by these names and the loops were
        # looking up backtest names that don't exist in it.
        _np_portfolio_names = [name for name, _, _ in _np_portfolios]

        try:
            _period_data = _compute_period_returns(_np_portfolios)
        except Exception as _np_err:
            _period_data = None
            story.append(Paragraph(
                f"<i>Notable Market Periods unavailable: {_np_err}</i>",
                body_small,
            ))

        if _period_data:
            # Per advisor preference, the summary numeric table at the
            # top of this section has been REMOVED. The per-event line
            # charts below show the full return path (drawdown + recovery)
            # which is more informative than the final number. The
            # _period_data computation above is no longer used for the
            # table, but is left in place because future iterations may
            # want to surface specific period numbers (peak drawdown,
            # recovery time, etc.) and the data is already cached.

            # ── Line charts: cumulative return curves per period ─────
            # Per-period mini line charts showing how each portfolio's
            # cumulative return evolved through the event window. Renders
            # below the summary table — the table shows the final
            # number, the charts show the path (drawdown depth + recovery
            # speed). Five charts arranged in a 2-up grid (or 1-up final
            # row for an odd count).
            #
            # Colors:
            #   Recommended → gold (ACCENT)
            #   Current     → navy (NAVY)
            #   SPY         → light gray
            #   BND         → lighter gray
            # Gold + navy lines are drawn thicker (1.6pt) than the gray
            # benchmarks (0.9pt) so the client's two portfolios pop.
            try:
                _series_data = _compute_period_price_series(_np_portfolios)
            except Exception:
                _series_data = None

            if _series_data:
                # Period chart styling constants. Layout restructured
                # per advisor: each event now stacks (title + date) →
                # (2x2 mini-table) → (full-width chart) → (italic desc)
                # vertically, instead of the previous "mini-table left /
                # chart right" two-column layout. Result: the chart
                # spans the full landscape content width (9.85") and
                # gets a taller height (1.30" vs 0.85") for maximum
                # visibility. Content runs across 2 pages — 3 events
                # on page 1, 2 on page 2 — which advisor accepted.
                _CHART_W = 9.85 * inch
                _CHART_H = 1.30 * inch
                # Palette C — palette match with the existing chart palette.
                # The benchmarks pick up the same teal and berry-pink that
                # the reader has already seen in the pie charts on pages 1
                # and 3, so the colors feel native to the document. Gold
                # and navy remain reserved for the client's two portfolios
                # so they're never confused with benchmarks.
                _BENCH_TEAL = colors.HexColor("#2c8a8f")  # SPY  (matches pie teal)
                _BENCH_BERRY = colors.HexColor("#a64664") # BND  (matches pie berry)

                # Per-portfolio styling map: (color, stroke_width, z_order).
                # Higher z_order renders on top — gold (recommended) is
                # drawn last so it sits above navy and the benchmarks.
                # All four lines are the SAME stroke width — color carries
                # the distinction. Per advisor feedback, mixing 1.6pt
                # client lines with 0.9pt benchmarks made the benchmark
                # lines look "spidery" and inconsistent. Uniform 1.1pt
                # reads as a clean four-line chart where each portfolio
                # is treated as a peer.
                _UNIFORM_WIDTH = 1.1
                _series_style = {
                    "Recommended":     (ACCENT,        _UNIFORM_WIDTH, 4),
                    "Current":         (NAVY,          _UNIFORM_WIDTH, 3),
                    "S&P 500 (SPY)":   (_BENCH_TEAL,   _UNIFORM_WIDTH, 2),
                    "Agg Bond (BND)":  (_BENCH_BERRY,  _UNIFORM_WIDTH, 1),
                }

                def _build_period_chart(label, start, end):
                    """Build a Drawing showing cumulative-return curves
                    for all portfolios across the given period window."""
                    d = Drawing(_CHART_W, _CHART_H)

                    # Collect all valid series for this period and compute
                    # the global y-range across them (so the axis is
                    # consistent across portfolios on the same chart).
                    chart_series = []  # list of (pname, points)
                    all_vals = []
                    for pname in _np_portfolio_names:
                        pts = _series_data.get(pname, {}).get(label)
                        if not pts:
                            continue
                        # pts is list[(date_str, decimal_return)]
                        rets = [v for _, v in pts]
                        if not rets:
                            continue
                        chart_series.append((pname, rets))
                        all_vals.extend(rets)

                    if not chart_series or not all_vals:
                        # No data for this period — render a placeholder
                        d.add(String(_CHART_W/2, _CHART_H/2,
                                     "data unavailable",
                                     fontName="Helvetica-Oblique",
                                     fontSize=8, fillColor=GRAY,
                                     textAnchor="middle"))
                        return d

                    # Axis bounds — anchor zero in the range so 0% always
                    # shows as a reference line, with ~6% padding above/
                    # below so curves don't kiss the chart edge.
                    y_min = min(all_vals + [0.0])
                    y_max = max(all_vals + [0.0])
                    if y_max == y_min:
                        y_max = y_min + 0.01
                    y_pad = (y_max - y_min) * 0.08
                    y_min -= y_pad
                    y_max += y_pad

                    # Chart plot region. The event name + date range
                    # now live in the table row's left cell, so we drop
                    # the top padding that previously reserved space for
                    # an embedded title.
                    pad_l, pad_r = 30, 8
                    pad_t, pad_b = 8, 14
                    plot_w = _CHART_W - pad_l - pad_r
                    plot_h = _CHART_H - pad_t - pad_b
                    plot_x0 = pad_l
                    plot_y0 = pad_b

                    # Plot frame (thin border)
                    d.add(Rect(plot_x0, plot_y0, plot_w, plot_h,
                               strokeColor=BORDER_SOFT, strokeWidth=0.4,
                               fillColor=None))

                    # Zero line (dashed) if zero falls inside the range
                    if y_min < 0 < y_max:
                        zero_y = plot_y0 + (0 - y_min) / (y_max - y_min) * plot_h
                        d.add(Line(plot_x0, zero_y, plot_x0 + plot_w, zero_y,
                                   strokeColor=GRAY_SOFT, strokeWidth=0.5,
                                   strokeDashArray=[2, 2]))

                    # Y-axis tick labels: y_min, 0 (if visible), y_max
                    def _y_to_px(v):
                        return plot_y0 + (v - y_min) / (y_max - y_min) * plot_h

                    def _fmt_axis_pct(v):
                        return f"{v*100:+.0f}%"

                    # Top and bottom labels
                    d.add(String(plot_x0 - 4, _y_to_px(y_max) - 3,
                                 _fmt_axis_pct(y_max),
                                 fontName="Helvetica", fontSize=6.5,
                                 fillColor=GRAY, textAnchor="end"))
                    d.add(String(plot_x0 - 4, _y_to_px(y_min) - 3,
                                 _fmt_axis_pct(y_min),
                                 fontName="Helvetica", fontSize=6.5,
                                 fillColor=GRAY, textAnchor="end"))
                    # 0% reference label — but only if it has clearance.
                    # In some windows (e.g. 2022 Bear Market where almost
                    # everything was negative), y_max is close to 0% and
                    # the "+2%" label would stack on top of the "0%" label.
                    # We require at least ~10pt vertical clearance from
                    # both endpoints before drawing the 0% label.
                    if y_min < 0 < y_max:
                        _zero_y_px = _y_to_px(0)
                        _max_y_px  = _y_to_px(y_max)
                        _min_y_px  = _y_to_px(y_min)
                        _clear_top    = abs(_max_y_px - _zero_y_px)
                        _clear_bottom = abs(_zero_y_px - _min_y_px)
                        if _clear_top >= 10 and _clear_bottom >= 10:
                            d.add(String(plot_x0 - 4, _zero_y_px - 3, "0%",
                                         fontName="Helvetica", fontSize=6.5,
                                         fillColor=GRAY_SOFT, textAnchor="end"))

                    # Draw each series as a polyline. Z-order: lower-z
                    # first (drawn under), gold/navy last (on top).
                    chart_series_with_style = [
                        (pname, rets, _series_style.get(
                            pname, (GRAY, 0.8, 0))
                        )
                        for pname, rets in chart_series
                    ]
                    chart_series_with_style.sort(key=lambda x: x[2][2])

                    for pname, rets, (col, sw, _z) in chart_series_with_style:
                        n_pts = len(rets)
                        if n_pts < 2:
                            continue
                        # Build flat point list [x0,y0,x1,y1,...]
                        pts_flat = []
                        for i, v in enumerate(rets):
                            x = plot_x0 + (i / (n_pts - 1)) * plot_w
                            y = _y_to_px(v)
                            pts_flat.extend([x, y])
                        d.add(PolyLine(pts_flat,
                                        strokeColor=col,
                                        strokeWidth=sw,
                                        strokeLineJoin=1,  # round joins
                                        strokeLineCap=1))  # round caps

                    return d

                # Build the event-name + chart rows. Each row pairs a
                # left cell (event name + date range + per-portfolio
                # final-return mini-table) with the line chart on the
                # right. Per advisor revision pass (Page 4 landscape
                # restructure): the prior summary-numeric-table-at-top
                # treatment was reintroduced as inline per-event mini-
                # tables so each event's chart sits next to its own
                # numbers, rather than asking the reader to cross-
                # reference a table at the top with the charts below.
                #
                # Mini-table rows are color-coded swatches matching the
                # chart lines, so the reader can map line → portfolio
                # without consulting the legend strip separately. The
                # legend at the top of the page is preserved as a
                # secondary anchor for readers who scan the page
                # before reading any single event row.
                # ── Per-event styles ──
                # Old _chart_table_rows accumulator and _label_para_style /
                # _date_para_style for the 2-col event grid removed —
                # event blocks are now appended to story directly inside
                # the loop below, and header rendering uses an inline
                # _event_hdr_style defined just below.
                # Mini-table cell styles — name in charcoal, return in
                # navy bold so the number reads first. Returns are
                # rounded to whole percent for table-grade scannability;
                # the chart shows the actual path for anyone who wants
                # finer precision.
                _mini_name_style = ParagraphStyle(
                    "mini_name", fontSize=7.5, leading=10,
                    textColor=CHARCOAL, fontName="Helvetica",
                    alignment=TA_LEFT,
                )
                _mini_ret_style = ParagraphStyle(
                    "mini_ret", fontSize=8, leading=10,
                    textColor=NAVY, fontName="Helvetica-Bold",
                    alignment=TA_RIGHT,
                )
                _mini_ret_neg_style = ParagraphStyle(
                    "mini_ret_neg", fontSize=8, leading=10,
                    textColor=colors.HexColor("#993526"),
                    fontName="Helvetica-Bold", alignment=TA_RIGHT,
                )

                def _swatch(color):
                    """Small filled circle matching the chart-line
                    color, used as the leading column in each mini-
                    table row so the reader can map color → line."""
                    _sw = Drawing(10, 10)
                    _sw.add(Circle(5, 5, 3.2,
                                   fillColor=color, strokeColor=None))
                    return _sw

                def _fmt_pct_row(v):
                    """Format a decimal return as a signed whole-percent
                    string (+18%, -57%, 0%). Mirrors the y-axis tick
                    formatter used inside the chart so the table and
                    chart speak the same language."""
                    if v is None:
                        return "—"
                    try:
                        return f"{float(v)*100:+.0f}%"
                    except (TypeError, ValueError):
                        return "—"

                # Event header style — bigger Times title with the date
                # range in smaller gray on the same line.
                _event_hdr_style = ParagraphStyle(
                    "event_hdr", fontSize=12, leading=15,
                    textColor=NAVY, fontName="Times-Roman",
                    alignment=TA_LEFT, spaceBefore=0, spaceAfter=0,
                )
                # Description style — same italic Helvetica-Oblique as
                # the page-2 intro paragraph so the per-event "what
                # happened" caption visually matches.
                _event_desc_style = ParagraphStyle(
                    "event_desc", fontSize=8.5, leading=11,
                    textColor=CHARCOAL, fontName="Helvetica-Oblique",
                    alignment=TA_LEFT, spaceBefore=0, spaceAfter=0,
                )

                # Loop builds vertical event blocks (one per crisis).
                # Each block stacks: header (title + date range), 2x2
                # mini-table of portfolio returns, full-width chart,
                # italic description. KeepTogether prevents a single
                # event from splitting across pages — when ReportLab
                # can't fit the whole block on the current page it
                # moves the entire block to the next page.
                for _evt_idx, (label, start, end, _desc) in enumerate(
                        NOTABLE_PERIODS):
                    _chart_d = _build_period_chart(label, start, end)

                    # Header row: title in serif, date range in gray
                    # caption-size to its right on the same line.
                    _hdr_para = Paragraph(
                        f"<b>{label}</b>  "
                        f"<font color='{GRAY.hexval()}' size='9' "
                        f"face='Helvetica'>"
                        f"({start[:7]} – {end[:7]})</font>",
                        _event_hdr_style,
                    )

                    # 2x2 mini-table: 2 rows × 2 portfolios per row, 3
                    # cells per portfolio (swatch + name + return). Top
                    # row holds the two CLIENT portfolios (Recommended +
                    # Current); bottom row holds the two BENCHMARKS
                    # (SPY + BND). This groups same-class items
                    # together so the eye doesn't bounce around when
                    # scanning. Total table width 4.85" — sits at the
                    # left edge of the chart so the visual baseline
                    # below the title runs cleanly across.
                    def _mini_cell(pname):
                        col, _, _ = _series_style.get(
                            pname, (GRAY, 0.8, 0))
                        _ret = None
                        if _period_data and pname in _period_data:
                            _ret = _period_data[pname].get(label)
                        _ret_style = (_mini_ret_neg_style
                                       if (_ret is not None and _ret < 0)
                                       else _mini_ret_style)
                        _pname_safe = (pname
                                        .replace("&", "&amp;")
                                        .replace("<", "&lt;"))
                        return [_swatch(col),
                                Paragraph(_pname_safe, _mini_name_style),
                                Paragraph(_fmt_pct_row(_ret), _ret_style)]

                    # Bind portfolio order safely whether 2 or 4
                    # portfolios are present (current snapshot might be
                    # absent → only 3 names).
                    _pnames = list(_np_portfolio_names)
                    while len(_pnames) < 4:
                        _pnames.append(None)

                    def _row_cells(p1, p2):
                        _cells = []
                        if p1:
                            _cells.extend(_mini_cell(p1))
                        else:
                            _cells.extend([Spacer(1,1), Paragraph("", _mini_name_style), Paragraph("", _mini_ret_style)])
                        if p2:
                            _cells.extend(_mini_cell(p2))
                        else:
                            _cells.extend([Spacer(1,1), Paragraph("", _mini_name_style), Paragraph("", _mini_ret_style)])
                        return _cells

                    _mini_2x2 = Table(
                        [
                            _row_cells(_pnames[0], _pnames[1]),
                            _row_cells(_pnames[2], _pnames[3]),
                        ],
                        colWidths=[0.20*inch, 1.65*inch, 0.55*inch,
                                    0.30*inch, 1.65*inch, 0.50*inch],
                        hAlign="LEFT",
                    )
                    _mini_2x2.setStyle(TableStyle([
                        ("LEFTPADDING",  (0,0), (-1,-1), 0),
                        ("RIGHTPADDING", (0,0), (-1,-1), 2),
                        ("TOPPADDING",   (0,0), (-1,-1), 1),
                        ("BOTTOMPADDING",(0,0), (-1,-1), 1),
                        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                        ("ALIGN",        (2,0), (2,-1), "RIGHT"),
                        ("ALIGN",        (5,0), (5,-1), "RIGHT"),
                    ]))

                    # Italic description of what happened during the
                    # crisis — same style as the page-2 intro paragraph
                    # (italic Helvetica-Oblique 8.5pt). Was previously
                    # inline at the bottom of the section; now sits
                    # under each event's chart for direct association.
                    _desc_para = Paragraph(_desc, _event_desc_style)

                    # Compose: header → description (italic, beneath
                    # title) → mini-table → chart. Description moved
                    # from below-chart to beneath-title per advisor —
                    # event context now reads at the top of the block
                    # before the data viz. Tight spacers throughout.
                    _event_block = [
                        _hdr_para,
                        Spacer(1, 0.02*inch),
                        _desc_para,
                        Spacer(1, 0.04*inch),
                        _mini_2x2,
                        Spacer(1, 0.06*inch),
                        _chart_d,
                        Spacer(1, 0.08*inch),
                    ]
                    story.append(KeepTogether(_event_block))

                    # Force a page break after the 3rd event so the
                    # first 3 land on the section's first page and the
                    # remaining 2 flow to the next page. Without this
                    # ReportLab's flow logic puts 2 events per page
                    # because of cumulative height.
                    if _evt_idx == 2:
                        story.append(NextPageTemplate('landscape'))
                        story.append(PageBreak())

                # No "Return paths" sub-header — with the summary table
                # removed, the charts ARE the section content and don't
                # need a title differentiating them from a sibling block.
                # The section header "Notable Market Periods" at the top
                # of the page covers what these charts represent.

                # Top legend strip REMOVED per advisor — the 2x2
                # mini-table in each event block already names the
                # portfolios with their color swatches; a separate
                # top-of-section legend duplicated that info.

            # Bottom inline _periods_inline list REMOVED — each event's
            # description now sits in italic Helvetica-Oblique 8.5pt
            # directly below its own chart (inside the event block).
            # The final italic disclaimer below stays so the legal
            # language about past performance remains visible.
            story.append(Spacer(1, 0.04*inch))
            story.append(Paragraph(
                "<i>Returns are total returns over the full event window "
                "(crash + recovery). Hypothetical comparisons assume the "
                "proposed allocation was held throughout without "
                "rebalancing. Past performance does not guarantee future "
                "results.</i>",
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

    # Restore portrait orientation for the remaining pages (Backtest,
    # Fees, Advisor, Methodology, Disclosures). Page 4 above was a
    # one-off landscape page; everything after it is portrait again.
    story.append(NextPageTemplate('portrait'))

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
            # Intro paragraph styled to match pages 2/3 — small italic
            # Helvetica-Oblique sitting tight beneath the header rule.
            # Removed the trailing "The chart shows..." sentence — the
            # 3-year growth chart was removed when this section was
            # converted to a transposed metrics-only table, so the
            # reference is no longer accurate.
            _bt_desc_tbl = Table(
                [[Paragraph(
                    "The table below compares the compound annual growth "
                    "rate (CAGR), annualized volatility, maximum drawdown, "
                    "and Sharpe ratio across each proposed portfolio, your "
                    "current portfolio, and the S&amp;P 500 and aggregate-bond "
                    "benchmarks over 1, 3, 5, and 10 year horizons.",
                    _intro_desc_style)]],
                colWidths=[7.4*inch],
                hAlign="LEFT",
            )
            _bt_desc_tbl.setStyle(TableStyle([
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))
            _bt_desc_tbl.spaceBefore = -4
            story.append(_bt_desc_tbl)
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
            for idx, (lbl, sub, tk, ptks, pws, pscore) in enumerate(ordered_picks):
                if ptks and pws:
                    # Label each backtest portfolio by its DESCRIPTOR,
                    # mirroring the comparison cards on page 3 (which show
                    # PROPOSED / MORE CONSERVATIVE / MORE AGGRESSIVE as the
                    # headline). The previous code used a positional
                    # "Comparison #{idx+1}" index — but ordered_picks sorts
                    # conservative → balanced → aggressive, so the balanced
                    # tier (which IS the proposed portfolio) always landed
                    # at idx 1 and got mislabeled "Comparison #2". The
                    # backtest legend now names the proposed portfolio
                    # "Proposed" outright.
                    #
                    # Compact single-word forms ("Proposed" / "Conservative"
                    # / "Aggressive") keep the backtest table's column
                    # headers within their ~1.3"-per-column budget — the
                    # "More" prefix the cards use is dropped here for fit.
                    if tk == "balanced":
                        _bt_label = "Proposed"
                    elif tk == "conservative":
                        _bt_label = "Conservative"
                    elif tk == "aggressive":
                        _bt_label = "Aggressive"
                    else:
                        # Custom / alternate pick with no standard tier
                        # key — inspect the human-readable label the same
                        # way the page-3 card descriptor logic does.
                        _u = (lbl or "").upper()
                        if "PROPOSED" in _u or "RECOMMENDED" in _u:
                            _bt_label = "Proposed"
                        elif "CONSERVATIVE" in _u:
                            _bt_label = "Conservative"
                        elif "AGGRESSIVE" in _u:
                            _bt_label = "Aggressive"
                        else:
                            _bt_label = f"Comparison #{idx+1}"
                    bt_portfolios.append(
                        (_bt_label, ptks, pws)
                    )

            # Fixed market benchmarks — added so the grid compares the
            # client's portfolios against a 100% S&P 500 and a 100%
            # aggregate-bond reference. These render as two extra rows under
            # every metric group, alongside Current / Proposed / Conservative
            # / Aggressive, for six comparisons in total. Tracked by name so
            # the table builder can style them as muted "benchmark" rows.
            _bt_benchmark_names = ("S&P 500 (SPY)", "Agg Bond (BND)")
            bt_portfolios.append(("S&P 500 (SPY)",  ["SPY"], [100.0]))
            bt_portfolios.append(("Agg Bond (BND)", ["BND"], [100.0]))

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
                        """Return (total, vol, sharpe, maxdd, cagr) over last N days.

                        Sharpe is excess-return Sharpe ((CAGR - rf) / vol) so it
                        matches the rest of the codebase's definition. Old code
                        used (mean*√252)/std which biased Sharpe upward by both
                        omitting rf and using arithmetic mean instead of CAGR.
                        """
                        if r is None or len(r) < days * 0.3:
                            return (None, None, None, None, None)
                        r_w = r.iloc[-days:] if len(r) > days else r
                        if len(r_w) < 10: return (None, None, None, None, None)
                        total = float((1 + r_w).prod() - 1)
                        vol = float(r_w.std() * _np.sqrt(252))
                        # CAGR over the actual window
                        actual_yrs = max(len(r_w) / 252.0, 0.08)
                        ann_r = (1 + total) ** (1.0 / actual_yrs) - 1
                        sharpe = _shared_sharpe(ann_r, vol)
                        equity = (1 + r_w).cumprod()
                        dd = float(((equity / equity.cummax()) - 1).min())
                        return (total, vol, sharpe, dd, ann_r)

                    # ── Build the table: cols = periods, rows = portfolios within metric ──
                    # Per advisor feedback: periods (1Y / 3Y / 5Y / 10Y)
                    # run across the top, portfolios (Current,
                    # Comparison #1/2/3) run down the left grouped under
                    # each metric heading. Same one-number-per-cell
                    # density as before, just transposed so a reader
                    # scanning a single portfolio's track record reads
                    # across one row instead of jumping down a column.

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

                    # Build header row: blank top-left, then period labels
                    _period_labels = [p[0] for p in _periods]
                    _portfolio_names = [name for name, _, _ in bt_portfolios]
                    hdr = ["Metric / Portfolio"] + _period_labels
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
                    # and per-portfolio rows under each.
                    _metric_groups = [
                        ("Compound Annual Growth Rate (CAGR)", 4, "pct"),
                        ("Maximum Drawdown",      3, "pct"),
                        ("Annualized Volatility", 1, "pct_abs"),
                        ("Sharpe Ratio",          2, "ratio"),
                    ]

                    # Track which rows are group headers (for styling)
                    _group_header_rows = []
                    # Track benchmark rows (S&P 500 / Agg Bond) so they can be
                    # styled as muted reference lines, distinct from the
                    # client's proposed/current portfolios.
                    _benchmark_rows = []
                    for _grp_label, _stat_idx, _kind in _metric_groups:
                        # Group header row: spans all columns, just the label
                        _group_header_rows.append(len(bt_rows))
                        bt_rows.append([_grp_label] + [""] * len(_period_labels))
                        # Then one row per portfolio
                        for _pi, _pname in enumerate(_portfolio_names):
                            if _pname in _bt_benchmark_names:
                                _benchmark_rows.append(len(bt_rows))
                            _row = [f"   {_pname}"]
                            for _period_idx, _ in enumerate(_periods):
                                _stat_tuple = _stats[_pi][_period_idx]
                                if _stat_tuple is None:
                                    _row.append("—")
                                else:
                                    _row.append(_fmt(_stat_tuple[_stat_idx], _kind))
                            bt_rows.append(_row)

                    # Column widths: first col wider for the portfolio
                    # label, remaining cols split evenly across the 4
                    # period columns.
                    _n_period_cols = len(_period_labels)
                    _label_w = 1.7*inch
                    _period_w  = (5.0*inch) / max(1, _n_period_cols)
                    bt_tbl = Table(bt_rows,
                                   colWidths=[_label_w] + [_period_w] * _n_period_cols)
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
                        # Padding tightened 5→3.5 so six portfolios (Current,
                        # Proposed, Conservative, Aggressive + 2 benchmarks)
                        # still fit on one portrait page within KeepTogether.
                        ("TOPPADDING",   (0,0), (-1,-1), 3.5),
                        ("BOTTOMPADDING",(0,0), (-1,-1), 3.5),
                        ("BOX",          (0,0), (-1,-1), 1.0, NAVY),
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
                        _tbl_style.append(("TOPPADDING", (0,_gi), (-1,_gi), 5))
                        _tbl_style.append(("BOTTOMPADDING",(0,_gi),(-1,_gi), 3))
                    # Benchmark rows (S&P 500 / Agg Bond) render italic + muted
                    # gray so they read as reference context, not as one of the
                    # client's proposed portfolios — mirroring how the Notable
                    # Market Periods charts treat the SPY/BND benchmark lines.
                    for _br in _benchmark_rows:
                        _tbl_style.append(("FONTNAME",  (0,_br), (-1,_br), "Helvetica-Oblique"))
                        _tbl_style.append(("TEXTCOLOR", (0,_br), (-1,_br), GRAY))
                    bt_tbl.setStyle(TableStyle(_tbl_style))
                    # Wrap table + caption together so they never split
                    _bt_caption_centered = ParagraphStyle(
                        "bt_caption_centered", parent=caption,
                        alignment=TA_CENTER,
                    )
                    story.append(KeepTogether([
                        bt_tbl,
                        Paragraph(
                            "<i>Each row reports a single metric over 1, 3, 5, "
                            "and 10-year windows. CAGR is the compound annual "
                            "growth rate over each window; Sharpe ratio is the "
                            "annualized return in excess of the risk-free rate "
                            "divided by volatility (standard deviation) — i.e., "
                            "return earned per unit of risk.</i>",
                            _bt_caption_centered,
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
    # OPTIONAL — SECTION 5: FEE COMPARISON TABLE (own page)
    # ═══════════════════════════════════════════════════════════
    # When the advisor ticks "Fee comparison table" in the section picker,
    # render the table on its own page so the client can study it without
    # the visual weight of disclosure paragraphs around it. The table
    # itself stays in the disclosures section as well (smaller, tighter)
    # for compliance — this version is the larger comparison-focused one.
    if sections.get("fee_comparison", False):
        story.append(PageBreak())
        story.append(section_header("Section 5", "Fee Comparison"))

        _adv_fee_pct_for_compare = _resolve_advisory_fee_pct(
            proposal, client_profile, _firm_settings,
        )

        # Client's current advisory fee (optional) — shown alongside the
        # proposed fee so the prospect can compare what they pay now vs the
        # proposal. None/0 means "not applicable" and nothing extra renders.
        _cur_fee_pct = None
        try:
            _cfv = float((proposal or {}).get("current_advisory_fee_pct"))
            if 0 < _cfv <= 10:
                _cur_fee_pct = round(_cfv, 2)
        except (TypeError, ValueError):
            _cur_fee_pct = None

        # Intro paragraph styled to match pages 2/3 — small italic
        # Helvetica-Oblique sitting tight beneath the header rule.
        _cur_clause = ""
        if _cur_fee_pct is not None:
            _cur_clause = (f" Your current advisory fee of "
                           f"<b>{_cur_fee_pct:.2f}%</b> (\u25CF) is included for "
                           f"comparison.")
        _fee_desc_tbl = Table(
            [[Paragraph(
                f"This table illustrates how different annual advisory "
                f"fee levels affect the growth of a hypothetical $100 "
                f"starting balance over common time horizons. Your firm's "
                f"fee of <b>{_adv_fee_pct_for_compare:.2f}%</b> (\u2605) is shown "
                f"alongside industry benchmark levels (0%, 0.25%, 0.5%, "
                f"0.75%, 1%, 1.5%, 2%, 2.5%) so you can compare.{_cur_clause} "
                f"All figures assume a 7% gross annual return compounded "
                f"monthly. Actual returns will differ.",
                _intro_desc_style)]],
            colWidths=[7.4*inch],
            hAlign="LEFT",
        )
        _fee_desc_tbl.setStyle(TableStyle([
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        _fee_desc_tbl.spaceBefore = -4
        story.append(_fee_desc_tbl)
        story.append(Spacer(1, 0.15*inch))

        # Build a fee table that includes the firm's actual rate (and the
        # client's current rate, if given) alongside the standard benchmarks.
        # Rates are inserted in sorted order so the table reads as a smooth
        # gradient; duplicates of a benchmark are collapsed.
        _benchmark_fees = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
        _extra_fees = [round(_adv_fee_pct_for_compare, 2)]
        if _cur_fee_pct is not None:
            _extra_fees.append(_cur_fee_pct)
        _all_fees = sorted(set(_benchmark_fees + _extra_fees))
        _fee_rows_full = _fee_impact_table_data(fee_levels=_all_fees)

        # Mark the firm's proposed fee and the client's current fee. The
        # firm's fee is flagged with the firm logo (the uploaded Spartan
        # mark) when one is configured, falling back to a ★ if not; the
        # client's current fee stays a ●. If both land on one rate, that
        # row carries both.
        _firm_fee_str = f"{_adv_fee_pct_for_compare:.2f}%"
        _cur_fee_str = (f"{_cur_fee_pct:.2f}%" if _cur_fee_pct is not None else None)
        _logo_ok = os.path.exists(FIRM_LOGO_PATH)
        _fee_cell_style = ParagraphStyle(
            "fee_cell", fontName="Helvetica", fontSize=9, leading=11,
            textColor=CHARCOAL, alignment=TA_LEFT)
        for i, row in enumerate(_fee_rows_full):
            if i == 0:
                continue
            _is_firm = (row[0] == _firm_fee_str)
            _is_cur = (_cur_fee_str is not None and row[0] == _cur_fee_str)
            if _is_firm and _logo_ok:
                _cur_mark = " \u25CF" if _is_cur else ""
                row[0] = Paragraph(
                    f'{row[0]}&nbsp;&nbsp;<img src="{FIRM_LOGO_PATH}" '
                    f'width="12" height="13" valign="middle"/>{_cur_mark}',
                    _fee_cell_style)
            else:
                _marks = ""
                if _is_firm:
                    _marks += " \u2605"
                if _is_cur:
                    _marks += " \u25CF"
                if _marks:
                    row[0] = row[0] + _marks

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
            ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.2, ACCENT),
        ]))
        story.append(_fee_tbl_full)
        story.append(Spacer(1, 0.10*inch))

        # When the client's current fee is known and differs from the
        # proposed fee, quantify the gap at the illustrated return: how much
        # more of every $100 is kept over 10 years at the lower fee.
        if _cur_fee_pct is not None and abs(_cur_fee_pct - _adv_fee_pct_for_compare) >= 0.005:
            def _fv100(_fee):
                _nm = (0.07 - _fee / 100.0) / 12.0
                return 100.0 * ((1.0 + _nm) ** 120)
            _diff10 = abs(_fv100(_adv_fee_pct_for_compare) - _fv100(_cur_fee_pct))
            _lower = ("proposed" if _adv_fee_pct_for_compare < _cur_fee_pct
                      else "current")
            _spread = abs(_cur_fee_pct - _adv_fee_pct_for_compare)
            story.append(Paragraph(
                f"At the illustrated 7% gross return, the "
                f"<b>{_spread:.2f}%</b> difference between the current "
                f"(<b>{_cur_fee_pct:.2f}%</b>) and proposed "
                f"(<b>{_adv_fee_pct_for_compare:.2f}%</b>) fee is worth about "
                f"<b>${_diff10:,.2f}</b> per $100 over 10 years — kept under "
                f"the {_lower} fee.",
                _intro_desc_style,
            ))
            story.append(Spacer(1, 0.08*inch))

        _legend = "<i>" + (
            "Your firm's mark denotes the proposed fee."
            if _logo_ok else "\u2605 marks your firm's proposed fee.")
        if _cur_fee_pct is not None:
            _legend += " \u25CF marks the client's current fee."
        _legend += (" Performance figures are hypothetical and for "
                    "illustration only. Past performance does not guarantee "
                    "future results.</i>")
        story.append(Paragraph(_legend, caption))

        # ── Total cost of ownership — proposed portfolios ──────────────
        # The fee table above covers the *advisory* fee only. Clients also
        # pay the underlying funds' expense ratios. This block shows each
        # proposed portfolio's weighted fund expense ratio, the advisory
        # fee, and the all-in total so the reader sees the complete cost.
        _toc_opts = [o for o in (_opt1_resolved, _opt2_resolved, _opt3_resolved)
                     if o and o[3]]   # o[3] = tickers; skip external/saved
        if _toc_opts:
            _toc_rows = [["Proposed Portfolio", "Fund Expenses",
                          "Advisory Fee", "All-In Cost"]]
            _adv_dec = _adv_fee_pct_for_compare  # already in percent units
            _toc_partial = False
            for _o in _toc_opts:
                _lbl, _tks, _wts = _o[0], _o[3], _o[4]
                try:
                    _wer, _cov = weighted_expense_ratio(_tks, _wts)
                except Exception:
                    _wer, _cov = 0.0, 0.0
                _wer_pct = (_wer or 0.0) * 100.0
                _mark = ""
                if _cov is not None and _cov < 99.5:
                    _toc_partial = True
                    _mark = "*"
                _toc_rows.append([
                    str(_lbl),
                    f"{_wer_pct:.2f}%{_mark}",
                    f"{_adv_dec:.2f}%",
                    f"{_wer_pct + _adv_dec:.2f}%",
                ])
            _toc_tbl = Table(
                _toc_rows,
                colWidths=[2.9*inch, 1.5*inch, 1.5*inch, 1.5*inch],
            )
            _toc_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",      (-1, 1), (-1, -1), "Helvetica-Bold"),
                ("TEXTCOLOR",     (-1, 1), (-1, -1), NAVY),
                ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, BG_SOFT]),
                ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
                ("LINEBELOW",     (0, 0), (-1, 0),  1.2, ACCENT),
            ]))
            story.append(Spacer(1, 0.22*inch))
            story.append(Paragraph(
                "<b>Total Cost of Ownership — Proposed Portfolios</b>",
                _intro_desc_style,
            ))
            story.append(Spacer(1, 0.06*inch))
            story.append(_toc_tbl)
            _toc_note = ("<i>Fund Expenses is the weighted average net expense "
                         "ratio of the portfolio's underlying funds. All-In Cost "
                         "adds the advisory fee. Fund expenses are paid to the "
                         "fund companies, not the advisor.")
            if _toc_partial:
                _toc_note += (" *Expense-ratio data was unavailable for some "
                              "holdings; the weighted figure covers the rest.")
            _toc_note += "</i>"
            story.append(Spacer(1, 0.08*inch))
            story.append(Paragraph(_toc_note, caption))

    # Implementation Plan (formerly Section 5 here) has been moved to the
    # second-to-last page, beneath the Advisor signature card. See the
    # ADVISOR INFO + IMPLEMENTATION block further down.
    #
    # Advisor Notes (formerly Section 6 here, on its own page) has also
    # been moved to the second-to-last page where it now sits between the
    # Your Advisor signature card and the Implementation Plan, per
    # advisor request — keeps the human-context narrative directly
    # adjacent to "who said it" and "what's next."

    # ═══════════════════════════════════════════════════════════
    # SECOND-TO-LAST PAGE — Advisor Info → Notes → Implementation
    # ═══════════════════════════════════════════════════════════
    _has_photo = os.path.exists(ADVISOR_PHOTO_PATH)
    _show_advisor_card = (_has_photo or _has_any_firm_text)
    _show_notes = sections.get("notes", True)
    _show_implementation = sections.get("proposals", True)

    if _show_advisor_card or _show_notes or _show_implementation:
        story.append(PageBreak())

        # ── Build all three sections into a single KeepTogether block ──
        # Per advisor request: advisor info, advisor notes, and implementation
        # plan must all appear on ONE page (not split across two). Wrapping
        # the trio in a single KeepTogether instructs ReportLab to evaluate
        # them as one unit — if the whole thing doesn't fit on the current
        # page, it'll start fresh on the next, but the three sections won't
        # split apart from each other.
        #
        # Spacers are kept tight (0.12-0.15") so the trio comfortably fits
        # in a single page's vertical budget (~9" of usable space after
        # margins). With aggressive trimming the page reads:
        #   Section 6 header + sig card  (~1.7")
        #   Section 7 header + notes     (~1.0" empty / ~2-4" with content)
        #   Section 8 header + 5-row tbl (~2.2")
        #   ── plus ~0.4" of inter-section spacers
        # Total: ~5.3"-7.3" depending on notes length. Fits comfortably.
        _combined_block = []

        if _show_advisor_card:
            _combined_block.append(section_header("Section 6", "Your Advisor"))

            def _circular_photo(path, diameter):
                """Advisor headshot clipped to a circle — mirrors the
                client portal, which clips the photo into an SVG
                <circle> with preserveAspectRatio='xMidYMid slice'.

                ReportLab's Image flowable can't clip to a shape, so
                the photo is center-cropped to a square and given a
                circular alpha mask here with PIL; the masked PNG
                (transparent outside the circle) is then embedded, so
                the card background shows through the corners.
                Rendered at 4x and downsampled so the circular edge is
                anti-aliased rather than stair-stepped.
                """
                from PIL import Image as PILImage, ImageDraw
                _SS = 4
                _R  = 256 * _SS
                src = PILImage.open(path).convert("RGBA")
                _w, _h = src.size
                # Center-crop to a square (= the portal's xMidYMid slice).
                _side = min(_w, _h)
                _l = (_w - _side) // 2
                _t = (_h - _side) // 2
                src = src.crop((_l, _t, _l + _side, _t + _side)
                               ).resize((_R, _R), PILImage.LANCZOS)
                # Circular alpha mask — hard-edged at 4x, then photo and
                # mask are downsampled together for a smooth edge.
                _mask = PILImage.new("L", (_R, _R), 0)
                ImageDraw.Draw(_mask).ellipse((0, 0, _R - 1, _R - 1),
                                              fill=255)
                _out = PILImage.new("RGBA", (_R, _R), (0, 0, 0, 0))
                _out.paste(src, (0, 0), _mask)
                _out = _out.resize((_R // _SS, _R // _SS),
                                   PILImage.LANCZOS)
                _buf = BytesIO()
                _out.save(_buf, format="PNG")
                _buf.seek(0)
                return Image(_buf, width=diameter, height=diameter)

            if _has_photo:
                try:
                    _photo_flow = _circular_photo(ADVISOR_PHOTO_PATH,
                                                  0.95*inch)
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

            # Contact-row icons matching the client portal's outline glyphs.
            # The emoji characters used previously (envelope, telephone, globe)
            # rendered as missing-glyph boxes because the embedded Helvetica
            # fonts in this PDF don't carry color emoji. Drawing them as
            # small vector shapes guarantees they render identically across
            # PDF viewers and stays consistent with the client portal's
            # outline-icon treatment for email / phone / website.
            # Contact-row icons — geometry translated from the client
            # portal's Lucide icon set (mail / phone / globe) so the PDF
            # advisor card and the portal's contact rows show the same
            # symbols. Lucide icons are authored on a 24-unit grid; the
            # X()/Y() helpers scale that grid into this `size`-unit
            # Drawing and flip Y (SVG is y-down, ReportLab is y-up).
            def _email_icon(size=10, color=SLATE):
                """Envelope — rounded-rectangle body + chevron flap.
                Translated from the client portal's Lucide 'mail' icon
                (rect x3 y5 w18 h14 rx2; flap path M3 7 l9 6 l9 -6)."""
                d = Drawing(size, size)
                k = size / 24.0
                def X(v): return v * k
                def Y(v): return size - v * k
                # Envelope body — rounded rectangle. ReportLab Rect takes
                # the BOTTOM-left corner, so y is the SVG bottom edge (19).
                d.add(Rect(X(3), Y(19), X(18), X(14),
                           rx=X(2), ry=X(2),
                           fillColor=None, strokeColor=color,
                           strokeWidth=0.8))
                # Chevron flap — polyline through (3,7) (12,13) (21,7).
                d.add(PolyLine([X(3),  Y(7),
                                X(12), Y(13),
                                X(21), Y(7)],
                               strokeColor=color, strokeWidth=0.8,
                               strokeLineJoin=1))
                return d

            def _phone_icon(size=10, color=SLATE):
                """Phone handset — a thick, gently-curved stroke with
                round caps. The round caps form the earpiece /
                mouthpiece bulbs and the quarter-bend curve is the
                receiver's handle, matching the form of the client
                portal's Lucide 'phone' handset.

                A verbatim trace of the Lucide path isn't possible
                here: that path relies on SVG arc ('a') commands, and
                ReportLab's Path object supports only moveTo / lineTo /
                curveTo / closePath. The handset is therefore
                reconstructed from a single thick cubic-bezier stroke
                — chunky enough that the round line caps read as the
                two bulbs.
                """
                d = Drawing(size, size)
                s = size
                p = Path(strokeColor=color, fillColor=None,
                         strokeWidth=s * 0.23,   # chunky → caps read as bulbs
                         strokeLineCap=1)        # round caps = the bulbs
                # Quarter-bend: earpiece upper-left → mouthpiece
                # lower-right, concave side facing the upper-right.
                p.moveTo(s * 0.30, s * 0.72)
                p.curveTo(s * 0.30, s * 0.46,
                          s * 0.46, s * 0.30,
                          s * 0.72, s * 0.30)
                d.add(p)
                return d

            def _globe_icon(size=10, color=SLATE):
                """Globe — circle + straight equator + vertical meridian
                lens. Translated from the client portal's Lucide 'globe'
                icon (circle r10; equator M2 12 h20; the curved meridian
                lens is approximated by an ellipse rx≈4 ry≈10)."""
                d = Drawing(size, size)
                k = size / 24.0
                def X(v): return v * k
                def Y(v): return size - v * k
                cx, cy = X(12), Y(12)
                # Outer circle (Lucide r=10)
                d.add(Circle(cx, cy, X(10), fillColor=None,
                             strokeColor=color, strokeWidth=0.8))
                # Equator — straight horizontal line (Lucide "M2 12 h20")
                d.add(Line(X(2), cy, X(22), cy,
                           strokeColor=color, strokeWidth=0.8))
                # Meridian lens — Lucide draws a curved lens via SVG
                # arcs; an ellipse (rx≈4, ry≈10) is a faithful stand-in.
                d.add(Ellipse(cx, cy, X(4), X(10), fillColor=None,
                              strokeColor=color, strokeWidth=0.8))
                return d

            def _contact_row(icon_draw, text):
                """Small 2-column row — icon on the left, contact text on
                the right. Used for the three contact lines below the
                advisor name so each icon aligns vertically with its text.
                colWidths sum (0.25 + 5.60 = 5.85") fits within the parent
                cell's effective width (6.15" outer − 0.14" L/R padding =
                5.87")."""
                t = Table(
                    [[icon_draw, Paragraph(text, _sig_contact)]],
                    colWidths=[0.25*inch, 5.60*inch],
                )
                t.setStyle(TableStyle([
                    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                    ("LEFTPADDING",  (0,0), (-1,-1), 0),
                    ("RIGHTPADDING", (0,0), (-1,-1), 0),
                    ("TOPPADDING",   (0,0), (-1,-1), 1),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 1),
                ]))
                return t

            _sig_right = [Paragraph("YOUR ADVISOR", _sig_label)]
            if _adv_name:
                _sig_right.append(Paragraph(_adv_name, _sig_name))
            _role_bits = []
            if _adv_title: _role_bits.append(_adv_title)
            if _firm_name: _role_bits.append(_firm_name)
            if _role_bits:
                _sig_right.append(Paragraph(" · ".join(_role_bits), _sig_role))
            if _adv_email:
                _sig_right.append(_contact_row(_email_icon(), _adv_email))
            if _adv_phone:
                _sig_right.append(_contact_row(_phone_icon(), _adv_phone))
            if _firm_website:
                _sig_right.append(_contact_row(_globe_icon(), _firm_website))

            _sig_card = Table(
                [[_photo_flow, _sig_right]],
                colWidths=[1.15*inch, 6.15*inch],
            )
            _sig_card.setStyle(TableStyle([
                ("VALIGN",       (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",  (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING",   (0,0), (-1,-1), 8),
                ("BOTTOMPADDING",(0,0), (-1,-1), 8),
                ("BOX",          (0,0), (-1,-1), 1.0, NAVY),
                ("BACKGROUND",   (0,0), (-1,-1), BG_LIGHT),
            ]))
            _combined_block.append(Spacer(1, 0.06*inch))
            _combined_block.append(_sig_card)

        if _show_notes:
            # Advisor Notes — sits between the advisor signature card
            # and the implementation plan. The note shown is the
            # per-proposal note if one was attached, otherwise the
            # firm-wide generic note from the PDF Content tab.
            _combined_block.append(Spacer(1, 0.18*inch))
            _combined_block.append(section_header("Section 7", "Advisor Notes"))
            _notes = (proposal.get("advisor_notes") or "").strip()
            if not _notes:
                _notes = (_pdf_content.get("advisor_notes") or "").strip()
            if _notes:
                for _flow in _render_pdf_prose(_notes, body):
                    _combined_block.append(_flow)
            else:
                _combined_block.append(Paragraph(
                    "<i>No additional notes were attached to this proposal version.</i>",
                    body_small,
                ))

        if _show_implementation:
            _combined_block.append(Spacer(1, 0.18*inch))
            _combined_block.append(section_header("Section 8", "Implementation Plan"))
            # Header row is fixed; body rows come from the PDF Content
            # tab (advisor edits) or DEFAULT_PDF_CONTENT.
            # Body cells are Paragraphs (not bare strings) so long Action
            # text wraps within the column and stacks onto extra lines
            # instead of overflowing the table's right edge. Header row stays
            # string-based so the navy/white TableStyle treatment applies.
            _impl_stage_style = ParagraphStyle(
                "impl_stage", fontSize=9, leading=12,
                fontName="Helvetica-Bold", textColor=NAVY_MID)
            _impl_body_style = ParagraphStyle(
                "impl_body", fontSize=9, leading=12,
                fontName="Helvetica", textColor=CHARCOAL)

            def _impl_esc(_s):
                return str(_s).replace("&", "&amp;").replace("<", "&lt;")

            impl_rows = [["Stage", "Cadence", "Action"]]
            for _ir in _pdf_content.get("implementation_plan", []):
                _ir = (list(_ir) + ["", "", ""])[:3]
                impl_rows.append([
                    Paragraph(_impl_esc(_ir[0]), _impl_stage_style),
                    Paragraph(_impl_esc(_ir[1]), _impl_body_style),
                    Paragraph(_impl_esc(_ir[2]), _impl_body_style),
                ])
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
                ("TOPPADDING",   (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0), (-1,-1), 5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, BG_SOFT]),
                ("BOX",          (0,0), (-1,-1), 1.0, NAVY),
                ("LINEBELOW",    (0,0), (-1,0), 1.2, ACCENT),
            ]))
            _combined_block.append(tbl)

        # Single KeepTogether for all three sections so ReportLab evaluates
        # them as a unit. If the unit doesn't fit, it moves to next page
        # whole — but the three sections won't split apart.
        story.append(KeepTogether(_combined_block))


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
    # Body text comes from the PDF Content tab (advisor edits) or
    # DEFAULT_PDF_CONTENT — blank-line-separated paragraphs.
    for _flow in _render_pdf_prose(_pdf_content.get("methodology", ""), body):
        story.append(_flow)
    story.append(Spacer(1, 0.20*inch))

    # ── Disclosures (final page below methodology) ──────────────
    story.append(section_header("Disclosures", "Important Information"))

    story.append(Paragraph("Key Definitions", h3))
    # Glossary rows come from the PDF Content tab or DEFAULT_PDF_CONTENT.
    # Same wrapping fix as Implementation Plan — Paragraph cells so a long
    # custom Definition wraps within its column instead of running off-table.
    _gl_term_style = ParagraphStyle(
        "gl_term", fontSize=9, leading=12,
        fontName="Helvetica-Bold", textColor=NAVY)
    _gl_def_style = ParagraphStyle(
        "gl_def", fontSize=9, leading=12,
        fontName="Helvetica", textColor=CHARCOAL)

    def _gl_esc(_s):
        return str(_s).replace("&", "&amp;").replace("<", "&lt;")

    glossary = []
    for _gr in _pdf_content.get("key_definitions", []):
        _gr = (list(_gr) + ["", ""])[:2]
        glossary.append([
            Paragraph(_gl_esc(_gr[0]), _gl_term_style),
            Paragraph(_gl_esc(_gr[1]), _gl_def_style),
        ])
    if glossary:
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
            ("BOX",           (0,0), (-1,-1), 1.0, NAVY),
        ]))
        story.append(gl)

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Important Performance Disclosures", h3))

    # Advisory fee resolved via the fallback chain so {advisory_fee}
    # tokens in the (possibly advisor-edited) disclosure text reflect
    # what the client is actually charged. Falls back to 1.00% if
    # nothing is configured — the firm default SHOULD be set in
    # firm_settings so the disclosure matches the actual ADV.
    _adv_fee_pct = _resolve_advisory_fee_pct(
        proposal, client_profile, _firm_settings)
    _disc_text = (_pdf_content.get("disclosures") or "").replace(
        "{advisory_fee}", f"{_adv_fee_pct:.2f}%")
    for _flow in _render_pdf_prose(_disc_text, body_small):
        story.append(_flow)

    # NOTE: the "Proposal v… · Prepared for … · Generated …" trailing footer
    # credit that previously rendered here was removed May 2026. The disclosure
    # paragraphs above were tall enough to overflow that final block onto an
    # otherwise-empty trailing page (showing only the footer credit on a blank
    # page). The same identifying info already appears on every page in the
    # page footer ("Confidential — Client · Prepared Date" + "Page N"), so
    # the standalone block was redundant.

    doc.build(story)
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


def _portfolio_label_with_score(label):
    """Decorate a portfolio-selectbox label with its risk score in
    parentheses for display.

    Used as `format_func` on the firm-wide portfolio pickers (Load
    portfolio, Client's current portfolio, Select saved portfolio) so
    the advisor can see each portfolio's risk score at a glance without
    expanding the dropdown — "Schwab Core ETF 56/44" becomes
    "Schwab Core ETF 56/44 (61)" in the rendered list. The selectbox
    VALUE remains the original undecorated label, so every downstream
    lookup keyed off the selection (load_saved, _resolve_preset, etc.)
    continues to work unchanged.

    Sentinel rows (Custom, Use analyzed portfolio, separators) and
    anything that fails to resolve are passed through unchanged — the
    function silently fails open so a misconfigured saved portfolio
    can never break the picker.

    Score computation is cheap: compute_portfolio_risk_score does
    arithmetic, and the per-ticker security_risk_score it depends on
    is @st.cache_data'd at 1h TTL, so on a warm cache every call here
    is effectively free.
    """
    if not label:
        return label
    # Sentinel / non-portfolio rows pass through unchanged
    if label in (
        "Custom — Enter Your Own Tickers",
        "Use analyzed portfolio (Section 1)",
    ) or label.startswith("── "):
        return label
    try:
        # Saved portfolio: "📁 <name>". Saved entries store weights as
        # decimals (0.6 = 60%); compute_portfolio_risk_score expects
        # percentages, so convert.
        if label.startswith("📁 "):
            sp = load_saved().get(label[2:])
            if not sp:
                return label
            _tks = sp.get("tickers", []) or []
            _ws_dec = sp.get("weights", []) or []
            if not _tks or not _ws_dec:
                return label
            _ws_pct = [float(w) * 100 for w in _ws_dec]
            _score = compute_portfolio_risk_score(_tks, _ws_pct)
            return f"{label} ({_score})"
        # Preset portfolio — _resolve_preset returns weights as a
        # ticker→percent dict, so flatten back to a parallel list
        # for compute_portfolio_risk_score.
        _tks, _wmap = _resolve_preset(label)
        if _tks and _wmap:
            _score = compute_portfolio_risk_score(
                _tks, [_wmap.get(t, 0.0) for t in _tks])
            return f"{label} ({_score})"
        # Raw saved-portfolio name (without "📁 " prefix). Used by the
        # benchmark picker, which passes load_saved().keys() directly.
        _sp_raw = load_saved().get(label)
        if _sp_raw:
            _tks = _sp_raw.get("tickers", []) or []
            _ws_dec = _sp_raw.get("weights", []) or []
            if _tks and _ws_dec:
                _ws_pct = [float(w) * 100 for w in _ws_dec]
                _score = compute_portfolio_risk_score(_tks, _ws_pct)
                return f"{label} ({_score})"
    except Exception:
        # Fall through to undecorated label — never let a scoring
        # failure break the dropdown render.
        pass
    return label


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
def _pf_quick_score(tickers_tuple, weights_tuple):
    """Cached portfolio risk score (0–100) for a (tickers, weights) pair.

    Used by the sidebar Portfolio manager to label entries with their risk
    score. Cached per portfolio (1h) since the underlying security_risk_score
    fetches 10y of history per ticker — without this, relabeling the picker on
    every rerun would refetch repeatedly. Returns None on any failure so a
    data hiccup degrades to "no score" rather than breaking the picker.
    """
    try:
        if not tickers_tuple:
            return None
        return int(round(compute_portfolio_risk_score(
            list(tickers_tuple), list(weights_tuple))))
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def security_risk_score(ticker):
    """Compute annualized vol and max drawdown for a single ticker over 10 years.

    Cached for 1 hour: this function gets called many times per render (PCM
    rows, optimizer tier gauges, PDF holdings table) for the same tickers.
    Caching here eliminates 80%+ of redundant price fetches and makes the
    optimizer tab feel snappy after the first load.

    Window: 10 years. The recalibration that anchors SPY at score ~65 in
    shared.py was tuned to 10-year vol (~17%) and 10-year max drawdown
    (~34% — COVID 2020 was 34%). Previously this function used a 3-year
    window which produced lower vol numbers (~14%) and shallower drawdowns
    (~11%, no major crashes in 2023-2026), giving SPY a score around
    53-58 instead of the calibrated ~65. Aligning this window to the
    shared.py anchor fixes the mismatch across every score that flows
    from security_risk_score (single-ticker tables, PCM weighted scores,
    Optimizer tier gauges, PDF holdings table risk numbers).
    """
    try:
        end_dt   = date.today()
        start_dt = end_dt - relativedelta(years=10)

        # Use proxy-stitched prices so short-history tickers (SGOV, FBTC,
        # etc.) get back-filled with their proxy's older history. Matches
        # how every other scoring path in the app fetches data — without
        # proxies, a short-history ticker would only score against its own
        # limited window and produce a different number than the same
        # ticker scored elsewhere.
        raw, _ = get_prices_with_proxies(
            (ticker,), str(start_dt), str(end_dt),
            min_days=max(60, 2400),  # ~10yr business days
        )
        if raw is None or (hasattr(raw, "empty") and raw.empty):
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
    _load_skfolio()
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
    _load_skfolio()
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
    _saved_dict   = load_saved()
    _saved_names  = sorted(_saved_dict.keys())
    # Institution filter. "Custom" is the folder for your saved portfolios
    # (📁); a provider option shows only that provider's models. Saved
    # portfolios show under "Custom" and "All institutions" — not inside a
    # provider's view.
    _pf_insts = _portfolio_institutions()
    _pf_inst = st.selectbox(
        "Institution", [_INSTITUTION_ALL, "Custom"] + _pf_insts,
        key="pf_mgr_inst_filter",
        help="Pick a provider, or Custom for your own saved portfolios.")
    _preset_names = _preset_labels_for(_pf_inst)  # Custom → []
    _show_saved_pf = _pf_inst in (_INSTITUTION_ALL, "Custom")

    # Single grouped picker: saved portfolios first (📁), then built-in
    # Standard presets (📊). Selecting only PREVIEWS the allocation — it
    # never loads into the Analyzer (Step 1 has its own Securities loader).
    # Saved entries are labeled with their risk score ("· Risk NN"); presets
    # are not (scoring the whole preset universe each render would refetch
    # 10y of history for dozens of tickers — the selected preset's score
    # still shows in the preview card below). A display→(is_saved, name) map
    # keeps the appended score from interfering with name resolution.
    _SAVED_PFX, _PRESET_PFX = "📁 ", "📊 "
    _pf_meta = {}
    _pf_options = [_PLACEHOLDER]
    if _show_saved_pf:
        for _n in _saved_names:
            _sp = _saved_dict.get(_n) or {}
            _sc = _pf_quick_score(tuple(_sp.get("tickers", [])),
                                  tuple(_sp.get("weights", [])))
            _disp = f"{_SAVED_PFX}{_n}" + (f"  ·  Risk {_sc}"
                                           if _sc is not None else "")
            _pf_meta[_disp] = (True, _n)
            _pf_options.append(_disp)
    for _n in _preset_names:
        _disp = f"{_PRESET_PFX}{_n}"
        _pf_meta[_disp] = (False, _n)
        _pf_options.append(_disp)

    # Deferred reset (after a delete) — runs before the widget instantiates.
    if st.session_state.pop("_pf_mgr_reset", False):
        st.session_state["pf_mgr_select"] = _PLACEHOLDER

    _sel = st.selectbox(
        "Select portfolio", _pf_options, key="pf_mgr_select",
        label_visibility="collapsed",
    )

    def _pf_alloc_for(_is_saved, _name):
        """[(ticker, weight_pct), …] for a saved or preset selection."""
        if _is_saved:
            _sp = load_saved().get(_name) or {}
            return [(_t, float(_w) * 100.0)
                    for _t, _w in zip(_sp.get("tickers", []),
                                      _sp.get("weights", []))]
        _tks, _wmap = _resolve_preset(_name)
        if not _tks:
            return []
        if _wmap:
            return [(_t, float(_wmap.get(_t, 0.0))) for _t in _tks]
        _n2 = len(_tks)
        return [(_t, 100.0 / _n2) for _t in _tks] if _n2 else []

    if _sel != _PLACEHOLDER and _sel in _pf_meta:
        _is_saved, _pf_name = _pf_meta[_sel]
        _alloc    = _pf_alloc_for(_is_saved, _pf_name)
        if _alloc:
            # Selected portfolio's risk score (cheap — one portfolio, cached).
            _sel_score = _pf_quick_score(
                tuple(_t for _t, _ in _alloc),
                tuple(_w for _, _w in _alloc),
            )
            # Allocation preview — top 8 by weight, remainder rolled to OTHER.
            _alloc_sorted = sorted(_alloc, key=lambda x: -x[1])
            _top, _rest = _alloc_sorted[:8], _alloc_sorted[8:]
            _rows = "".join(
                '<div style="display:flex;justify-content:space-between;'
                'font-size:0.72rem;color:#475569;padding:2px 0">'
                f'<span style="font-weight:700;color:#1a2b4a">{_t}</span>'
                f'<span>{_w:.1f}%</span></div>'
                for _t, _w in _top
            )
            if _rest:
                _other_w = sum(_w for _, _w in _rest)
                _rows += (
                    '<div style="display:flex;justify-content:space-between;'
                    'font-size:0.72rem;color:#64748b;padding:2px 0">'
                    '<span style="font-weight:700">OTHER</span>'
                    f'<span>{_other_w:.1f}%</span></div>'
                )
            st.markdown(
                '<div style="background:#ffffff;border:1px solid #e2e8f0;'
                'border-radius:8px;padding:8px 10px;margin:6px 0">'
                '<div style="font-size:0.6rem;letter-spacing:0.08em;'
                'text-transform:uppercase;color:#64748b;margin-bottom:4px">'
                f'Allocations · {len(_alloc)} holdings'
                + (f' · Risk {_sel_score}' if _sel_score is not None else '')
                + f'</div>{_rows}</div>',
                unsafe_allow_html=True,
            )
            # Download CSV (saved or preset)
            _csv = "Ticker,Weight (%)\n" + "\n".join(
                f"{_t},{_w:.2f}" for _t, _w in _alloc_sorted)
            st.download_button(
                "⬇ Download CSV", _csv,
                file_name=f"{_pf_name.replace(' ', '_')}.csv",
                mime="text/csv", use_container_width=True, key="pf_mgr_dl",
            )
            # Delete — saved portfolios only (presets live in code).
            if _is_saved:
                _del_flag = f"_pf_del_confirm_{_pf_name}"
                if not st.session_state.get(_del_flag):
                    if st.button("🗑 Delete", key="pf_mgr_del",
                                 use_container_width=True):
                        st.session_state[_del_flag] = True
                        st.rerun()
                else:
                    st.caption(
                        f"Delete **{_pf_name}**? This can't be undone.")
                    _dc1, _dc2 = st.columns(2)
                    if _dc1.button("Yes, delete", key="pf_mgr_del_yes",
                                   use_container_width=True):
                        _shared_update_json(
                            SAVE_FILE,
                            lambda d, n=_pf_name: d.pop(n, None),
                        )
                        st.session_state.pop(_del_flag, None)
                        st.session_state["_pf_mgr_reset"] = True
                        st.rerun()
                    if _dc2.button("Cancel", key="pf_mgr_del_no",
                                   use_container_width=True):
                        st.session_state.pop(_del_flag, None)
                        st.rerun()
        else:
            st.caption("_Couldn't resolve this portfolio's holdings._")

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
    st.caption("Excel (.xlsx) or CSV with ticker + weight or share columns. "
               "Saved to your portfolios (does not load the Analyzer).")
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
                        # Stash the parsed portfolio and prompt for a name —
                        # the sidebar SAVES uploads to saved_portfolios.json
                        # rather than loading them into the Analyzer (Step 1).
                        st.session_state["_pf_upload_parsed"] = {
                            "tickers": _tks,
                            "weights": [float(w) for w in _ws],
                            "fname": _upload.name,
                        }
                        st.session_state["_last_upload_processed"] = _upload.name
                        st.rerun()
        except Exception as _up_err:
            st.error(f"Couldn't parse file: {_up_err}")
            st.caption(
                "Expected columns (any case): Ticker/Symbol + one of: "
                "Weight, Allocation, %, Value, or Shares."
            )

    # ── Save an uploaded portfolio (name prompt) ──────────────
    # Uploads are SAVED to saved_portfolios.json (so they appear in the
    # picker above and in the Optimizer), never loaded into the Analyzer.
    _pf_parsed = st.session_state.get("_pf_upload_parsed")
    if _pf_parsed:
        st.caption(
            f"Parsed **{len(_pf_parsed['tickers'])}** holdings from "
            f"`{_pf_parsed['fname']}`. Name it to save:"
        )
        _up_name = st.text_input(
            "Save uploaded portfolio as",
            value=os.path.splitext(_pf_parsed["fname"])[0],
            key="pf_upload_name", label_visibility="collapsed",
            placeholder="Name to save as",
        )
        _us1, _us2 = st.columns(2)
        if _us1.button("💾 Save", key="pf_upload_save_btn",
                       use_container_width=True):
            _nm = (_up_name or "").strip()
            if not _nm:
                st.warning("Enter a name to save.")
            else:
                _tks = _pf_parsed["tickers"]
                _ws  = _pf_parsed["weights"]
                _tot = sum(_ws) or 1.0
                _dec = [round(float(w) / _tot, 6) for w in _ws]
                _shared_update_json(
                    SAVE_FILE,
                    lambda d, n=_nm, t=_tks, w=_dec: d.update({
                        n: {"tickers": t, "weights": w,
                            "saved_at":
                                datetime.now().isoformat(timespec="minutes")}
                    }),
                )
                st.session_state.pop("_pf_upload_parsed", None)
                st.success(f"✅ Saved **{_nm}** to your portfolios.")
                st.rerun()
        if _us2.button("Cancel", key="pf_upload_cancel_btn",
                       use_container_width=True):
            st.session_state.pop("_pf_upload_parsed", None)
            st.rerun()

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
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tab order: Analyzer, Results & Charts, Optimizer, Fee Drag, Client Records, Settings.
# The `with main_tabN:` blocks farther down in this file are keyed by variable
# name, NOT by position — so to move a tab in the UI we just rebind its
# variable to a new st.tabs() position. main_tab6 (Fee Drag) is bound to the
# 4th position; main_tab4 (Client Records) is bound to the 5th. main_tab7
# (PDF Content) is the 6th; main_tab5 (Settings) stays rightmost.
main_tab1, main_tab2, main_tab3, main_tab6, main_tab4, main_tab7, main_tab5 = st.tabs([
    "Analyzer", "Results & Charts", "Optimizer", "Fee Drag Analyzer",
    "Client Records", "PDF Content", "Settings"
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

    # Institution filter. "Custom" is the folder for your own saved
    # portfolios (📁); the provider options (Schwab / Zacks / WisdomTree / …)
    # show only that provider's models. Saved/custom portfolios appear under
    # "Custom" and "All institutions" — not inside a provider's view.
    _insts = _portfolio_institutions()
    _inst_filter = st.selectbox(
        "Institution", [_INSTITUTION_ALL, "Custom"] + _insts,
        key="portfolio_inst_filter",
        help="Pick a provider to see only its models, or Custom for your "
             "own saved portfolios.",
    )
    _show_saved = _inst_filter in (_INSTITUTION_ALL, "Custom")
    preset_names = _preset_labels_for(_inst_filter)  # Custom → []

    source_opts   = (
        ["Custom — Enter Your Own Tickers"] +
        ([f"📁 {n}" for n in saved_names] if (_show_saved and saved_names) else []) +
        preset_names
    )
    # Keep the active selection visible even if the current filter would hide
    # it, so changing the filter never silently drops a loaded portfolio.
    _cur_src = st.session_state.portfolio_source
    if _cur_src and _cur_src not in source_opts:
        source_opts.append(_cur_src)
    src_col, rand_col = st.columns([3, 1])
    sel_src = src_col.selectbox(
        "Load portfolio",
        source_opts,
        index=source_opts.index(st.session_state.portfolio_source)
              if st.session_state.portfolio_source in source_opts else 0,
        key="portfolio_source_sel",
        format_func=_portfolio_label_with_score,
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
        if total <= 0:
            st.caption("Set weights above 0 to include your portfolio.")
        elif abs(total - 100.0) < 0.5:
            st.success(f"✅ Total: {total:.1f}% — Your portfolio will be included!")
            custom_weights_valid = True
        else:
            # Auto-normalize to 100% (proportions preserved) rather than
            # discarding the allocation. Previously a sum even slightly off
            # 100% — common when entering an optimizer-rounded Option 1/3
            # allocation that lands at e.g. 99.8% or 100.2% — left
            # custom_weights_valid False, which made BOTH the backtest
            # (cw=None → re-optimized) and submitted_weights (→ equal weight)
            # silently ignore the advisor's entered weights. Normalizing here
            # means the portfolio the advisor actually entered is the one
            # that gets analyzed and fed to the Optimizer.
            custom_weights = {t: round(w * 100.0 / total, 4)
                              for t, w in custom_weights.items()}
            custom_weights_valid = True
            st.info(
                f"ℹ️ Total was {total:.1f}% — normalized to 100% with your "
                f"proportions kept, so your portfolio is used as entered."
            )


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
        format_func=_portfolio_label_with_score,
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
            sel_saved = bm_col2.selectbox("Select saved portfolio", saved_names,
                                          key="bm_saved_sel",
                                          format_func=_portfolio_label_with_score)
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
            #   Option 3 = Slightly more aggressive (corridor max-return)
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

                    # ── Save this tier to my saved portfolios ────────
                    # Replaces the prior save-to-profile checkbox. Writes
                    # the tier's current tickers + weights to SAVE_FILE
                    # (the same store the Analyzer's "Save Portfolio"
                    # writes to via _shared_update_json), so the tier
                    # then appears as a 📁 option in the Step 2 ·
                    # Select Final Proposal for Report dropdowns below
                    # as well as throughout the Analyzer.
                    #
                    # Pattern mirrors the Analyzer save row at line
                    # ~11867: text_input + button, deferred name-clear
                    # on a successful save (Streamlit forbids widget-key
                    # writes after the widget has rendered, so the actual
                    # clear is staged via a session_state flag and runs
                    # at the top of the next rerun before the widget
                    # instantiates).
                    _save_name_key = f"save_to_list_name_{_ck}_{_tk}"
                    _clear_flag_key = f"_clear_{_save_name_key}_next"
                    if st.session_state.pop(_clear_flag_key, False):
                        st.session_state[_save_name_key] = ""
                    _default_save_name = (
                        f"Recommended — {prop.get('label', _tk.title())}"
                    )
                    _name_col, _btn_col = st.columns([2, 1])
                    with _name_col:
                        _save_name = st.text_input(
                            "Save name",
                            value=st.session_state.get(_save_name_key,
                                                         _default_save_name),
                            key=_save_name_key,
                            label_visibility="collapsed",
                            placeholder=_default_save_name,
                        )
                    with _btn_col:
                        _do_save = st.button(
                            "💾 Save to my portfolios",
                            key=f"save_to_list_btn_{_ck}_{_tk}",
                            use_container_width=True,
                            help=(
                                "Saves this tier's current tickers + "
                                "weights to your saved portfolios. It "
                                "will then appear as a 📁 option in "
                                "the Step 2 dropdowns below and in the "
                                "Analyzer."
                            ),
                        )
                    if _do_save:
                        _name_clean = (_save_name or "").strip()
                        _tier_tks = list(prop.get("tickers", []) or [])
                        _tier_wts_raw = list(prop.get("weights", []) or [])
                        if not _name_clean:
                            st.warning("Enter a name above before saving.")
                        elif not _tier_tks or not _tier_wts_raw:
                            st.warning(
                                "This tier has no tickers/weights yet."
                            )
                        else:
                            _w_total = sum(float(w or 0)
                                           for w in _tier_wts_raw)
                            if _w_total <= 0:
                                st.warning(
                                    "Weights must sum to a positive total."
                                )
                            else:
                                # Tier weights are already stored as
                                # decimals (0.0-1.0) on the proposal,
                                # but normalize defensively so the saved
                                # schema matches what
                                # load_portfolio_into_session expects.
                                _decimals = [
                                    round(float(w or 0) / _w_total, 6)
                                    for w in _tier_wts_raw
                                ]
                                _payload = {
                                    "tickers": _tier_tks,
                                    "weights": _decimals,
                                    "saved_at": datetime.now().isoformat(
                                        timespec="minutes"),
                                }
                                _shared_update_json(
                                    SAVE_FILE,
                                    lambda d, n=_name_clean, p=_payload:
                                        d.update({n: p}),
                                )
                                st.success(
                                    f"✅ Saved as **{_name_clean}** — "
                                    "available as a 📁 option in the "
                                    "Step 2 dropdowns below."
                                )
                                st.session_state[_clear_flag_key] = True
                                st.rerun()

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
        #   Option 3 → Recommended Aggressive tier (corridor max-return)
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
    # User records carry the referral fields the portal writes: referral_code,
    # referred_by / referred_by_name, and referrals / referrals_sent.
    all_users            = _safe_load_json("ra_users.json")

    def _user_record_for(profile_key, profile):
        """Resolve a profile's matching ra_users record. Profiles and users are
        both keyed by the normalized email, but fall back to the client_email
        field and a scan in case the normalization ever differs."""
        u = all_users.get(profile_key)
        if u:
            return u
        pe = (profile.get("client_email") or "").strip().lower()
        if pe and all_users.get(pe):
            return all_users.get(pe)
        target = pe or profile_key
        for v in all_users.values():
            if isinstance(v, dict) and (v.get("email", "") or "").strip().lower() == target:
                return v
        return {}

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
        _total_referred = sum(1 for v in all_users.values()
                              if isinstance(v, dict) and v.get("referred_by"))
        if _total_referred:
            st.caption(
                f"🔗 {_total_referred} "
                f"{'client' if _total_referred == 1 else 'clients'} joined via an "
                f"existing client's invite link."
            )
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

                # ── Referrals ──────────────────────────────────────────
                # "Referred by" (this client arrived via someone's invite link)
                # and who this client has, in turn, brought in. Pulled from the
                # ra_users record the portal writes; section hidden when there's
                # nothing to show.
                _uref = _user_record_for(profile_key, p)
                _referred_by_name = (_uref.get("referred_by_name") or "").strip()
                _refs_list = _uref.get("referrals") if isinstance(_uref.get("referrals"), list) else []
                _refs_sent = _uref.get("referrals_sent")
                if not isinstance(_refs_sent, int):
                    _refs_sent = len(_refs_list)
                if _referred_by_name or _refs_sent:
                    st.markdown("---")
                    st.markdown("**Referrals**")
                    if _referred_by_name:
                        st.markdown(f"Referred by **{_referred_by_name}**")
                    if _refs_sent:
                        st.markdown(
                            f"Brought in **{_refs_sent}** "
                            f"{'client' if _refs_sent == 1 else 'clients'} who signed up:"
                        )
                        if _refs_list:
                            _rrows = [{"Name":   (r.get("name") or "—"),
                                       "Email":  (r.get("email") or "—"),
                                       "Joined": (r.get("at") or "—")}
                                      for r in _refs_list]
                            st.dataframe(pd.DataFrame(_rrows),
                                         use_container_width=True, hide_index=True)

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

                            # ── Risk gauges for the 3 proposals ──────────
                            # Cleaned-up view: just the risk-score gauges for
                            # Option 1/2/3. The Broad-ETF Alternate column and
                            # the allocation pie charts were removed per advisor
                            # preference. Order matches the tier-tab strip:
                            # Option 1 = balanced (proposed), Option 2 =
                            # conservative, Option 3 = aggressive. Internal keys
                            # (and all save/load/PDF code) unchanged.
                            _tier_cfg_full = [
                                ("balanced",     "Option 1 (proposed)"),
                                ("conservative", "Option 2"),
                                ("aggressive",   "Option 3"),
                            ]
                            _present = [(k, l) for k, l in _tier_cfg_full
                                        if k in _prop.get("tiers", {})]
                            if _present:
                                _gauge_cols = st.columns(len(_present))
                                for _gc, (_ptk, _ptlbl) in zip(_gauge_cols, _present):
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
                                        with _gc:
                                            # The gauge's built-in plotly number
                                            # (gauge+number mode) gets clipped when
                                            # the gauge scales up to fill a wide
                                            # column under use_container_width — the
                                            # arc well slides past the fixed figure
                                            # height. So we render the gauge as the
                                            # arc only and draw the score ourselves
                                            # as HTML nested into the well, which
                                            # shows reliably at any screen width.
                                            _sc = int(max(1, min(99, _saved_score)))
                                            if _sc <= 33:   _sc_col = "#00A84F"   # green
                                            elif _sc <= 66: _sc_col = "#E0930A"   # amber
                                            else:           _sc_col = "#E0302A"   # red
                                            _g = make_risk_gauge(_saved_score, height=160)
                                            _g.update_traces(mode="gauge")  # drop built-in number
                                            st.plotly_chart(
                                                _g,
                                                use_container_width=True,
                                                config={"displayModeBar": False},
                                                key=f"saved_prop_gauge_{profile_key}_{_vid}_{_ptk}",
                                            )
                                            # Score (nested up into the arc well) +
                                            # option label below. Negative margin
                                            # pulls the number into the gauge.
                                            st.markdown(
                                                f"<div style='text-align:center;margin-top:-66px'>"
                                                f"<div style='font-size:2.1rem;font-weight:700;"
                                                f"color:{_sc_col};line-height:1'>{_sc}</div>"
                                                f"<div style='font-weight:700;font-size:0.95rem;"
                                                f"color:#111827;margin-top:14px'>{_ptlbl}</div>"
                                                f"</div>",
                                                unsafe_allow_html=True,
                                            )

                            # (Compact per-tier summary cards removed — the
                            # saved-proposal view now shows only the risk gauges
                            # for the 3 proposals. Full allocation / target /
                            # holdings detail still lives in the PDF and the
                            # Optimizer tab.)

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
# TAB 7 — PDF CONTENT (customizable closing sections)
# ═══════════════════════════════════════════════════════════════
with main_tab7:
    st.session_state["optimizer_tab_active"] = False
    st.markdown(
        "<h3 style='color:#0E5C5E;font-weight:600;letter-spacing:-0.015em;"
        "margin:0 0 6px 0'>PDF Content</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Customize the closing sections of the client proposal PDF — "
        "everything from Advisor Notes onward. Each box is pre-filled "
        "with the current wording; edit what you like, then Save. Use the "
        "↩ Restore default button inside any section to put just that "
        "section back to its original wording."
    )
    st.markdown("---")

    # Per-section "restore default" helper. Drops the section's saved
    # override (so get_pdf_content() falls back to DEFAULT_PDF_CONTENT for
    # it) and clears the editor's widget state so the box re-seeds from the
    # default on the rerun. Persists immediately; other sections are left
    # exactly as they are, including any unsaved edits in their boxes.
    def _pdfc_restore_button(_section_key, _widget_key, _btn_key):
        if st.button("↩ Restore default", key=_btn_key,
                     help="Reset this section to its original wording. "
                          "Saved immediately; other sections are unaffected."):
            _cur = load_pdf_content()
            if isinstance(_cur, dict):
                _cur.pop(_section_key, None)
                try:
                    save_pdf_content(_cur)
                except Exception as _re_err:
                    st.error(f"Couldn't restore default: {_re_err}")
                    st.stop()
            st.session_state.pop(_widget_key, None)
            st.toast("Section restored to its default.")
            st.rerun()

    # Effective content (advisor edits layered over defaults) pre-fills
    # the editors so the advisor always starts from real text.
    _pdfc = get_pdf_content()

    with st.expander("Advisor Notes", expanded=False):
        st.caption(
            "Firm-wide default note — used on a proposal only when that "
            "proposal doesn't carry its own advisor note. Separate "
            "paragraphs with a blank line."
        )
        _pdfc_notes = st.text_area(
            "Advisor note", value=_pdfc["advisor_notes"], height=160,
            key="pdfc_notes_input", label_visibility="collapsed",
        )
        _pdfc_restore_button("advisor_notes", "pdfc_notes_input",
                             "pdfc_restore_notes")

    with st.expander("Implementation Plan", expanded=False):
        st.caption(
            "Rows of the Implementation Plan table. Use the controls at "
            "the bottom of the grid to add or remove rows."
        )
        _pdfc_impl = st.data_editor(
            pd.DataFrame(_pdfc["implementation_plan"],
                         columns=["Stage", "Cadence", "Action"]),
            num_rows="dynamic", use_container_width=True,
            key="pdfc_impl_input",
        )
        _pdfc_restore_button("implementation_plan", "pdfc_impl_input",
                             "pdfc_restore_impl")

    with st.expander("How This Proposal Was Built", expanded=False):
        st.caption(
            "Separate paragraphs with a blank line. Wrap text in "
            "`<b>...</b>` to bold it."
        )
        _pdfc_method = st.text_area(
            "Methodology", value=_pdfc["methodology"], height=220,
            key="pdfc_method_input", label_visibility="collapsed",
        )
        _pdfc_restore_button("methodology", "pdfc_method_input",
                             "pdfc_restore_method")

    with st.expander("Key Definitions", expanded=False):
        st.caption(
            "Term / definition rows of the Key Definitions table. Use "
            "the controls at the bottom of the grid to add or remove "
            "rows."
        )
        _pdfc_kd = st.data_editor(
            pd.DataFrame(_pdfc["key_definitions"],
                         columns=["Term", "Definition"]),
            num_rows="dynamic", use_container_width=True,
            key="pdfc_kd_input",
        )
        _pdfc_restore_button("key_definitions", "pdfc_kd_input",
                             "pdfc_restore_kd")

    with st.expander("Disclosures", expanded=False):
        st.caption(
            "Separate paragraphs with a blank line. Wrap text in "
            "`<b>...</b>` to bold it. Use the token `{advisory_fee}` "
            "where the firm's advisory fee should appear — it's filled "
            "in automatically on each PDF."
        )
        _pdfc_disc = st.text_area(
            "Disclosures", value=_pdfc["disclosures"], height=300,
            key="pdfc_disc_input", label_visibility="collapsed",
        )
        _pdfc_restore_button("disclosures", "pdfc_disc_input",
                             "pdfc_restore_disc")

    st.markdown("---")
    if st.button("💾 Save PDF content", type="primary",
                 key="pdfc_save_btn"):
        def _rows_from_editor(_df):
            """Edited grid → list of string rows; fully-blank rows are
            dropped and NaN/None cells become empty strings."""
            _out = []
            for _row in _df.fillna("").values.tolist():
                _cells = [str(_c).strip() for _c in _row]
                if any(_cells):
                    _out.append(_cells)
            return _out

        _pdfc_payload = {
            "advisor_notes":       _pdfc_notes.strip(),
            "implementation_plan": _rows_from_editor(_pdfc_impl),
            "methodology":         _pdfc_method.strip(),
            "key_definitions":     _rows_from_editor(_pdfc_kd),
            "disclosures":         _pdfc_disc.strip(),
        }
        try:
            save_pdf_content(_pdfc_payload)
        except Exception as _pdfc_err:
            st.error(f"Couldn't save PDF content: {_pdfc_err}")
            with st.expander("Debug info"):
                import traceback as _pdfc_tb
                st.code(_pdfc_tb.format_exc())
        else:
            st.success(
                "✅ PDF content saved. Newly generated proposal PDFs "
                "will use this wording."
            )
            st.caption(f"Saved to: `{PDF_CONTENT_FILE}`")

    # ── Client portal agreement (Terms & Privacy popups) ───────────────
    st.markdown("---")
    st.markdown(
        "<h4 style='color:#0E5C5E;font-weight:600;margin:10px 0 4px 0'>"
        "Client Portal — Terms &amp; Privacy</h4>",
        unsafe_allow_html=True,
    )
    st.caption(
        "These show as popups on the client portal's registration screen, "
        "where clients must agree before registering. Edit and Save — the "
        "portal picks up changes within ~60s. Bump the version whenever the "
        "wording changes so the record of which version each client accepted "
        "stays accurate. Have your CCO/counsel review the language before launch."
    )
    _legal = get_legal_content()
    _legal_ver = st.text_input(
        "Agreement version", value=_legal["version"], key="legal_ver_input",
        help="Shown in the Terms popup and recorded with each client's acceptance.",
    )
    with st.expander("Terms & Conditions", expanded=False):
        st.caption("Markdown supported — e.g. **bold**, line breaks, lists.")
        _legal_terms = st.text_area(
            "Terms & Conditions", value=_legal["terms"], height=320,
            key="legal_terms_input", label_visibility="collapsed",
        )
    with st.expander("Privacy Policy", expanded=False):
        st.caption("Markdown supported.")
        _legal_privacy = st.text_area(
            "Privacy Policy", value=_legal["privacy"], height=320,
            key="legal_privacy_input", label_visibility="collapsed",
        )
    if st.button("💾 Save client agreement", type="primary",
                 key="legal_save_btn"):
        try:
            save_legal_content({
                "version": (_legal_ver or "").strip() or DEFAULT_LEGAL_CONTENT["version"],
                "terms":   (_legal_terms or "").strip(),
                "privacy": (_legal_privacy or "").strip(),
            })
        except Exception as _le:
            st.error(f"Couldn't save client agreement: {_le}")
            with st.expander("Debug info"):
                import traceback as _ltb
                st.code(_ltb.format_exc())
        else:
            st.success(
                "✅ Client agreement saved. The portal registration popups "
                "will use this wording within ~60s."
            )
            st.caption(f"Saved to: `{LEGAL_CONTENT_FILE}`")


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
    _fs  = load_firm_settings()
    # v2.4 nested schema — the client portal + mrb_design read firm.* and
    # advisor.* (not the legacy flat keys). Pre-fill from there so a save
    # round-trips the canonical values instead of blanking them.
    _adv = _fs.get("advisor", {}) or {}
    _frm = _fs.get("firm", {}) or {}

    fb_l, fb_r = st.columns(2)
    with fb_l:
        firm_name = st.text_input(
            "Firm name",
            value=_frm.get("name", ""),
            placeholder="Foresight Wealth Partners",
            key="fb_firm_name",
        )
        advisor_name = st.text_input(
            "Advisor name",
            value=_adv.get("name", ""),
            placeholder="Sarah Whitfield, CFP®",
            key="fb_advisor_name",
        )
        advisor_title = st.text_input(
            "Advisor title",
            value=_adv.get("title", ""),
            placeholder="Senior Financial Advisor",
            key="fb_advisor_title",
        )
    with fb_r:
        advisor_email = st.text_input(
            "Advisor email",
            value=_adv.get("email", ""),
            placeholder="sarah@foresightwealth.com",
            key="fb_advisor_email",
        )
        advisor_phone = st.text_input(
            "Advisor phone",
            value=_adv.get("phone", ""),
            placeholder="(612) 555-0142",
            key="fb_advisor_phone",
        )
        firm_website = st.text_input(
            "Firm website",
            value=_frm.get("website", ""),
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
        value=float(_fs.get("default_advisory_fee_pct", _frm.get("fee_pct", 1.00))),
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
        value=_frm.get("address", ""),
        placeholder="200 South Sixth Street, Suite 1200, Minneapolis, MN 55402",
        key="fb_firm_address",
    )
    advisor_bio = st.text_area(
        "Advisor bio (short paragraph)",
        value=_adv.get("bio", ""),
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
                except OSError:
                    pass
                try:
                    _set_brand_image("logo", None)
                except Exception:
                    pass
                st.rerun()
        else:
            st.caption("_No logo uploaded._")
        _logo_up = st.file_uploader(
            "Upload firm logo",
            type=["png", "jpg", "jpeg"],
            key="fb_logo_uploader",
            label_visibility="collapsed",
        )
        if _logo_up is not None:
            _raw = _logo_up.getvalue()
            with open(FIRM_LOGO_PATH, "wb") as _f:
                _f.write(_raw)
            try:
                _set_brand_image("logo", _img_bytes_to_data_uri(_raw))
            except Exception as _e:
                st.warning(f"Saved locally, but couldn't sync to the portal: {_e}")
            st.success("Logo saved.")
            st.rerun()

    with img_r:
        st.markdown("**Advisor photo**")
        if os.path.exists(ADVISOR_PHOTO_PATH):
            st.image(ADVISOR_PHOTO_PATH, width=140)
            if st.button("Remove photo", key="fb_photo_remove"):
                try:
                    os.remove(ADVISOR_PHOTO_PATH)
                except OSError:
                    pass
                try:
                    _set_brand_image("photo", None)
                except Exception:
                    pass
                st.rerun()
        else:
            st.caption("_No photo uploaded._")
        _photo_up = st.file_uploader(
            "Upload advisor photo",
            type=["png", "jpg", "jpeg"],
            key="fb_photo_uploader",
            label_visibility="collapsed",
        )
        if _photo_up is not None:
            _raw = _photo_up.getvalue()
            with open(ADVISOR_PHOTO_PATH, "wb") as _f:
                _f.write(_raw)
            try:
                _set_brand_image("photo", _img_bytes_to_data_uri(_raw))
            except Exception as _e:
                st.warning(f"Saved locally, but couldn't sync to the portal: {_e}")
            st.success("Photo saved.")
            st.rerun()

    st.markdown("---")
    if st.button("💾 Save firm details", type="primary", key="fb_save"):
        # Merge into the v2.4 NESTED schema the client portal + mrb_design
        # actually read (firm.* / advisor.*). update_json is a read-modify-write
        # under a lock, so it preserves the rest of firm_settings.json (brand,
        # typography, proposal_copy) instead of the old flat full-overwrite,
        # which would have clobbered the entire design system.
        _fee = round(float(default_advisory_fee_pct), 2)
        _fields = {
            "firm.name":     firm_name.strip(),
            "firm.website":  firm_website.strip(),
            "firm.address":  firm_address.strip(),
            "advisor.name":  advisor_name.strip(),
            "advisor.title": advisor_title.strip(),
            "advisor.email": advisor_email.strip(),
            "advisor.phone": advisor_phone.strip(),
            "advisor.bio":   advisor_bio.strip(),
        }

        def _merge_branding(s):
            s.setdefault("firm", {})
            s.setdefault("advisor", {})
            s["firm"]["name"]     = firm_name.strip()
            s["firm"]["website"]  = firm_website.strip()
            s["firm"]["address"]  = firm_address.strip()
            s["advisor"]["name"]  = advisor_name.strip()
            s["advisor"]["title"] = advisor_title.strip()
            s["advisor"]["email"] = advisor_email.strip()
            s["advisor"]["phone"] = advisor_phone.strip()
            s["advisor"]["bio"]   = advisor_bio.strip()
            # Fee stays top-level - the proposal fee resolver reads it there.
            s["default_advisory_fee_pct"] = _fee

        # Write, surface any error visibly, then re-read to verify the data
        # landed. A silent failure would otherwise look like a dead button.
        try:
            _shared_update_json(FIRM_SETTINGS_FILE, _merge_branding)
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
            # Verify against the ACTUAL store, not the write cache (which would
            # mask a local-only write). clear_cache() forces a fresh read, so a
            # green check genuinely means it persisted where the portal reads.
            _ds.clear_cache()
            _verify = load_firm_settings()
            _vadv   = _verify.get("advisor", {}) or {}
            _vfrm   = _verify.get("firm", {}) or {}
            _resolved = {
                "firm.name":     _vfrm.get("name", ""),
                "firm.website":  _vfrm.get("website", ""),
                "firm.address":  _vfrm.get("address", ""),
                "advisor.name":  _vadv.get("name", ""),
                "advisor.title": _vadv.get("title", ""),
                "advisor.email": _vadv.get("email", ""),
                "advisor.phone": _vadv.get("phone", ""),
                "advisor.bio":   _vadv.get("bio", ""),
            }
            _total_filled = sum(1 for v in _fields.values() if v)
            _populated = sum(
                1 for k, v in _fields.items()
                if v and _resolved.get(k) == v
            )
            if _total_filled == 0:
                st.warning(
                    "All fields are blank \u2014 nothing was saved. Fill in at "
                    "least one field (firm name, advisor name, etc.) and retry."
                )
            elif _populated == _total_filled:
                if _ds.is_remote():
                    st.success(
                        f"\u2705 Saved {_populated} field(s) to the shared repo. "
                        "Refresh or reboot the portal to see the change."
                    )
                else:
                    st.error(
                        f"Saved {_populated} field(s), but to LOCAL disk only \u2014 "
                        "this app has no [github] secret, so the client portal will "
                        "NOT see these changes. Add the [github] token + data_repo "
                        "to this app's Streamlit secrets (same values as the portal)."
                    )
                st.caption(f"Wrote to: `{FIRM_SETTINGS_FILE}` (firm.* / advisor.*)")
            else:
                st.warning(
                    f"Save partially verified: {_populated} of {_total_filled} "
                    "non-empty fields confirmed on re-read. If this persists, the "
                    "portal may read a different settings source \u2014 confirm "
                    "mrb_design.load_settings() reads via data_store."
                )
                st.caption(f"Wrote to: `{FIRM_SETTINGS_FILE}`")

    # ── Sync diagnostics ──────────────────────────────────────────
    # #1 reason branding doesn't reach the portal: this app is in
    # data_store LOCAL-FALLBACK mode (no [github] secret), so saves stay on
    # this container and never reach the shared repo the portal reads.
    with st.expander("Sync diagnostics — is branding reaching the portal?"):
        if _ds.is_remote():
            try:
                _repo = _ds._config()[1]
            except Exception:
                _repo = "?"
            st.success(
                f"Data store: remote (GitHub) -> `{_repo}`. Saves here reach "
                "the shared repo the portal reads."
            )
        else:
            st.error(
                "Data store: LOCAL fallback - no [github] secret on this app. "
                "Saves stay on this container and never reach the shared repo, "
                "so the client portal can't see them. Add the [github] token + "
                "data_repo to THIS app's Streamlit secrets (same values the "
                "portal uses)."
            )
        if st.button("Check what's actually in the shared store", key="fb_diag"):
            _ds.clear_cache()
            _live = load_firm_settings()
            _ladv = _live.get("advisor", {}) or {}
            _lfrm = _live.get("firm", {}) or {}
            st.json({
                "mode": "remote" if _ds.is_remote() else "local",
                "firm.name": _lfrm.get("name", "(none)"),
                "advisor.name": _ladv.get("name", "(none)"),
                "advisor.bio_chars": len(str(_ladv.get("bio", "") or "")),
                "legacy_flat_advisor_bio_chars": len(str(_live.get("advisor_bio", "") or "")),
                "top_level_keys": sorted(_live.keys()),
            })

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

        # 4. Cache stats + clear button
        try:
            _disk_cache = _load_er_cache()
            _ses_cache = getattr(_expense_ratio_for_ticker, "_session_cache", {}) or {}
            st.caption(
                f"Cache: **{len(_disk_cache)}** tickers on disk · "
                f"**{len(_ses_cache)}** in this session"
            )
            if st.button("🗑️ Clear ER cache (disk + session)",
                         key="_er_cache_clear_btn",
                         use_container_width=True,
                         help="Wipes both the on-disk expense-ratio cache "
                              "and the in-memory session cache. Next ER "
                              "lookup runs the full waterfall fresh — "
                              "use this if old stock-fallback 0.0 values "
                              "are shadowing yfinance results."):
                # Wipe disk cache via the data_store layer
                try:
                    _save_er_cache({})
                except Exception:
                    pass
                # Wipe the in-process memoized copy too
                if hasattr(_load_er_cache, "_cache"):
                    _load_er_cache._cache = {}
                # Wipe the session-level memoization on the resolver
                _expense_ratio_for_ticker._session_cache = {}
                st.success("ER cache cleared. Re-render to refresh.")
                st.rerun()
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
