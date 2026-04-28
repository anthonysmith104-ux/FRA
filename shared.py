"""shared.py — code shared between app.py (advisor) and risk_assessment.py (client).

Centralizes:
  • atomic JSON read/modify/write with file locking (eliminates race conditions)
  • canonical compute_risk_score() — was duplicated with two different formulas
  • email validation
  • risk-free-rate aware Sharpe ratio

Import from here in both apps. Do NOT redefine any of these locally.
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

# ── File locking ───────────────────────────────────────────────────────────────
# POSIX (fcntl) on Linux/Mac, msvcrt on Windows. Streamlit reruns + multi-user =
# concurrent writers, so non-atomic read-modify-write loses data without this.
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


@contextmanager
def _file_lock(lock_path: str):
    """Cross-platform exclusive file lock. Best-effort: if no locking primitive
    is available, falls back to no-op (single-process safety only)."""
    # Ensure parent dir exists
    parent = os.path.dirname(os.path.abspath(lock_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = open(lock_path, "a+")
    try:
        if _HAS_FCNTL:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        elif _HAS_MSVCRT:
            # msvcrt.locking blocks; loop with retries
            for _ in range(50):
                try:
                    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    import time; time.sleep(0.1)
        yield
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            elif _HAS_MSVCRT:
                try: msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError: pass
        except Exception:
            pass
        fd.close()


def load_json(path: str, default: Any = None) -> Any:
    """Load JSON file. Returns `default` (or {}) for missing/empty/corrupted files."""
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else default
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def save_json(path: str, data: Any) -> None:
    """Atomic JSON write: write to temp file then os.replace.
    Holds an exclusive file lock for the duration to serialize concurrent writers.
    """
    lock_path = path + ".lock"
    with _file_lock(lock_path):
        # Write to tempfile in same directory (so os.replace is atomic on the same FS)
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise


def update_json(path: str, mutator) -> Any:
    """Atomic read-modify-write under a single lock.
    `mutator` receives the loaded dict (or {}) and should mutate it in place
    OR return a new dict. Returns whatever the file ends up containing.

    Use this instead of load_json + save_json when both operations need to be
    atomic (e.g. add to dict, increment counter).
    """
    lock_path = path + ".lock"
    with _file_lock(lock_path):
        # Inline load (we already hold the lock, don't recurse)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                data = json.loads(content) if content else {}
            except (json.JSONDecodeError, ValueError, OSError):
                data = {}
        else:
            data = {}

        result = mutator(data)
        if result is not None:
            data = result

        # Inline atomic save
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise
        return data


# ── Email validation ──────────────────────────────────────────────────────────
# RFC 5322 is too permissive for this use case; this regex covers ~99% of real
# emails users actually have. Rejects whitespace-only, missing @ or TLD, etc.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def is_valid_email(email: Optional[str]) -> bool:
    """Returns True iff `email` is a non-empty, well-formed email address."""
    if not email or not isinstance(email, str):
        return False
    e = email.strip()
    if not e or len(e) > 254:
        return False
    return bool(_EMAIL_RE.match(e))


def normalize_email(email: Optional[str]) -> Optional[str]:
    """Strip + lowercase. Returns None if invalid."""
    if not is_valid_email(email):
        return None
    return email.strip().lower()


# ── Risk-free rate ────────────────────────────────────────────────────────────
# Used by Sharpe ratio computations. Default ~current 13-week T-bill yield.
# Override via env var if desired. For a more sophisticated approach, fetch
# ^IRX from yfinance — but the simple constant is enough to make Sharpe values
# meaningful relative to cash and consistent across time periods.
DEFAULT_RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.045"))  # 4.5%


def sharpe_ratio(ann_return: float, ann_vol: float,
                 risk_free: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Excess-return Sharpe ratio. Returns 0 if vol is non-positive."""
    if ann_vol is None or ann_vol <= 0:
        return 0.0
    return float((ann_return - risk_free) / ann_vol)


# ── Canonical risk score (was duplicated in app.py and risk_assessment.py) ────
# Spec from app.py:
#   • vol_score        = vol × 4.2 × 100        (vol expressed as decimal)
#   • dd_score         = |max_dd| × 2.0 × 100   (dd expressed as decimal)
#   • real_rate_score  = duration × 1.2 × 4.2   (bonds only, when provided)
#   • raw              = max(vol_score, dd_score, real_rate_score) + credit_adj
#   • credit_adj       = 0/+1/+3/+5  (govt/IG/HY/EM)
#   • Equity LOG COMPRESSION above raw 80:
#         compressed = 80 + (excess) ** 0.45 × 1.4
#   • Class caps: cash ≤ 5, bonds ≤ 55, equity ≤ 95 (with compression),
#     crypto BTC = 91, crypto alt 90-95, leveraged 96-99.
#
# `sharpe` accepted for backward-compat but no longer used.

_CREDIT_ADJ = {"govt": 0, "ig": 1, "hy": 3, "em": 5}


def compute_risk_score(ann_vol: float,
                       max_drawdown: float,
                       sharpe: float = 0,
                       ticker: Optional[str] = None,
                       duration: Optional[float] = None,
                       asset_class: Optional[str] = None,
                       credit_tier: Optional[str] = None,
                       classifier=None) -> int:
    """Score from 1 (very low risk) to 99 (very high risk).

    `classifier` is an optional callable: ticker -> (asset_class, credit_tier).
    If provided and asset_class is None, it will be used to auto-classify.
    Pass `_classify_ticker` from app.py to reproduce its behavior.
    """
    vol_pct = float(ann_vol or 0) * 100
    dd_pct  = abs(float(max_drawdown or 0)) * 100
    dur     = float(duration or 0)

    # Auto-classify if class not provided AND a classifier was injected
    if asset_class is None and ticker and classifier is not None:
        try:
            ac, ct = classifier(ticker)
            asset_class = ac
            if credit_tier is None:
                credit_tier = ct
        except Exception:
            pass

    asset_class = asset_class or "equity"
    credit_tier = credit_tier or "govt"

    # Component scores
    vol_score = vol_pct * 4.2
    dd_score  = dd_pct  * 2.0
    rr_score  = dur * 1.2 * 4.2 if dur > 0 else 0

    raw = max(vol_score, dd_score, rr_score) + _CREDIT_ADJ.get(credit_tier, 0)

    if asset_class == "cash":
        score = min(5, max(1, raw))
    elif asset_class == "bond":
        score = min(55, max(1, raw))
    elif asset_class == "leveraged":
        score = max(96, min(99, 96 + (raw - 80) / 25))
    elif asset_class == "crypto_btc":
        score = 91
    elif asset_class == "crypto_alt":
        score = 90 + min(5, max(0, (vol_pct - 50) / 12))
    else:
        # equity — log compression above 80, hard cap 95
        if raw <= 80:
            score = raw
        else:
            score = 80 + ((raw - 80) ** 0.45) * 1.4
        score = min(95, max(1, score))

    cap = 95 if asset_class == "equity" else 99
    return int(round(min(cap, max(1, score))))


def score_to_label(score: int) -> tuple[str, str]:
    """(label, hex_color) for a 1-99 score."""
    if score <= 15: return "Conservative",              "#16a34a"
    if score <= 30: return "Moderately Conservative",   "#65a30d"
    if score <= 45: return "Moderate",                  "#d97706"
    if score <= 60: return "Moderately Aggressive",     "#ea580c"
    if score <= 75: return "Aggressive",                "#dc2626"
    return              "Very Aggressive",              "#991b1b"


def score_to_allocation(score: int) -> tuple[int, int, str]:
    """(equity%, bond%, description) for a 1-99 score."""
    if score <= 15: return 20, 80,  "Very conservative — capital preservation focus"
    if score <= 30: return 35, 65,  "Conservative — income with limited growth"
    if score <= 45: return 55, 45,  "Balanced — moderate growth and stability"
    if score <= 60: return 70, 30,  "Growth-oriented — higher equity allocation"
    if score <= 75: return 85, 15,  "Aggressive growth — significant equity exposure"
    return              100,  0,   "Maximum growth — primarily equities"


# ── Secure token generation (for proposal sharing) ────────────────────────────

def make_secure_token() -> str:
    """22 chars of urlsafe base64 = 128 bits of entropy. Use at proposal-creation
    time and STORE on the proposal — don't derive tokens from client_key/version_id
    (those are guessable)."""
    return _secrets.token_urlsafe(16)
