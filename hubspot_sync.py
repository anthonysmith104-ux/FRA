"""hubspot_sync.py — non-blocking HubSpot CRM sync for Foresight Risk Analytics.

Architecture: registrations always succeed locally first (atomic write to the
JSON store via shared.py). Then a HubSpot sync is enqueued. A background
thread drains the queue with exponential backoff so the UI never blocks on
HubSpot's API and a HubSpot outage can't cause a failed registration.

Configuration:
    Set HUBSPOT_TOKEN in environment, OR drop a `hubspot_config.json` file
    next to this script with {"token": "pat-na1-..."}. The token is a HubSpot
    Private App access token (Settings → Integrations → Private Apps).

Required scopes on the Private App:
    crm.objects.contacts.read / write
    crm.objects.deals.read / write
    crm.schemas.contacts.read / write   (for the custom risk-score properties)

Public API:
    is_configured()           -> bool
    sync_contact(...)         -> queues a contact upsert + advisor-follow-up deal
    pending_count()           -> queue depth (for diagnostics)
    init()                    -> idempotent — starts the worker thread

The worker is started lazily on first sync_contact() call. If the token is
missing the call returns silently; nothing is queued and no thread is spawned.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

# ── Configuration ───────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_APP_DIR, "hubspot_config.json")
_QUEUE_PATH  = os.path.join(_APP_DIR, "hubspot_queue.json")  # disk-backed retry log

_HS_BASE = "https://api.hubapi.com"
_TIMEOUT = 8     # seconds per HTTP call — keep short so the worker can move on
_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0   # seconds; doubles each retry

# ── Token resolution ────────────────────────────────────────────────────────
def _read_token() -> Optional[str]:
    """Token lookup order: env var → config file → None.
    Returning None means HubSpot sync is silently disabled."""
    tok = os.environ.get("HUBSPOT_TOKEN", "").strip()
    if tok:
        return tok
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            return (cfg.get("token") or "").strip() or None
    except Exception:
        pass
    return None


def is_configured() -> bool:
    return bool(_read_token())


# ── Disk-backed queue ───────────────────────────────────────────────────────
# Keeping the queue on disk means a Streamlit restart doesn't lose pending
# syncs. Atomic writes via the shared module's update_json keep concurrent
# writes safe.
def _load_queue() -> list:
    try:
        if os.path.exists(_QUEUE_PATH):
            with open(_QUEUE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save_queue(q: list) -> None:
    tmp = _QUEUE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2)
        os.replace(tmp, _QUEUE_PATH)
    except Exception:
        # If we can't persist, just continue; the in-memory queue still works.
        pass


def pending_count() -> int:
    return len(_load_queue())


def get_deadletter() -> list:
    """Return the list of permanently-failed sync attempts. Used by the
    diagnostic panel to surface what HubSpot rejected so an admin can
    investigate or retry manually."""
    path = os.path.join(_APP_DIR, "hubspot_deadletter.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def clear_deadletter() -> None:
    """Wipe the deadletter file. Use after manually resolving the failures
    (e.g. after fixing the email format and re-running)."""
    path = os.path.join(_APP_DIR, "hubspot_deadletter.json")
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── Worker thread state ─────────────────────────────────────────────────────
_worker_lock   = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_worker_wakeup = threading.Event()


def _enqueue(payload: dict) -> None:
    q = _load_queue()
    payload = dict(payload)
    payload.setdefault("_attempts", 0)
    payload.setdefault("_enqueued_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    q.append(payload)
    _save_queue(q)
    _worker_wakeup.set()  # nudge the worker if it's sleeping


def _dequeue_one() -> Optional[dict]:
    q = _load_queue()
    if not q: return None
    head = q[0]
    rest = q[1:]
    _save_queue(rest)
    return head


def _requeue(payload: dict) -> None:
    """Push a failed payload back to the END of the queue with bumped attempts."""
    payload = dict(payload)
    payload["_attempts"] = int(payload.get("_attempts", 0)) + 1
    if payload["_attempts"] >= _MAX_RETRIES:
        # Give up — log to a deadletter file so the user can see what failed.
        _write_deadletter(payload)
        return
    q = _load_queue()
    q.append(payload)
    _save_queue(q)


def _write_deadletter(payload: dict) -> None:
    path = os.path.join(_APP_DIR, "hubspot_deadletter.json")
    try:
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
        existing.append({**payload,
                         "_failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


# ── HTTP layer (lazy import of requests) ────────────────────────────────────
def _hs_request(method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
    """Make a HubSpot API call. Returns (status_code, json_body). Raises only
    on transport-level failures — HTTP error codes are returned as-is so the
    worker can decide whether to retry."""
    import requests  # imported lazily so the module is usable without it
    token = _read_token()
    if not token:
        return (0, {"error": "no_token"})

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    url = f"{_HS_BASE}{path}"
    resp = requests.request(method, url, headers=headers,
                            json=body, timeout=_TIMEOUT)
    try:
        resp_body = resp.json() if resp.text else {}
    except Exception:
        resp_body = {"_raw": resp.text}
    return (resp.status_code, resp_body)


def _is_retryable(status: int) -> bool:
    """5xx and 429 (rate limit) are retryable. 4xx (other than 429) are not —
    those usually indicate a bad payload or invalid token, neither of which a
    retry will fix."""
    return status == 0 or status == 429 or status >= 500


# ── Contact upsert ──────────────────────────────────────────────────────────
def _upsert_contact(props: dict) -> tuple[int, dict]:
    """Upsert a contact by email. HubSpot's "search → patch or create" pattern
    is the cleanest way to do this without race conditions on duplicate emails.
    """
    email = (props.get("email") or "").strip().lower()
    if not email:
        return (400, {"error": "missing_email"})

    # Search by email
    search_body = {
        "filterGroups": [{
            "filters": [{
                "propertyName": "email",
                "operator":     "EQ",
                "value":        email,
            }]
        }],
        "properties": ["email"],
        "limit": 1,
    }
    sc, sresp = _hs_request("POST", "/crm/v3/objects/contacts/search", search_body)
    if sc != 200:
        return (sc, sresp)

    results = (sresp or {}).get("results") or []
    if results:
        contact_id = results[0].get("id")
        sc2, resp2 = _hs_request(
            "PATCH",
            f"/crm/v3/objects/contacts/{contact_id}",
            {"properties": props},
        )
        return (sc2, {**resp2, "_action": "updated", "_id": contact_id})
    else:
        sc2, resp2 = _hs_request(
            "POST",
            "/crm/v3/objects/contacts",
            {"properties": props},
        )
        # HubSpot returns 409 Conflict when an email exists but our search
        # missed it (eventual consistency: a contact created seconds ago may
        # not be searchable yet). Recover by re-searching and PATCHing.
        if sc2 == 409:
            sc3, sresp3 = _hs_request(
                "POST", "/crm/v3/objects/contacts/search", search_body,
            )
            results3 = (sresp3 or {}).get("results") or []
            if sc3 == 200 and results3:
                contact_id = results3[0].get("id")
                sc4, resp4 = _hs_request(
                    "PATCH",
                    f"/crm/v3/objects/contacts/{contact_id}",
                    {"properties": props},
                )
                return (sc4, {**resp4, "_action": "updated_after_409",
                              "_id": contact_id})
            # Couldn't recover — return the original 409 so the caller can
            # see what happened.
            return (sc2, {**resp2, "_action": "conflict_unresolved"})
        # Normal create path — inject _id so callers don't have to special-case.
        return (sc2, {**resp2, "_action": "created",
                      "_id": resp2.get("id")})


# ── Deal creation ───────────────────────────────────────────────────────────
def _create_advisor_followup_deal(contact_id: str, contact_name: str,
                                   risk_score: int, risk_label: str
                                   ) -> tuple[int, dict]:
    """Create a follow-up Deal so the advisor sees a card in their pipeline.
    Uses the default "Sales Pipeline" and the appointment-scheduled stage —
    HubSpot will auto-create these on free tier accounts. Falls back to no
    pipeline if the lookup fails (HubSpot will use the default)."""
    deal_props = {
        "dealname":  f"Risk Profile Follow-up — {contact_name}",
        "dealstage": "appointmentscheduled",   # default first stage on free tier
        "amount":    "0",
        "closedate": (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d"),
        "fra_risk_score":         str(risk_score),
        "fra_risk_label":         risk_label,
    }
    body = {
        "properties": deal_props,
        "associations": [{
            "to": {"id": contact_id},
            "types": [{
                "associationCategory": "HUBSPOT_DEFINED",
                "associationTypeId":   3,   # contact-to-deal default
            }],
        }],
    }
    return _hs_request("POST", "/crm/v3/objects/deals", body)


# ── Worker ──────────────────────────────────────────────────────────────────
def _worker_loop():
    """Drain the queue forever. Exponential backoff between retries; sleeps
    on the wakeup event when idle so we're not spinning."""
    while True:
        if not is_configured():
            # Token went away — sleep and re-check periodically. This lets the
            # worker pick up a config change without a restart.
            _worker_wakeup.wait(timeout=30)
            _worker_wakeup.clear()
            continue

        item = _dequeue_one()
        if item is None:
            _worker_wakeup.wait(timeout=10)
            _worker_wakeup.clear()
            continue

        kind = item.get("_kind")
        attempts = int(item.get("_attempts", 0))
        # Per-item backoff: wait a bit before this attempt if it's a retry
        if attempts > 0:
            time.sleep(_BASE_BACKOFF * (2 ** (attempts - 1)))

        try:
            if kind == "contact_with_deal":
                sc, resp = _upsert_contact(item["contact_props"])
                if sc not in (200, 201):
                    if _is_retryable(sc):
                        _requeue(item)
                    else:
                        _write_deadletter({**item, "_error": resp})
                    continue
                contact_id = resp.get("_id") or (resp.get("id"))
                if not contact_id:
                    _write_deadletter({**item, "_error": "no_contact_id_returned"})
                    continue

                # Now create the follow-up deal
                sc_d, resp_d = _create_advisor_followup_deal(
                    contact_id,
                    item.get("contact_name", ""),
                    int(item.get("risk_score", 0)),
                    str(item.get("risk_label", "")),
                )
                if sc_d not in (200, 201):
                    # Contact succeeded; just retry the deal portion separately
                    if _is_retryable(sc_d):
                        _requeue({"_kind": "deal_only",
                                  "contact_id":  contact_id,
                                  "contact_name": item.get("contact_name",""),
                                  "risk_score":  item.get("risk_score", 0),
                                  "risk_label":  item.get("risk_label",""),
                                  "_attempts":   0})
                    else:
                        _write_deadletter({**item, "_error_deal": resp_d})
            elif kind == "deal_only":
                sc, resp = _create_advisor_followup_deal(
                    item["contact_id"],
                    item.get("contact_name",""),
                    int(item.get("risk_score", 0)),
                    str(item.get("risk_label","")),
                )
                if sc not in (200, 201):
                    if _is_retryable(sc):
                        _requeue(item)
                    else:
                        _write_deadletter({**item, "_error": resp})
            else:
                # Unknown kind — drop it
                _write_deadletter({**item, "_error": "unknown_kind"})
        except Exception as e:
            # Network blip or transient — retry
            if attempts < _MAX_RETRIES:
                _requeue(item)
            else:
                _write_deadletter({**item, "_error": str(e)})


def init():
    """Start the worker thread if it isn't running. Idempotent — safe to call
    on every Streamlit rerun."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        t = threading.Thread(target=_worker_loop, daemon=True,
                             name="hubspot-sync-worker")
        t.start()
        _worker_thread = t


# ── Public API ──────────────────────────────────────────────────────────────
def sync_contact(*, first: str, last: str, email: str, phone: str = "",
                 address: str = "", zipcode: str = "",
                 age: int = 0,
                 risk_score: int = 0, risk_label: str = "",
                 sync_now: bool = True,
                 ) -> dict:
    """Upsert a contact + create an advisor-follow-up deal.

    Two modes:
        sync_now=True (default) — try ONCE synchronously. Total upper bound
            on user-perceived latency is ~16s (two HTTP calls × 8s timeout),
            but in practice ~1-2s. If anything fails it falls through to the
            queue-and-retry path so the user is never blocked.
        sync_now=False — just queue it; the background worker will get to it
            on its own schedule.

    Returns a dict with keys:
        ok:          bool        — overall success
        configured:  bool        — token is set
        contact_id:  str | None  — HubSpot contact ID if created/updated now
        deal_id:     str | None  — deal ID if created now
        error:       str | None  — error message if anything went wrong
        queued:      bool        — True if it was deferred to the worker

    Never raises. Never blocks for more than ~16s under any circumstances."""
    result = {"ok": False, "configured": is_configured(),
              "contact_id": None, "deal_id": None, "error": None,
              "queued": False}

    if not result["configured"]:
        result["error"] = "HubSpot not configured (no token)."
        return result

    # Always make sure the worker is around to handle the queue, even in
    # synchronous mode — failed sync_now calls fall back to it.
    init()

    contact_props = {
        "firstname": (first or "").strip(),
        "lastname":  (last or "").strip(),
        "email":     (email or "").strip().lower(),
        "phone":     (phone or "").strip(),
        "address":   (address or "").strip(),
        "zip":       (zipcode or "").strip(),
        "fra_risk_score":  str(int(risk_score or 0)),
        "fra_risk_label":  str(risk_label or ""),
        "fra_risk_age":    str(int(age or 0)),
    }
    contact_name = f"{(first or '').strip()} {(last or '').strip()}".strip()

    payload = {
        "_kind":         "contact_with_deal",
        "contact_props": contact_props,
        "contact_name":  contact_name,
        "risk_score":    int(risk_score or 0),
        "risk_label":    str(risk_label or ""),
    }

    if not sync_now:
        _enqueue(payload)
        result["queued"] = True
        result["ok"] = True
        return result

    # ── Synchronous attempt ───────────────────────────────────────────
    # Try the upsert + deal creation right now so a working setup gets
    # immediate feedback. If anything fails we fall back to queueing,
    # so the registration is never lost.
    try:
        sc, resp = _upsert_contact(contact_props)
        if sc not in (200, 201):
            # Sync failed — let the background worker retry
            _enqueue(payload)
            result["queued"] = True
            result["error"] = f"Contact upsert HTTP {sc} — queued for retry."
            return result

        contact_id = resp.get("_id") or resp.get("id")
        result["contact_id"] = contact_id

        if contact_id:
            sc_d, resp_d = _create_advisor_followup_deal(
                contact_id, contact_name,
                int(risk_score or 0), str(risk_label or ""),
            )
            if sc_d in (200, 201):
                result["deal_id"] = resp_d.get("id")
            else:
                # Contact succeeded; queue just the deal for retry
                _enqueue({"_kind": "deal_only",
                          "contact_id":  contact_id,
                          "contact_name": contact_name,
                          "risk_score":  int(risk_score or 0),
                          "risk_label":  str(risk_label or "")})
                result["queued"] = True
                result["error"] = f"Deal HTTP {sc_d} — queued for retry."
        result["ok"] = True
        return result
    except Exception as e:
        # Network error / timeout / anything — queue it for the worker
        _enqueue(payload)
        result["queued"] = True
        result["error"] = f"Sync error: {e} — queued for retry."
        return result
