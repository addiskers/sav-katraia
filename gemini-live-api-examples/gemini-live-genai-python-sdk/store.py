"""
JSON file persistence for call records — one file per call.

Each call is stored at  DATA_DIR/calls/<call_id>.json  (human-readable).
One file per call means each live call owns its own file, so there is no
concurrent-writer contention. Writes are atomic (temp file + os.replace).

A lightweight in-memory index (everything except the transcript/tool_calls
arrays) is kept so list/summary endpoints don't re-read every file. Full
records are only read from disk for the call-detail view.

All public functions are async and run blocking disk I/O in a thread executor
so they never block the FastAPI event loop. Storage is intentionally isolated
behind this module so it can be swapped for SQLite/Postgres later.
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
CALLS_DIR = os.path.join(DATA_DIR, "calls")

# call_id -> lightweight meta (full record minus transcript/tool_calls)
_INDEX = {}
_LOCK = threading.Lock()

# Heavy fields excluded from the in-memory index.
_HEAVY_FIELDS = ("transcript", "tool_calls")


def _meta_from_call(call):
    return {k: v for k, v in call.items() if k not in _HEAVY_FIELDS}


def _path(call_id):
    return os.path.join(CALLS_DIR, f"{call_id}.json")


# ---- sync internals (run inside executor) ----------------------------------

def _init_sync():
    os.makedirs(CALLS_DIR, exist_ok=True)
    with _LOCK:
        _INDEX.clear()
        for name in os.listdir(CALLS_DIR):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(CALLS_DIR, name), "r", encoding="utf-8") as f:
                    call = json.load(f)
                _INDEX[call["id"]] = _meta_from_call(call)
            except Exception as e:
                logger.warning(f"Skipping unreadable call file {name}: {e}")
    logger.info(f"Call store initialized at {CALLS_DIR} ({len(_INDEX)} calls)")


def _save_sync(call):
    os.makedirs(CALLS_DIR, exist_ok=True)
    path = _path(call["id"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(call, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)
    with _LOCK:
        _INDEX[call["id"]] = _meta_from_call(call)


def _load_sync(call_id):
    try:
        with open(_path(call_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Failed to load call {call_id}: {e}")
        return None


# ---- async public API ------------------------------------------------------

async def _run(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


async def init():
    await _run(_init_sync)


async def save_call(call):
    await _run(_save_sync, call)


async def load_call(call_id):
    return await _run(_load_sync, call_id)


def _date_of(meta):
    s = meta.get("started_at") or ""
    return s[:10]


def _matches(meta, filters):
    src = filters.get("source")
    if src and meta.get("source") != src:
        return False
    booking = filters.get("booking")
    if booking is not None:
        want = booking in (True, "1", "true", "yes")
        if bool(meta.get("booking_created")) != want:
            return False
    frm = filters.get("from")
    to = filters.get("to")
    d = _date_of(meta)
    if frm and d and d < frm:
        return False
    if to and d and d > to:
        return False
    q = (filters.get("q") or "").strip().lower()
    if q:
        hay = " ".join(str(meta.get(k, "")) for k in
                       ("caller", "call_sid", "language", "status", "source")).lower()
        if q not in hay:
            return False
    return True


async def list_calls(filters=None):
    """Return {items: [...meta], total: n} filtered + paginated, newest first."""
    filters = filters or {}
    with _LOCK:
        metas = list(_INDEX.values())
    rows = [m for m in metas if _matches(m, filters)]
    rows.sort(key=lambda m: m.get("started_at") or "", reverse=True)
    total = len(rows)
    offset = int(filters.get("offset") or 0)
    limit = filters.get("limit")
    if limit is not None:
        rows = rows[offset:offset + int(limit)]
    elif offset:
        rows = rows[offset:]
    return {"items": rows, "total": total}


async def summary(filters=None):
    """Aggregate project costing across all (filtered) calls."""
    filters = filters or {}
    with _LOCK:
        metas = [m for m in _INDEX.values() if _matches(m, filters)]

    now = datetime.now(timezone.utc)
    month_prefix = now.strftime("%Y-%m")

    total_cost = ai_cost = twilio_cost = 0.0
    total_secs = 0
    bookings = 0
    by_source = {}
    by_lang = {}
    by_day = {}
    pending_twilio = 0
    month_cost = 0.0
    month_calls = 0

    for m in metas:
        g = m.get("ai_cost_usd") or 0.0
        tw = ((m.get("twilio") or {}).get("price_usd")) or 0.0
        t = m.get("total_cost_usd")
        if t is None:
            t = g + tw
        ai_cost += g
        twilio_cost += tw
        total_cost += t
        total_secs += m.get("duration_seconds") or 0
        if m.get("booking_created"):
            bookings += 1
        src = m.get("source") or "unknown"
        by_source[src] = by_source.get(src, 0) + 1
        lang = m.get("language") or "unknown"
        by_lang[lang] = by_lang.get(lang, 0) + 1
        if m.get("source") == "twilio" and (m.get("twilio") or {}).get("price_usd") is None:
            pending_twilio += 1

        d = _date_of(m)
        if d:
            day = by_day.setdefault(d, {"date": d, "calls": 0, "cost_usd": 0.0,
                                        "ai_cost_usd": 0.0})
            day["calls"] += 1
            day["cost_usd"] = round(day["cost_usd"] + t, 6)
            day["ai_cost_usd"] = round(day["ai_cost_usd"] + g, 6)

        if d.startswith(month_prefix):
            month_cost += t
            month_calls += 1

    total_calls = len(metas)
    avg_cost = round(total_cost / total_calls, 6) if total_calls else 0.0

    # Projected month cost = run rate so far this month, extrapolated to month end.
    day_of_month = now.day
    import calendar
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    projected = round(month_cost / day_of_month * days_in_month, 4) if day_of_month else month_cost

    return {
        "total_calls": total_calls,
        "by_source": by_source,
        "by_language": by_lang,
        "total_minutes": round(total_secs / 60.0, 2),
        "total_seconds": total_secs,
        "total_cost_usd": round(total_cost, 6),
        "ai_cost_usd": round(ai_cost, 6),
        "twilio_cost_usd": round(twilio_cost, 6),
        "avg_cost_per_call": avg_cost,
        "bookings": bookings,
        "booking_conversion_rate": round(bookings / total_calls, 4) if total_calls else 0.0,
        "this_month": {"calls": month_calls, "cost_usd": round(month_cost, 6)},
        "projected_month_cost": projected,
        "by_day": sorted(by_day.values(), key=lambda x: x["date"]),
        "pending_twilio_price": pending_twilio,
    }


async def sweep_stale(max_age_minutes=30):
    """Mark long-running 'in_progress' records (orphaned by a crash) as abandoned."""
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_minutes * 60
    with _LOCK:
        stale_ids = []
        for cid, m in _INDEX.items():
            if m.get("status") != "in_progress":
                continue
            try:
                started = datetime.fromisoformat(m["started_at"]).timestamp()
            except Exception:
                continue
            if started < cutoff:
                stale_ids.append(cid)
    for cid in stale_ids:
        call = await load_call(cid)
        if call and call.get("status") == "in_progress":
            call["status"] = "abandoned"
            await save_call(call)
            logger.info(f"Marked stale call {cid} as abandoned")
