"""
Cost calculation for Voice Hero calls.

The voice pipeline uses three AI providers plus telephony:
- Deepgram STT  -> USD per minute of caller audio streamed.
- LLM (OpenRouter -> Groq preferred / Cerebras fallback)
                -> USD per 1M prompt / completion tokens (from `usage`).
- Sarvam TTS    -> INR per 10k characters synthesized (Bulbul) -> converted to USD.
- Twilio        -> Twilio's REAL billed Call.price (USD, fetched from the API).

All rates are env-overridable so they can be corrected without a code change.
Everything is normalized to USD (Sarvam INR -> USD via USD_INR).

Default LLM rates match OpenRouter's Groq route for openai/gpt-oss-120b
($0.15 in / $0.60 out per 1M) — our preferred fastest provider.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---- Deepgram STT (USD per minute) -----------------------------------------
# nova-2/nova-3 streaming pay-as-you-go, ~$0.0077/min (Growth: ~$0.0058/min).
DEEPGRAM_PER_MINUTE = float(os.getenv("RATE_DEEPGRAM_PER_MIN", "0.0077"))

# ---- LLM via OpenRouter (USD per 1M tokens) --------------------------------
# Prefer RATE_LLM_*; fall back to legacy RATE_GROQ_* names.
# Defaults = Groq on OpenRouter for openai/gpt-oss-120b ($0.15 / $0.60).
# Cerebras on OpenRouter is ~$0.35 / $0.75 if you force that route.
LLM_IN_PER_1M = float(
    os.getenv("RATE_LLM_IN_PER_1M")
    or os.getenv("RATE_GROQ_IN_PER_1M")
    or "0.15"
)
LLM_OUT_PER_1M = float(
    os.getenv("RATE_LLM_OUT_PER_1M")
    or os.getenv("RATE_GROQ_OUT_PER_1M")
    or "0.60"
)

# Back-compat aliases used by older call sites / docs.
GROQ_IN_PER_1M = LLM_IN_PER_1M
GROQ_OUT_PER_1M = LLM_OUT_PER_1M

LLM_PROVIDER_LABEL = os.getenv("LLM_PROVIDER_LABEL", "openrouter/groq")

# ---- Sarvam Bulbul TTS (INR per 10k chars) ---------------------------------
SARVAM_TTS_PER_10K_INR = float(os.getenv("RATE_SARVAM_TTS_PER_10K", "30.0"))
USD_INR = float(os.getenv("USD_INR", "88.0"))

# ---- Twilio fallback rate (USD per minute) ---------------------------------
# Only used when Twilio's real Call.price is not (yet) available.
TWILIO_PER_MINUTE = float(os.getenv("RATE_TWILIO_PER_MIN", "0.014"))


def _empty_usage():
    """Zeroed AI usage structure used by a call record."""
    return {
        "stt_seconds": 0.0,   # caller audio streamed to Deepgram
        "llm_in": 0,          # LLM prompt tokens
        "llm_out": 0,         # LLM output tokens
        "tts_chars": 0,       # characters synthesized by Bulbul
    }


def inr_to_usd(inr):
    return round((inr or 0.0) / USD_INR, 6)


def compute_ai_cost(usage):
    """Real AI (STT + LLM + TTS) cost in USD from accumulated usage."""
    u = usage or {}
    stt = (u.get("stt_seconds", 0.0) / 60.0) * DEEPGRAM_PER_MINUTE
    llm = ((u.get("llm_in", 0) / 1_000_000.0) * LLM_IN_PER_1M
           + (u.get("llm_out", 0) / 1_000_000.0) * LLM_OUT_PER_1M)
    tts_inr = (u.get("tts_chars", 0) / 10_000.0) * SARVAM_TTS_PER_10K_INR
    tts = tts_inr / USD_INR
    return round(stt + llm + tts, 6)


def ai_cost_breakdown(usage):
    """Per-provider USD breakdown for the call-detail view."""
    u = usage or {}
    stt = round((u.get("stt_seconds", 0.0) / 60.0) * DEEPGRAM_PER_MINUTE, 6)
    llm_in = round((u.get("llm_in", 0) / 1e6) * LLM_IN_PER_1M, 6)
    llm_out = round((u.get("llm_out", 0) / 1e6) * LLM_OUT_PER_1M, 6)
    tts_inr = round((u.get("tts_chars", 0) / 1e4) * SARVAM_TTS_PER_10K_INR, 6)
    tts = round(tts_inr / USD_INR, 6)
    per = {"stt": stt, "llm_in": llm_in, "llm_out": llm_out, "tts": tts}
    return {
        "usage": u,
        "providers": {
            "stt": "deepgram",
            "llm": LLM_PROVIDER_LABEL,
            "tts": "sarvam-bulbul",
        },
        "rates": {
            "llm_in_per_1m_usd": LLM_IN_PER_1M,
            "llm_out_per_1m_usd": LLM_OUT_PER_1M,
        },
        "cost_by_stage_usd": per,
        "tts_cost_inr": tts_inr,
        "cost_usd": round(sum(per.values()), 6),
    }


def compute_total(call):
    """
    Total real cost for a call in USD = AI cost (STT+LLM+TTS) + Twilio cost.

    Twilio cost prefers the real billed price; falls back to a per-minute
    estimate when the price has not arrived yet. Returns (total_usd, estimated).
    """
    ai_usd = call.get("ai_cost_usd")
    if ai_usd is None:
        ai_usd = compute_ai_cost(call.get("ai_usage"))

    twilio = call.get("twilio") or {}
    twilio_price = twilio.get("price_usd")
    estimated = bool(call.get("cost_estimated"))

    if twilio_price is None and call.get("source") == "twilio":
        # Estimate from duration until the real price is fetched.
        secs = twilio.get("duration_seconds") or call.get("duration_seconds") or 0
        twilio_price = round((secs / 60.0) * TWILIO_PER_MINUTE, 6)
        estimated = True

    total = round((ai_usd or 0) + (twilio_price or 0), 6)
    return total, estimated


def fetch_twilio_price(call_sid):
    """
    Fetch the REAL billed price for a Twilio call. Synchronous (Twilio SDK is
    sync) -> call this inside an executor. Returns a dict or None on failure.

    Twilio reports `price` as a negative string (a charge) and only populates
    it a few seconds AFTER the call completes, so the caller should retry while
    `price` is None.
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and token and call_sid):
        return None
    try:
        from twilio.rest import Client

        call = Client(sid, token).calls(call_sid).fetch()
        price = abs(float(call.price)) if call.price else None
        return {
            "price_usd": price,
            "price_unit": call.price_unit,
            "duration_seconds": int(call.duration) if call.duration else None,
            "status": call.status,
        }
    except Exception as e:
        logger.warning(f"Twilio price fetch failed for {call_sid}: {e}")
        return None
