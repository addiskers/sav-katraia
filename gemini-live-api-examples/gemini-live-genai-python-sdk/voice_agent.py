"""
Low-latency voice agent engine.

  caller audio (PCM16 16kHz) --> Deepgram nova-3 streaming STT (WebSocket)
                                       |
                                 final transcript
                                       |
              OpenRouter -> Cerebras gpt-oss-120b chat completions (tool calling)
                                       |
                                 reply text
                                       |
           Sarvam Bulbul TTS (WebSocket stream) --> PCM16 24kHz --> caller

Deepgram gives sub-200ms interim transcripts + endpointing, Cerebras serves
gpt-oss-120b at ~3000 tok/s with prompt caching, and Sarvam Bulbul WebSocket
streaming starts audio mid-generation. Only TTS stays on Sarvam.

Exposes the SAME session interface the rest of the app already uses
(`start_session(...)` async generator yielding event dicts), so the Twilio
bridge, recorder, live dashboard and browser client all keep working.

Emitted events:
  {"type": "user", "text": ...}          final user utterance transcript
  {"type": "agent", "text": ...}         agent reply transcript
  {"type": "tool_call", "name", "args", "result"}
  {"type": "turn_complete"}
  {"type": "interrupted"}                user barged in while agent spoke
  {"type": "usage", ...}                 real usage for costing:
        stt_seconds  - seconds of caller audio streamed to Deepgram
        llm_in / llm_out - chat completion tokens
        tts_chars    - characters synthesized by Bulbul
  {"type": "error", "error": ...}
"""

import asyncio
import base64
import inspect
import json
import logging
import os
import re
import statistics
import time
import traceback
from urllib.parse import urlencode

import httpx
import websockets

from system_prompt import build_system_instruction

logger = logging.getLogger(__name__)

# ---- Providers -------------------------------------------------------------
# NOTE: API keys and most knobs are read at session START (see start_session),
# not import time, because main.py imports this module before load_dotenv().
# Legacy GROQ_* names remain as fallbacks.
DEFAULT_LLM_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# ---- Deepgram STT (env-overridable; re-read in start_session) --------------
# nova-3 supports language=multi (Hindi/English/Gujarati code-mixing).
STT_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
STT_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "hi")
DEFAULT_CALL_LANGUAGE = os.getenv("DEFAULT_CALL_LANGUAGE", "hi-IN")

# ---- Sarvam TTS (env-overridable) ------------------------------------------
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "rahul")
TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "24000"))

# Call-start trigger from Twilio (and similar) — used for greeting shortcut.
CALL_START_RE = re.compile(
    r"(start the call|picked up the phone|__start__|begin the call)",
    re.IGNORECASE,
)


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def get_system_instruction():
    return build_system_instruction()


# Tool declarations in OpenAI/Groq chat-completions format.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_info",
            "description": "Get complete vehicle info including owner name, model, service history, warranty, and next service due. Call this FIRST at the start of every call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "Customer phone number"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_pickup",
            "description": "Schedule vehicle pickup for service when customer agrees to date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_number": {"type": "string", "description": "Vehicle registration number"},
                    "date": {"type": "string", "description": "Pickup date (YYYY-MM-DD or natural language like 'tomorrow')"},
                    "time": {"type": "string", "description": "Pickup time like '9:30 AM'"},
                    "pickup_address": {"type": "string", "description": "Customer's confirmed pickup address (use address from vehicle record if confirmed, or new address if customer provides one)"},
                    "special_instructions": {"type": "string", "description": "Any special request like 'need car back by 8 PM'"}
                },
                "required": ["vehicle_number", "date", "time", "pickup_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_cost_estimate",
            "description": "Get estimated cost range for a service type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_type": {"type": "string", "description": "e.g. 'Third Service', 'Second Service'"}
                },
                "required": ["service_type"]
            }
        }
    }
]


def _strip_wav_header(data: bytes) -> bytes:
    """Return raw PCM from a WAV blob (pass-through if already raw)."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return data
    idx = data.find(b"data", 12)
    if idx == -1:
        return data[44:]
    return data[idx + 8:]


def _script_counts(text: str):
    gujarati = devanagari = latin = 0
    for ch in text or "":
        o = ord(ch)
        if 0x0A80 <= o <= 0x0AFF:
            gujarati += 1
        elif 0x0900 <= o <= 0x097F:
            devanagari += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    return gujarati, devanagari, latin


def _has_indic_script(text: str) -> bool:
    gu, dev, _ = _script_counts(text)
    return (gu + dev) > 0


def _tts_language_for(text: str) -> str:
    """Pick Bulbul language from script counts (fallback when no call language set)."""
    gu, dev, latin = _script_counts(text)
    if gu > dev and gu > latin:
        return "gu-IN"
    if _looks_marathi(text):
        return "mr-IN"
    if dev >= latin and dev > 0:
        return "hi-IN"
    if latin > 0:
        return "en-IN"
    return DEFAULT_CALL_LANGUAGE


_EXPLICIT_LANG_PATTERNS = (
    (re.compile(r"(talk|speak)\s+in\s+english|english\s+me(in)?|in\s+english\b", re.I), "en-IN"),
    (re.compile(r"gujarati\s+ma|gujarati\s+me|speak\s+gujarati|ગુજરાતી", re.I), "gu-IN"),
    (re.compile(r"marathi\s+madhe|marathi\s+me|speak\s+marathi|मराठी", re.I), "mr-IN"),
    (re.compile(r"hindi\s+me(in)?|speak\s+hindi|हिंदी\s+में", re.I), "hi-IN"),
)

_MARATHI_MARKERS = re.compile(
    r"ळ|आहे|नाही|होय|काय|मला|तुम्ही|नको|बरो|माझ|तुमच",
    re.I,
)

# Short Latin acks — do not treat as English on first response (STT noise).
_LATIN_ACKS = frozenset({
    "hello", "hi", "hey", "yes", "yeah", "ok", "okay", "haan", "han", "ji",
    "hmm", "hm", "speaking", "bolo", "boliye",
})

_INDIC_ACKS = frozenset({
    "हां", "हाँ", "haan", "han", "ji", "जी", "bolo", "बोलो", "boliye", "बोलिए",
    "theek", "ठीक", "ok", "okay", "yes", "hello", "hi", "ha", "hmm", "achha",
    "अच्छा", "sahi", "सही",
})


def _resolve_explicit_language(text: str) -> str | None:
    for pattern, lang in _EXPLICIT_LANG_PATTERNS:
        if pattern.search(text or ""):
            return lang
    return None


def _looks_marathi(text: str) -> bool:
    return bool(_MARATHI_MARKERS.search(text or ""))


def _looks_english_turn(text: str) -> bool:
    gu, dev, latin = _script_counts(text)
    if gu > 0 or dev > 0:
        return False
    words = [w.strip(".,!?").lower() for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return False
    if all(w in _LATIN_ACKS for w in words):
        return False
    if len(words) >= 2 and latin >= len(words):
        return True
    return len(words) >= 3 and latin >= 2


def _update_customer_language(
    text: str,
    current: str,
    en_streak: dict | None = None,
    *,
    is_first_response: bool = False,
) -> str:
    """Update reply/TTS language from customer speech."""
    explicit = _resolve_explicit_language(text)
    if explicit:
        if en_streak is not None:
            en_streak["n"] = 0
        return explicit

    gu, dev, latin = _script_counts(text)
    if gu > 2 and gu > dev:
        if en_streak is not None:
            en_streak["n"] = 0
        return "gu-IN"
    if _looks_marathi(text):
        if en_streak is not None:
            en_streak["n"] = 0
        return "mr-IN"
    if dev > 0:
        if en_streak is not None:
            en_streak["n"] = 0
        return "hi-IN"

    if latin > 0 and gu == 0 and dev == 0:
        # Gemini prompt: auto-detect language on first customer response.
        if is_first_response and _looks_english_turn(text):
            if en_streak is not None:
                en_streak["n"] = 0
            return "en-IN"

        words = [w for w in re.split(r"\s+", text.strip()) if w]
        min_en_words = int(_env_float("ENGLISH_SWITCH_MIN_WORDS", 8))
        need_streak = int(_env_float("ENGLISH_SWITCH_STREAK", 2))
        if len(words) >= min_en_words:
            if en_streak is not None:
                en_streak["n"] = en_streak.get("n", 0) + 1
                if en_streak["n"] >= need_streak:
                    return "en-IN"
                return current or DEFAULT_CALL_LANGUAGE
            return "en-IN"
        if en_streak is not None:
            en_streak["n"] = 0
    return current or DEFAULT_CALL_LANGUAGE


def _turn_confidence_floor(text: str, base_min: float) -> float:
    """Confidence threshold for accepting a user turn."""
    if _has_indic_script(text):
        return base_min
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    # Short acks ("hello", "haan") — Hindi STT often scores 0.3–0.5; trust structure.
    if len(words) <= 3:
        return base_min
    if len(words) <= 4:
        return max(base_min, _env_float("STT_LATIN_MEDIUM_CONF", 0.55))
    return max(base_min, _env_float("STT_LATIN_MIN_CONF", 0.50))


def _language_hint(lang_code: str) -> str:
    labels = {
        "hi-IN": "Hindi/Hinglish (Devanagari script)",
        "gu-IN": "Gujarati script only",
        "mr-IN": "Marathi (Devanagari script)",
        "en-IN": "English only",
    }
    label = labels.get(lang_code, lang_code)
    return (
        f"LANGUAGE LOCK: Reply ONLY in {label} for this turn and all following turns "
        f"until the customer switches language. Never mix scripts or languages. "
        f"Never use generic help-desk filler."
    )


def _split_sentences(text: str, max_len: int = 140):
    """Split reply into TTS-friendly chunks (< ~500 chars for Sarvam WS)."""
    pieces = re.split(r"(?<=[।.!?])\s+|\n+", text)
    parts = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        # Only split on commas for very long clauses — mid-clause commas
        # caused truncated TTS ("...Baleno ki" then silence).
        if len(p) > 100:
            parts.extend(s.strip() for s in re.split(r"(?<=[,।])\s+", p) if s.strip())
        else:
            parts.append(p)

    chunks = []
    for part in parts:
        while len(part) > max_len:
            cut = part.rfind(" ", 0, max_len)
            if cut <= 0:
                cut = max_len
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks or ([text.strip()] if text.strip() else [])


def _first_name(owner_name: str) -> str:
    name = (owner_name or "").strip()
    if not name:
        return "ग्राहक"
    return name.split()[0]


_GARBAGE_PUNCT_RE = re.compile(r"[¿¡]")
_GARBAGE_ACCENT_RE = re.compile(
    r"[àáâãäåæçèéêëìíîïñòóôõöùúûüýÿ]", re.I)
# Punctuation that should not count against speech-content ratio.
_NEUTRAL_PUNCT = frozenset(".,!?;:'\"()-…")


def _is_speech_char(ch: str) -> bool:
    """True for letters in any script (incl. Indic matras Deepgram emits)."""
    if not ch or ch.isspace() or ch in _NEUTRAL_PUNCT:
        return False
    o = ord(ch)
    if 0x0900 <= o <= 0x097F:   # Devanagari
        return True
    if 0x0A80 <= o <= 0x0AFF:   # Gujarati
        return True
    return ch.isalpha() or ch.isdigit()


def _speech_char_ratio(text: str) -> float:
    """Share of chars that look like speech, not whitespace/punctuation."""
    stripped = (text or "").strip()
    if not stripped:
        return 0.0
    speech = sum(1 for ch in stripped if _is_speech_char(ch))
    return speech / len(stripped)


def _median_word_confidence(words) -> float | None:
    confs = [
        float(w["confidence"])
        for w in (words or [])
        if w.get("confidence") is not None
    ]
    if not confs:
        return None
    return float(statistics.median(confs))


def _utterance_structure_ok(text: str):
    """Language-agnostic shape check. Returns (ok, reason)."""
    text = (text or "").strip()
    if not text:
        return False, "empty"

    min_ratio = _env_float("STT_MIN_LETTER_RATIO", 0.30)
    if _speech_char_ratio(text) < min_ratio:
        return False, "low_letter_ratio"

    if _GARBAGE_PUNCT_RE.search(text) or _GARBAGE_ACCENT_RE.search(text):
        return False, "garbage_punctuation"

    return True, "ok"


def _utterance_confidence(text, confidence=None, words=None):
    """Confidence score when structure is OK; None if structure fails."""
    ok, reason = _utterance_structure_ok(text)
    if not ok:
        return None, reason

    word_conf = _median_word_confidence(words)
    if word_conf is not None:
        return word_conf, "ok"
    if confidence is not None:
        return float(confidence), "ok"
    # Deepgram omitted scores — trust structural pass.
    return 0.55, "ok"


def _allows_turn(score, min_confidence):
    return score is not None and score >= min_confidence


def _allows_barge_in(text: str, *, recent_agent: str) -> bool:
    """Whether to interrupt the in-flight agent turn for this utterance."""
    ok, _ = _utterance_structure_ok(text)
    if not ok:
        return False
    if _is_echo_of_agent(text, recent_agent):
        return False
    if _is_ack_only(text):
        return False
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) >= 3:
        return True
    if _has_interrupt_intent(text):
        return True
    return False


def _interim_allows_barge_in(text: str, recent_agent: str) -> bool:
    """Fast barge-in on interim transcripts while the agent is speaking.

    Stricter than final-transcript barge-in: interims are noisier, so require
    a real multi-word utterance (or explicit interrupt intent) that is not an
    echo of the agent's own TTS.
    """
    ok, _ = _utterance_structure_ok(text)
    if not ok:
        return False
    if _is_echo_of_agent(text, recent_agent):
        return False
    if _is_ack_only(text):
        return False
    if _has_interrupt_intent(text):
        return True
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return len(words) >= 3


def _normalize_echo_text(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "").lower())


def _is_echo_of_agent(user_text: str, agent_text: str) -> bool:
    """Reject STT that is mostly the agent's own TTS picked up by the mic."""
    u = _normalize_echo_text(user_text)
    a = _normalize_echo_text(agent_text)
    if not u or not a:
        return False
    if len(u) >= 12 and (u in a or a in u):
        return True
    u_words = [w for w in u.split() if len(w) > 1]
    a_words = set(w for w in a.split() if len(w) > 1)
    if len(u_words) >= 3 and a_words:
        overlap = sum(1 for w in u_words if w in a_words) / len(u_words)
        if overlap >= 0.7:
            return True
    return False


def _is_ack_only(text: str) -> bool:
    words = [
        re.sub(r"[^\w]", "", w).lower()
        for w in re.split(r"\s+", (text or "").strip())
        if w
    ]
    if not words:
        return True
    return all(w in _LATIN_ACKS or w in _INDIC_ACKS for w in words)


_INTERRUPT_MARKERS = re.compile(
    r"\b(wait|stop|ruko|ruk|sun|suno|suniye|listen|nahi|no|galat|wrong|"
    r"band|chup|hold|busy|baad|later|callback)\b|"
    r"(रुको|सुन|सुनो|सुनिए|नही|गलत|बंद|बाद|मत)",
    re.I,
)


def _has_interrupt_intent(text: str) -> bool:
    return bool(_INTERRUPT_MARKERS.search(text or ""))


def _sanitize_speech(text: str) -> str:
    """Strip parenthetical stage directions the LLM sometimes emits."""
    if not text:
        return text
    cleaned = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# TTS pronunciation lexicon: Bulbul's Indic voices garble Latin-script brand
# names (e.g. "Suzuki" -> "Sogeni"). Send them in Devanagari for Indic langs.
_TTS_DEVANAGARI_LEXICON = [
    (re.compile(r"maruti\s+suzuki", re.I), "मारुति सुज़ुकी"),
    (re.compile(r"\bsuzuki\b", re.I), "सुज़ुकी"),
    (re.compile(r"\bmaruti\b", re.I), "मारुति"),
    (re.compile(r"\bbaleno\b", re.I), "बलेनो"),
    (re.compile(r"\bkataria\b", re.I), "कटारिया"),
    (re.compile(r"\bautomobiles\b", re.I), "ऑटोमोबाइल्स"),
    (re.compile(r"\bahmedabad\b", re.I), "अहमदाबाद"),
]


def _tts_pronounce(text: str, lang: str) -> str:
    """Rewrite brand names for correct TTS pronunciation (Indic langs only)."""
    if not text or lang == "en-IN":
        return text
    for pat, rep in _TTS_DEVANAGARI_LEXICON:
        text = pat.sub(rep, text)
    return text


def _resolve_llm_config():
    """Prefer OpenRouter (LLM_*/OPENROUTER_*); fall back to Groq as a set.

    Never mix an OpenRouter URL with a Groq key (or vice versa).
    """
    openrouter_key = (
        os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()

    if openrouter_key:
        chat_url = (
            os.getenv("LLM_CHAT_URL")
            or DEFAULT_LLM_CHAT_URL
        )
        model = (
            os.getenv("LLM_MODEL")
            or DEFAULT_LLM_MODEL
        )
        return chat_url, model, openrouter_key

    if groq_key:
        chat_url = (
            os.getenv("GROQ_CHAT_URL")
            or "https://api.groq.com/openai/v1/chat/completions"
        )
        model = (
            os.getenv("GROQ_LLM_MODEL")
            or os.getenv("LLM_MODEL")
            or "openai/gpt-oss-20b"
        )
        return chat_url, model, groq_key

    # No keys — still return OpenRouter defaults so errors are clear.
    return (
        os.getenv("LLM_CHAT_URL") or DEFAULT_LLM_CHAT_URL,
        os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL,
        "",
    )


class VoiceAgent:
    """
    Deepgram STT -> OpenRouter/Cerebras LLM -> Sarvam streaming TTS voice agent.

    Exposes the app's standard engine interface: same constructor shape and
    the same `start_session` async-generator interface.
    """

    def __init__(self, api_key=None, model=None, input_sample_rate=16000,
                 tools=None, tool_mapping=None):
        """
        Args:
            api_key: unused (kept for interface compatibility) — provider keys
                     come from env (DEEPGRAM_API_KEY, LLM_API_KEY / GROQ_API_KEY,
                     SARVAM_API_KEY).
            model: chat model id (e.g. 'openai/gpt-oss-120b').
            input_sample_rate: caller audio sample rate (PCM16 mono).
            tools: tool declarations (chat-completions format).
            tool_mapping: tool name -> python callable.
        """
        _, default_model, _ = _resolve_llm_config()
        self.model = model or default_model
        self.input_sample_rate = input_sample_rate
        self.tools = tools or TOOLS
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue,
                            audio_output_callback, audio_interrupt_callback=None):
        # Read keys / knobs now (env is guaranteed loaded by the time a session starts).
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
        llm_chat_url, _, llm_key = _resolve_llm_config()
        # Prefer constructor model, else fresh env (may have changed after import).
        model = self.model or _resolve_llm_config()[1]
        sarvam_key = os.getenv("SARVAM_API_KEY", "")
        stt_model = os.getenv("DEEPGRAM_MODEL", STT_MODEL)
        stt_language = os.getenv("DEEPGRAM_LANGUAGE", STT_LANGUAGE)
        stt_endpointing_ms = os.getenv("DEEPGRAM_ENDPOINTING_MS", "300")
        stt_utterance_end_ms = os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1000")
        turn_debounce = _env_float("TURN_DEBOUNCE", 0.08)
        stt_min_confidence = _env_float("STT_MIN_CONFIDENCE", 0.45)
        use_openrouter = "openrouter.ai" in llm_chat_url

        try:
            http = httpx.AsyncClient(
                http2=True,
                timeout=45.0,
                limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60.0),
            )
        except Exception:
            # h2 package missing — fall back to HTTP/1.1
            http = httpx.AsyncClient(
                timeout=45.0,
                limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60.0),
            )

        event_queue = asyncio.Queue()
        turn_queue = asyncio.Queue()
        messages = [{"role": "system", "content": get_system_instruction()}]

        speaking = {"on": False}       # agent audio is being sent out right now
        speaking_since = {"t": 0.0}    # monotonic time when current audio began
        agent_busy = {"v": False}      # LLM+TTS turn in flight (broader than speaking)
        current_turn = {"task": None}  # in-flight agent turn task
        greeted = {"done": False}
        default_lang = os.getenv("DEFAULT_CALL_LANGUAGE", DEFAULT_CALL_LANGUAGE)
        customer_lang = {"v": default_lang, "hint_sent": False}
        en_streak = {"n": 0}
        customer_turns = {"n": 0}
        tts_lock = asyncio.Lock()      # one Sarvam WS speaker at a time
        # Barge-in disabled during greeting / until first real audio plays.
        # Also ignored for BARGE_IN_GRACE_S after audio starts (mic hears TTS echo).
        barge_in_ok = {"v": False}
        # Grace only for VAD soft-mute (echo protection).
        BARGE_IN_GRACE_S = float(os.getenv("BARGE_IN_GRACE_S", "0.8"))
        # After a reply starts, ignore trailing speech_final that would re-cancel it.
        REPLY_GUARD_S = float(os.getenv("REPLY_GUARD_S", "1.0"))
        reply_guard_until = {"t": 0.0}
        last_utt = {"text": "", "t": 0.0}
        recent_agent = {"text": ""}


        async def emit(event):
            await event_queue.put(event)

        async def _flush_client_audio():
            if audio_interrupt_callback:
                if inspect.iscoroutinefunction(audio_interrupt_callback):
                    await audio_interrupt_callback()
                else:
                    audio_interrupt_callback()
            await emit({"type": "interrupted"})

        # ---- barge-in -------------------------------------------------------
        async def interrupt_agent(reason="speech", *, force=False):
            """Cancel the in-flight agent turn and flush client audio.

            force=True: skip echo grace (use when Deepgram has real transcript text).
            """
            if not barge_in_ok["v"]:
                return False
            task = current_turn["task"]
            if not task or task.done() or not agent_busy["v"]:
                return False
            # Trailing speech_final must not kill a reply that just started
            # generating (same utterance that triggered the turn).
            if time.monotonic() < reply_guard_until["t"] and not speaking["on"]:
                logger.info(f"Barge-in ignored (reply guard): {reason}")
                return False
            if not force and speaking["on"]:
                if (time.monotonic() - speaking_since["t"]) < BARGE_IN_GRACE_S:
                    return False
            logger.info(f"Barge-in ({reason}, force={force})")
            speaking["on"] = False
            agent_busy["v"] = False
            task.cancel()
            await _flush_client_audio()
            return True

        # ---- Deepgram STT: stream caller audio, receive transcripts ---------
        async def stt_loop():
            params = {
                "model": stt_model,
                "language": stt_language,
                "encoding": "linear16",
                "sample_rate": self.input_sample_rate,
                "channels": 1,
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "true",
                "endpointing": stt_endpointing_ms,
                "utterance_end_ms": stt_utterance_end_ms,
                "vad_events": "true",
                "filler_words": "false",
            }
            url = f"{DEEPGRAM_WS_URL}?{urlencode(params)}"
            headers = {"Authorization": f"Token {deepgram_key}"}
            bytes_per_sec = 2 * self.input_sample_rate

            failures = 0
            while True:
                stt_secs_pending = 0.0
                pending_final = []   # is_final segments awaiting UtteranceEnd
                try:
                    async with websockets.connect(url, additional_headers=headers,
                                                  max_size=None) as ws:
                        logger.info(f"Deepgram STT connected (model={stt_model}, "
                                    f"lang={stt_language}, endpointing={stt_endpointing_ms}ms)")
                        failures = 0

                        async def pump_audio():
                            nonlocal stt_secs_pending
                            while True:
                                try:
                                    chunk = await asyncio.wait_for(
                                        audio_input_queue.get(), timeout=5.0)
                                except asyncio.TimeoutError:
                                    # keep the socket warm during silence
                                    await ws.send(json.dumps({"type": "KeepAlive"}))
                                    continue
                                if not chunk:
                                    continue
                                await ws.send(chunk)
                                stt_secs_pending += len(chunk) / bytes_per_sec
                                if stt_secs_pending >= 2.0:
                                    await emit({"type": "usage",
                                                "stt_seconds": round(stt_secs_pending, 3)})
                                    stt_secs_pending = 0.0

                        pump = asyncio.create_task(pump_audio())
                        try:
                            async for raw in ws:
                                if isinstance(raw, bytes):
                                    continue
                                msg = json.loads(raw)
                                mtype = msg.get("type")

                                if mtype == "SpeechStarted":
                                    # Ignore VAD during agent speech — soft-mute caused
                                    # mid-sentence silence when STT was echo/noise.
                                    pass
                                elif mtype == "Results":
                                    alt = (((msg.get("channel") or {}).get("alternatives")
                                            or [{}])[0])
                                    text = (alt.get("transcript") or "").strip()
                                    if not text:
                                        continue
                                    if not msg.get("is_final"):
                                        # Fast barge-in: stop agent audio as soon
                                        # as a real multi-word interim shows up,
                                        # instead of waiting for speech_final.
                                        if speaking["on"] and _interim_allows_barge_in(
                                                text, recent_agent["text"]):
                                            await interrupt_agent(
                                                "interim", force=True)
                                        continue
                                    pending_final.append({
                                        "text": text,
                                        "confidence": alt.get("confidence"),
                                        "words": alt.get("words") or [],
                                    })
                                    if msg.get("speech_final"):
                                        await _finish_utterance(
                                            pending_final, try_barge_in=True)
                                        pending_final = []
                                elif mtype == "UtteranceEnd":
                                    await _finish_utterance(
                                        pending_final, try_barge_in=True)
                                    pending_final = []
                        finally:
                            pump.cancel()
                        # graceful re-loop (idle disconnect) — reconnect
                except asyncio.CancelledError:
                    if stt_secs_pending > 0:
                        await emit({"type": "usage",
                                    "stt_seconds": round(stt_secs_pending, 3)})
                    raise
                except Exception as e:
                    failures += 1
                    logger.warning(f"Deepgram STT error ({failures}): {e}")
                    if failures >= 5:
                        await emit({"type": "error", "error": f"STT failed: {e}"})
                        return
                    await asyncio.sleep(min(2 ** failures, 8))

        async def _finish_utterance(segments, *, try_barge_in=False):
            text = " ".join(
                (s.get("text") if isinstance(s, dict) else s)
                for s in segments if s
            ).strip()
            if not text:
                return False

            all_words = []
            confidences = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                all_words.extend(seg.get("words") or [])
                if seg.get("confidence") is not None:
                    confidences.append(float(seg["confidence"]))
            alt_confidence = (
                float(statistics.median(confidences)) if confidences else None
            )

            score, reason = _utterance_confidence(text, alt_confidence, all_words)

            if reason != "ok":
                logger.info(f"STT rejected ({reason}): text={text!r}")
                return False

            if not _allows_turn(score, _turn_confidence_floor(text, stt_min_confidence)):
                logger.info(
                    f"STT rejected (low_confidence): score={score} text={text!r}")
                return False

            agent_active = speaking["on"] or agent_busy["v"]
            if agent_active and not _allows_barge_in(
                text, recent_agent=recent_agent["text"],
            ):
                logger.info(f"Barge-in ignored (echo/ack): text={text!r}")
                return False

            # Drop duplicate finals (speech_final + UtteranceEnd of same text).
            norm = re.sub(r"\s+", " ", text.strip().lower())
            now = time.monotonic()
            if norm == last_utt["text"] and (now - last_utt["t"]) < 1.2:
                logger.info(f"STT deduped: {text!r}")
                return False
            last_utt["text"] = norm
            last_utt["t"] = now

            prev_lang = customer_lang["v"]
            is_first = customer_turns["n"] == 0
            customer_lang["v"] = _update_customer_language(
                text, customer_lang["v"], en_streak, is_first_response=is_first)
            customer_turns["n"] += 1
            if customer_lang["v"] != prev_lang:
                customer_lang["hint_sent"] = False
                logger.info(f"Customer language updated: {customer_lang['v']}")

            if try_barge_in and agent_busy["v"]:
                await interrupt_agent("speech_final", force=True)

            await emit({"type": "user", "text": text})
            logger.info(f"User turn accepted: {text!r}")
            await emit({"type": "turn_complete"})
            await turn_queue.put(text)
            return True

        # ---- text inputs (browser text box / Twilio call trigger) -----------
        async def text_loop():
            while True:
                text = await text_input_queue.get()
                if text and text.strip():
                    await turn_queue.put(text.strip())

        # ---- video inputs: not supported by this pipeline, drain ------------
        async def video_drain_loop():
            while True:
                await video_input_queue.get()

        # ---- LLM (OpenAI-compatible streaming) ------------------------------
        # Streams token deltas. As soon as a complete sentence/clause of spoken
        # text is available it is pushed to `on_sentence` so TTS can start while
        # the model is still generating.
        async def chat_completion_stream(msgs, on_sentence=None):
            payload = {
                "model": model,
                "messages": msgs,
                "temperature": 0.4,
                "max_tokens": 600,
                "tools": self.tools,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if use_openrouter:
                # Fastest measured route: Groq TTFT beat Cerebras for this prompt.
                # Cerebras stays as high-tok/s fallback if Groq is rate-limited.
                payload["provider"] = {
                    "order": ["Groq", "Cerebras"],
                    "allow_fallbacks": True,
                }
                # gpt-oss is a reasoning model: cap thinking so it can't burn
                # the whole token budget reasoning and return empty speech.
                # low = fastest but degenerates (looped tokens spoken aloud);
                # medium keeps answers coherent. Env-tunable.
                payload["reasoning"] = {
                    "effort": os.getenv("LLM_REASONING_EFFORT", "medium")}

            headers = {
                "Authorization": f"Bearer {llm_key}",
                "Content-Type": "application/json",
            }
            if use_openrouter:
                headers["HTTP-Referer"] = os.getenv(
                    "PUBLIC_URL", "https://voice-v1.onrender.com")
                headers["X-Title"] = "Kataria Voice Agent"

            content_parts = []
            tool_calls = {}   # index -> {id, name, arguments}
            pending = ""      # spoken text not yet flushed as a sentence
            finish_reason = None
            emitted_any = False  # True after first TTS chunk was handed off

            async def flush_sentences(force=False):
                nonlocal pending, emitted_any
                if not on_sentence:
                    return
                while True:
                    chunks = _split_sentences(pending)
                    if force:
                        for c in chunks:
                            await on_sentence(c)
                            emitted_any = True
                        pending = ""
                        return
                    if len(chunks) > 1:
                        first = chunks[0]
                        await on_sentence(first)
                        emitted_any = True
                        idx = pending.find(first)
                        pending = (
                            pending[idx + len(first):].lstrip(" ,।")
                            if idx >= 0 else ""
                        )
                        continue
                    return
            for attempt in range(4):
                try:
                    async with http.stream("POST", llm_chat_url, headers=headers,
                                           json=payload) as resp:
                        if resp.status_code == 429:
                            await resp.aread()
                            retry_after = 2.0
                            try:
                                retry_after = float(resp.headers.get("retry-after") or 0) or 2.0
                            except ValueError:
                                pass
                            wait = min(retry_after + 0.3, 8.0)
                            logger.warning(f"LLM rate limit, waiting {wait:.1f}s "
                                           f"(attempt {attempt + 1}/4)")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()

                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            usage = obj.get("usage")
                            if usage:
                                await emit({"type": "usage",
                                            "llm_in": int(usage.get("prompt_tokens") or 0),
                                            "llm_out": int(usage.get("completion_tokens") or 0)})
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            ch = choices[0]
                            delta = ch.get("delta") or {}
                            if ch.get("finish_reason"):
                                finish_reason = ch["finish_reason"]
                            piece = delta.get("content")
                            if piece:
                                content_parts.append(piece)
                                pending += piece
                                await flush_sentences(force=False)
                            for tcd in (delta.get("tool_calls") or []):
                                i = tcd.get("index", 0)
                                slot = tool_calls.setdefault(
                                    i, {"id": None, "name": "", "arguments": ""})
                                if tcd.get("id"):
                                    slot["id"] = tcd["id"]
                                fn = tcd.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["arguments"] += fn["arguments"]
                        await flush_sentences(force=True)
                        break
                except httpx.HTTPStatusError:
                    raise
            else:
                raise RuntimeError("LLM streaming failed after retries")

            content = "".join(content_parts)
            if not content.strip() and not tool_calls:
                logger.warning(
                    f"LLM returned no speech (finish_reason={finish_reason})")
            tcs = [
                {"id": v["id"] or v["name"], "type": "function",
                 "function": {"name": v["name"], "arguments": v["arguments"]}}
                for _, v in sorted(tool_calls.items())
            ] if tool_calls else []
            return {"content": content, "tool_calls": tcs,
                    "finish_reason": finish_reason}

        async def run_tool(name, args):
            func = self.tool_mapping.get(name)
            if not func:
                return f"Error: unknown tool {name}"
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(**args)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: func(**args))
            except Exception as e:
                return f"Error: {e}"

        # ---- TTS helpers (session-scoped reusable WebSocket) ----------------
        tts = {"ws": None, "lang": None}
        vehicle_prefetch = {"data": None, "done": False}

        async def _close_tts():
            ws = tts["ws"]
            tts["ws"] = None
            tts["lang"] = None
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

        def _tts_alive():
            ws = tts["ws"]
            if ws is None:
                return False
            try:
                # websockets v12+: State.OPEN; older: .closed flag
                state = getattr(ws, "state", None)
                if state is not None:
                    return getattr(state, "name", str(state)) == "OPEN"
                return not getattr(ws, "closed", False)
            except Exception:
                return False

        async def _ensure_tts(lang):
            """Reuse one Sarvam WS for the whole call; reconfigure on lang change."""
            if not _tts_alive():
                tts["ws"] = None
                tts["lang"] = None
            if tts["ws"] is None:
                url = (f"{SARVAM_TTS_WS_URL}?model={TTS_MODEL}"
                       f"&send_completion_event=true")
                headers = {"Api-Subscription-Key": sarvam_key}
                tts["ws"] = await websockets.connect(
                    url, additional_headers=headers, max_size=None,
                    open_timeout=20, close_timeout=5)
                tts["lang"] = None
            if tts["lang"] != lang:
                await tts["ws"].send(json.dumps({
                    "type": "config",
                    "data": {
                        "language_code": lang,
                        "target_language_code": lang,
                        "speaker": TTS_SPEAKER,
                        "model": TTS_MODEL,
                        "speech_sample_rate": str(TTS_SAMPLE_RATE),
                        "output_audio_codec": "linear16",
                        "min_buffer_size": 30,
                        "max_chunk_length": 80,
                    },
                }))
                tts["lang"] = lang
            return tts["ws"]

        async def _emit_audio(pcm):
            # Mark speaking only when PCM actually leaves — never while waiting
            # on the TTS lock (that caused false barge-ins canceling the greeting).
            if not speaking["on"]:
                speaking["on"] = True
                speaking_since["t"] = time.monotonic()
            # Send in ~60ms chunks so barge-in cancellation is responsive.
            step = int(TTS_SAMPLE_RATE * 2 * 0.06)
            for i in range(0, len(pcm), step):
                if inspect.iscoroutinefunction(audio_output_callback):
                    await audio_output_callback(pcm[i:i + step])
                else:
                    audio_output_callback(pcm[i:i + step])
                await asyncio.sleep(0)  # yield so a cancel can land between chunks

        async def speak_stream(sentence_queue, lang):
            """Stream text into Sarvam TTS WebSocket; play PCM as it arrives."""
            chars_sent = 0
            sender_done = asyncio.Event()
            sender_done_at = {"t": None}
            last_audio_at = {"t": 0.0}
            got_audio = {"v": False}
            clean_final = {"v": False}
            flush_timeout = float(os.getenv("TTS_FLUSH_TIMEOUT_S", "15"))
            idle_after_flush = float(os.getenv("TTS_IDLE_AFTER_FLUSH_S", "2.5"))

            try:
                async with tts_lock:
                    ws = await _ensure_tts(lang)

                    async def sender():
                        nonlocal chars_sent
                        while True:
                            sentence = await sentence_queue.get()
                            if sentence is None:
                                await ws.send(json.dumps({"type": "flush"}))
                                sender_done.set()
                                sender_done_at["t"] = time.monotonic()
                                return
                            if not sentence.strip():
                                continue
                            sentence = _tts_pronounce(sentence, lang)
                            chars_sent += len(sentence)
                            await ws.send(json.dumps({
                                "type": "text",
                                "data": {"text": sentence},
                            }))

                    send_task = asyncio.create_task(sender())
                    try:
                        while True:
                            if sender_done.is_set() and sender_done_at["t"]:
                                elapsed = time.monotonic() - sender_done_at["t"]
                                if got_audio["v"]:
                                    if (time.monotonic() - last_audio_at["t"]
                                            >= idle_after_flush):
                                        break
                                elif elapsed >= 1.5:
                                    logger.warning(
                                        "Sarvam TTS: no audio after flush")
                                    break
                                if elapsed >= flush_timeout:
                                    logger.warning(
                                        "Sarvam TTS flush timeout")
                                    break
                            try:
                                wait = (idle_after_flush if sender_done.is_set()
                                        else flush_timeout)
                                raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                            except asyncio.TimeoutError:
                                if sender_done.is_set():
                                    break
                                logger.warning("Sarvam TTS waiting for first audio")
                                continue
                            if isinstance(raw, bytes):
                                continue
                            msg = json.loads(raw)
                            mtype = msg.get("type")
                            if mtype == "audio":
                                audio_b64 = (msg.get("data") or {}).get("audio")
                                if not audio_b64:
                                    continue
                                pcm = _strip_wav_header(
                                    base64.b64decode(audio_b64))
                                if pcm:
                                    got_audio["v"] = True
                                    last_audio_at["t"] = time.monotonic()
                                    await _emit_audio(pcm)
                            elif mtype == "event":
                                et = (msg.get("data") or {}).get("event_type")
                                if et == "final" and sender_done.is_set():
                                    clean_final["v"] = True
                                    break
                            elif mtype == "error":
                                err = (msg.get("data") or {}).get("message") or str(msg)
                                raise RuntimeError(f"Sarvam TTS WS error: {err}")
                    finally:
                        if not send_task.done():
                            send_task.cancel()
                            try:
                                await send_task
                            except (asyncio.CancelledError, Exception):
                                pass
                    # If we exited on a timeout (not a clean flush-final), the
                    # socket may still hold undelivered frames from THIS turn.
                    # Reusing it would poison the NEXT turn with stale audio
                    # and stray `final` events (heard as cut-offs / wrong audio).
                    if not clean_final["v"]:
                        logger.info("Sarvam TTS: unclean stream end — closing WS")
                        await _close_tts()
            except asyncio.CancelledError:
                # Close dirty mid-stream socket, rewarm in background during LLM time.
                speaking["on"] = False
                keep_lang = tts.get("lang")
                await _close_tts()
                if keep_lang:
                    async def _rewarm():
                        try:
                            await _ensure_tts(keep_lang)
                        except Exception:
                            pass
                    asyncio.create_task(_rewarm())
                raise
            except Exception:
                await _close_tts()
                raise
            finally:
                if chars_sent:
                    await emit({"type": "usage", "tts_chars": chars_sent})
                logger.info(
                    f"TTS done: {chars_sent} chars, audio_played={got_audio['v']}")

        async def speak_text(text, lang=None):
            """Speak a fully-known string via streaming TTS."""
            if not text or not text.strip():
                return
            lang = lang or _tts_language_for(text)
            text = _sanitize_speech(text)
            if not text:
                return
            sentence_queue = asyncio.Queue()
            speak_task = asyncio.create_task(speak_stream(sentence_queue, lang))
            for chunk in _split_sentences(text):
                await sentence_queue.put(chunk)
            await sentence_queue.put(None)
            try:
                await speak_task
            finally:
                speaking["on"] = False

        # Ack filler removed — it caused users to only hear "हाँ" while real
        # replies were barge-in-cancelled by TTS echo into the mic.

        # ---- connection prewarm --------------------------------------------
        async def prewarm_connections():
            # Prefetch vehicle so greeting doesn't wait on the tool.
            try:
                vehicle_prefetch["data"] = await run_tool("get_vehicle_info", {})
                vehicle_prefetch["done"] = True
            except Exception as e:
                logger.warning(f"Vehicle prefetch failed: {e}")

            # Warm LLM TLS + prompt-prefix cache (short system prompt).
            try:
                headers = {
                    "Authorization": f"Bearer {llm_key}",
                    "Content-Type": "application/json",
                }
                if use_openrouter:
                    headers["HTTP-Referer"] = os.getenv(
                        "PUBLIC_URL", "https://voice-v1.onrender.com")
                    headers["X-Title"] = "Kataria Voice Agent"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": get_system_instruction()},
                        {"role": "user", "content": "."},
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                if use_openrouter:
                    payload["provider"] = {
                        "order": ["Groq", "Cerebras"],
                        "allow_fallbacks": True,
                    }
                resp = await http.post(llm_chat_url, headers=headers, json=payload)
                logger.info(f"LLM prewarm status={resp.status_code}")
            except Exception as e:
                logger.warning(f"LLM prewarm failed: {e}")

            # TTS is opened on first speak (greeting) — do NOT prewarm here;
            # racing _ensure_tts() with greeting corrupted the Sarvam WS.

        # ---- greeting shortcut (skip 2 LLM roundtrips on call start) --------
        async def run_greeting():
            # No barge-in during greeting — browser mic / TTS lock wait used to
            # cancel this before any audio played (logs: "interrupted by caller").
            barge_in_ok["v"] = False
            speaking["on"] = False
            agent_busy["v"] = True
            try:
                if vehicle_prefetch["done"] and vehicle_prefetch["data"] is not None:
                    vehicle = vehicle_prefetch["data"]
                else:
                    vehicle = await run_tool("get_vehicle_info", {})
                await emit({
                    "type": "tool_call",
                    "name": "get_vehicle_info",
                    "args": {},
                    "result": vehicle,
                })

                owner = ""
                if isinstance(vehicle, dict):
                    owner = _first_name(vehicle.get("owner_name") or "")
                else:
                    owner = "ग्राहक"

                greeting = (
                    f"Namaste! Main Rahul bol raha hoon, Kataria Automobiles se. "
                    f"Kya main {owner} ji se baat kar sakta hoon?"
                )
                record_note = (
                    "Yeh call training aur quality ke liye record ho rahi hai."
                )
                full = f"{greeting} {record_note}"

                # Seed chat history so later turns have full vehicle context.
                tool_call_id = "call_greeting_get_vehicle_info"
                messages.append({"role": "user", "content": "[call connected]"})
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "get_vehicle_info",
                            "arguments": "{}",
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(vehicle, ensure_ascii=False, default=str),
                })
                messages.append({"role": "assistant", "content": full})

                await emit({"type": "agent", "text": full})
                recent_agent["text"] = full[-500:]
                logger.info("Greeting: starting TTS")
                try:
                    await speak_text(full, lang="hi-IN")
                    logger.info("Greeting: TTS complete")
                except Exception as e:
                    logger.error(f"Greeting TTS failed: {e}")
                    await emit({"type": "error", "error": f"TTS failed: {e}"})
                greeted["done"] = True
                barge_in_ok["v"] = True
                await emit({"type": "turn_complete"})
            except asyncio.CancelledError:
                speaking["on"] = False
                # Allow a retry — do not leave greeted stuck True with no audio.
                raise
            except Exception:
                speaking["on"] = False
                raise
            finally:
                agent_busy["v"] = False

        # ---- one agent turn: LLM (+tools) then TTS, all streamed ------------
        async def run_turn(user_text):
            barge_in_ok["v"] = True
            agent_busy["v"] = True
            reply_guard_until["t"] = time.monotonic() + REPLY_GUARD_S
            try:
                if not customer_lang["hint_sent"]:
                    messages.append({
                        "role": "system",
                        "content": (
                            f"{_language_hint(customer_lang['v'])} "
                            "ONE call-flow step this turn only (max 2 short sentences), "
                            "then wait for the customer."
                        ),
                    })
                    customer_lang["hint_sent"] = True

                user_content = user_text
                messages.append({"role": "user", "content": user_content})
                turn_tts_lang = customer_lang["v"]
                nudged = {"v": False}
                for _ in range(5):  # tool-call loop guard
                    sentence_queue = asyncio.Queue()
                    agent_text_parts = []
                    started_speaking = {"v": False}
                    speak_task = None
                    speak_lang = {"v": turn_tts_lang}

                    async def on_sentence(s):
                        nonlocal speak_task
                        s = _sanitize_speech(s)
                        if not s:
                            return
                        if not started_speaking["v"]:
                            started_speaking["v"] = True
                            speak_lang["v"] = turn_tts_lang or DEFAULT_CALL_LANGUAGE
                            speak_task = asyncio.create_task(
                                speak_stream(sentence_queue, speak_lang["v"]))
                        agent_text_parts.append(s)
                        recent_agent["text"] = (
                            f"{recent_agent['text']} {s}".strip()[-500:]
                        )
                        await emit({"type": "agent", "text": s})
                        await sentence_queue.put(s)

                    try:
                        reply = await chat_completion_stream(messages, on_sentence)
                    finally:
                        if started_speaking["v"]:
                            await sentence_queue.put(None)
                            if speak_task:
                                try:
                                    await speak_task
                                finally:
                                    speaking["on"] = False
                        else:
                            speaking["on"] = False

                    tool_calls = reply.get("tool_calls") or []
                    if tool_calls:
                        messages.append({
                            "role": "assistant",
                            "content": reply.get("content") or "",
                            "tool_calls": tool_calls,
                        })
                        for tc in tool_calls:
                            fn = (tc.get("function") or {})
                            name = fn.get("name") or ""
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except json.JSONDecodeError:
                                args = {}
                            result = await run_tool(name, args)
                            await emit({"type": "tool_call", "name": name,
                                        "args": args, "result": result})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id") or name,
                                "content": json.dumps(result, ensure_ascii=False, default=str),
                            })
                        continue

                    content = _sanitize_speech("".join(agent_text_parts).strip())
                    if content:
                        messages.append({"role": "assistant", "content": content})
                    elif not nudged["v"]:
                        # Reasoning model burned its budget thinking, or chose
                        # silence — retry ONCE with an explicit speak nudge so
                        # the caller never gets dead air.
                        nudged["v"] = True
                        logger.warning(
                            "Agent turn produced no speech — retrying with nudge")
                        messages.append({
                            "role": "system",
                            "content": (
                                "You MUST reply with spoken text now. Answer the "
                                "customer's last message directly in 1-2 short "
                                "sentences. Do not stay silent."
                            ),
                        })
                        continue
                    else:
                        logger.warning("Agent turn still empty after nudge")
                    await emit({"type": "turn_complete"})
                    return
            finally:
                agent_busy["v"] = False

        async def agent_loop():
            while True:
                user_text = await turn_queue.get()
                # Debounce: if the caller keeps talking, merge the continuation
                # fragments into ONE turn instead of answering each half-sentence.
                while True:
                    try:
                        more = await asyncio.wait_for(
                            turn_queue.get(), timeout=turn_debounce)
                        user_text = f"{user_text} {more}".strip()
                    except asyncio.TimeoutError:
                        break

                # First turn: always run greeting until it succeeds.
                # greeted["done"] is set inside run_greeting only on success.
                if not greeted["done"]:
                    if user_text and not CALL_START_RE.search(user_text):
                        # Keep the real user utterance for after the greeting.
                        await turn_queue.put(user_text)
                    current_turn["task"] = asyncio.create_task(run_greeting())
                elif CALL_START_RE.search(user_text):
                    continue  # ignore duplicate call-start triggers
                else:
                    current_turn["task"] = asyncio.create_task(run_turn(user_text))

                try:
                    await current_turn["task"]
                except asyncio.CancelledError:
                    logger.info("Agent turn interrupted by caller")
                    speaking["on"] = False
                except Exception as e:
                    logger.error(f"Agent turn failed: {type(e).__name__}: {e}\n"
                                 f"{traceback.format_exc()}")
                    await emit({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    current_turn["task"] = None
                    speaking["on"] = False

        tasks = [
            asyncio.create_task(stt_loop()),
            asyncio.create_task(text_loop()),
            asyncio.create_task(video_drain_loop()),
            asyncio.create_task(agent_loop()),
            asyncio.create_task(prewarm_connections()),
        ]

        logger.info(
            f"Voice agent started (stt=deepgram/{stt_model}, "
            f"llm={llm_chat_url.split('//')[-1].split('/')[0]}/{model}, "
            f"tts=sarvam-ws/{TTS_MODEL}/{TTS_SPEAKER}, "
            f"endpointing={stt_endpointing_ms}ms, debounce={turn_debounce}s)"
        )
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
                if event.get("type") == "error":
                    break
        finally:
            for task in tasks:
                task.cancel()
            inflight = current_turn["task"]
            if inflight and not inflight.done():
                inflight.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await _close_tts()
            await http.aclose()
            logger.info("Voice agent session closed")
