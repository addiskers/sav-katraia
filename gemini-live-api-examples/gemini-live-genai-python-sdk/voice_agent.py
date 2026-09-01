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
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
import websockets

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
STT_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi")

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
    # Keep this short — TTFT scales with prompt size (~250ms saved vs the old
    # 3k-char prompt on Groq in our R&D bench).
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    return (
        f"You are Rahul (ONLY Rahul) at Kataria Automobiles (spell Kataria, never Katrina), "
        f"Maruti Suzuki dealership, Ahmedabad. LIVE phone call — words are spoken aloud.\n"
        f"OUTPUT: plain speech only (no markdown/bullets/emoji). Native script "
        f"(Hindi/Marathi Devanagari, Gujarati script, English Latin). Never romanize "
        f"Indian languages. 1–2 short sentences per turn; one step then wait. Never "
        f"narrate reasoning; if on hold stay silent.\n"
        f"DATES: Today {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}); "
        f"kal/tomorrow={tomorrow.strftime('%Y-%m-%d')}; "
        f"parso/day-after={day_after.strftime('%Y-%m-%d')}. Pickup dates are near-future "
        f"(today+0–14d only). Never use warranty/purchase/service-history dates.\n"
        f"LANGUAGE: open Hindi; after customer's first reply switch fully to their "
        f"language (en/gu/mr/hi) and stay. Never mix.\n"
        f"FLOW (wait after each): (1) name vehicle+service due (2) offer free pickup/drop "
        f"(3) confirm address (4) ask day/time (5) MUST call schedule_pickup when all "
        f"confirmed (6) share booking ID/driver (7) warm close. Call get_vehicle_info at "
        f"start (never invent data); get_service_cost_estimate on price questions. If wrong "
        f"car: discard data, apologize, ask name/Maruti status."
    )

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


def _tts_language_for(text: str) -> str:
    """Pick the Bulbul language code from the script of the reply text."""
    gujarati = devanagari = latin = 0
    for ch in text:
        o = ord(ch)
        if 0x0A80 <= o <= 0x0AFF:
            gujarati += 1
        elif 0x0900 <= o <= 0x097F:
            devanagari += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    if gujarati > devanagari and gujarati > latin:
        return "gu-IN"
    if devanagari >= latin and devanagari > 0:
        return "hi-IN"
    if latin > 0:
        return "en-IN"
    return "hi-IN"


def _split_sentences(text: str, max_len: int = 140):
    """Split reply into TTS-friendly chunks (< ~500 chars for Sarvam WS)."""
    pieces = re.split(r"(?<=[।.!?])\s+|\n+", text)
    parts = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(p) > 60:
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
        stt_endpointing_ms = os.getenv("DEEPGRAM_ENDPOINTING_MS", "250")
        stt_utterance_end_ms = os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1000")
        turn_debounce = _env_float("TURN_DEBOUNCE", 0.15)
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
        current_turn = {"task": None}  # in-flight agent turn task
        greeted = {"done": False}
        tts_lock = asyncio.Lock()      # one Sarvam WS speaker at a time


        async def emit(event):
            await event_queue.put(event)

        # ---- barge-in -------------------------------------------------------
        async def interrupt_agent():
            task = current_turn["task"]
            if task and not task.done() and speaking["on"]:
                task.cancel()
                if audio_interrupt_callback:
                    if inspect.iscoroutinefunction(audio_interrupt_callback):
                        await audio_interrupt_callback()
                    else:
                        audio_interrupt_callback()
                await emit({"type": "interrupted"})

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
                                    # Fastest barge-in signal (~100-300ms): the
                                    # caller began speaking. Stop the agent NOW.
                                    if speaking["on"]:
                                        await interrupt_agent()
                                elif mtype == "Results":
                                    alt = (((msg.get("channel") or {}).get("alternatives")
                                            or [{}])[0])
                                    text = (alt.get("transcript") or "").strip()
                                    if not text:
                                        continue
                                    # Backup barge-in in case SpeechStarted was missed.
                                    if speaking["on"]:
                                        await interrupt_agent()
                                    if msg.get("is_final"):
                                        pending_final.append(text)
                                        if msg.get("speech_final"):
                                            await _finish_utterance(pending_final)
                                            pending_final = []
                                elif mtype == "UtteranceEnd":
                                    await _finish_utterance(pending_final)
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

        async def _finish_utterance(segments):
            text = " ".join(s for s in segments if s).strip()
            if not text:
                return
            await emit({"type": "user", "text": text})
            await emit({"type": "turn_complete"})
            # Stale in-flight (not yet speaking) reply -> cancel; merged turn re-runs.
            task = current_turn["task"]
            if task and not task.done() and not speaking["on"]:
                task.cancel()
            await turn_queue.put(text)

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
                "max_tokens": 220,
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
                    # Early first-audio: don't wait for a full sentence — flush
                    # ~25 chars at a word/clause break so TTS starts while LLM
                    # is still generating (R&D: ~200-300ms earlier first audio).
                    if not emitted_any and len(pending) >= 25:
                        break_at = -1
                        for sep in ("।", "!", "?", ".", ",", " "):
                            pos = pending.rfind(sep)
                            if pos > break_at:
                                break_at = pos
                        if break_at >= 15:
                            if pending[break_at] in "।!?,.":
                                first = pending[: break_at + 1].strip()
                            else:
                                first = pending[:break_at].strip()
                            if first:
                                await on_sentence(first)
                                emitted_any = True
                                pending = pending[len(first):].lstrip(" ,।")
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
        # Pre-synthesized PCM clips (filled during prewarm). Keys are stable
        # phrase ids — used for ack-filler and zero-latency stock lines.
        pcm_cache = {}
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
                    url, additional_headers=headers, max_size=None)
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

        async def _synth_to_pcm(text, lang="hi-IN"):
            """One-shot synthesize via the shared WS; return raw PCM bytes."""
            async with tts_lock:
                ws = await _ensure_tts(lang)
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send(json.dumps({"type": "flush"}))
                parts = []
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "audio":
                        audio_b64 = (msg.get("data") or {}).get("audio")
                        if audio_b64:
                            parts.append(_strip_wav_header(base64.b64decode(audio_b64)))
                    elif mtype == "event":
                        if (msg.get("data") or {}).get("event_type") == "final":
                            break
                    elif mtype == "error":
                        err = (msg.get("data") or {}).get("message") or str(msg)
                        raise RuntimeError(f"Sarvam TTS WS error: {err}")
                pcm = b"".join(parts)
                if pcm:
                    await emit({"type": "usage", "tts_chars": len(text)})
                return pcm

        async def _emit_audio(pcm):
            # Send in ~60ms chunks so barge-in cancellation is responsive.
            step = int(TTS_SAMPLE_RATE * 2 * 0.06)
            for i in range(0, len(pcm), step):
                if inspect.iscoroutinefunction(audio_output_callback):
                    await audio_output_callback(pcm[i:i + step])
                else:
                    audio_output_callback(pcm[i:i + step])
                await asyncio.sleep(0)  # yield so a cancel can land between chunks

        async def speak_stream(sentence_queue, lang):
            """Stream text into Sarvam TTS WebSocket; play PCM as it arrives.

            Reuses the session WebSocket (R&D: ~40-50ms faster than reconnect).
            """
            chars_sent = 0
            try:
                async with tts_lock:
                    ws = await _ensure_tts(lang)

                    async def sender():
                        nonlocal chars_sent
                        while True:
                            sentence = await sentence_queue.get()
                            if sentence is None:
                                await ws.send(json.dumps({"type": "flush"}))
                                return
                            if not sentence.strip():
                                continue
                            chars_sent += len(sentence)
                            await ws.send(json.dumps({
                                "type": "text",
                                "data": {"text": sentence},
                            }))

                    send_task = asyncio.create_task(sender())
                    try:
                        async for raw in ws:
                            if isinstance(raw, bytes):
                                continue
                            msg = json.loads(raw)
                            mtype = msg.get("type")
                            if mtype == "audio":
                                audio_b64 = (msg.get("data") or {}).get("audio")
                                if not audio_b64:
                                    continue
                                pcm = _strip_wav_header(base64.b64decode(audio_b64))
                                if pcm:
                                    await _emit_audio(pcm)
                            elif mtype == "event":
                                et = (msg.get("data") or {}).get("event_type")
                                if et == "final":
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
            except asyncio.CancelledError:
                # Barge-in: drop the socket so the next turn starts clean.
                await _close_tts()
                raise
            except Exception:
                await _close_tts()
                raise
            finally:
                if chars_sent:
                    await emit({"type": "usage", "tts_chars": chars_sent})

        async def speak_text(text, lang=None):
            """Speak a fully-known string via streaming TTS."""
            if not text or not text.strip():
                return
            lang = lang or _tts_language_for(text)
            sentence_queue = asyncio.Queue()
            speak_task = asyncio.create_task(speak_stream(sentence_queue, lang))
            for chunk in _split_sentences(text):
                await sentence_queue.put(chunk)
            await sentence_queue.put(None)
            await speak_task

        async def play_ack_filler(started_speaking):
            """If LLM is slow (>200ms), play a cached 'हाँ' to mask dead air.

            Cancelled as soon as the real reply starts so audio never overlaps.
            """
            try:
                await asyncio.sleep(0.20)
            except asyncio.CancelledError:
                return
            if started_speaking["v"]:
                return
            pcm = pcm_cache.get("haan")
            if not pcm:
                return
            speaking["on"] = True
            try:
                await _emit_audio(pcm)
            except asyncio.CancelledError:
                if audio_interrupt_callback:
                    if inspect.iscoroutinefunction(audio_interrupt_callback):
                        await audio_interrupt_callback()
                    else:
                        audio_interrupt_callback()
                raise
            finally:
                if not started_speaking["v"]:
                    speaking["on"] = False

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

            # Open Sarvam TTS WS + cache a short ack filler ("हाँ") for dead-air mask.
            try:
                await _ensure_tts("hi-IN")
                haan = await _synth_to_pcm("हाँ।", lang="hi-IN")
                if haan:
                    pcm_cache["haan"] = haan
                    logger.info(f"Ack filler cached ({len(haan)} bytes PCM)")
                logger.info("Sarvam TTS WebSocket prewarmed (reused for call)")
            except Exception as e:
                logger.warning(f"Sarvam TTS prewarm failed: {e}")
                await _close_tts()

        # ---- greeting shortcut (skip 2 LLM roundtrips on call start) -------
        async def run_greeting():
            # IMPORTANT: do NOT set speaking["on"] until TTS actually starts.
            # Browser mic is already open; early speaking=True lets Deepgram
            # SpeechStarted cancel the greeting before any audio plays.
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
                    f"नमस्ते! मैं राहुल बोल रहा हूँ, Kataria Automobiles से. "
                    f"क्या मैं {owner} जी से बात कर सकता हूँ?"
                )
                record_note = "यह कॉल training और quality के लिए record हो रही है."
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
                speaking["on"] = True
                try:
                    await speak_text(full, lang="hi-IN")
                finally:
                    speaking["on"] = False
                await emit({"type": "turn_complete"})
            except asyncio.CancelledError:
                speaking["on"] = False
                raise
            except Exception:
                speaking["on"] = False
                raise

        # ---- one agent turn: LLM (+tools) then TTS, all streamed ------------
        async def run_turn(user_text):
            messages.append({"role": "user", "content": user_text})
            for hop in range(5):  # tool-call loop guard
                sentence_queue = asyncio.Queue()
                agent_text_parts = []
                started_speaking = {"v": False}
                speak_task = None
                speak_lang = {"v": None}
                # Ack filler only on the first hop of a user turn (masks LLM TTFT).
                # Skipped on tool follow-up hops so we don't say "हाँ" mid-tool.
                filler_task = (
                    asyncio.create_task(play_ack_filler(started_speaking))
                    if hop == 0 else None
                )

                async def stop_filler():
                    if not filler_task or filler_task.done():
                        return
                    filler_task.cancel()
                    try:
                        await filler_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    if audio_interrupt_callback and speaking["on"]:
                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                            await audio_interrupt_callback()
                        else:
                            audio_interrupt_callback()
                        speaking["on"] = False

                async def on_sentence(s):
                    nonlocal speak_task
                    await stop_filler()
                    if not started_speaking["v"]:
                        started_speaking["v"] = True
                        speaking["on"] = True
                        speak_lang["v"] = _tts_language_for(s)
                        speak_task = asyncio.create_task(
                            speak_stream(sentence_queue, speak_lang["v"]))
                    agent_text_parts.append(s)
                    await emit({"type": "agent", "text": s})
                    await sentence_queue.put(s)

                try:
                    reply = await chat_completion_stream(messages, on_sentence)
                finally:
                    await stop_filler()
                    if started_speaking["v"]:
                        await sentence_queue.put(None)  # end the audio stream
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
                    continue  # loop again for the model's spoken answer

                content = "".join(agent_text_parts).strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
                await emit({"type": "turn_complete"})
                return

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

                # First call-start trigger: skip LLM, greet immediately.
                if (not greeted["done"]) and CALL_START_RE.search(user_text):
                    greeted["done"] = True
                    current_turn["task"] = asyncio.create_task(run_greeting())
                else:
                    if not greeted["done"]:
                        greeted["done"] = True
                    current_turn["task"] = asyncio.create_task(run_turn(user_text))

                try:
                    await current_turn["task"]
                except asyncio.CancelledError:
                    logger.info("Agent turn interrupted by caller")
                except Exception as e:
                    logger.error(f"Agent turn failed: {type(e).__name__}: {e}\n"
                                 f"{traceback.format_exc()}")
                    await emit({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    current_turn["task"] = None

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
