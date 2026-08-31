"""
Low-latency voice agent engine.

  caller audio (PCM16 16kHz) --> Deepgram nova-2 streaming STT (WebSocket)
                                       |
                                 final transcript
                                       |
                       Groq Llama-3.3-70B chat completions (tool calling)
                                       |
                                 reply text
                                       |
                  Sarvam Bulbul TTS (REST) --> PCM16 24kHz --> caller

Deepgram gives sub-200ms interim transcripts + endpointing, Groq serves Llama
at ~300 tok/s so replies start almost instantly, and Sarvam Bulbul keeps the
natural Indian "rahul" voice. Only TTS stays on Sarvam.

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
        llm_in / llm_out - Groq chat completion tokens
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
# NOTE: API keys are read at session START (see start_session), not import time,
# because main.py imports this module before it calls load_dotenv().
GROQ_CHAT_URL = os.getenv("GROQ_CHAT_URL", "https://api.groq.com/openai/v1/chat/completions")
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

# ---- Deepgram STT (env-overridable) ----------------------------------------
# nova-3 supports language=multi (Hindi/English/Gujarati code-mixing);
# nova-2 does NOT support 'multi' — use nova-3 for Indian-language calls.
STT_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
STT_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi")
# Deepgram closes a turn after `endpointing` ms of silence. Lower = snappier
# replies; the turn debounce below re-merges any over-eager splits.
STT_ENDPOINTING_MS = os.getenv("DEEPGRAM_ENDPOINTING_MS", "300")
STT_UTTERANCE_END_MS = os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1000")  # min Deepgram allows

# ---- Sarvam TTS (env-overridable) ------------------------------------------
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "rahul")
TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "24000"))

# After a transcript arrives, wait this long for a continuation fragment and
# merge it into the same turn instead of answering each fragment separately.
TURN_DEBOUNCE_SECONDS = float(os.getenv("TURN_DEBOUNCE", "0.15"))


def get_system_instruction():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)

    date_context = f"""## TODAY'S DATE — USE THIS FOR ALL SCHEDULING
- Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}).
- "Kal" / "Tomorrow" = {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).
- "Parso" / "Day after tomorrow" = {day_after.strftime('%Y-%m-%d')} ({day_after.strftime('%A')}).
- Use TODAY as the reference for ALL pickup scheduling. NEVER confuse pickup dates with warranty, purchase, or service history dates — those are COMPLETELY DIFFERENT.
- Pickup dates are ALWAYS in the near future (within 1-2 weeks from today).
"""

    return date_context + SYSTEM_INSTRUCTION


SYSTEM_INSTRUCTION = """
You are Rahul (ONLY ever "Rahul"), a warm, natural service advisor at Kataria
Automobiles (spell it Kataria, NEVER "Katrina"), an authorized Maruti Suzuki
dealership in Ahmedabad. You are on a LIVE PHONE CALL — your words are spoken aloud.

OUTPUT (spoken):
- Plain sentences only. NO markdown, asterisks, bullets, emojis.
- Write in the NATIVE SCRIPT of the language you speak: Hindi/Marathi in Devanagari
  (e.g. "नमस्ते, मैं राहुल बोल रहा हूँ"), Gujarati in Gujarati script, English in Latin.
  NEVER write Hindi/Gujarati in roman letters. English tech words (service, pickup,
  booking ID) may stay in Latin inside an Indian-language sentence.
- Keep EVERY reply to 1–2 short sentences. Say ONE step, then wait for the reply.
- NEVER speak your internal reasoning/plans. Only say what a real advisor would say
  aloud. If the customer is busy or on hold, stay SILENT (don't narrate).

FIRST STEP: At call start, IMMEDIATELY call get_vehicle_info. Never invent vehicle/
owner details — only use tool data. Then say EXACTLY:
"नमस्ते! मैं राहुल बोल रहा हूँ, Kataria Automobiles से. क्या मैं {owner_name} जी से बात कर सकता हूँ?"
(use the real owner_name, first name only) then: "यह कॉल training और quality के लिए record हो रही है."

LANGUAGE (top priority): Open in Hindi. From the customer's FIRST reply, auto-detect
their language and switch FULLY to it (English→English, Gujarati→Gujarati,
Marathi→Marathi, Hindi→Hindi) and STAY there. Also switch if they explicitly ask.
Never mix languages after switching.

CALL FLOW (one step at a time, wait after each):
1. Mention their {vehicle_model} ({vehicle_number}) has its {Nth} service due.
2. Ask to schedule it; pickup & drop are free.
3. Confirm the pickup address (use the record's address, or a new one they give —
   repeat it back).
4. Ask preferred day & time.
5. When day, time AND address are confirmed, CALL schedule_pickup (vehicle_number,
   date, time, pickup_address). A verbal "confirmed" is NOT enough — you MUST call it.
6. Share the returned booking ID and driver details.
7. Close warmly: "धन्यवाद {name} जी. आपका दिन शुभ हो!"

TOOLS (mandatory): get_vehicle_info at call start; schedule_pickup whenever they
agree to a date/time; get_service_cost_estimate when they ask about price.

IF customer says it's not their car: discard ALL prior vehicle data, apologize, ask
their name and whether they have a Maruti with you. Never reuse the old data.

PICKUP DATES are always in the near future (today +0–14 days). NEVER use warranty,
purchase, or service-history dates. "14th" means the 14th of the CURRENT month.
If unsure, ask.
"""

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
    """Split reply into small TTS chunks so the FIRST chunk plays fast.

    Sarvam TTS latency scales with text length (~0.6s for 20 chars, ~3.2s for
    160). We break on sentence ends AND commas so the first audible chunk is
    short and starts quickly; the pipeline in speak() synthesizes the rest in
    the background while the first chunk plays, leaving no gap.
    """
    # Primary split on sentence terminators, secondary on commas/clause breaks.
    pieces = re.split(r"(?<=[।.!?])\s+|\n+", text)
    parts = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        # further split long pieces on commas to shorten the first chunk
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


class VoiceAgent:
    """
    Deepgram STT -> Groq LLM -> Sarvam TTS voice agent session.

    Exposes the app's standard engine interface: same constructor shape and
    the same `start_session` async-generator interface.
    """

    def __init__(self, api_key=None, model=None, input_sample_rate=16000,
                 tools=None, tool_mapping=None):
        """
        Args:
            api_key: unused (kept for interface compatibility) — provider keys
                     come from env (DEEPGRAM_API_KEY, GROQ_API_KEY, SARVAM_API_KEY).
            model: Groq chat model (e.g. 'llama-3.3-70b-versatile').
            input_sample_rate: caller audio sample rate (PCM16 mono).
            tools: tool declarations (chat-completions format).
            tool_mapping: tool name -> python callable.
        """
        self.model = model or os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
        self.input_sample_rate = input_sample_rate
        self.tools = tools or TOOLS
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue,
                            audio_output_callback, audio_interrupt_callback=None):
        # Read keys now (env is guaranteed loaded by the time a session starts).
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        sarvam_key = os.getenv("SARVAM_API_KEY", "")
        http = httpx.AsyncClient(timeout=45.0)

        event_queue = asyncio.Queue()
        turn_queue = asyncio.Queue()
        messages = [{"role": "system", "content": get_system_instruction()}]

        speaking = {"on": False}       # agent audio is being sent out right now
        current_turn = {"task": None}  # in-flight agent turn task

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
                "model": STT_MODEL,
                "language": STT_LANGUAGE,
                "encoding": "linear16",
                "sample_rate": self.input_sample_rate,
                "channels": 1,
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "true",
                "endpointing": STT_ENDPOINTING_MS,
                "utterance_end_ms": STT_UTTERANCE_END_MS,
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
                        logger.info(f"Deepgram STT connected (model={STT_MODEL}, lang={STT_LANGUAGE})")
                        failures = 0

                        async def pump_audio():
                            nonlocal stt_secs_pending
                            keepalive = 0.0
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

        # ---- LLM (Groq, OpenAI-compatible, STREAMING) -----------------------
        # Streams token deltas. As soon as a complete sentence/clause of spoken
        # text is available it is pushed to `on_sentence` so TTS can start while
        # the model is still generating. Returns the assembled message
        # {content, tool_calls} once the stream ends. Tool-call deltas are
        # accumulated by index (OpenAI streaming format).
        async def chat_completion_stream(msgs, on_sentence=None):
            payload = {
                "model": self.model,
                "messages": msgs,
                "temperature": 0.4,
                "max_tokens": 400,
                "tools": self.tools,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            headers = {"Authorization": f"Bearer {groq_key}",
                       "Content-Type": "application/json"}

            content_parts = []
            tool_calls = {}   # index -> {id, name, arguments}
            pending = ""      # spoken text not yet flushed as a sentence
            finish_reason = None

            async def flush_sentences(force=False):
                nonlocal pending
                if not on_sentence:
                    return
                # Emit each complete sentence/clause; keep the tail buffered.
                while True:
                    chunks = _split_sentences(pending)
                    # If not forcing, only emit chunks we're sure are complete
                    # (i.e. there is text after the last boundary, or force).
                    if force:
                        for c in chunks:
                            await on_sentence(c)
                        pending = ""
                        return
                    if len(chunks) <= 1:
                        return
                    first = chunks[0]
                    await on_sentence(first)
                    # remove the emitted chunk from the front of `pending`
                    idx = pending.find(first)
                    pending = pending[idx + len(first):].lstrip(" ,।") if idx >= 0 else ""

            for attempt in range(4):
                try:
                    async with http.stream("POST", GROQ_CHAT_URL, headers=headers,
                                           json=payload) as resp:
                        if resp.status_code == 429:
                            await resp.aread()
                            retry_after = 2.0
                            try:
                                retry_after = float(resp.headers.get("retry-after") or 0) or 2.0
                            except ValueError:
                                pass
                            wait = min(retry_after + 0.3, 8.0)
                            logger.warning(f"Groq rate limit, waiting {wait:.1f}s "
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
                        # stream finished
                        await flush_sentences(force=True)
                        break
                except httpx.HTTPStatusError:
                    raise
            else:
                raise RuntimeError("Groq streaming failed after retries")

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

        # ---- TTS (Sarvam Bulbul, REST) --------------------------------------
        async def synth(sentence, lang):
            payload = {
                "text": sentence,
                "target_language_code": lang,
                "speaker": TTS_SPEAKER,
                "model": TTS_MODEL,
                "speech_sample_rate": TTS_SAMPLE_RATE,
                "output_audio_codec": "wav",
            }
            resp = await http.post(
                SARVAM_TTS_URL,
                headers={"api-subscription-key": sarvam_key,
                         "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("audios") or []

        async def _emit_audio(pcm):
            # Send in ~120ms chunks so barge-in cancellation is responsive.
            step = int(TTS_SAMPLE_RATE * 2 * 0.12)
            for i in range(0, len(pcm), step):
                if inspect.iscoroutinefunction(audio_output_callback):
                    await audio_output_callback(pcm[i:i + step])
                else:
                    audio_output_callback(pcm[i:i + step])
                await asyncio.sleep(0)  # yield so a cancel can land between chunks

        async def synth_chunk(s):
            audios = await synth(s, _tts_language_for(s))
            await emit({"type": "usage", "tts_chars": len(s)})
            return b"".join(_strip_wav_header(base64.b64decode(a)) for a in audios)

        async def speak_stream(sentence_queue):
            """Consume sentences from a queue as the LLM streams them, synthesize
            them (each synthesis kicked off the instant the sentence arrives, so
            they overlap) and emit audio strictly in order. None ends the stream."""
            synth_tasks = []   # ordered in-flight/complete synthesis tasks
            play_idx = 0
            try:
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is not None:
                        synth_tasks.append(asyncio.create_task(synth_chunk(sentence)))
                    # play everything we can, in order, without getting ahead
                    if sentence is None:
                        while play_idx < len(synth_tasks):
                            await _emit_audio(await synth_tasks[play_idx])
                            play_idx += 1
                        return
                    # opportunistically play any already-finished leading chunks
                    while play_idx < len(synth_tasks) and synth_tasks[play_idx].done():
                        await _emit_audio(await synth_tasks[play_idx])
                        play_idx += 1
            finally:
                for t in synth_tasks[play_idx:]:
                    if not t.done():
                        t.cancel()

        # ---- one agent turn: LLM (+tools) then TTS, all streamed ------------
        async def run_turn(user_text):
            messages.append({"role": "user", "content": user_text})
            for _ in range(5):  # tool-call loop guard
                # A sentence queue drives TTS in parallel with LLM generation.
                sentence_queue = asyncio.Queue()
                agent_text_parts = []
                started_speaking = {"v": False}
                speak_task = None

                async def on_sentence(s):
                    # First spoken chunk: flip speaking flag + start the player.
                    nonlocal speak_task
                    if not started_speaking["v"]:
                        started_speaking["v"] = True
                        speaking["on"] = True
                        speak_task = asyncio.create_task(speak_stream(sentence_queue))
                    agent_text_parts.append(s)
                    await emit({"type": "agent", "text": s})
                    await sentence_queue.put(s)

                try:
                    reply = await chat_completion_stream(messages, on_sentence)
                finally:
                    if started_speaking["v"]:
                        await sentence_queue.put(None)  # end the audio stream
                        if speak_task:
                            try:
                                await speak_task
                            finally:
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
                            turn_queue.get(), timeout=TURN_DEBOUNCE_SECONDS)
                        user_text = f"{user_text} {more}".strip()
                    except asyncio.TimeoutError:
                        break
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
        ]

        logger.info(f"Voice agent started (stt=deepgram/{STT_MODEL}, "
                    f"llm=groq/{self.model}, tts=sarvam/{TTS_MODEL}/{TTS_SPEAKER})")
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
            await http.aclose()
            logger.info("Voice agent session closed")
