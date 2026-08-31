# Kataria Voice Agent — low-latency stack

A real-time voice service advisor ("Rahul" from Kataria Automobiles) with browser
calls, Twilio phone calls, a live transcript dashboard, and an admin dashboard
with real per-call costing. Built for **low latency** by picking the fastest
provider for each stage.

## Pipeline

```
caller audio (PCM16 16kHz)
    │
    ▼
Deepgram nova-3 streaming STT ─ WebSocket, language=multi (Hindi/English/Gujarati),
    │  final transcript      interim results + endpointing, barge-in
    ▼
Groq gpt-oss-20b chat completions ─ fast, native tool calling
    │  reply text
    ▼
Sarvam Bulbul v3 TTS (speaker: rahul) ─ PCM16 24kHz back to the caller
```

Deepgram nova-3 (with `language=multi`) transcribes Hindi/English code-mixing, Groq
replies fast, and Sarvam Bulbul keeps the natural Indian voice. Only TTS is on Sarvam.

Notes:
- Use **nova-3**, not nova-2 — nova-2 doesn't support `language=multi` and returns
  nothing for Hindi.
- Groq's free tier is **8,000 tokens/min per org**; the engine retries on 429 and the
  system prompt is kept compact so multi-turn calls stay under the limit. For heavy
  use, upgrade the Groq tier or point `GROQ_LLM_MODEL` at a larger model.

## Quick Start

### 1. Backend Setup

```bash
# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate       # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# Configure
copy .env.example .env       # then set SARVAM_API_KEY (get one at dashboard.sarvam.ai)

# Start the server
python main.py
```

### 2. Frontend

Open [http://localhost:8000](http://localhost:8000) and press Start Call.

- `/live` — real-time transcript of an in-progress phone call
- `/admin` — call logs, transcripts and costing (key = `ANALYTICS_SECRET`)

## Costing — real measured usage (all USD)

Every call records the actual usage of each stage and prices it (rates
env-overridable in `.env`):

| Stage | Provider | Measured | Rate |
|---|---|---|---|
| STT | Deepgram nova-3 | minutes of caller audio | ~$0.0077 / min |
| LLM | Groq gpt-oss-20b | prompt + completion tokens | $0.15 in / $0.75 out per 1M |
| TTS | Sarvam Bulbul v3 | characters synthesized | ₹30 / 10k chars (→ USD via `USD_INR`) |

Twilio's real billed `Call.price` (USD) is fetched after each phone call and
added to produce the combined USD total.

## Project Structure

```
/
├── main.py             # FastAPI server, WebSocket endpoints, dashboards
├── voice_agent.py      # Deepgram STT → Groq LLM → Sarvam TTS engine
├── twilio_handler.py   # Twilio Media Streams bridge (mulaw 8k <-> PCM)
├── pricing.py          # Deepgram + Groq + Sarvam TTS rates + Twilio price
├── recorder.py         # Per-call transcript + usage + cost recording
├── store.py            # JSON call-record persistence + summaries
├── requirements.txt    # Python dependencies
└── frontend/
    ├── index.html       # User Interface
    ├── main.js          # Application logic
    ├── sarvam-client.js # WebSocket client for backend communication
    ├── media-handler.js # Audio capture and playback
    └── pcm-processor.js # AudioWorklet for PCM processing
```

## Configuration

Set `DEEPGRAM_API_KEY`, `GROQ_API_KEY` and `SARVAM_API_KEY` in `.env` (see
`.env.example` for every option: models, voice, Twilio credentials, cost rates).
