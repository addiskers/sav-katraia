"""
Twilio Voice <-> Sarvam voice agent bridge.

Handles:
- Twilio Media Streams WebSocket (mulaw 8kHz)
- Audio conversion: mulaw 8kHz <-> PCM 16kHz (STT input) / PCM 24kHz (TTS output)
- Bridges the two in real-time
- Pure Python audio conversion (no audioop, works on Python 3.13+)
"""

import asyncio
import array
import base64
import json
import logging
import struct

logger = logging.getLogger(__name__)

# ===== Mulaw codec tables (ITU-T G.711) =====

# Mulaw -> Linear PCM16 decode table (256 entries)
_MULAW_DECODE = array.array("h")  # signed short
for _i in range(256):
    _v = ~_i
    _sign = _v & 0x80
    _exponent = (_v >> 4) & 0x07
    _mantissa = _v & 0x0F
    _sample = ((_mantissa << 3) + 0x84) << _exponent
    _sample -= 0x84
    if _sign:
        _sample = -_sample
    _MULAW_DECODE.append(max(-32768, min(32767, _sample)))

# Linear PCM16 -> Mulaw encode
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635
_MULAW_EXP_TABLE = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


def _pcm16_to_mulaw_sample(sample: int) -> int:
    """Encode one PCM16 sample to mulaw byte."""
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP
    sample += _MULAW_BIAS
    exponent = _MULAW_EXP_TABLE[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


# ===== Audio conversion functions =====

def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz (Twilio) -> PCM 16-bit 16kHz (STT input)."""
    # Decode mulaw to PCM16 at 8kHz
    samples_8k = [_MULAW_DECODE[b] for b in mulaw_bytes]
    # Upsample 8kHz -> 16kHz by linear interpolation
    samples_16k = []
    for i in range(len(samples_8k)):
        samples_16k.append(samples_8k[i])
        if i + 1 < len(samples_8k):
            samples_16k.append((samples_8k[i] + samples_8k[i + 1]) >> 1)
        else:
            samples_16k.append(samples_8k[i])
    return struct.pack(f"<{len(samples_16k)}h", *samples_16k)


def pcm24k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert PCM 16-bit 24kHz (TTS output) -> mulaw 8kHz (Twilio)."""
    # Read PCM16 samples
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    # Downsample 24kHz -> 8kHz (take every 3rd sample)
    samples_8k = samples[::3]
    # Encode to mulaw
    return bytes(_pcm16_to_mulaw_sample(s) for s in samples_8k)


class TwilioMediaBridge:
    """Bridges a Twilio Media Stream WebSocket with a Sarvam voice session."""

    def __init__(self, websocket, agent_client, text_trigger, on_event=None):
        self.ws = websocket
        self.agent = agent_client
        self.stream_sid = None
        self.call_sid = None
        self.text_trigger = text_trigger
        self.on_event = on_event  # async callback for live transcript

        # Queues for the voice agent
        self.audio_input_queue = asyncio.Queue()
        self.video_input_queue = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()

    async def audio_output_callback(self, data: bytes):
        """Called when the agent produces audio. Convert and send to Twilio."""
        if not self.stream_sid:
            return
        try:
            mulaw = pcm24k_to_mulaw(data)
            payload = base64.b64encode(mulaw).decode("utf-8")
            msg = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }
            await self.ws.send_json(msg)
        except Exception as e:
            logger.error(f"Error sending audio to Twilio: {e}")

    async def audio_interrupt_callback(self):
        """Called when the agent detects user interruption. Clear Twilio buffer."""
        if not self.stream_sid:
            return
        try:
            await self.ws.send_json({
                "event": "clear",
                "streamSid": self.stream_sid,
            })
        except Exception:
            pass

    async def handle_twilio_messages(self):
        """Receive messages from Twilio Media Streams WebSocket."""
        try:
            while True:
                message = await self.ws.receive_text()
                data = json.loads(message)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio Media Stream connected")

                elif event == "start":
                    self.stream_sid = data["start"]["streamSid"]
                    self.call_sid = data["start"].get("callSid", "")
                    logger.info(f"Twilio stream started: sid={self.stream_sid}, call={self.call_sid}")

                    # Trigger the AI to start talking
                    await self.text_input_queue.put(self.text_trigger)

                elif event == "media":
                    # Twilio sends base64 mulaw audio
                    payload = data["media"]["payload"]
                    mulaw_bytes = base64.b64decode(payload)
                    pcm_16k = mulaw_to_pcm16k(mulaw_bytes)
                    await self.audio_input_queue.put(pcm_16k)

                elif event == "stop":
                    logger.info("Twilio stream stopped")
                    break

        except Exception as e:
            logger.error(f"Twilio receive error: {e}")

    async def _emit(self, event):
        """Send event to live transcript watchers."""
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception:
                pass

    async def run(self):
        """Run the bridge: Twilio <-> Sarvam agent."""
        twilio_task = asyncio.create_task(self.handle_twilio_messages())

        await self._emit({"type": "call_start", "call_sid": self.call_sid or ""})

        try:
            async for event in self.agent.start_session(
                audio_input_queue=self.audio_input_queue,
                video_input_queue=self.video_input_queue,
                text_input_queue=self.text_input_queue,
                audio_output_callback=self.audio_output_callback,
                audio_interrupt_callback=self.audio_interrupt_callback,
            ):
                if event:
                    # Broadcast transcript/tool events to live watchers
                    await self._emit(event)
                    if event.get("type") == "error":
                        logger.error(f"Sarvam error during Twilio call: {event}")
                        break
        except Exception as e:
            logger.error(f"Sarvam session error: {e}")
        finally:
            await self._emit({"type": "call_end"})
            twilio_task.cancel()
            logger.info("Twilio-Sarvam bridge closed")
