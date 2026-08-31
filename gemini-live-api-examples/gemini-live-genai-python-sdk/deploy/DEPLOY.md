# Deploying the Voice Agent to a GCP Compute Engine VM

Target: `sar-kataria.globalvoxinc.ai` serving the web UI + Twilio phone calls over
HTTPS/WSS. Stack per request: **Deepgram STT → Groq LLM → Sarvam TTS**.

```
Internet ──HTTPS/WSS──> nginx (TLS, :443) ──HTTP/WS──> uvicorn (127.0.0.1:8000)
                             │                              │
                        Let's Encrypt                 systemd: voice-agent
```

---

## 1. Create the VM

In GCP Console → Compute Engine → Create instance:
- **Machine type**: `e2-small` is enough (this app is I/O-bound, not CPU-bound).
- **Boot disk**: Ubuntu 22.04 or 24.04 LTS.
- **Firewall**: check **Allow HTTP** and **Allow HTTPS** (or add a rule for tcp:80,443).
- Note the **external IP**.

Reserve a **static external IP** (VPC network → IP addresses) so it doesn't change
on reboot — your DNS points at it.

## 2. Point DNS

Add an **A record** for `sar-kataria.globalvoxinc.ai` → the VM's static external IP.
Wait for it to resolve (`ping sar-kataria.globalvoxinc.ai` shows the right IP).

## 3. Run the setup script

SSH into the VM (the browser SSH button in GCP works), then:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/addiskers/sav-katraia.git /tmp/sav && \
  bash /tmp/sav/gemini-live-api-examples/gemini-live-genai-python-sdk/deploy/setup-vm.sh
```

The script installs Python/nginx/certbot, clones the repo to `/opt/voice-agent`,
builds the venv, installs the systemd service and nginx site, and prints the
remaining manual steps.

## 4. Fill in secrets

```bash
sudo nano /opt/voice-agent/gemini-live-api-examples/gemini-live-genai-python-sdk/.env
```

Set at minimum:
```
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
SARVAM_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
PUBLIC_URL=https://sar-kataria.globalvoxinc.ai
ANALYTICS_SECRET=<pick a strong admin key>
DATA_DIR=/var/lib/voice-agent/data
```

## 5. TLS certificate

```bash
sudo certbot --nginx -d sar-kataria.globalvoxinc.ai
```
Certbot edits the nginx site to add the real cert paths and reloads nginx.
Renewal is automatic (systemd timer). DNS from step 2 must already resolve.

## 6. Start it

```bash
sudo systemctl start voice-agent
sudo systemctl status voice-agent      # should be "active (running)"
curl -I https://sar-kataria.globalvoxinc.ai   # 200 OK
```

Logs: `sudo journalctl -u voice-agent -f`

## 7. Twilio webhook

In the Twilio Console → your phone number → Voice → "A call comes in":
- Webhook: `https://sar-kataria.globalvoxinc.ai/twilio/voice`
- Method: **HTTP POST**

The app already emits `wss://` for the media stream on this domain, so calls work
over secure WebSockets. Test with the **Call Me** button on the site or by dialing
the Twilio number.

---

## Updating after a code change

```bash
cd /opt/voice-agent && git pull
cd gemini-live-api-examples/gemini-live-genai-python-sdk
sudo ./.venv/bin/pip install -r requirements.txt   # only if deps changed
sudo systemctl restart voice-agent
```

## Endpoints
- `/`        — browser voice UI
- `/admin`   — call logs + costing (key = `ANALYTICS_SECRET`)
- `/live`    — live transcript of an in-progress phone call
- `/twilio/voice` — Twilio webhook (POST)

## Notes
- **Groq free tier is 8k tokens/min per org** — under load you'll hit brief 429
  waits (the app retries). Upgrade the Groq tier for production traffic.
- The app binds to `127.0.0.1:8000`; only nginx is public. Don't open 8000 in the
  GCP firewall.
- `.env`, `.venv/`, and `data/` are gitignored — secrets never leave the VM.
