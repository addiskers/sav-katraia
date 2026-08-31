#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 22.04/24.04 GCP Compute Engine VM.
# Run as a sudo-capable user:  bash setup-vm.sh
set -euo pipefail

DOMAIN="sar-kataria.globalvoxinc.ai"
APP_ROOT="/opt/voice-agent"
REPO="https://github.com/addiskers/sav-katraia.git"
SDK_DIR="$APP_ROOT/gemini-live-api-examples/gemini-live-genai-python-sdk"

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git nginx certbot python3-certbot-nginx

echo "==> Cloning repo to $APP_ROOT"
sudo mkdir -p "$APP_ROOT"
sudo chown -R "$USER":"$USER" "$APP_ROOT"
if [ -d "$APP_ROOT/.git" ]; then
  git -C "$APP_ROOT" pull
else
  git clone "$REPO" "$APP_ROOT"
fi

echo "==> Creating Python venv + installing deps"
cd "$SDK_DIR"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
# uvicorn is used by the systemd unit; ensure it's present
./.venv/bin/pip install "uvicorn[standard]"

echo "==> Creating data dir for call logs"
sudo mkdir -p /var/lib/voice-agent/data
sudo chown -R www-data:www-data /var/lib/voice-agent

echo "==> .env"
if [ ! -f "$SDK_DIR/.env" ]; then
  cp "$SDK_DIR/.env.example" "$SDK_DIR/.env"
  echo "    !! Created $SDK_DIR/.env from example."
  echo "    !! EDIT IT NOW and fill in DEEPGRAM_API_KEY, GROQ_API_KEY, SARVAM_API_KEY,"
  echo "       TWILIO_* and set PUBLIC_URL=https://$DOMAIN"
fi
# make the app files readable by www-data
sudo chown -R www-data:www-data "$APP_ROOT"

echo "==> systemd service"
sudo cp "$SDK_DIR/deploy/voice-agent.service" /etc/systemd/system/voice-agent.service
sudo systemctl daemon-reload
sudo systemctl enable voice-agent

echo "==> nginx: websocket upgrade map"
sudo tee /etc/nginx/conf.d/websocket-upgrade.conf >/dev/null <<'EOF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
EOF

echo "==> nginx: site"
sudo cp "$SDK_DIR/deploy/nginx-voice-agent.conf" /etc/nginx/sites-available/voice-agent
sudo ln -sf /etc/nginx/sites-available/voice-agent /etc/nginx/sites-enabled/voice-agent
sudo rm -f /etc/nginx/sites-enabled/default || true

echo
echo "======================================================================"
echo " NEXT STEPS (manual):"
echo " 1. Point DNS: an A record for $DOMAIN -> this VM's external IP."
echo " 2. Open firewall for ports 80 and 443 (GCP firewall rule / tag)."
echo " 3. Fill in secrets:   nano $SDK_DIR/.env"
echo " 4. Get TLS cert:      sudo certbot --nginx -d $DOMAIN"
echo " 5. Start the app:     sudo systemctl start voice-agent"
echo "    Check it:          sudo systemctl status voice-agent"
echo "                       curl -I https://$DOMAIN"
echo " 6. Point Twilio number's Voice webhook to:"
echo "        https://$DOMAIN/twilio/voice   (HTTP POST)"
echo "======================================================================"
