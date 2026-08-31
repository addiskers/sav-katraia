# Optional container image (the VM deploy uses systemd + venv instead; this is
# here for reproducible builds / Cloud Run if you ever want it).
FROM python:3.13-slim

WORKDIR /app

# System deps kept minimal; app is pure-python + audio math in Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]"

COPY . .

# Call-log JSON store (mount a volume here in production for persistence).
ENV DATA_DIR=/data
RUN mkdir -p /data

ENV PORT=8000
EXPOSE 8000

# Secrets come from the environment (--env-file .env), never baked into the image.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
