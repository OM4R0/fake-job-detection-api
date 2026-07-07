# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Environment variables ─────────────────────────────────────────────────────
# Force Python to flush stdout immediately — required for logs to appear in
# real time on cloud platforms like Render instead of being buffered and delayed.
ENV PYTHONUNBUFFERED=1

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy requirements before source code so Docker caches this layer.
# If only source code changes, packages won't reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
# Copy src/ files flat into /app (not into a src/ subfolder).
# This means `from schemas import` and `import preprocess` resolve naturally
# without needing PYTHONPATH tricks, and `models/` paths stay correct.
COPY src/ .
COPY models/ ./models/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the API ─────────────────────────────────────────────────────────────
# Read $PORT at runtime — Render sets this automatically (default: 10000).
# Falls back to 8000 for local Docker runs where PORT is not set.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
