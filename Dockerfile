# ── Stage 1: Build the React frontend ────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, frozen lockfile)
RUN uv sync --frozen --no-dev

# Pre-download the embedding model so it's baked into the image.
# Railway's filesystem is ephemeral — without this, the model is
# re-downloaded from HuggingFace on every cold start.
# Using huggingface_hub directly avoids importing torch (which would
# crash because the GPU torch wheel tries to load CUDA libraries).
ENV HF_HOME=/app/.cache/huggingface
RUN uv run --no-sync hf download sentence-transformers/all-MiniLM-L6-v2

# Copy the rest of the source
COPY . .

# Drop in the pre-built frontend so FastAPI can serve it
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Railway injects PORT at runtime; default to 8000 for local use
ENV PORT=8000

EXPOSE 8000

CMD /app/.venv/bin/uvicorn server:app --host 0.0.0.0 --port ${PORT}
