# Build the dashboard, then run the API with the built dashboard served at "/".
#
#   docker build -t visionpilot .
#   docker run -p 8000:8000 visionpilot

FROM node:20-slim AS dashboard
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend
COPY ml ./ml
COPY benchmark ./benchmark
# Editable, so backend/main.py finds frontend/dist next to the code.
RUN pip install -e .

COPY --from=dashboard /app/frontend/dist ./frontend/dist

# SQLite is created here on first run. It is lost when the container is replaced unless
# DB_PATH points at a mounted volume. Runs as root so a mounted disk is writable.
RUN mkdir -p /app/data
EXPOSE 8000

# Hosts such as Render set PORT; locally it falls back to 8000.
CMD ["sh", "-c", "uvicorn backend.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
