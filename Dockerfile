# Etapa 1: build
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim as builder

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir -e .

# Etapa 2: runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/streamlit
COPY src/ ./src/

EXPOSE 8501
CMD ["streamlit", "run", "src/pl_solver/app.py", "--server.port=8501", "--server.address=0.0.0.0"]