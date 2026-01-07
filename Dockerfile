FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project first (so README.md and src/ exist)
COPY . /app

# Create venv + install deps
RUN uv venv && uv sync

# Install package (editable)
RUN uv pip install -e .

CMD ["uv", "run", "train"]
