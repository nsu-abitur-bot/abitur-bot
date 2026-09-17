FROM python:3.12-slim

# Install uv (версия запинена вместо плывущего :latest)
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen ensures we use the exact versions from uv.lock
# --no-install-project installs only dependencies first (better caching)
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv sync --frozen

# Контейнер работает не от root; фиксированный UID 1000 — чтобы владельцем
# bind-монтируемых ./logs ./data ./faq на хосте можно было сделать тем же пользователем
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /usr/sbin/nologin app \
    && chown -R app:app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER app

# Run the application
CMD ["python", "main.py"]
