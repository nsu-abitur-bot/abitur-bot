FROM python:3.12-slim

# Install uv (версия запинена вместо плывущего :latest)
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Контейнер работает не от root; фиксированный UID 1000 — чтобы владельцем
# bind-монтируемых ./logs ./data ./faq на хосте можно было сделать тем же пользователем.
# Пользователь создаётся до копирования кода: файлы сразу приходят с нужным
# владельцем через COPY --chown. Отдельный `chown -R` после установки зависимостей
# переписал бы всё дерево вместе с .venv в новый слой и почти удвоил образ.
RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 -s /usr/sbin/nologin app

WORKDIR /app
RUN chown app:app /app
USER app

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock ./

# Install dependencies
# --frozen ensures we use the exact versions from uv.lock
# --no-install-project installs only dependencies first (better caching)
RUN uv sync --frozen --no-install-project

# Copy the rest of the application.
# Код принадлежит app, а не root: приложение пишет не только в смонтированные
# каталоги — например, api/routes/evals.py сохраняет evals/eval_results.json.
COPY --chown=app:app . .

# Install the project itself
RUN uv sync --frozen

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
