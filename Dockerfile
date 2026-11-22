# One image for both services. docker-compose.yml picks the command (API or UI).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /code

# Install dependencies first so this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY ui ./ui

ENV PATH="/code/.venv/bin:$PATH"
