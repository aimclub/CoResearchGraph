FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md .
RUN uv sync --no-dev

COPY . . 

ENV UV_SYSTEM_PYTHON=1

EXPOSE 7331

CMD ["uv", "run", "python", "-m", "server.aggregator_server"]
