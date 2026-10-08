FROM python:3.12-slim

# RDKit Draw (Cairo) needs X11/Cairo runtime libs; slim image omits them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libfontconfig1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# The build context is the repository root, so every COPY path starts there.
COPY mcp-servers/chemical-mcp-server/pyproject.toml mcp-servers/chemical-mcp-server/README.md ./
RUN uv sync --no-dev

COPY mcp-servers/chemical-mcp-server/ ./
# The shared S3 client. The papers-search and dataset-collection images copy the
# same path, so all three talk to S3 through one class.
COPY CoScientist/paper_parser /app/CoScientist/paper_parser

ENV UV_SYSTEM_PYTHON=1

EXPOSE 7331

CMD ["uv", "run", "python", "-m", "server.chemical_server"]
