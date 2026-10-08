FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY mcp-servers/papers-search-mcp-server/pyproject.toml mcp-servers/papers-search-mcp-server/uv.lock ./

RUN uv sync --frozen --no-install-project

COPY mcp-servers/papers-search-mcp-server/ ./
COPY CoScientist/paper_parser /app/CoScientist/paper_parser

CMD ["uv", "run", "--no-project", "python", "papers_search_server.py"]