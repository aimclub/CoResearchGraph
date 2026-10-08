FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY mcp-servers/dataset-collection-mcp-server/pyproject.toml mcp-servers/dataset-collection-mcp-server/uv.lock ./

RUN uv sync --frozen --no-install-project

COPY mcp-servers/dataset-collection-mcp-server/ ./
COPY CoScientist/paper_parser /app/CoScientist/paper_parser
COPY CoScientist/chemical_utils /app/CoScientist/chemical_utils

CMD ["uv", "run", "--no-project", "python", "dataset_collection_server.py"]
