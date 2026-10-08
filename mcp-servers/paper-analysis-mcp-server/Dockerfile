FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY mcp-servers/paper-analysis-mcp-server/pyproject.toml mcp-servers/paper-analysis-mcp-server/uv.lock ./

RUN uv sync --frozen --no-install-project

COPY mcp-servers/paper-analysis-mcp-server/ ./
COPY CoScientist/chemical_utils /app/CoScientist/chemical_utils
COPY CoScientist/paper_analysis /app/CoScientist/paper_analysis
COPY CoScientist/paper_parser/__init__.py /app/CoScientist/paper_parser/__init__.py
COPY CoScientist/paper_parser/s3_connection.py /app/CoScientist/paper_parser/s3_connection.py
COPY CoScientist/paper_parser/utils.py /app/CoScientist/paper_parser/utils.py
COPY CoScientist/papers_processing_refactoring /app/CoScientist/papers_processing_refactoring

CMD ["uv", "run", "--no-project", "python", "paper_analysis_server.py"]