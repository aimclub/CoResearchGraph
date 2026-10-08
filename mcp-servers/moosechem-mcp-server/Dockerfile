FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    sed \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# ── Шаг 1: MOOSE-Chem (меняется редко — кешируется) ──────────────────────
RUN git clone --depth 1 https://github.com/ZonglinY/MOOSE-Chem /app/MOOSE-Chem \
    # CPU-only torch: MOOSE-Chem never calls .cuda() or checks torch.cuda.is_available(),
    # so the CUDA build (2-3GB extra download: cudnn, cusparselt, cupti, etc.) is dead
    # weight for this container, which never runs with GPU access anyway.
    && pip install --no-cache-dir --default-timeout=300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=300 --retries 5 -r /app/MOOSE-Chem/requirements.txt

RUN sed -i \
    -e 's|^api_type=.*|api_type=0|' \
    -e 's|^model_name_insp_retrieval=.*|model_name_insp_retrieval="gpt-4o-mini"|' \
    -e 's|^model_name_gene=.*|model_name_gene="gpt-4o-mini"|' \
    -e 's|^model_name_eval=.*|model_name_eval="gpt-4o-mini"|' \
    -e 's|--custom_research_background_path "${custom_research_background_path} \\"|--custom_research_background_path "${custom_research_background_path}" \\|' \
    -e 's|--custom_inspiration_corpus_path ${custom_inspiration_corpus_path}|--custom_inspiration_corpus_path "${custom_inspiration_corpus_path}"|' \
    -e 's|--custom_research_background_path ${custom_research_background_path}|--custom_research_background_path "${custom_research_background_path}"|' \
    /app/MOOSE-Chem/main.sh

# ── Шаг 2: зависимости сервера (сервер самодостаточный) ──────────────────
RUN pip install --no-cache-dir aiohttp litellm pydantic requests

# ── Шаг 3: зависимости MCP-сервера (меняется редко — кешируется) ─────────
COPY mcp-servers/moosechem-mcp-server/pyproject.toml ./
RUN uv sync --no-install-project

# ── Шаг 4: сам сервер (меняется часто — последний слой) ──────────────────
COPY mcp-servers/moosechem-mcp-server/moosechem_server.py ./

RUN mkdir -p /app/jobs

ENV MOOSECHEM_PATH=/app/MOOSE-Chem
ENV MOOSECHEM_JOBS_DIR=/app/jobs
ENV MOOSECHEM_MODEL=gpt-4o-mini
ENV PYTHONPATH=/app

CMD ["uv", "run", "--no-project", "python", "moosechem_server.py"]
