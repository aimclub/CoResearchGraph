# Build a SERVING MCP image from an ALREADY-BUILT alembic output — fully
# algorithmic: no LLM, no `docker commit`. The venvs are NOT copied (they are
# host-specific and non-relocatable); they are rebuilt inside the image from the
# recorded, now-portable setup.sh plus a fixed fastmcp install.
#
#   docker build -f docker/alembic/serve.Dockerfile --build-arg REPO_NAME=<name> -t <img> .
#   docker run -p 8000:8000 <img> serve <repo_url>
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALEMBIC_WORKDIR=/work/.alembic \
    MCP_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential pkg-config \
        libcairo2 libfontconfig1 libx11-6 libxext6 libxrender1 \
        libgl1 libglib2.0-0 libsm6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG REPO_NAME
WORKDIR /work

# The cloned repo (needed by the editable install) + the portable code artefacts.
# Venvs are deliberately excluded and rebuilt below.
COPY .alembic/${REPO_NAME}/repos            /work/.alembic/${REPO_NAME}/repos
COPY .alembic/${REPO_NAME}/output/tools     /work/.alembic/${REPO_NAME}/output/tools
COPY .alembic/${REPO_NAME}/output/helpers   /work/.alembic/${REPO_NAME}/output/helpers
COPY .alembic/${REPO_NAME}/output/server.py /work/.alembic/${REPO_NAME}/output/server.py
COPY .alembic/${REPO_NAME}/output/setup.sh  /work/.alembic/${REPO_NAME}/output/setup.sh

# 1. Rebuild the main .venv (repo + deps + editable install) from the transcript.
#    setup.sh does `cd /work` and uses workdir-relative, --python-targeted commands.
RUN bash /work/.alembic/${REPO_NAME}/output/setup.sh

# 2. Build the isolated server venv with the MCP runtime (a wrapper-stage concern,
#    not recorded in setup.sh) — server.py + serve.py import fastmcp from here.
#    boto3 backs helpers/s3_transfer.py (lazy-imported there); this package list
#    is duplicated in alembic/tools/venv.py's SERVER_PACKAGES — keep in sync.
RUN uv venv /work/.alembic/${REPO_NAME}/output/.venv-server --python 3.11 \
 && uv pip install --python /work/.alembic/${REPO_NAME}/output/.venv-server/bin/python fastmcp mcp boto3

COPY docker/alembic/serve.py      /usr/local/bin/serve.py
COPY docker/alembic/entrypoint.py /usr/local/bin/entrypoint.py
RUN chmod +x /usr/local/bin/entrypoint.py

EXPOSE 8000
ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
CMD ["help"]
