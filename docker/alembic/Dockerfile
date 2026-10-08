# Base image for the Alembic pipeline.
#
# python:3.11 (Debian-based, multi-arch) — Python is preinstalled at
# /usr/local/bin/python, so we only add git, build tools and the C libs
# scientific repos commonly need at install/runtime.
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    ALEMBIC_WORKDIR=/work/.alembic \
    MCP_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        build-essential pkg-config \
        libcairo2 libfontconfig1 libx11-6 libxext6 libxrender1 \
        libgl1 libglib2.0-0 libsm6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY docker/alembic/requirements.txt /tmp/alembic-requirements.txt
RUN uv pip install --system --no-cache -r /tmp/alembic-requirements.txt

COPY CoScientist/alembic /app/alembic

COPY docker/alembic/entrypoint.py /usr/local/bin/entrypoint.py
COPY docker/alembic/serve.py      /usr/local/bin/serve.py
RUN chmod +x /usr/local/bin/entrypoint.py

RUN mkdir -p /work
WORKDIR /work

EXPOSE 8000

ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
CMD ["help"]
