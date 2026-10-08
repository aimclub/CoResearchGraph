FROM python:3.11-slim

WORKDIR /app

# Docker Compose fills these from the .env file. pip reads them by itself and
# sends its traffic through the proxy.
ARG HTTP_PROXY
ARG HTTPS_PROXY

# The build context is the repository root, so every COPY path starts there.
COPY mcp-servers/vault-mcp-server/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY mcp-servers/vault-mcp-server/vault_server.py ./

CMD ["python", "vault_server.py"]
