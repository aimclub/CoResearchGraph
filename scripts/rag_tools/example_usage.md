## Example usage

1. Add one server
```bash
python3 scripts/rag_tools/cli.py add \
  --url http://localhost:8080/mcp \
  --name papers \
  --description "Academic search"
```

2. Add with token
```bash
python3 scripts/rag_tools/cli.py add \
  --url http://localhost:8080/mcp \
  --name private-api \
  --token abc123
```

3. Load multiple servers
```bash
python3 scripts/rag_tools/cli.py load scripts/rag_tools/example_servers.json
```

4. Remove server
```bash
python3 scripts/rag_tools/cli.py remove <server_id>
```

5. Sync one
```bash
python3 scripts/rag_tools/cli.py sync <server_id>
```

6. Sync all
```bash
python3 scripts/rag_tools/cli.py sync-all
```

7. List servers
```bash
python3 scripts/rag_tools/cli.py list
```
