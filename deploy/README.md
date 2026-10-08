# Deploy the CoScientist web UI

The web UI is a long-running uvicorn server. A GitHub Actions job is short. So
the job does not host the server. A user-level `systemd` service owns the
process. The job updates the code and restarts the service.

Parts:

- `coscientist-web.service` — the systemd user unit that runs the server.
- `../.github/workflows/deploy-web.yml` — the deploy job for the self-hosted
  runner.

## One-time setup on the server

Run these steps once, as the same Linux user that runs the GitHub Actions
runner. The steps need no root.

1. Install `uv` for this user.

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   This puts `uv` at `~/.local/bin/uv`. Confirm with `~/.local/bin/uv --version`.

2. Clone the repository to the fixed deploy directory. The runner work
   directory changes each job, so the service uses a stable path instead. If
   the clone already exists, skip this step.

   ```
   git clone https://github.com/aimclub/CoScientist.git ~/cosci/CoScientist
   cd ~/cosci/CoScientist
   git checkout main
   ```

   The deploy job, the systemd unit, and these steps all use the same fixed
   path `$HOME/cosci/CoScientist`. Keep them the same. To move it, change
   `DEPLOY_DIR` in `deploy-web.yml` and `WorkingDirectory` in
   `coscientist-web.service` together.

3. Put the secrets file at a protected path outside the repository.

   ```
   mkdir -p ~/.config/coscientist
   cp /path/to/your/.env ~/.config/coscientist/.env
   chmod 600 ~/.config/coscientist/.env
   ```

   Keep secrets out of git. Do not commit this file. See "The .env file" below.

4. Install and start the service.

   ```
   mkdir -p ~/.config/systemd/user
   cp ~/cosci/CoScientist/deploy/coscientist-web.service ~/.config/systemd/user/
   ln -sfn ~/.config/coscientist/.env ~/cosci/CoScientist/.env
   systemctl --user daemon-reload
   systemctl --user enable --now coscientist-web
   ```

5. Keep the service alive after logout and after a reboot.

   ```
   loginctl enable-linger "$USER"
   ```

6. Check the state.

   ```
   systemctl --user status coscientist-web --no-pager
   curl -fsS http://127.0.0.1:7000/ >/dev/null && echo up
   ```

Open the firewall for the port if clients are not on the server. The service
binds `0.0.0.0:7000`.

## The deploy job

The job runs on the self-hosted runner. On each run it:

1. Resets the deploy directory to the pushed commit (`git reset --hard`).
2. Runs `uv sync --frozen` to match `uv.lock`.
3. Re-links `.env` into the deploy directory.
4. Restarts the user service.
5. Polls `http://127.0.0.1:7000/` until it answers, or fails the job.

Triggers: a push to `main`, or a manual run from the Actions tab. To deploy a
different branch, change `DEPLOY_BRANCH` in `deploy-web.yml`.

## The .env file

The application reads configuration from a `.env` file in its working
directory. `.env` is gitignored, so it never enters the repository or the
workflow. The service reads it through the symlink at
`~/cosci/CoScientist/.env`.

Split of duties:

- Secrets and endpoints stay in `~/.config/coscientist/.env` (LLM keys, S3,
  Postgres, service URLs).
- Runtime toggles stay in the unit file as `Environment=` lines
  (`HITL__ENABLED=true`, `RESEARCH_FRAME=true`). `load_dotenv` does not
  override real environment variables, so the unit wins over the same keys in
  `.env`.

Two notes on the `.env` content:

1. The frame form needs `HITL__ENABLED=true`. The unit sets it. Do not set it
   to `false` in `.env` and expect the form.
2. Remove any line that is not `KEY=VALUE`. A bare path line such as
   `/root/.opik.config` is not a valid variable and only adds noise.

## Config keys the web UI needs

The server starts without a database, but the agents need live model access.
At minimum set these in `~/.config/coscientist/.env`:

- `LLM__OPENAI_API_KEY`, `LLM__MAIN_MODEL`, `LLM__MAIN_URL`.
- `LLM__ALLOWED_PROVIDERS` for the providers you use.
- `SERVICES__TAVILY_API_KEY` and `SERVICES__OPENALEX_API_KEY` for web and
  literature search.

Keys use the nested form `SECTION__FIELD` (double underscore).

## OpenRouter proxy

The VM egress blocks `openrouter.ai`. The institution runs a tunnel on the
docker0 bridge to reach it. The systemd unit sets these variables, so the
service uses the tunnel with no code change:

```
Environment=HTTP_PROXY=http://172.17.0.1:7890
Environment=HTTPS_PROXY=http://172.17.0.1:7890
Environment=NO_PROXY=localhost,127.0.0.1
```

Facts to know:

- The tunnel carries only `openrouter.ai`. All other hosts go out the normal
  route, so this global setting does not affect Tavily, OpenAlex, or PubMed.
- TLS stays end to end. The proxy moves bytes only. The client checks the
  certificate as usual.
- The proxy is reachable only from this server. The address `172.17.0.1` does
  not route from outside.
- It is fail-closed. If the proxy is down, OpenRouter calls fail. They do not
  fall back to a direct connection.

Test the tunnel from the VM. A working tunnel answers with country `LV`:

```
HTTPS_PROXY=http://172.17.0.1:7890 curl -s https://api.myip.com
```

If the port refuses the connection, the proxy container is down:

```
docker ps --filter name=openhands-xray-proxy --format '{{.Status}}'
```

## Common operations

```
# Watch logs
journalctl --user -u coscientist-web -f

# Restart by hand
systemctl --user restart coscientist-web

# Stop
systemctl --user stop coscientist-web
```
