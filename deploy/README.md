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
   curl -fsS http://127.0.0.1:7000/healthz >/dev/null && echo up
   ```

   `/healthz` is the one path that answers without a login. Every other path,
   including `/`, redirects to `/login`.

Open the firewall for the port if clients are not on the server. The service
binds `0.0.0.0:7000`.

Open **only** that port. These services bind `0.0.0.0` by default and have no
authentication of their own:

- The code-exec server on `:8131`. It runs any shell command that is posted to
  `/submit`. Set `CODE_EXEC_HOST=127.0.0.1`.
- The MCP servers on `:7331`-`:7338`. The vault server on `:7338` holds the S3
  credentials.
- The A2A agent ports and the graph service.

## The deploy job

The job runs on the self-hosted runner. On each run it:

1. Resets the deploy directory to the pushed commit (`git reset --hard`).
2. Runs `uv sync --frozen` to match `uv.lock`.
3. Re-links `.env` into the deploy directory.
4. Restarts the user service.
5. Polls `http://127.0.0.1:7000/healthz` until it answers, or fails the job.

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

## Authentication

One shared password gates the whole deployment. There are no user accounts.
Everybody who logs in sees every session, every report and every setting.

The `.env` file holds a salted digest of the password, not the password. Build
the line on any machine with the repository:

```
python3 ~/cosci/CoScientist/deploy/make_password_hash.py
```

The script needs nothing but a Python 3 interpreter. It asks for the password
twice without echoing it, then prints one line.

`uv run --frozen python -m CoScientist auth-hash` does the same thing from the
deploy directory. Use the script while the `.env` file is still incomplete:
importing the package builds the agent system, so the command needs the LLM
keys to be in place already. Put
that line in `~/.config/coscientist/.env` **before** the first deploy of this
feature. The gate fails closed: with `AUTH__ENABLED=true` and no usable digest,
every request gets `503`.

```
AUTH__PASSWORD_HASH=pbkdf2_sha256:200000:<salt>:<digest>
AUTH__SECRET_KEY=<long random string>
AUTH__ALLOWED_ORIGINS=https://<the host browsers use>
```

Keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `AUTH__PASSWORD_HASH` | unset | Salted digest of the one password. Missing or malformed means `503` on every path. |
| `AUTH__PASSWORD` | unset | The password in clear text. Local runs only. See note 5. |
| `AUTH__SECRET_KEY` | random per boot | Signs the session cookie. See note 3. |
| `AUTH__ENABLED` | `true` | Set to `false` only for a localhost-only instance. |
| `AUTH__COOKIE_SECURE` | auto | `Secure` on the session cookie. See note 1. |
| `AUTH__ALLOWED_ORIGINS` | empty | Comma-separated origins that may open the WebSocket. Empty falls back to same-origin. |
| `AUTH__SESSION_MAX_AGE` | `604800` | Cookie lifetime in seconds. |
| `AUTH__MAX_LOGIN_ATTEMPTS` | `10` | Failed logins per 15 minutes before the login page reports a throttle. Read note 4. |

Five things to get right:

1. **TLS.** Leave `AUTH__COOKIE_SECURE` out of the file. Do not write it with
   an empty value: an empty string is not a boolean, and the server stops at
   startup before it can report why. The server then reads
   `X-Forwarded-Proto` for each login and marks the cookie `Secure` under
   HTTPS only. This matters because a browser never sends a `Secure` cookie
   back over plain HTTP: pin it to `true` without TLS and the login appears to
   succeed, then every page bounces back to `/login`. Set it to `true` to
   demand HTTPS. On plain HTTP the password crosses the network in the clear,
   whatever this key says.
2. **The origin list.** `AUTH__ALLOWED_ORIGINS` must hold the origin the browser
   shows, not `127.0.0.1:7000`. An empty list falls back to same-origin, which
   works only while the proxy passes the `Host` header through unchanged. If
   the proxy rewrites `Host`, the UI loads and then reports `Disconnected`.
   The log names the refused origin, so check journalctl. Name the origin here
   and the check no longer depends on the proxy.
3. **The cookie key.** Leave `AUTH__SECRET_KEY` unset and the server mints a
   new key at every start. The service restarts on each deploy, so every open
   tab loses its session and jumps to `/login` — in the middle of a running
   job. Set a fixed key to keep sessions across deploys.
4. **The password must be long and random.** The login throttle counts failures
   per client address, but uvicorn runs without `--proxy-headers`, so behind a
   reverse proxy every caller arrives as the proxy address and shares one
   counter. The server therefore checks the password before the throttle and
   never refuses a correct one. The other order would let ten wrong guesses
   from anywhere on the internet lock out the whole team for 15 minutes. The
   cost is that the throttle slows guessing but does not stop it. Use 20 or
   more random characters, not a memorable phrase. Failed logins go to
   `logs/app.log` under the deploy directory, not to journalctl — that file is
   the only record of somebody guessing.
5. **Keep the clear-text key out of the deployment.** `AUTH__PASSWORD` still
   works, for local runs that do not want a generation step. On a server it
   puts the password in a file, which is what the digest exists to avoid. The
   digest wins when both are set, and the server prints a warning at startup
   while the clear-text key is in use. The digest uses PBKDF2-HMAC-SHA256 over
   200000 rounds. That is below the usual recommendation on purpose: against
   20 random characters the stretching adds almost nothing, and the digest is
   there so a leaked file carries no password.

What this does not do:

- It does not isolate users from each other. `user_id` still comes from the
  caller, so anyone who logs in can read any session.
- `POST /api/settings` still changes settings for everybody.
- Anyone who logs in reaches `/alembic`, which builds arbitrary git
  repositories. Set `ALEMBIC_WEB_CONTROLS=0` in the unit file unless somebody
  is using it.
- It does not stop a determined password-guessing attack. Read note 4.

To rotate the password, run `deploy/make_password_hash.py` again, replace
`AUTH__PASSWORD_HASH` and restart. This also logs everybody out: the cookie
signing key mixes in the stored digest, so every outstanding cookie stops
verifying. Re-running the command over the same password has the same effect,
because the salt is new each time. Moving from `AUTH__PASSWORD` to
`AUTH__PASSWORD_HASH` logs everybody out once, for the same reason. Changing
`AUTH__SECRET_KEY` and restarting also does it, without touching the
password.

## Artifact links on a cluster

The web UI rewrites S3 links in reports to local `/api/artifact/` links. Each
request to that link mints a fresh presigned URL. The links then work for
every user, and they do not expire. Use this checklist on a cluster.

1. Set `S3__EXTERNAL_ENDPOINT_URL` in `~/.config/coscientist/.env` to the S3
   address that the users' browsers can reach. Example:
   `https://s3.example.org`. The server signs presigned URLs for this host.
2. Add the MinIO host to `NO_PROXY` in `coscientist-web.service`. The unit
   sets a global `HTTP_PROXY`. A presigned URL request must not go through
   that proxy.
3. Keep `S3__ENDPOINT_URL` on the internal address. The agents use it for
   uploads and reads inside the cluster.
4. Legacy reports carry plain unsigned URLs for `permanent/` objects. These
   URLs need an anonymous download policy. Run
   `mc anonymous set download <alias>/<bucket>/permanent` once. New reports
   do not need this policy. The new endpoint mints a signed URL for each
   click.

Restart the service after a change to `.env` or to the unit file.

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
