"""MCP hub: alembic servers shared as images in a public Docker Hub namespace.

Uploads use DOCKERHUB_USERNAME and DOCKERHUB_TOKEN from the project's .env only;
DOCKERHUB_NAMESPACE (default: the username) picks the namespace. The source
repository, tools and validation go into the repository description, so the hub
is browsed without pulling images. A pulled image becomes a build with origin "hub".
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values

from CoScientist.tools import alembic_tools as at

logger = logging.getLogger(__name__)

ENV_FILE = at.PROJECT_ROOT / ".env"
HUB_API = "https://hub.docker.com/v2"
REPO_PREFIX = "alembic-tool-"
_HTTP_TIMEOUT = 15
_PULL_PUSH_TIMEOUT = 3600
_LIST_CACHE_SECONDS = 60
_SHORT_MAX = 100       # Docker Hub limit for the short description
_FULL_MAX = 25000      # Docker Hub limit for the full description
_NAME_RE = re.compile(r"^alembic-tool-[a-z0-9][a-z0-9._-]*$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_META_RE = re.compile(r"<!-- alembic-hub (\{.*?\}) -->", re.S)
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
# Set by the official python base image: the public id of the Python release signing key.
_ENV_ALLOWED = {"GPG_KEY"}
# Published files that may hold what the build agents printed, or secrets.
_FORBIDDEN_FILES_SCRIPT = (
    "find /work \\( -name .venv -o -name .server_venv -o -name node_modules -o -name .git \\) "
    "-prune -o -type f \\( -name pipeline.log -size +0c -o -name .env \\) -print"
)
_REPO_DIR_SCRIPT = 'for d in /work/.alembic/*/; do [ -f "$d/output/server.py" ] && basename "$d"; done'
# The tools a generated server exposes, read from its source: images built before
# the reports carried a plan have nothing else naming them.
_TOOL_RE = re.compile(r"@mcp\.tool\([^)]*\)\s*(?:async\s+)?def\s+(\w+)\s*\(.*?\)[^:]*:\s*(?:[ru]?\"\"\"(.*?)\"\"\")?", re.S)
_GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+")
# URLError, HTTPError and TimeoutError are OSErrors; a bad JSON body is a ValueError.
_HUB_ERRORS = (OSError, ValueError)

_run = subprocess.run  # replaced in tests


@dataclass(frozen=True)
class HubConfig:
    namespace: str
    username: str = ""
    token: str = ""

    @property
    def can_upload(self) -> bool:
        return bool(self.username and self.token)

    @classmethod
    def load(cls) -> Optional["HubConfig"]:
        """The hub settings from .env; None when it names no namespace or user."""
        values = dotenv_values(ENV_FILE)
        username = (values.get("DOCKERHUB_USERNAME") or "").strip()
        namespace = (values.get("DOCKERHUB_NAMESPACE") or username).strip().lower()
        if not namespace:
            return None
        return cls(namespace, username, (values.get("DOCKERHUB_TOKEN") or "").strip())


def search_enabled() -> bool:
    """Whether build_mcp_server looks for a server on the hub before building."""
    return at._web_flag("alembic_hub_search_enabled", True)


def auto_upload_enabled() -> bool:
    """Whether every successful build is uploaded to the hub."""
    return at._web_flag("alembic_hub_auto_upload", False)


def hub_config() -> Dict[str, Any]:
    cfg = HubConfig.load()
    return {"configured": cfg is not None, "namespace": cfg.namespace if cfg else None,
            "can_upload": bool(cfg and cfg.can_upload), "search_enabled": search_enabled(),
            "auto_upload": auto_upload_enabled()}


# ── names and descriptions ───────────────────────────────────────────────────

def repository_name(repo_url: str) -> str:
    """The Docker Hub repository of a converted repository: alembic-tool-<repo>, lowercase."""
    name = re.sub(r"[^a-z0-9._-]+", "-", at._repo_name(repo_url).lower()).strip("-._")
    return REPO_PREFIX + (name or "tool")


def build_tag(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", job_id)[:128]


def normalize_repo_url(url: str) -> str:
    url = (url or "").strip().lower().rstrip("/")
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url[:-4] if url.endswith(".git") else url


# The short description is all a reader sees in a hub listing, so it says what
# the server does. The explorer already wrote that sentence in its report.
_MARKUP_RE = re.compile(r"[`*_#]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s")
_ASIDE_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
# "a/an/the ..." or "this <noun> ..." at the start.
_LEAD_RE = re.compile(r"^(?:a|an|the)\s+|^this\s+\w+\s+", re.I)
# "mordred is a molecular descriptor calculator" -> "molecular descriptor calculator";
# "AutoGen is Microsoft's framework" -> "Microsoft's framework".
_COPULA_RE = re.compile(r"^(?:\S+\s+)?(?:is|are|was|were)\s+(?:a|an|the)\s+"
                        r"|^(?:\S+\s+)?(?:is|are|was|were)\s+", re.I)
_SUMMARY_WORDS = 15
_DESCRIPTION_RE = re.compile(r"^##\s+Description\s*\n(.+?)(?=\n##\s|\Z)", re.M | re.S | re.I)
_DANGLING = {"a", "an", "the", "and", "or", "of", "for", "with", "from", "to", "in", "on",
             "by", "at", "into", "over", "around", "using", "that", "which", "as", "its",
             "given", "each", "such", "some", "any", "one", "this", "these", "those",
             "their", "based", "via", "per", "between", "across", "now", "still", "when",
             "while", "where", "who", "both"}


def summarise_repository(reports: Path, repo_name: str) -> Optional[str]:
    """What a converted repository does, from its exploration report.

    One sentence of at most _SUMMARY_WORDS words and _SHORT_MAX characters, so
    the same line fits the Docker Hub short description, the hub listing and the
    builds list. None when the report says nothing usable.
    """
    try:
        text = (reports / "exploration.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = _DESCRIPTION_RE.search(text)
    if not found:
        return None
    body = found.group(1).strip()
    if not body:
        return None
    # An aside in brackets ("(PyPI: biopsykit, v0.13.2)") eats half the words and
    # says nothing about what the server does.
    plain = _ASIDE_RE.sub(" ", _MARKUP_RE.sub("", body).replace("\n", " "))
    first = _SENTENCE_RE.split(" ".join(plain.split()))[0]
    # The name is dropped on a word boundary, or "NeuroKit" left "2 is a ..." of
    # "NeuroKit2 is a ...". The copula rule then covers a name the report spells
    # its own way ("nnU-Net is a ..." for the repository nnUNet).
    first = re.sub(rf"^{re.escape(repo_name)}\d*\b[:,]?\s*", "", first.strip(), flags=re.I)
    first = _COPULA_RE.sub("", first.strip())
    first = _LEAD_RE.sub("", first.strip())
    words = _trim(first.split()[:_SUMMARY_WORDS])
    while words and len(" ".join(words)) > _SHORT_MAX:
        words = _trim(words[:-1])
    if len(words) < 2:
        return None
    phrase = " ".join(words).strip(" ,;:.")
    return phrase[0].upper() + phrase[1:]


def _trim(words: List[str]) -> List[str]:
    """``words`` without a tail that asks for a continuation ("... based on")."""
    while len(words) > 2 and words[-1].lower().strip(",") in _DANGLING:
        words.pop()
    return words


def _first_line(text: Any) -> str:
    return (str(text or "").strip().splitlines() or [""])[0][:200]


def _escape_pipes(text: Any) -> str:
    """A table cell; the text itself is stored unescaped."""
    return _first_line(text).replace("|", "\\|")


def render_description(namespace: str, repository: str, meta: Dict[str, Any]) -> Tuple[str, str]:
    """(short, full) Docker Hub descriptions: markdown for people, and the
    metadata as JSON in an HTML comment for the hub page and the agent."""
    repo_url = meta.get("repo_url") or ""
    source = normalize_repo_url(repo_url).removeprefix("github.com/") or "an unknown repository"
    tools = meta.get("tools") or []
    counts = meta.get("tool_counts") or {}
    short = (meta.get("summary") or f"MCP server for {source}")[:_SHORT_MAX]

    def full_text(meta_out: Dict[str, Any], with_descriptions: bool) -> str:
        lines = [f"# {repository}", "",
                 f"MCP server generated by Alembic (CoScientist) from {repo_url or 'an unknown repository'}.", ""]
        if counts.get("tools_total"):
            lines += [f"Tools that passed validation: {counts.get('tools_passed', 0)} of "
                      f"{counts['tools_total']}.", ""]
        if tools:
            lines += ["## Tools", ""]
            if any(t.get("status") for t in tools):
                lines += ["A tool marked `passed` ran its generated tests; `perfect` also answered "
                          "every invocation example correctly. `failed` and `untested` tools are "
                          "served too, and may or may not work.", ""]
            if with_descriptions:
                lines += ["| Tool | Validation | Description |", "|---|---|---|"]
                lines += [f"| `{t.get('name')}` | {_verdict_cell(t)} | {_escape_pipes(t.get('description'))} |"
                          for t in tools]
            else:
                lines += [f"- `{t.get('name')}` {_verdict_cell(t)}" for t in tools]
            lines.append("")
        lines += ["## Run", "",
                  "Pull and start it from the MCP hub page of CoScientist, or serve it yourself:", "",
                  "```", f"docker run -d -p 8000:8000 {namespace}/{repository}:latest serve {repo_url}",
                  "```", "", "The server answers MCP at http://localhost:8000/mcp.", ""]
        # ">" is escaped inside the JSON so no description can close the comment.
        blob = json.dumps(meta_out, ensure_ascii=False).replace(">", "\\u003e")
        lines.append(f"<!-- alembic-hub {blob} -->")
        return "\n".join(lines)

    full = full_text(meta, True)
    if len(full) > _FULL_MAX:
        slim = {**meta, "tools": [{k: v for k, v in t.items() if k != "description"} for t in tools]}
        full = full_text(slim, False)[-_FULL_MAX:]
    return short, full


def _verdict_cell(tool: Dict[str, Any]) -> str:
    """How a tool validated, for one table cell: its verdict and its invocations."""
    status = tool.get("status")
    if not status:
        return ""
    invoc = tool.get("invoc")
    return f"{status} ({invoc} calls)" if invoc else str(status)


def parse_meta(full_description: str) -> Optional[Dict[str, Any]]:
    found = _META_RE.findall(full_description or "")
    if not found:
        return None
    try:
        meta = json.loads(found[-1])
    except ValueError:
        return None
    return meta if isinstance(meta, dict) else None


# ── Docker Hub API ───────────────────────────────────────────────────────────

def _request(method: str, url: str, *, token: Optional[str] = None,
             body: Optional[Dict[str, Any]] = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _unreachable(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"Docker Hub answered {exc.code} {exc.reason}"
    return f"Docker Hub is not reachable: {exc}"


def list_servers(refresh: bool = False) -> Dict[str, Any]:
    """Every alembic server in the namespace, with the metadata from its description."""
    cfg = HubConfig.load()
    if cfg is None:
        return {"ok": False, "configured": False,
                "error": "no DOCKERHUB_NAMESPACE or DOCKERHUB_USERNAME in .env"}
    with _cache_lock:
        cached = _cache.get(cfg.namespace)
        if cached and not refresh and time.monotonic() - cached[0] < _LIST_CACHE_SECONDS:
            return cached[1]
    try:
        repos: List[Dict[str, Any]] = []
        url: Optional[str] = f"{HUB_API}/repositories/{cfg.namespace}/?page_size=100"
        while url:
            page = _request("GET", url)
            repos += page.get("results") or []
            url = page.get("next")
        repos = [r for r in repos if str(r.get("name", "")).startswith(REPO_PREFIX)]
        # The list leaves out the full description, which holds the metadata.
        with ThreadPoolExecutor(max_workers=8) as pool:
            details = list(pool.map(
                lambda r: _request("GET", f"{HUB_API}/repositories/{cfg.namespace}/{r['name']}/"), repos))
    except _HUB_ERRORS as exc:
        return {"ok": False, "configured": True, "namespace": cfg.namespace, "error": _unreachable(exc)}
    servers = [{
        "name": repo["name"],
        "image": f"{cfg.namespace}/{repo['name']}",
        "description": repo.get("description") or "",
        "last_updated": repo.get("last_updated"),
        "pull_count": repo.get("pull_count"),
        "meta": parse_meta(detail.get("full_description") or ""),
    } for repo, detail in zip(repos, details)]
    servers.sort(key=lambda s: s.get("last_updated") or "", reverse=True)
    listing = {"ok": True, "configured": True, "namespace": cfg.namespace,
               "can_upload": cfg.can_upload, "servers": servers}
    with _cache_lock:
        _cache[cfg.namespace] = (time.monotonic(), listing)
    return listing


def server_tags(name: str) -> Dict[str, Any]:
    cfg = HubConfig.load()
    if cfg is None or not _NAME_RE.match(name or ""):
        return {"ok": False, "error": "unknown hub server"}
    try:
        page = _request("GET", f"{HUB_API}/repositories/{cfg.namespace}/{name}/tags/?page_size=50")
    except _HUB_ERRORS as exc:
        return {"ok": False, "error": _unreachable(exc)}
    tags = [{"tag": t.get("name"), "last_updated": t.get("last_updated"),
             "size": at._docker_size(t["full_size"]) if isinstance(t.get("full_size"), int) else None}
            for t in page.get("results") or []]
    return {"ok": True, "image": f"{cfg.namespace}/{name}", "tags": tags}


def find_for_repo(repo_url: str) -> Optional[Dict[str, Any]]:
    """The hub server converted from ``repo_url``; None when there is none or the hub does not answer.

    A server whose description names its repository matches by that. One
    without metadata (uploaded by hand) matches by name, and the pull checks
    the repository the image names before serving it.
    """
    listing = list_servers()
    if not listing.get("ok"):
        return None
    want = normalize_repo_url(repo_url)
    servers = listing["servers"]
    by_meta = [s for s in servers if s.get("meta")
               and normalize_repo_url(s["meta"].get("repo_url") or "") == want]
    if by_meta:
        return by_meta[0]
    name = repository_name(repo_url)
    return next((s for s in servers if not s.get("meta") and s["name"] == name), None)


def _hub_token(cfg: HubConfig) -> str:
    """A Docker Hub API token for the credentials in .env."""
    try:
        body = _request("POST", f"{HUB_API}/auth/token",
                        body={"identifier": cfg.username, "secret": cfg.token})
        if body.get("access_token"):
            return body["access_token"]
    except urllib.error.HTTPError:
        pass
    body = _request("POST", f"{HUB_API}/users/login",
                    body={"username": cfg.username, "password": cfg.token})
    return body["token"]


def _write_description(cfg: HubConfig, repository: str, meta: Dict[str, Any]) -> Optional[str]:
    """Write the repository description; a warning when that failed."""
    short, full = render_description(cfg.namespace, repository, meta)
    try:
        token = _hub_token(cfg)
        _request("PATCH", f"{HUB_API}/repositories/{cfg.namespace}/{repository}/", token=token,
                 body={"description": short, "full_description": full})
    except (*_HUB_ERRORS, KeyError) as exc:
        return f"the image is on Docker Hub, but its description was not written: {_unreachable(exc)}"
    return None


# ── upload ───────────────────────────────────────────────────────────────────

def check_image(image: str) -> List[str]:
    """What stops ``image`` from being published: secret-like environment values,
    a non-empty pipeline.log, a .env file. Empty when it can go."""
    problems = []
    r = at._docker("image", "inspect", "-f", "{{json .Config.Env}}", image, timeout=30)
    try:
        env = json.loads(r.stdout) or []
    except ValueError:
        return ["could not read the image's environment"]
    filled = sorted(name for name, _, value in (item.partition("=") for item in env)
                    if _SECRET_NAME.search(name) and name not in _ENV_ALLOWED and value.strip())
    if filled:
        problems.append("environment variables with secret-like values: " + ", ".join(filled))
    r = at._docker("run", "--rm", "--network", "none", "--entrypoint", "sh", image,
                   "-c", _FORBIDDEN_FILES_SCRIPT, timeout=600)
    if r.returncode != 0:
        problems.append("could not look inside the image: " + (r.stderr or r.stdout).strip()[-300:])
    else:
        files = [line for line in r.stdout.splitlines() if line.strip()]
        if files:
            problems.append("files that must not be published: " + ", ".join(files[:10]))
    return problems


def _masked(proc: subprocess.CompletedProcess, token: str) -> str:
    text = ((proc.stderr or "") + (proc.stdout or "")).strip()[-500:]
    return text.replace(token, "***") if token else text


def _push(cfg: HubConfig, image: str, refs: List[str]) -> Optional[str]:
    """Tag and push ``image`` as ``refs``; an error message, or None.

    The login goes into a throwaway docker config, so the user's own
    ~/.docker/config.json is never read or changed.
    """
    config_dir = tempfile.mkdtemp(prefix="alembic-hub-")
    try:
        login = _run(["docker", "--config", config_dir, "login", "--username", cfg.username,
                      "--password-stdin"], input=cfg.token, capture_output=True, text=True,
                     timeout=120, check=False)
        if login.returncode != 0:
            return "docker login failed: " + _masked(login, cfg.token)
        for ref in refs:
            tagged = at._docker("tag", image, ref, timeout=60)
            if tagged.returncode != 0:
                return f"docker tag {ref} failed: " + (tagged.stderr or tagged.stdout).strip()[-300:]
            pushed = _run(["docker", "--config", config_dir, "push", ref], capture_output=True,
                          text=True, timeout=_PULL_PUSH_TIMEOUT, check=False)
            if pushed.returncode != 0:
                return f"docker push {ref} failed: " + _masked(pushed, cfg.token)
        return None
    except (OSError, subprocess.SubprocessError) as exc:
        return f"docker failed: {exc}"
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)


def tool_verdicts(reports: Path) -> Dict[str, Dict[str, Any]]:
    """name -> {status, invoc} for a build's tools; {} when nothing reports them.

    From validation.json, else from the validation report the oldest builds wrote
    as markdown alone.
    """
    try:
        listed = json.loads((reports / "validation.json").read_text(encoding="utf-8"))["tools"]
    except (OSError, ValueError, KeyError, TypeError):
        return _verdicts_from_markdown(reports / "validation.md")
    out: Dict[str, Dict[str, Any]] = {}
    for tool in listed if isinstance(listed, list) else []:
        name = tool.get("name")
        if not name:
            continue
        verdict: Dict[str, Any] = {"status": tool.get("status") or "untested"}
        total = tool.get("invoc_total")
        if isinstance(total, int) and total:
            verdict["invoc"] = f"{tool.get('invoc_passed') or 0}/{total}"
        out[name] = verdict
    return out


# "- **get_spectra_genesets** — PASSED" under "## Tool Invocations", and the table
# row "| reconstruction_error | 3/3 | CRASHED | 2/2 | failed |" of a later format.
_MD_LIST_RE = re.compile(r"^\s*[-*]\s+\**(\w+)\**\s*[—:-]+\s*(PASSED|FAILED)\b", re.M | re.I)
_MD_ROW_RE = re.compile(r"^\|\s*(\w+)\s*\|(?:[^|\n]*\|){1,6}?\s*(perfect|passed|failed|untested)\s*\|",
                        re.M | re.I)


def _verdicts_from_markdown(report: Path) -> Dict[str, Dict[str, Any]]:
    try:
        text = report.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    found = {name: verdict.lower() for name, verdict in _MD_ROW_RE.findall(text)}
    for name, verdict in _MD_LIST_RE.findall(text):
        found.setdefault(name, verdict.lower())
    return {name: {"status": verdict} for name, verdict in found.items()}


def _with_verdicts(tools: List[Dict[str, Any]], reports: Path) -> List[Dict[str, Any]]:
    verdicts = tool_verdicts(reports)
    return [{**tool, **verdicts.get(tool["name"], {})} for tool in tools]


def image_artifacts(workdir: Optional[str], repo_dir: str) -> Dict[str, Any]:
    """What the artifacts copied out of a tool image say about the server.

    ``repo_url``: from the plan, else a link in the reports naming this repository.
    ``tools``: the names and first doc lines in server.py, with their verdicts.
    ``tool_counts``: counted from those verdicts; validation.json counts are read on import.
    """
    out: Dict[str, Any] = {"repo_url": None, "tools": [], "tool_counts": None, "summary": None}
    if not workdir:
        return out
    base = Path(workdir) / repo_dir
    out["repo_url"] = at._plan_repo_url(workdir, repo_dir) or _repo_url_from_reports(base, repo_dir)
    try:
        source = (base / "output" / "server.py").read_text(encoding="utf-8", errors="replace")
        out["tools"] = _with_verdicts([{"name": m.group(1), "description": _first_line(m.group(2))}
                                       for m in _TOOL_RE.finditer(source)], base / "reports")
    except OSError:
        pass
    if any(tool.get("status") for tool in out["tools"]):
        out["tool_counts"] = counts_from_verdicts(out["tools"])
    out["summary"] = summarise_repository(base / "reports", repo_dir)
    return out


def counts_from_verdicts(tools: List[Dict[str, Any]]) -> Dict[str, int]:
    """The counts the builds page shows, for an image whose validation report is
    markdown alone: how many of its tools the validator cleared."""
    statuses = [tool.get("status") for tool in tools]
    return {"tools_total": len(tools),
            "tools_passed": sum(s in ("passed", "perfect") for s in statuses),
            "tools_perfect": statuses.count("perfect")}


def _repo_url_from_reports(base: Path, repo_dir: str) -> Optional[str]:
    """A repository link in the reports whose name is the one in the image. The
    reports also cite other projects, so only a name match counts."""
    for report in sorted(base.glob("reports/*.md")):
        try:
            text = report.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for url in _GITHUB_RE.findall(text):
            if at._repo_name(url).lower() == repo_dir.lower():
                return url
    return None


def build_summary(workdir: Optional[str], repo_url: Optional[str]) -> Optional[str]:
    """summarise_repository for a build's workdir."""
    if not workdir or not repo_url:
        return None
    from CoScientist.alembic.web import artifacts

    return summarise_repository(artifacts.reports_dir(Path(workdir), repo_url), at._repo_name(repo_url))


def _build_meta(job: Dict[str, Any]) -> Dict[str, Any]:
    tools = [{"name": t.get("name"), "description": _first_line(t.get("description"))}
             for t in job.get("tools") or [] if t.get("name")]
    summary = None
    if job.get("workdir"):
        try:  # the description stands without any of these
            from CoScientist.alembic.web import artifacts

            workdir = Path(job["workdir"])
            if not tools:
                tools = [{"name": t["name"], "description": _first_line(t.get("desc") or t.get("description"))}
                         for t in artifacts.build_tools(workdir, job["repo_url"]).get("tools") or []
                         if t.get("name")]
            tools = _with_verdicts(tools, artifacts.reports_dir(workdir, job["repo_url"]))
            summary = build_summary(job["workdir"], job["repo_url"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read the reports of %s: %s", job.get("job_id"), exc)
    counts = job.get("tool_counts") or (
        counts_from_verdicts(tools) if any(t.get("status") for t in tools) else {})
    return {"repo_url": job.get("repo_url"), "job_id": job.get("job_id"), "tools": tools,
            "summary": summary, "tool_counts": counts,
            "uploaded_at": int(time.time())}


def _hub_is_better(repository: str, counts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why the hub's :latest of ``repository`` beats a build with ``counts``, or None.

    Validated tools decide, then perfect ones. A hub that does not answer, or a
    server without counts in its description, never blocks an upload.
    """
    listing = list_servers(refresh=True)
    if not listing.get("ok"):
        return None
    server = next((s for s in listing["servers"] if s["name"] == repository), None)
    theirs = ((server or {}).get("meta") or {}).get("tool_counts") or {}
    if not theirs:
        return None

    def rank(c: Dict[str, Any]) -> tuple:
        return (c.get("tools_passed") or 0, c.get("tools_perfect") or 0)

    if rank(counts or {}) >= rank(theirs):
        return None
    return (f"the hub has a better build of this repository ({rank(theirs)[0]}/{theirs.get('tools_total')} "
            f"tools valid, this one {rank(counts or {})[0]}/{(counts or {}).get('tools_total')})")


def upload_build(job_id: str, keep_better: bool = False) -> Dict[str, Any]:
    """Push a finished build's image to the hub as <ns>/alembic-tool-<repo>:<job_id>
    and :latest, then write the repository description.

    With ``keep_better`` (auto-upload) nothing is pushed when the hub already
    serves a build with more valid tools; the upload button always replaces it.
    """
    cfg = HubConfig.load()
    if cfg is None or not cfg.can_upload:
        return {"ok": False, "error": "uploading needs DOCKERHUB_USERNAME and DOCKERHUB_TOKEN in .env"}
    rec, job = at._job(job_id)
    if job.get("status") != "done" or not job.get("repo_url"):
        return {"ok": False, "error": "only a finished build can be uploaded"}
    image = at.job_image(job, at.docker_inventory())
    if not image:
        return {"ok": False, "error": "this build has no image of its own on this host"}
    repository = repository_name(job["repo_url"])
    better = _hub_is_better(repository, job.get("tool_counts")) if keep_better else None
    if better:
        note = f"not uploaded: {better}; the upload button replaces it"
        at._update_job(job, rec, {"hub": {"status": "skipped", "error": note, "at": time.time()}})
        at._log_event(job, note)
        return {"ok": False, "skipped": True, "error": note}
    problems = check_image(image)
    if problems:
        error = "the image was not uploaded: " + "; ".join(problems)
        at._log_event(job, error)
        return {"ok": False, "error": error}
    tags = [build_tag(job_id), "latest"]
    refs = [f"{cfg.namespace}/{repository}:{tag}" for tag in tags]
    at._log_event(job, f"uploading {image} to Docker Hub as {', '.join(refs)}")
    error = _push(cfg, image, refs)
    if error:
        at._update_job(job, rec, {"hub": {"status": "failed", "error": error, "at": time.time()}})
        at._log_event(job, error)
        return {"ok": False, "error": error}
    warning = _write_description(cfg, repository, _build_meta(job))
    hub = {"status": "uploaded", "image": f"{cfg.namespace}/{repository}", "tags": tags,
           "uploaded_at": time.time()}
    if warning:
        hub["warning"] = warning
    at._update_job(job, rec, {"hub": hub})
    at._log_event(job, f"uploaded to Docker Hub as {', '.join(refs)}" + (f"; {warning}" if warning else ""))
    with _cache_lock:
        _cache.pop(cfg.namespace, None)
    return {"ok": True, **hub}


# ── pull ─────────────────────────────────────────────────────────────────────

def start_pull(name: str, tag: str = "latest", scope: Optional[list] = None,
               repo_url: Optional[str] = None) -> Dict[str, Any]:
    """Pull <ns>/<name>:<tag> and serve it, in the background, as a build record.

    Returns the record's snapshot (status "running"); check_mcp_build and the
    builds page follow it like a build.
    """
    cfg = HubConfig.load()
    if cfg is None:
        return {"status": "error", "error": "no DOCKERHUB_NAMESPACE or DOCKERHUB_USERNAME in .env"}
    if not _NAME_RE.match(name or "") or not _TAG_RE.match(tag or ""):
        return {"status": "error", "error": f"not a hub server: {name}:{tag}"}
    ref = f"{cfg.namespace}/{name}:{tag}"
    job_id = f"{name[len(REPO_PREFIX):]}-hub-{secrets.token_hex(3)}"
    rec: Dict[str, Any] = {
        "job_id": job_id, "repo_url": repo_url or "", "status": "running",
        "started_at": time.time(), "log_file": str(at.LOG_DIR / f"{job_id}.log"),
        "origin": "hub", "scopes": [scope] if scope else [],
        "hub": {"status": "pulling", "image": f"{cfg.namespace}/{name}", "tags": [tag]},
    }
    at.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with at._LOCK:
        at._evict_finished_jobs()
        at._JOBS[job_id] = rec
        snap = at._snapshot(rec, with_log_tail=False)
    at._write_job_meta(rec)
    threading.Thread(target=_pull_runner, args=(rec, ref), daemon=True,
                     name=f"hub-pull-{job_id}").start()
    snap["note"] = (f"This repository is on the MCP hub as {ref}. It is being pulled and "
                    "started instead of a new build; pulling a few GB takes minutes. Track it "
                    f"with check_mcp_build('{job_id}'). If it fails, call build_mcp_server with "
                    "force_rebuild=true to build from source.")
    return snap


def _fail_pull(rec: Dict[str, Any], message: str) -> None:
    with at._LOCK:
        rec.update(status="failed", error=message, finished_at=time.time())
        rec["hub"] = {**rec.get("hub", {}), "status": "failed"}
    at._write_job_meta(rec)
    at._log_event(rec, message)


def _pull_runner(rec: Dict[str, Any], ref: str) -> None:
    try:
        _pull(rec, ref)
    except Exception as exc:  # noqa: BLE001 - the record must not stay "running"
        logger.exception("pulling %s failed", ref)
        if rec.get("status") == "running":
            _fail_pull(rec, f"pulling {ref} failed: {type(exc).__name__}: {exc}")


def _pull(rec: Dict[str, Any], ref: str) -> None:
    job_id = rec["job_id"]

    def fail(message: str) -> None:
        _fail_pull(rec, message)

    at._log_event(rec, f"pulling {ref} from the MCP hub")
    try:
        pulled = _run(["docker", "pull", ref], capture_output=True, text=True,
                      timeout=_PULL_PUSH_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(f"docker pull {ref} failed: {exc}")
    if pulled.returncode != 0:
        return fail(f"docker pull {ref} failed: {(pulled.stderr or pulled.stdout).strip()[-500:]}")
    local_tag = f"{at._TOOL_IMAGE}:{job_id}"
    tagged = at._docker("tag", ref, local_tag, timeout=60)
    if tagged.returncode != 0:
        return fail(f"docker tag {local_tag} failed: {(tagged.stderr or tagged.stdout).strip()[-300:]}")
    image_id, _, labels_out = at._docker("image", "inspect", "-f", "{{.Id}} {{json .Config.Labels}}",
                                         ref, timeout=30).stdout.strip().partition(" ")
    try:
        labels = json.loads(labels_out) or {}
    except ValueError:
        labels = {}
    dirs = at._docker("run", "--rm", "--network", "none", "--entrypoint", "sh", ref,
                      "-c", _REPO_DIR_SCRIPT, timeout=300).stdout.split()
    if not dirs:
        return fail(f"{ref} holds no alembic server (no /work/.alembic/<repo>/output/server.py)")
    repo_dir = dirs[0]
    container = f"alembic-hub-import-{job_id}"
    created = at._docker("create", "--name", container, ref, timeout=120)
    workdir, counts = (at._import_image_artifacts(job_id, container, repo_dir)
                       if created.returncode == 0 else (None, None))
    at._docker("rm", container, timeout=60)
    details = image_artifacts(workdir, repo_dir)
    named = labels.get("alembic.repo_url") or details["repo_url"]
    wanted = rec.get("repo_url")
    if wanted and named and normalize_repo_url(named) != normalize_repo_url(wanted):
        return fail(f"{ref} was converted from {named}, not from {wanted}; it was not served")
    repo_url = wanted or named
    if not repo_url:
        return fail(f"nothing in {ref} names the repository it was converted from; "
                    "pull it again with that repository URL")
    with at._LOCK:
        rec.update(repo_url=repo_url, status="done", finished_at=time.time(), image=local_tag,
                   image_id=image_id or None, workdir=workdir,
                   tool_counts=counts or details["tool_counts"])
        if details["tools"]:
            rec["tools"] = details["tools"]
        rec["hub"] = {**rec["hub"], "status": "pulled", "pulled_at": time.time()}
    at._write_job_meta(rec)
    at._log_event(rec, f"pulled {ref} as {local_tag}")
    started = at.start_build_server(job_id)
    if not started.get("ok"):
        at._log_event(rec, f"the pulled server did not start: {started.get('error')}")
        return
    at._await_answer(rec, started["mcp_url"])


__all__ = ["HubConfig", "auto_upload_enabled", "build_summary", "find_for_repo", "hub_config",
           "list_servers", "normalize_repo_url", "parse_meta", "render_description", "repository_name",
           "search_enabled", "server_tags", "start_pull", "summarise_repository", "tool_verdicts",
           "upload_build"]
