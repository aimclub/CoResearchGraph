"""Artifact matching, normalisation, and durable-evidence policy."""
from __future__ import annotations

import copy
import html
import logging
import mimetypes
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from uuid import uuid4

from CoScientist.config import get_settings
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.shared import artifact_name_key, schema_offers_s3_upload
from CoScientist.experiments.reporting.models import artifact_name_from_location
from CoScientist.experiments.schemas import (
    ArtifactRef,
    CriterionCheck,
    ExecutionRoute,
    ExperimentTask,
    is_presigned_url,
    utc_now,
)

logger = logging.getLogger(__name__)

ARTIFACT_KEYS = ("mcp_artifacts", "fedot_artifacts", "coder_artifacts")
_DATA_ROLES = frozenset({"data", "model", "mcp_server"})
_REPORT_EVIDENCE_ROLES = frozenset({"data", "report", "log"})
EVIDENCE_AGENT_ROUTES = frozenset({
    ExecutionRoute.RESEARCH.value,
    ExecutionRoute.MEDICAL.value,
})
_ATTESTABLE_CRITERION_KINDS = frozenset({"execution", "artifact_exists", "schema"})
_UNKNOWN_TOOL_SNIPPET = "do not exist — they are not in your tool list"
_TABULAR_MEDIA = frozenset({
    "text/csv", "text/tab-separated-values", "text/plain", "application/json",
})


def task_requires_managed_s3(task: ExperimentTask) -> bool:
    return any(
        schema_offers_s3_upload(tool.input_schema)
        for server in task.mcp_servers
        for tool in server.tools
    )


def _artifact_suffix(name: str) -> str:
    return Path(str(name)).suffix.lower()


def artifact_extension_compatible(captured: str, expected: str) -> bool:
    """Suffixes agree, expected has no suffix, or both look tabular."""
    c_suf, e_suf = _artifact_suffix(captured), _artifact_suffix(expected)
    if not e_suf or not c_suf or c_suf == e_suf:
        return True
    tabular = {".csv", ".tsv", ".txt"}
    return c_suf in tabular and e_suf in tabular


def match_expected_artifact(
    name: str,
    expected: list[Any],
    *,
    role: str | None = None,
    claimed_names: set[str] | None = None,
) -> Any | None:
    claimed = claimed_names or set()
    pool = [item for item in expected if item.name not in claimed]
    if (exact := next((item for item in pool if item.name == name), None)) is not None:
        return exact
    if key := artifact_name_key(name):
        soft = [item for item in pool if artifact_name_key(item.name) == key]
        if len(soft) == 1:
            return soft[0]
    role_key = (role or "data").strip().lower()
    same_role = [
        item for item in pool
        if str(getattr(item, "role", "data") or "data").strip().lower() == role_key
    ]
    compatible = [item for item in same_role if artifact_extension_compatible(name, item.name)]
    if len(compatible) == 1:
        return compatible[0]
    if role_key == "data" and len(same_role) == 1:
        # Last resort, and deliberately loose about the suffix: a planner names
        # an expected artifact before the tool runs, so `generated_molecules.json`
        # routinely arrives as a CSV. Renaming it to the name the plan asked for
        # is right.
        #
        # But a PICTURE, a PAPER or an ARCHIVE is never the data a plan asked
        # for. A task that failed to produce `dataset_overview.json` had a
        # cluster-map PNG captured in the same attempt; this branch claimed the
        # PNG as that dataset, and the graph then showed an attachment named
        # `dataset_overview.json` that was a figure — belonging to a task whose
        # own summary said the artifact was never obtained.
        if not _crosses_media_boundary(name, same_role[0].name):
            return same_role[0]
    return None


#: What a file *is*, coarsely. Renaming within a family is the whole point of
#: the branch above — a planner names `generated_molecules.json` before the tool
#: runs and a CSV comes back. Renaming ACROSS one is always a lie.
#:
#: The image family alone used to be guarded, which left the same bug one
#: category over: a Semantic Scholar PDF that a literature search pulled in was
#: claimed as the expected `metabolite_smiles.json`, and the research graph then
#: offered a `.json` chip that downloaded a paper. Worse than cosmetic — the
#: rename also let that PDF satisfy a required artifact and pass a success
#: criterion the run had not actually met.
_MEDIA_FAMILIES: dict[str, frozenset[str]] = {
    "image": frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
                        ".bmp", ".avif", ".tif", ".tiff"}),
    "document": frozenset({".pdf", ".doc", ".docx", ".odt", ".ppt", ".pptx",
                           ".rtf", ".epub"}),
    "archive": frozenset({".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z",
                          ".rar"}),
    "model": frozenset({".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".onnx",
                        ".h5", ".hdf5", ".joblib", ".safetensors"}),
}

#: Anything not in a named family. Data is the open category on purpose: a
#: plan's expected artifacts are overwhelmingly data, and a suffix nobody
#: anticipated should behave like `.json`, not like a refusal.
_FAMILY_DATA = "data"


def _media_family(name: str) -> str:
    suffix = _artifact_suffix(name)
    if not suffix:
        return ""
    for family, suffixes in _MEDIA_FAMILIES.items():
        if suffix in suffixes:
            return family
    return _FAMILY_DATA


def _crosses_media_boundary(captured: str, expected: str) -> bool:
    """These two are different kinds of thing, whatever the plan called them.

    An unknown suffix on either side is not a crossing: `output` with no
    extension at all is exactly the case the loose branch exists to serve.
    """
    got, want = _media_family(captured), _media_family(expected)
    if not got or not want:
        return False
    return got != want


def find_artifact(
    runtime: dict[str, Any],
    artifact_ref: str,
    *,
    source_task_id: str | None = None,
) -> dict[str, Any]:
    """Resolve prior-task artifact by ART-* id or expected name (latest wins)."""
    want = str(artifact_ref or "").strip()
    if not want:
        raise ExperimentRuntimeError("artifact_not_found", "Required artifact ref is empty.")

    matches: list[dict[str, Any]] = []
    for result in runtime.get("results") or []:
        if source_task_id and result.get("task_id") != source_task_id:
            continue
        if result.get("status") not in {"success", "partial"}:
            continue
        for artifact in result.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("artifact_id") == want:
                return artifact
            name = str(artifact.get("name") or "")
            if name == want or Path(name).name == Path(want).name:
                matches.append(artifact)
            elif artifact_name_key(name) and artifact_name_key(name) == artifact_name_key(want):
                matches.append(artifact)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[-1]
    raise ExperimentRuntimeError("artifact_not_found", f"Required artifact {want!r} does not exist.")


def captured_delta(state: MutableMapping[str, Any], attempt: dict[str, Any]) -> list[dict[str, Any]]:
    cursor = attempt["artifact_cursor"]
    return [
        copy.deepcopy(item)
        for key in ARTIFACT_KEYS
        for item in (state.get(key) or [])[int(cursor.get(key, 0)):]
        if isinstance(item, dict)
    ]


def _materialize_signed_artifact(url: str, *, task_id: str, attempt_id: str, name: str) -> str | None:
    """Copy URL-only signed output into the report workspace."""
    try:
        import requests
        folder = Path(get_settings().code_exec.workspace_root) / "experiment_artifacts" / task_id / attempt_id
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / (Path(name).name or f"artifact-{uuid4().hex}")
        response = requests.get(html.unescape(url), timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return str(destination)
    except Exception:
        return None


def normalise_artifacts(
    raw_artifacts: list[dict[str, Any]],
    *,
    runtime: dict[str, Any],
    task_runtime: dict[str, Any],
    attempt: dict[str, Any],
    state: Any = None,
) -> tuple[list[ArtifactRef], list[str]]:
    task = ExperimentTask.model_validate(task_runtime["task"])
    expected = task.expected_artifacts
    artifacts, warnings = [], []
    seen: set[tuple[Any, ...]] = set()
    claimed_names: set[str] = set()

    for raw in raw_artifacts:
        name = artifact_name_from_location(raw)
        role_hint = raw.get("role") or "data"
        match = match_expected_artifact(
            name, expected, role=str(role_hint), claimed_names=claimed_names
        )
        role = match.role if match else role_hint
        if match is not None:
            name = match.name
            claimed_names.add(match.name)

        bucket = raw.get("bucket") or raw.get("bucket_name")
        s3_key = raw.get("s3_key") or raw.get("results_s3_key")
        workspace_path = raw.get("workspace_path")
        external_url = raw.get("external_url") or raw.get("url")
        durability = raw.get("durability")

        if s3_key and not bucket:
            bucket = get_settings().s3.bucket_name

        if bucket and s3_key:
            external_url, durability = None, "managed"
            location_key = ("s3", bucket, s3_key)
        elif workspace_path:
            external_url, durability = None, durability or "workspace"
            location_key = ("workspace", workspace_path)
        elif external_url and is_presigned_url(external_url):
            if not (materialized := _materialize_signed_artifact(
                str(external_url), task_id=task.id, attempt_id=attempt["attempt_id"], name=name,
            )):
                warnings.append(f"Could not materialize signed artifact {name!r}.")
                continue
            workspace_path, external_url, durability = materialized, None, "workspace"
            location_key = ("workspace", workspace_path)
        elif external_url:
            durability = durability or "transient"
            location_key = ("external", str(external_url))
        else:
            warnings.append(f"Captured artifact {name!r} has no canonical location.")
            continue

        if location_key in seen:
            continue
        seen.add(location_key)

        # Mirror first, then believe the store about the name. Its ids are
        # content-addressed and first-wins, so identical bytes already captured
        # under another name come back as that record — and if we kept the
        # plan's name here, every consumer downstream (the media type below, the
        # GeneratedData description, the graph chip's label) would go on
        # describing a file by a name that is not the one being served.
        mirrored = _mirror_record_to_session(
            state, name=name, workspace_path=workspace_path,
            bucket=bucket if s3_key else None, s3_key=s3_key,
            external_url=external_url, tool=raw.get("producer_tool") or raw.get("tool"),
        )
        stored_name = str((mirrored or {}).get("filename") or "")
        if stored_name and stored_name != name:
            warnings.append(
                f"Artifact {name!r} was already stored as {stored_name!r}; "
                "using the stored name."
            )
            logger.warning(
                "experiment artifact %r resolved to a stored copy named %r",
                name, stored_name,
            )
            name = stored_name

        artifacts.append(
            ArtifactRef(
                artifact_id=raw.get("artifact_id") or f"ART-{uuid4().hex}",
                plan_id=runtime["plan_id"],
                task_id=task.id,
                attempt_id=attempt["attempt_id"],
                role=role,
                name=name,
                bucket=bucket if s3_key else None,
                s3_key=s3_key,
                workspace_path=workspace_path,
                external_url=external_url,
                media_type=raw.get("media_type") or mimetypes.guess_type(name)[0],
                size_bytes=raw.get("size_bytes"),
                checksum_sha256=raw.get("checksum_sha256"),
                producer_route=attempt["route"],
                producer_tool=raw.get("producer_tool") or raw.get("tool"),
                derived_from=raw.get("derived_from") or [],
                created_at=utc_now(),
                durability=durability,
                session_artifact_id=(mirrored or {}).get("artifact_id"),
            )
        )
    return artifacts, warnings


def _mirror_to_session(
    state: Any, *, name: str, workspace_path: Any, bucket: Any, s3_key: Any,
    external_url: Any, tool: Any,
) -> str | None:
    """The id of our own copy of one experiment artifact.

    Thin wrapper over :func:`_mirror_record_to_session` for callers that only
    need the reference.
    """
    record = _mirror_record_to_session(
        state, name=name, workspace_path=workspace_path, bucket=bucket,
        s3_key=s3_key, external_url=external_url, tool=tool,
    )
    return (record or {}).get("artifact_id")


def _mirror_record_to_session(
    state: Any, *, name: str, workspace_path: Any, bucket: Any, s3_key: Any,
    external_url: Any, tool: Any,
) -> dict[str, Any] | None:
    """Take our own copy of one experiment artifact, and return its record.

    Everything above settles on a *canonical location*, and for most experiment
    output that location is a path on the machine that ran the task. A path is
    not something a reader can open: the research graph showed
    ``dataset_overview.json`` as an attachment whose href was
    ``D:\\projects26\\...``, which no browser will follow. Copying the bytes into
    the session's own store is what turns that row into a link — and it is the
    same copy the report collector and an exported bundle then use.

    The record, not just the id, because the store is content-addressed and
    first-wins: the same bytes mirrored earlier under another name come back as
    *that* record, and the id then ends in that name's extension. The caller has
    to be able to see the name it actually got, or it goes on believing the
    artifact is called whatever the plan asked for — which is how a graph chip
    came to read ``metabolite_smiles.json`` while serving a PDF.

    Best-effort: a failure here leaves the artifact exactly as it was.
    """
    try:
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )
        from CoScientist.reporting.mirror import mirror_artifact

        user_id = str((state or {}).get(GRAPH_SCOPE_USER_KEY) or "")
        session_id = str((state or {}).get(GRAPH_SCOPE_SESSION_KEY) or "")
        if not (user_id and session_id):
            return None
        record = mirror_artifact(
            user_id=user_id, session_id=session_id,
            path=workspace_path, bucket=bucket, s3_key=s3_key,
            url=external_url, filename=name, label=name,
            tool=str(tool or ""), source_kind="experiment",
        )
        if record.get("state") == "stored":
            return record
    except Exception as exc:  # noqa: BLE001 — the artifact is still usable
        logger.warning("experiment artifact mirror failed for %s (%s)", name, exc)
    return None
def synthesize_mcp_server_artifacts(
    task: ExperimentTask,
    artifacts: list[ArtifactRef],
    *,
    mcp_url: str,
    runtime: dict[str, Any],
    attempt: dict[str, Any],
) -> list[ArtifactRef]:
    """Give a served Alembic MCP the artifact names the planner promised.

    A build task's expected artifacts of role ``mcp_server`` (or named
    ``mcp_endpoint``/``mcp_url``) exist as soon as the server answers, but the
    executor reports an address, not a file. Downstream tasks reference these
    names as ``task_artifact`` inputs; without an ArtifactRef carrying the name,
    ``find_artifact`` fails and readiness blocks the consumer while its producer
    is already done. One transient artifact per unclaimed name, pointing at the
    served URL, keeps the plan's names resolvable.
    """
    url = str(mcp_url or "").strip()
    if not url.startswith("http"):
        return artifacts
    claimed = {str(a.name) for a in artifacts}
    extra: list[ArtifactRef] = []
    for item in task.expected_artifacts:
        if item.role != "mcp_server" and item.name not in {"mcp_endpoint", "mcp_url"}:
            continue
        if item.name in claimed:
            continue
        extra.append(
            ArtifactRef(
                artifact_id=f"ART-{uuid4().hex}",
                plan_id=runtime["plan_id"],
                task_id=task.id,
                attempt_id=attempt["attempt_id"],
                role="mcp_server",
                name=item.name,
                external_url=url,
                media_type="application/json",
                producer_route=attempt["route"],
                producer_tool="build_mcp_server",
                created_at=utc_now(),
                durability="transient",
            )
        )
        claimed.add(item.name)
    return [*artifacts, *extra]


def artifact_exists(artifact: ArtifactRef) -> bool:
    return bool(artifact.bucket and artifact.s3_key) or (
        Path(artifact.workspace_path).is_file() if artifact.workspace_path else bool(artifact.external_url)
    )


def _media_compatible(artifact: ArtifactRef, expected: Any) -> bool:
    got = str(artifact.media_type or "").split(";", 1)[0].strip().lower()
    want = str(getattr(expected, "media_type", None) or "").split(";", 1)[0].strip().lower()
    if not got or not want:
        return True
    if got == want:
        return True
    return got in _TABULAR_MEDIA and want in _TABULAR_MEDIA


def _artifact_family_compatible(artifact: ArtifactRef, expected: Any) -> bool:
    got_role = str(artifact.role or "data")
    want_role = str(getattr(expected, "role", "data") or "data")
    if got_role != want_role and not (got_role in _DATA_ROLES and want_role in _DATA_ROLES):
        return False
    return artifact_extension_compatible(artifact.name, expected.name) or _media_compatible(
        artifact, expected
    )


def artifact_matches(artifact: ArtifactRef, expected: Any) -> bool:
    if artifact.name == expected.name:
        return True
    if artifact_name_key(artifact.name) == artifact_name_key(expected.name):
        return True
    return artifact_exists(artifact) and _artifact_family_compatible(artifact, expected)


def is_durable_artifact(artifact: ArtifactRef) -> bool:
    if artifact.bucket and artifact.s3_key:
        return True
    if artifact.workspace_path and Path(artifact.workspace_path).is_file():
        return True
    return bool(artifact.external_url)


def has_durable_family_evidence(
    task: ExperimentTask,
    artifacts: list[ArtifactRef],
    *,
    route: str,
    outputs: Mapping[str, Any] | None,
) -> bool:
    """Accept when this attempt produced durable family evidence (not planner names)."""
    if route == ExecutionRoute.ALEMBIC_BUILD.value:
        from CoScientist.experiments.runtime.alembic_bridge import extract_mcp_url

        return bool(extract_mcp_url(outputs if isinstance(outputs, dict) else {}))
    # If any artifact has S3 or workspace/external URL, or outputs has results URL or molecules
    if artifacts and any(is_durable_artifact(a) for a in artifacts):
        return True
    if outputs and (outputs.get("results_url") or outputs.get("molecules_generated")):
        return True
    if task_requires_managed_s3(task):
        return any(bool(a.bucket and a.s3_key) for a in artifacts)
    roles = _REPORT_EVIDENCE_ROLES if route in EVIDENCE_AGENT_ROUTES else _DATA_ROLES
    durable = [
        a for a in artifacts
        if is_durable_artifact(a) and str(a.role or "data") in roles
    ]
    return bool(durable)


def route_response_text(state: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    chunks: list[str] = [str(result.get("summary") or "").strip()]
    last = state.get("experiment_last_route_response") if isinstance(state, Mapping) else None
    if isinstance(last, str):
        chunks.append(last.strip())
    elif isinstance(last, Mapping):
        for key in ("summary", "message", "text", "answer"):
            if last.get(key):
                chunks.append(str(last.get(key)).strip())
        content = last.get("content")
        if isinstance(content, Mapping) and content.get("text"):
            chunks.append(str(content.get("text")).strip())
        elif isinstance(content, str):
            chunks.append(content.strip())
    elif last is not None:
        text = str(last).strip()
        if text and text not in {"None", "{}"}:
            chunks.append(text[:8000])
    return "\n\n".join(part for part in chunks if part)


def append_notes_artifact(
    *,
    task: ExperimentTask,
    attempt: Mapping[str, Any],
    raw_artifacts: list[dict[str, Any]],
    text: str,
) -> None:
    from CoScientist.experiments.runtime.inline_artifacts import _write_artifact

    if not text.strip() or _UNKNOWN_TOOL_SNIPPET in text:
        return
    name = next(
        (
            str(item.name)
            for item in task.expected_artifacts
            if str(item.role or "") in _REPORT_EVIDENCE_ROLES
        ),
        f"{task.id.lower()}-notes.md",
    )
    if any(str(item.get("name") or "") == name for item in raw_artifacts):
        return
    written = _write_artifact(
        task_id=str(task.id),
        attempt_id=str(attempt.get("attempt_id") or "attempt"),
        name=name,
        media_type="text/markdown",
        payload=text.strip().encode("utf-8"),
        producer_tool="record_result_route_notes",
    )
    if written:
        written["role"] = "report"
        raw_artifacts.append(written)


def attest_durable_criteria(
    task: ExperimentTask, checks: list[CriterionCheck],
) -> list[CriterionCheck]:
    """Pass execution/artifact/schema criteria when durable family evidence exists."""
    by_id = {check.criterion_id: check for check in checks}
    out: list[CriterionCheck] = []
    seen: set[str] = set()
    for crit in task.success_criteria:
        cid = crit.criterion_id
        existing = by_id.get(cid)
        kind = str(crit.kind or "execution")
        if kind in _ATTESTABLE_CRITERION_KINDS:
            out.append(CriterionCheck.model_validate({
                "criterion_id": cid,
                "passed": True,
                "observed": existing.observed if existing is not None else True,
                "details": (
                    (existing.details if existing is not None else "")
                    or "attested via durable family evidence"
                ),
            }))
            seen.add(cid)
            continue
        if existing is not None:
            out.append(existing)
            seen.add(cid)
    for check in checks:
        if check.criterion_id not in seen:
            out.append(check)
    return out


def runtime_has_durable_data_evidence(runtime: Mapping[str, Any], task_id: str) -> bool:
    """Prior TaskResults already hold S3/file evidence (not alembic mcp_url)."""
    for result in runtime.get("results") or []:
        if not isinstance(result, dict) or result.get("task_id") != task_id:
            continue
        for raw in result.get("artifacts") or []:
            if not isinstance(raw, dict):
                continue
            if raw.get("bucket") and raw.get("s3_key"):
                return True
            path = raw.get("workspace_path")
            if path and Path(str(path)).is_file():
                return True
    return False


def _evidence_expected_artifacts(task: ExperimentTask, *, route: str) -> list:
    required = [item for item in task.expected_artifacts if item.required]
    if route != ExecutionRoute.ALEMBIC_BUILD.value:
        return required
    return [
        item for item in required
        if item.role == "mcp_server" or item.name in {"mcp_endpoint", "mcp_url"}
    ]


def _evidence_success_criteria(task: ExperimentTask, *, route: str) -> list:
    required = [item for item in task.success_criteria if item.required]
    if route != ExecutionRoute.ALEMBIC_BUILD.value:
        return required
    return [item for item in required if item.kind == "execution"]


def required_artifacts_present(
    task: ExperimentTask,
    artifacts: list[ArtifactRef],
    *,
    route: str | None = None,
) -> tuple[bool, list[str]]:
    missing, claimed = [], set()
    expected_items = (
        _evidence_expected_artifacts(task, route=route)
        if route is not None
        else [item for item in task.expected_artifacts if item.required]
    )
    for expected in expected_items:
        hit = next(
            (
                a
                for a in artifacts
                if a.artifact_id not in claimed and artifact_exists(a) and artifact_matches(a, expected)
            ),
            None,
        )
        if hit is None:
            missing.append(expected.name)
        else:
            claimed.add(hit.artifact_id)
    return not missing, missing


def criteria_valid(
    task: ExperimentTask,
    checks: list[CriterionCheck],
    *,
    route: str | None = None,
) -> tuple[bool, list[str]]:
    by_id = {check.criterion_id: check for check in checks}
    if unknown := set(by_id) - {c.criterion_id for c in task.success_criteria}:
        raise ExperimentRuntimeError("criterion_unknown", f"Unknown criterion checks: {sorted(unknown)}.")
    required = (
        _evidence_success_criteria(task, route=route)
        if route is not None
        else [c for c in task.success_criteria if c.required]
    )
    failed = [
        c.criterion_id
        for c in required
        if c.criterion_id not in by_id or by_id[c.criterion_id].passed is not True
    ]
    return not failed, failed
