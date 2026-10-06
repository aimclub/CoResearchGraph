"""Strict Pydantic contracts for Experiment Module."""
from __future__ import annotations

import json
import re
from contextvars import ContextVar, Token
from pathlib import PurePosixPath
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.json_schema import WithJsonSchema

# Kept for callers / env (EXPERIMENTS__LENIENT_PLANNER). Design lists are no
# longer invented: empty stays empty; unspecified* names are dropped.
_LENIENT_PLANNER: ContextVar[bool] = ContextVar("experiment_lenient_planner", default=True)

_DESIGN_PLACEHOLDERS = frozenset({
    "comparative reference method",
    "primary_outcome",
    "task dataset",
    "analysis.py",
    "what measurable outcome does this task produce?",
    "operation-specified method",
    "operation inputs",
})


def is_design_placeholder(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return (not text) or text.startswith("unspecified") or text in _DESIGN_PLACEHOLDERS


def set_lenient_planner(enabled: bool) -> Token:
    return _LENIENT_PLANNER.set(bool(enabled))


def reset_lenient_planner(token: Token) -> None:
    _LENIENT_PLANNER.reset(token)


def lenient_planner_enabled() -> bool:
    return _LENIENT_PLANNER.get()

_SIGNED_QUERY_MARKERS = (
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-signature",
    "x-amz-signedheaders",
)
_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_TASK_ID_RE = re.compile(r"EXP-[1-9][0-9]*")


def _coerce_json_object(value: Any) -> dict[str, Any]:
    """Accept a dict or a JSON-object string (OpenAI structured-output safe)."""
    if not value:
        return {}
    if isinstance(value, str):
        if not (text := value.strip()):
            return {}
        if isinstance(parsed := json.loads(text), dict):
            return parsed
        raise ValueError("JSON object string must decode to an object")
    if isinstance(value, dict):
        return value
    raise ValueError("expected a JSON object or object-encoded string")


def _coerce_optional_json_object(value: Any) -> dict[str, Any] | None:
    return None if value is None or (isinstance(value, str) and not value.strip()) else _coerce_json_object(value)


def _json_object_schema(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


JsonObjectDict = Annotated[
    dict[str, Any],
    BeforeValidator(_coerce_json_object),
    WithJsonSchema(_json_object_schema('Arbitrary JSON object encoded as a string, e.g. "{\\"case\\":\\"cancer\\"}".')),
]
OptionalJsonObjectDict = Annotated[
    dict[str, Any] | None,
    BeforeValidator(_coerce_optional_json_object),
    WithJsonSchema({
        "anyOf": [
            _json_object_schema('Arbitrary JSON object encoded as a string, e.g. "{\\"type\\":\\"object\\"}".'),
            {"type": "null"},
        ]
    }),
]


def is_presigned_url(value: Any) -> bool:
    """Return True for AWS/MinIO SigV4 URLs, case-insensitively."""
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _SIGNED_QUERY_MARKERS)


def _http_url_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not _HTTP_URL_RE.match(text):
        raise ValueError("url must start with http:// or https://")
    return text


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime) -> datetime:
    if not value.tzinfo or value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be UTC")
    return value


def _utc_iso_str(value: Any) -> str:
    if isinstance(value, datetime):
        return _require_utc(value).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError("created_at must be a UTC ISO-8601 string")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    _require_utc(datetime.fromisoformat(normalized))
    return text


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutionRoute(str, Enum):
    REACT_TOOLS = "react_tools"
    FEDOT_MAS = "fedot_mas"
    CODER = "coder"
    ALEMBIC_BUILD = "alembic_build"
    RESEARCH = "research"  # Legacy snapshots only; review/runtime refuse new execution.
    MEDICAL = "medical"


class CodeRequirement(str, Enum):
    """Whether repository code can be executed as-is or must be changed."""

    REUSE = "reuse"
    MODIFY = "modify"
    UNKNOWN = "unknown"


class MCPToolRef(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: OptionalJsonObjectDict = None
    output_schema: OptionalJsonObjectDict = None
    data_contract: OptionalJsonObjectDict = None
    required_for_task: bool = True

    @model_validator(mode="before")
    @classmethod
    def coerce_string_tool(cls, data: Any) -> Any:
        if isinstance(data, str):
            name = data.strip()
            return {
                "name": name,
                "description": f"Registry tool {name}",
                "input_schema": None,
                "required_for_task": True,
            }
        if isinstance(data, dict):
            raw = dict(data)
            if not raw.get("name") and raw.get("tool"):
                raw["name"] = str(raw.pop("tool")).strip()
            elif "tool" in raw and raw.get("name"):
                raw.pop("tool", None)
            name = str(raw.get("name") or "").strip()
            if name and not str(raw.get("description") or "").strip():
                raw["description"] = f"Registry tool {name}"
            return raw
        return data


class MCPServerRef(StrictModel):
    name: str = Field(min_length=1)
    server_id: str | None = None
    url: str | None = None
    tools: list[MCPToolRef] = Field(min_length=1)
    source: Literal["registry", "explicit", "alembic"]
    health: Literal["unknown", "healthy", "unhealthy"] = "unknown"

    @field_validator("server_id")
    @classmethod
    def validate_server_id(cls, value: Any) -> str | None:
        if value is not None:
            sid = str(value).strip()
            if "/" in sid and "://" not in sid:
                raise ValueError(f"server_id must not contain composite '/tool': {sid!r}")
            return sid or None
        return None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: Any) -> str | None:
        return _http_url_str(value)

    @model_validator(mode="before")
    @classmethod
    def coerce_server_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        # Server description is not a schema field — drop LLM extras.
        raw.pop("description", None)
        if not raw.get("name"):
            raw["name"] = str(raw.get("server_id") or raw.get("url") or "mcp-server")
        # Planner often emits a singular "tool" instead of tools=[...].
        if not raw.get("tools") and raw.get("tool"):
            raw["tools"] = [raw.pop("tool")]
        elif "tool" in raw and raw.get("tools"):
            raw.pop("tool", None)
        tools = raw.get("tools")
        # Bare string: "tools": "fetch_activity_data"
        if isinstance(tools, str) and tools.strip():
            raw["tools"] = [tools.strip()]
            tools = raw["tools"]
        if isinstance(tools, list):
            fixed = []
            for tool in tools:
                if isinstance(tool, str):
                    name = tool.strip()
                    fixed.append(
                        {
                            "name": name,
                            "description": f"Registry tool {name}",
                            "input_schema": None,
                            "required_for_task": True,
                        }
                    )
                else:
                    fixed.append(tool)
            raw["tools"] = fixed
        return raw

    @model_validator(mode="after")
    def validate_location(self) -> "MCPServerRef":
        if not self.url:
            raise ValueError(f"MCP server requires url (source={self.source!r})")
        if self.source == "registry" and not self.server_id:
            raise ValueError("source='registry' requires server_id and url")
        return self


class DataRef(StrictModel):
    data_id: str = Field(min_length=1)
    kind: Literal["s3", "url", "workspace", "task_artifact", "to_prepare"]
    description: str = Field(min_length=1)
    bucket: str | None = None
    s3_key: str | None = None
    url: str | None = None
    workspace_path: str | None = None
    source_task_id: str | None = None
    source_artifact_id: str | None = None
    #: Logical output of the producer. The filename is not this id.
    source_output_id: str | None = None
    media_type: str | None = None
    required: bool = True
    prepare_instruction: str | None = None
    dataset_scope: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_dataref_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        # GLM sometimes nests producer refs instead of flat source_* fields.
        producer = raw.pop("producer", None)
        if isinstance(producer, dict):
            raw.setdefault("source_task_id", producer.get("source_task_id") or producer.get("task_id"))
            raw.setdefault(
                "source_artifact_id",
                producer.get("source_artifact_id") or producer.get("artifact_id") or producer.get("name"),
            )
        elif isinstance(producer, str) and producer.strip():
            raw.setdefault("source_task_id", producer.strip())
        # Weak planners put aliases / task-level fields on DataRef — absorb, don't revise-burn.
        name_alias = raw.pop("name", None)
        ref_alias = raw.pop("ref", None)
        # role/depends_on belong on ExpectedArtifact / ExperimentTask, not DataRef.
        raw.pop("role", None)
        raw.pop("depends_on", None)
        # Location aliases: path_or_tool / location / uri / path / filename.
        loc_alias = None
        for key in ("path_or_tool", "path_or_uri", "location", "uri", "path", "filename", "artifact"):
            if key in raw and loc_alias is None:
                val = raw.pop(key)
                if val is not None and str(val).strip():
                    loc_alias = str(val).strip()
            else:
                raw.pop(key, None)
        if name_alias and not str(raw.get("data_id") or "").strip():
            raw["data_id"] = str(name_alias).strip()
        if loc_alias:
            if _HTTP_URL_RE.match(loc_alias):
                raw.setdefault("url", loc_alias)
                raw.setdefault("kind", "url")
            elif not str(raw.get("source_artifact_id") or "").strip() and (
                str(raw.get("kind") or "").strip() == "task_artifact"
                or str(raw.get("source_task_id") or "").strip()
            ):
                raw.setdefault("source_artifact_id", loc_alias)
                raw.setdefault("kind", "task_artifact")
            elif str(raw.get("kind") or "").strip() == "workspace" and not str(
                raw.get("workspace_path") or ""
            ).strip():
                raw["workspace_path"] = loc_alias
            elif str(raw.get("kind") or "").strip() == "to_prepare" and not str(
                raw.get("prepare_instruction") or ""
            ).strip():
                raw["prepare_instruction"] = loc_alias
            if not str(raw.get("data_id") or "").strip():
                raw["data_id"] = loc_alias
        if ref_alias is not None and not str(raw.get("source_task_id") or "").strip():
            ref_text = str(ref_alias).strip()
            left, _, right = ref_text.partition("/")
            task_id = _normalize_task_id(left)
            if task_id.startswith("EXP-"):
                raw.setdefault("kind", "task_artifact")
                raw["source_task_id"] = task_id
                art = (right.strip() if right else "") or str(
                    raw.get("data_id") or name_alias or loc_alias or "artifact"
                ).strip()
                raw.setdefault("source_artifact_id", art or "artifact")
        if sid := raw.get("source_task_id"):
            raw["source_task_id"] = _normalize_task_id(sid) or sid
        # GLM often emits kind=task_artifact with only producer/ref=EXP-Tn (no artifact name).
        if str(raw.get("kind") or "").strip() == "task_artifact" or str(
            raw.get("source_task_id") or ""
        ).startswith("EXP-"):
            if not str(raw.get("source_artifact_id") or "").strip() and not str(
                raw.get("source_output_id") or ""
            ).strip():
                hint = (
                    str(raw.get("data_id") or "").strip()
                    or str(name_alias or "").strip()
                    or str(loc_alias or "").strip()
                    or "artifact"
                )
                raw["source_artifact_id"] = hint
            token = str(raw.get("source_artifact_id") or "").strip()
            if (
                not str(raw.get("source_output_id") or "").strip()
                and token
                and not token.startswith("ART-")
                and token != "artifact"
            ):
                raw["source_output_id"] = logical_output_id(token)
            raw.setdefault("kind", "task_artifact")
        if not str(raw.get("description") or "").strip():
            hint = (
                raw.get("source_artifact_id")
                or raw.get("data_id")
                or raw.get("workspace_path")
                or raw.get("s3_key")
                or name_alias
                or loc_alias
                or "upstream artifact"
            )
            raw["description"] = f"Input data: {hint}"
        if not str(raw.get("data_id") or "").strip():
            raw["data_id"] = str(
                raw.get("source_artifact_id")
                or raw.get("workspace_path")
                or name_alias
                or loc_alias
                or "input_data"
            ).strip() or "input_data"
        return raw

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: Any) -> str | None:
        return _http_url_str(value)

    @model_validator(mode="after")
    def validate_kind(self) -> "DataRef":
        ok = {
            "s3": bool(self.bucket and self.s3_key),
            "url": bool(self.url),
            "workspace": bool(self.workspace_path),
            "task_artifact": bool(
                self.source_task_id and (self.source_artifact_id or self.source_output_id)
            ),
            "to_prepare": bool(self.prepare_instruction),
        }[self.kind]
        if not ok:
            raise ValueError(f"DataRef kind={self.kind!r} is missing its location fields")
        if is_presigned_url(self.url):
            raise ValueError("canonical DataRef.url must not contain S3 signing parameters")
        return self


class SuccessCriterion(StrictModel):
    requirement_id: str = ""
    requirement_criterion_id: str = ""
    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: Literal["threshold", "artifact_exists", "schema", "execution", "expert"]
    # Execution criteria answer "did the requested operation deliver its core
    # output?".  Assessment criteria answer "was that output scientifically
    # good enough?".  The latter must never silently become a technical retry.
    purpose: Literal["execution", "assessment"] = "execution"
    metric: str | None = None
    operator: Literal["<", "<=", "==", ">=", ">", "in"] | None = None
    target: Any | None = None
    required: bool = True
    verification: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _clear_non_threshold_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        # Unlinked method criteria have no catalog reference. JSON null and
        # an omitted optional reference use the same canonical empty string.
        for field in ("requirement_id", "requirement_criterion_id"):
            if raw.get(field) is None:
                raw[field] = ""
        kind = str(raw.get("kind") or "").strip().lower()
        if not str(raw.get("purpose") or "").strip():
            raw["purpose"] = (
                "assessment" if kind in {"threshold", "expert"} else "execution"
            )
        if kind and kind != "threshold":
            # LLMs often copy metric/operator/target onto artifact_exists/etc.
            raw["metric"] = None
            raw["operator"] = None
            raw["target"] = None
        return raw

    @model_validator(mode="after")
    def validate_threshold(self) -> "SuccessCriterion":
        has_thresh = self.metric is not None or self.operator is not None or self.target is not None
        if self.kind == "threshold":
            if not self.metric or self.operator is None or self.target is None:
                raise ValueError("threshold criterion requires metric, operator and target")
        elif has_thresh:
            raise ValueError("non-threshold criteria must leave metric, operator and target null")
        return self


def logical_output_id(name: str) -> str:
    """Stable id of a logical output. A renamed file keeps this id."""
    stem = PurePosixPath(str(name or "").replace("\\", "/")).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "output"


class ExpectedArtifact(StrictModel):
    name: str = Field(min_length=1)
    #: Logical output. Planners may omit it; the filename stem is the adapter.
    output_id: str = ""
    role: Annotated[
        Literal["data", "model", "plot", "report", "code", "log", "mcp_server"],
        BeforeValidator(
            lambda v: {
                "metrics_table": "data",
                "table": "data",
                "csv": "data",
                "json": "data",
                "figure": "plot",
                "image": "plot",
                "script": "code",
                "notebook": "code",
                "cfg": "code",
                "config": "code",
            }.get(str(v or "").strip().lower(), str(v or "").strip().lower())
        ),
    ]
    media_type: str | None = None
    required: bool = True
    description: str = Field(min_length=1)
    #: Column names a tabular output must contain. Empty means "readable table".
    columns: list[str] = Field(default_factory=list)
    allow_empty: bool = False

    @model_validator(mode="before")
    @classmethod
    def fill_output_id(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if not str(raw.get("output_id") or "").strip():
            raw["output_id"] = logical_output_id(str(raw.get("name") or ""))
        else:
            raw["output_id"] = logical_output_id(str(raw.get("output_id")))
        columns = raw.get("columns")
        if isinstance(columns, str):
            raw["columns"] = [part.strip() for part in columns.split(",") if part.strip()]
        return raw


def _unique_output_ids(artifacts: list[ExpectedArtifact]) -> list[ExpectedArtifact]:
    """One logical id per expected output. A stem clash keeps both, distinctly."""
    used: set[str] = set()
    out: list[ExpectedArtifact] = []
    for art in artifacts:
        base = art.output_id or logical_output_id(art.name)
        oid = base
        if oid in used:
            suffix = PurePosixPath(art.name).suffix.lower().lstrip(".") or "alt"
            oid = f"{base}_{suffix}"
            n = 2
            while oid in used:
                oid = f"{base}_{suffix}_{n}"
                n += 1
        used.add(oid)
        out.append(art if art.output_id == oid else art.model_copy(update={"output_id": oid}))
    return out


class HypothesisSpec(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def coerce_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if not raw.get("hypothesis_id") and raw.get("id"):
            raw["hypothesis_id"] = raw.pop("id")
        elif "id" in raw and raw.get("hypothesis_id"):
            raw.pop("id", None)
        if not raw.get("statement"):
            for alt in ("description", "text", "hypothesis", "content", "summary", "details"):
                if raw.get(alt):
                    raw["statement"] = raw.pop(alt)
                    break
        # Keep only schema fields; drop any GLM extras (type/details/tests/…).
        for key in list(raw):
            if key not in {"hypothesis_id", "statement"}:
                raw.pop(key, None)
        return raw


def _coerce_hypothesis_ref(value: Any) -> str:
    """Accept a single id. Empty, None and placeholders stay empty — no invented H1."""
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    text = str(value or "").strip()
    if not text or text.upper() in {"H0", "N/A", "NA", "NONE", "NULL"}:
        return ""
    if "," in text:
        text = text.split(",", 1)[0].strip()
    return text.upper() if re.fullmatch(r"H\d+", text, flags=re.I) else text


def _coerce_also_tests(value: Any) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        for part in re.split(r"[,/;]+", text):
            token = part.strip()
            if not token:
                continue
            norm = token.upper() if re.fullmatch(r"H\d+", token, flags=re.I) else token
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def _split_hypothesis_list(value: Any) -> tuple[str, list[str]]:
    if value is None or value == "" or value == []:
        return "", []
    if isinstance(value, list):
        ids = _coerce_also_tests(value)
        if not ids:
            return "", []
        return ids[0], ids[1:]
    text = str(value or "").strip()
    if not text:
        return "", []
    if re.search(r"[,;/]", text):
        ids = _coerce_also_tests(re.split(r"[,;/]+", text))
        if ids:
            return ids[0], ids[1:]
    return _coerce_hypothesis_ref(value), []


def _coerce_design_dataset_ref(value: Any) -> DataRef | None:
    """Best-effort coerce; prefer null over rejecting the whole plan."""
    if value is None or value == "" or value == {}:
        return None
    if isinstance(value, DataRef):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "internal", "task_artifact", "n/a", "external"}:
            return None
        if _HTTP_URL_RE.match(text):
            return DataRef(
                data_id="design_dataset_url",
                kind="url",
                description="Dataset URL from plan design",
                url=text,
            )
        return None
    if isinstance(value, dict):
        raw = dict(value)
        # Common LLM aliases → canonical DataRef fields.
        if "path_or_uri" in raw and "url" not in raw and "workspace_path" not in raw:
            loc = str(raw.pop("path_or_uri") or "")
            if _HTTP_URL_RE.match(loc):
                raw.setdefault("url", loc)
                raw["kind"] = "url"
            elif loc:
                raw.setdefault("workspace_path", loc)
                raw["kind"] = "workspace"
        raw.pop("source", None)
        kind = str(raw.get("kind") or "").strip().lower()
        if kind in {"external", "internal", "simulated", "generation"}:
            if url := raw.get("url") or raw.get("path_or_uri"):
                if _HTTP_URL_RE.match(str(url)):
                    raw = {
                        "data_id": str(raw.get("data_id") or "design_dataset_url"),
                        "kind": "url",
                        "description": str(raw.get("description") or "Dataset URL from plan design"),
                        "url": str(url),
                    }
                else:
                    return None
            else:
                return None
        raw.setdefault("data_id", "design_dataset")
        raw.setdefault("description", "Dataset referenced by task design")
        try:
            return DataRef.model_validate(raw)
        except Exception:
            return None
    return None


def _coerce_analysis_role(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "data": "metrics_table",
        "table": "metrics_table",
        "metrics": "metrics_table",
        "plot": "report",
        "figure": "report",
        "image": "report",
        "log": "report",
        "logs": "report",
        "output": "report",
        "script": "code",
        "notebook": "code",
        "cfg": "config",
        "yaml": "config",
        "json": "config",
        "model": "config",
        "weights": "config",
        "checkpoint": "config",
    }
    return aliases.get(text, text)


def _coerce_optional_baselines(value: Any) -> list[Any]:
    kind_aliases = {
        "report": "prior_result",
        "paper": "external",
        "literature": "external",
        "publication": "external",
        "baseline": "method",
        "reference": "method",
        "control": "method",
        "comparator": "method",
        "heuristic": "method",
        "algorithm": "method",
        "dataset": "prior_result",
        "result": "prior_result",
    }
    allowed = {"method", "model", "prior_result", "external"}
    if not isinstance(value, list) or not value:
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            raw = dict(item)
            name = str(raw.get("name") or "").strip()
            if is_design_placeholder(name):
                continue
            kind = str(raw.get("kind") or "method").strip().lower()
            if kind not in allowed:
                kind = kind_aliases.get(kind, "method")
            raw["kind"] = kind
            out.append(raw)
        elif not is_design_placeholder(item):
            out.append(item)
    return out


def _coerce_optional_metrics(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            raw = dict(item)
            raw.pop("operator", None)
            name = str(raw.get("name") or "").strip()
            if is_design_placeholder(name):
                continue
            raw.setdefault("direction", "compare")
            out.append(raw)
        elif not is_design_placeholder(item):
            out.append(item)
    return out


class DesignDataset(StrictModel):
    name: str = ""
    dataset_scope: str | None = None
    ref: Annotated[
        DataRef | None,
        BeforeValidator(_coerce_design_dataset_ref),
    ] = None
    notes: str | None = None


class DesignBaseline(StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["method", "model", "prior_result", "external"]
    ref: str | None = None


class DesignMetric(StrictModel):
    name: str = Field(min_length=1)
    direction: Literal["maximize", "minimize", "compare"]
    threshold: Any | None = None
    test: str | None = None
    # Optional prose; planners often emit this — keep it, do not forbid.
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_criteria_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        # LLMs often put success_criteria.operator onto metrics — strip it.
        raw.pop("operator", None)
        return raw


def _coerce_prepare_via(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "code": "coder",
        "sandbox": "coder",
        "python": "coder",
        "tool": "mcp",
        "fedot": "mcp",
        "react": "mcp",
        "file": "existing",
        "present": "existing",
        "available": "existing",
        "literature": "research",
        "pubmed": "medical",
        "clinical": "medical",
    }
    return aliases.get(text, text or "coder")


class DesignAnalysisArtifact(StrictModel):
    name: str = Field(min_length=1)
    role: Annotated[
        Literal["code", "config", "metrics_table", "report"],
        BeforeValidator(_coerce_analysis_role),
    ]
    prepare_via: Annotated[
        Literal["coder", "mcp", "existing", "research", "medical"],
        BeforeValidator(_coerce_prepare_via),
    ] = "coder"
    path_or_tool: str | None = None


def _coerce_optional_analysis_artifacts(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    fixed: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            row = dict(item)
            name = str(row.get("name") or "").strip()
            path = str(row.get("path_or_tool") or "").strip()
            if is_design_placeholder(name):
                if is_design_placeholder(path):
                    continue
                row["name"] = path
            elif not name and path and not is_design_placeholder(path):
                row["name"] = path
            elif not name:
                continue
            if is_design_placeholder(row.get("path_or_tool")):
                row["path_or_tool"] = None
            fixed.append(row)
        elif not is_design_placeholder(item):
            fixed.append(item)
    return fixed


def _coerce_target_refs(value: Any) -> list[str]:
    """One list of question / hypothesis / deliverable ids a step serves."""
    return _coerce_also_tests(value)


class RequirementLink(StrictModel):
    """Whether a task prepares a requirement or produces its closing result."""

    requirement_id: str = Field(min_length=1)
    role: Literal["supports", "delivers"]


class TaskDesign(StrictModel):
    """Machine-readable scientific node. A hypothesis is optional.

    ``target_refs`` is the single answer to "what this step serves".
    ``hypothesis_ref`` and ``also_tests`` remain as a compatible projection of
    the hypothesis ids inside that list — they are not a second meaning.
    """

    hypothesis_ref: Annotated[str, BeforeValidator(_coerce_hypothesis_ref)] = ""
    also_tests: Annotated[list[str], BeforeValidator(_coerce_also_tests)] = Field(default_factory=list)
    target_refs: Annotated[list[str], BeforeValidator(_coerce_target_refs)] = Field(default_factory=list)
    target_links: list[RequirementLink] = Field(default_factory=list)
    step_key: str = ""
    operation_ref: Annotated[
        str,
        BeforeValidator(lambda v: "" if v is None else str(v).strip()),
    ] = ""
    question_ref: str = ""
    experiment_question: str = ""
    dataset: DesignDataset = Field(default_factory=DesignDataset)
    baselines: Annotated[
        list[DesignBaseline],
        BeforeValidator(_coerce_optional_baselines),
    ] = Field(default_factory=list)
    metrics: Annotated[
        list[DesignMetric],
        BeforeValidator(_coerce_optional_metrics),
    ] = Field(default_factory=list)
    analysis_artifacts: Annotated[
        list[DesignAnalysisArtifact],
        BeforeValidator(_coerce_optional_analysis_artifacts),
    ] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def promote_multi_hypothesis_ref(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if is_design_placeholder(data.get("experiment_question")):
            data["experiment_question"] = ""
        ds = data.get("dataset")
        if not isinstance(ds, dict):
            data["dataset"] = {"name": "", "ref": None, "notes": None}
        elif is_design_placeholder(ds.get("name")):
            data["dataset"] = {**ds, "name": ""}
        if "hypothesis_ref" in data:
            primary, extras = _split_hypothesis_list(data.get("hypothesis_ref"))
            data["hypothesis_ref"] = primary
            also = list(_coerce_also_tests(data.get("also_tests")))
            for hid in extras:
                if hid and hid != primary and hid not in also:
                    also.append(hid)
            data["also_tests"] = also
        targets = list(_coerce_target_refs(data.get("target_refs")))
        links = data.get("target_links")
        for link in links if isinstance(links, list) else []:
            ref = link.get("requirement_id") if isinstance(link, dict) else getattr(link, "requirement_id", "")
            if ref and ref not in targets:
                targets.append(ref)
        for hid in [data.get("hypothesis_ref"), *(data.get("also_tests") or [])]:
            text = str(hid or "").strip()
            if text and text not in targets:
                targets.append(text)
        data["target_refs"] = targets
        data["step_key"] = str(data.get("step_key") or "").strip()
        return data

    @field_validator("target_links")
    @classmethod
    def unique_requirement_roles(cls, links: list[RequirementLink]) -> list[RequirementLink]:
        ids = [link.requirement_id for link in links]
        if len(ids) != len(set(ids)):
            raise ValueError("target_links must declare one role per requirement")
        return links

    def covered_hypothesis_ids(self, requirement_refs: Any = ()) -> set[str]:
        """Hypothesis ids only. Question and deliverable targets are not hypotheses."""
        from CoScientist.requirements.coverage import hypothesis_ids

        return set(hypothesis_ids(self.model_dump(), requirement_refs))

    def covered_target_ids(self) -> set[str]:
        ids = {str(item).strip() for item in self.target_refs if str(item).strip()}
        if self.hypothesis_ref.strip():
            ids.add(self.hypothesis_ref.strip())
        ids.update(str(item).strip() for item in self.also_tests if str(item).strip())
        return ids


def _normalize_task_id(value: Any) -> str:
    """TASK-1 / EXP-T1 / T1 → EXP-1; leave other non-empty strings unchanged."""
    text = str(value or "").strip()
    if not text:
        return ""
    # EXP-T1 / EXP_T1 / TASK-T2 (GLM) in addition to EXP-1 / TASK-1 / T1.
    m = re.fullmatch(r"(?:TASK|EXP)[_-]?(?:T)?(\d+)", text, flags=re.I) or re.fullmatch(
        r"T(\d+)", text, flags=re.I
    )
    if m:
        return f"EXP-{int(m.group(1))}"
    return text


def _coerce_depends_on(value: Any) -> list[str]:
    """Accept EXP-n strings; unwrap accidental DataRef objects to source_task_id."""
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("source_task_id")
                or item.get("task_id")
                or item.get("data_id")
                or item.get("id")
                or ""
            ).strip()
        else:
            text = str(item or "").strip()
        text = _normalize_task_id(text)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


class CodeAssessment(StrictModel):
    requirement: CodeRequirement = CodeRequirement.UNKNOWN
    evidence: str = ""
    entrypoints: list[str] = Field(default_factory=list)

    @field_validator("entrypoints")
    @classmethod
    def validate_entrypoints(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("code assessment entrypoints must be non-empty strings")
        return cleaned


class ExperimentTask(StrictModel):
    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    route: ExecutionRoute
    design: TaskDesign
    code_assessment: CodeAssessment = Field(default_factory=CodeAssessment)
    coder_fallback_method: str | None = Field(default=None, min_length=1,
        description="Approved alternative implementation preserving the task's inputs, method constraints and outputs. Null forbids substitution by Coder.")
    mcp_servers: list[MCPServerRef] = Field(default_factory=list)
    repo_url: str | None = None
    post_build_route: Literal["fedot_mas", "react_tools"] | None = None
    input_data: list[DataRef] = Field(default_factory=list)
    launch_params: JsonObjectDict = Field(default_factory=dict)
    success_criteria: list[SuccessCriterion] = Field(min_length=1)
    expected_artifacts: list[ExpectedArtifact] = Field(min_length=1)
    est_duration_min: int = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)
    depends_on: Annotated[list[str], BeforeValidator(_coerce_depends_on)] = Field(default_factory=list)
    optional: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_task_shape(cls, data: Any) -> Any:
        """Absorb common planner shape mistakes before strict validation."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)

        # TASK-1 / EXP-T1 / T1 → EXP-1
        tid = _normalize_task_id(raw.get("id"))
        if tid.startswith("EXP-"):
            raw["id"] = tid

        design = raw.get("design")
        if not isinstance(design, dict):
            design = {}
        else:
            design = dict(design)

        # Fields planners often nest under design (forbidden there).
        for key in ("est_duration_min", "depends_on", "optional", "success_criteria",
                    "expected_artifacts", "mcp_servers", "input_data", "route",
                    "description", "rationale", "name", "code_assessment"):
            if key in design and key not in raw:
                raw[key] = design.pop(key)
            else:
                design.pop(key, None)

        # Hoist design pieces wrongly placed on the task root.
        for key in ("hypothesis_ref", "also_tests", "target_refs", "step_key",
                    "operation_ref", "experiment_question",
                    "dataset", "baselines", "metrics", "analysis_artifacts"):
            if key in raw and key not in design:
                design[key] = raw.pop(key)

        # Missing hypothesis_ref stays missing. Do not invent H1 / H0 / N/A.
        raw["design"] = design

        # Plan-only fields misplaced on tasks → drop (hoisted in coerce_plan_shape).
        raw.pop("risks", None)
        raw.pop("assumptions", None)

        # GLM sometimes puts depends_on on input_data items — hoist to task.depends_on.
        inputs = raw.get("input_data")
        if isinstance(inputs, list) and inputs:
            dep_extra: list[Any] = []
            cleaned: list[Any] = []
            for item in inputs:
                if isinstance(item, dict):
                    entry = dict(item)
                    nested = entry.pop("depends_on", None)
                    if nested is not None:
                        if isinstance(nested, list):
                            dep_extra.extend(nested)
                        else:
                            dep_extra.append(nested)
                    cleaned.append(entry)
                else:
                    cleaned.append(item)
            raw["input_data"] = cleaned
            if dep_extra:
                existing = raw.get("depends_on")
                base = list(existing) if isinstance(existing, list) else (
                    [existing] if existing else []
                )
                raw["depends_on"] = base + dep_extra
            # Ensure every task_artifact producer is listed in depends_on.
            producers: list[Any] = []
            for item in cleaned:
                if not isinstance(item, dict):
                    continue
                if str(item.get("kind") or "") != "task_artifact":
                    continue
                src = item.get("source_task_id") or item.get("task_id")
                if src:
                    producers.append(src)
            if producers:
                existing = raw.get("depends_on")
                base = list(existing) if isinstance(existing, list) else (
                    [existing] if existing else []
                )
                raw["depends_on"] = base + producers

        name = str(raw.get("name") or "").strip() or str(raw.get("id") or "task")
        raw.setdefault("name", name)
        if not str(raw.get("description") or "").strip():
            raw["description"] = name
        if not str(raw.get("rationale") or "").strip():
            raw["rationale"] = f"Required step toward the experiment goal: {name}"

        arts = raw.get("expected_artifacts")
        if not isinstance(arts, list) or not arts:
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "task"
            route = str(raw.get("route") or "").strip()
            # MCP routes: never invent a mandatory markdown report (feeds R2 incomplete→coder).
            if route in {"fedot_mas", "react_tools"}:
                raw["expected_artifacts"] = [{
                    "name": f"{slug}_output",
                    "role": "data",
                    "media_type": "application/octet-stream",
                    "required": True,
                    "description": f"Primary data deliverable for {name}",
                }]
            else:
                raw["expected_artifacts"] = [{
                    "name": f"{slug}_report",
                    "role": "report",
                    "media_type": "text/markdown",
                    "required": True,
                    "description": f"Primary deliverable for {name}",
                }]
            arts = raw["expected_artifacts"]

        crit = raw.get("success_criteria")
        if not isinstance(crit, list) or not crit:
            first = arts[0] if isinstance(arts, list) and arts else {}
            art_name = str((first or {}).get("name") or "primary_report")
            raw["success_criteria"] = [{
                "criterion_id": "C1",
                "description": f"Required artifact {art_name} exists",
                "kind": "artifact_exists",
                "metric": None,
                "operator": None,
                "target": None,
                "required": True,
                "verification": f"Confirm workspace artifact {art_name} was produced",
            }]

        if raw.get("est_duration_min") in (None, "", 0):
            raw["est_duration_min"] = 30
        # Do not invent route=coder when omitted — schema must force an explicit choice.
        return raw

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _TASK_ID_RE.fullmatch(value):
            raise ValueError("task id must match EXP-<n>")
        return value

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, value: Any) -> str | None:
        return _http_url_str(value)

    @model_validator(mode="after")
    def validate_route_and_refs(self) -> "ExperimentTask":
        if self.route in {ExecutionRoute.REACT_TOOLS, ExecutionRoute.FEDOT_MAS}:
            if not any(tool for server in self.mcp_servers for tool in server.tools):
                raise ValueError(f"route={self.route.value} requires an MCP server/tool")
        if self.route in {ExecutionRoute.RESEARCH, ExecutionRoute.MEDICAL}:
            if self.mcp_servers:
                raise ValueError(
                    f"route={self.route.value} must keep mcp_servers empty "
                    "(the route agent owns its toolset)"
                )
            if not any(item.required for item in self.expected_artifacts):
                raise ValueError(
                    f"route={self.route.value} requires ≥1 required evidence artifact"
                )
        if self.route == ExecutionRoute.ALEMBIC_BUILD:
            if not self.repo_url:
                raise ValueError("route=alembic_build requires repo_url")
            if self.post_build_route is None:
                raise ValueError("route=alembic_build requires post_build_route")
            if self.code_assessment.requirement != CodeRequirement.REUSE:
                raise ValueError(
                    "route=alembic_build requires code_assessment.requirement=reuse"
                )
        elif self.post_build_route is not None:
            raise ValueError("post_build_route is only valid with route=alembic_build")

        requirement = self.code_assessment.requirement
        if requirement == CodeRequirement.MODIFY and self.route != ExecutionRoute.CODER:
            raise ValueError("code_assessment.requirement=modify requires route=coder")
        if requirement == CodeRequirement.REUSE:
            post_build_mcp = (
                self.route in {ExecutionRoute.REACT_TOOLS, ExecutionRoute.FEDOT_MAS}
                and any(server.source == "alembic" for server in self.mcp_servers)
            )
            if (
                self.route not in {ExecutionRoute.CODER, ExecutionRoute.ALEMBIC_BUILD}
                and not post_build_mcp
            ):
                raise ValueError(
                    "code_assessment.requirement=reuse requires route=coder, alembic_build, "
                    "or an Alembic-produced post-build MCP"
                )
            if not self.repo_url:
                raise ValueError("code_assessment.requirement=reuse requires repo_url")
            if not self.code_assessment.evidence.strip():
                raise ValueError("code_assessment.requirement=reuse requires inspection evidence")
            if not self.code_assessment.entrypoints:
                raise ValueError("code_assessment.requirement=reuse requires entrypoints")

        c_ids = [c.criterion_id for c in self.success_criteria]
        if len(c_ids) != len(set(c_ids)):
            raise ValueError("criterion_id values must be unique inside a task")
        self.expected_artifacts = _unique_output_ids(self.expected_artifacts)

        for ref in self.input_data:
            if ref.kind == "task_artifact" and ref.source_task_id and ref.source_task_id not in self.depends_on:
                raise ValueError(f"task_artifact input from {ref.source_task_id!r} requires that producer in depends_on")
        return self


def _coerce_string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, dict):
        # LLM sometimes emits {} instead of []
        return []
    if not isinstance(value, list):
        return [str(value)]
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            if text := item.strip():
                out.append(text)
        elif isinstance(item, dict):
            text = str(
                item.get("description")
                or item.get("message")
                or item.get("risk")
                or item.get("assumption")
                or item.get("text")
                or item.get("source_task_id")
                or item.get("data_id")
                or ""
            ).strip()
            mitigation = str(item.get("mitigation") or "").strip()
            if text and mitigation:
                out.append(f"{text} Mitigation: {mitigation}")
            elif text:
                out.append(text)
            else:
                out.append(json.dumps(item, ensure_ascii=False)[:500])
        else:
            out.append(str(item))
    return out


class ExperimentPlan(StrictModel):
    schema_version: Literal["experiment-plan/1.0"]
    plan_id: str = Field(min_length=1)
    experiment_run_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    source_request: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    hypothesis: str | None = None
    hypotheses: list[HypothesisSpec] = Field(default_factory=list)
    methods: Annotated[list[str], BeforeValidator(_coerce_string_list)] = Field(min_length=1)
    context_digest: str = Field(min_length=1)
    context_refs: Annotated[list[str], BeforeValidator(_coerce_string_list)] = Field(default_factory=list)
    tasks: list[ExperimentTask] = Field(min_length=1, max_length=20)
    risks: Annotated[list[str], BeforeValidator(_coerce_string_list)] = Field(default_factory=list)
    assumptions: Annotated[list[str], BeforeValidator(_coerce_string_list)] = Field(default_factory=list)
    total_est_duration_min: int = Field(gt=0)
    created_at: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def coerce_plan_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        tasks = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
        # Hoist plan-only fields wrongly placed on tasks (avoid extra_forbidden revise burns).
        hoisted_risks: list[Any] = []
        hoisted_assumptions: list[Any] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if "risks" in t:
                val = t.pop("risks")
                if isinstance(val, list) and val:
                    hoisted_risks.extend(val)
                elif val and not isinstance(val, list):
                    hoisted_risks.append(val)
            if "assumptions" in t:
                val = t.pop("assumptions")
                if isinstance(val, list) and val:
                    hoisted_assumptions.extend(val)
                elif val and not isinstance(val, list):
                    hoisted_assumptions.append(val)
        plan_risks = raw.get("risks")
        if hoisted_risks and (not isinstance(plan_risks, list) or not plan_risks):
            raw["risks"] = hoisted_risks
        plan_assumptions = raw.get("assumptions")
        if hoisted_assumptions and (not isinstance(plan_assumptions, list) or not plan_assumptions):
            raw["assumptions"] = hoisted_assumptions
        methods = raw.get("methods")
        if not isinstance(methods, list) or not methods:
            derived = []
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                label = str(t.get("name") or t.get("id") or "").strip()
                route = str(t.get("route") or "").strip()
                if label:
                    derived.append(f"{label}" + (f" via {route}" if route else ""))
            raw["methods"] = derived or ["computational experiment pipeline"]
        if not str(raw.get("context_digest") or "").strip():
            raw["context_digest"] = "Plan derived from source_request and available MCP inventory."
        if not str(raw.get("goal") or "").strip():
            raw["goal"] = str(raw.get("source_request") or "Execute computational experiment")[:500]
        # Align total duration with task sum when missing/zero/mismatched.
        # Task durations may still be nested under design (hoisted later in
        # coerce_task_shape); read both so we don't under-sum then fail after.
        def _task_duration(t: dict) -> int:
            raw_dur = t.get("est_duration_min")
            if raw_dur in (None, "", 0):
                design = t.get("design") if isinstance(t.get("design"), dict) else {}
                raw_dur = design.get("est_duration_min")
            try:
                return int(raw_dur or 30)
            except (TypeError, ValueError):
                return 30

        task_sum = sum(_task_duration(t) for t in tasks if isinstance(t, dict))
        if task_sum > 0:
            try:
                if int(raw.get("total_est_duration_min") or 0) != task_sum:
                    raw["total_est_duration_min"] = task_sum
            except (TypeError, ValueError):
                raw["total_est_duration_min"] = task_sum
        return raw

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: Any) -> str:
        return _utc_iso_str(value)

    @model_validator(mode="after")
    def validate_plan_graph(self) -> "ExperimentPlan":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        known, by_id = set(ids), {t.id: t for t in self.tasks}
        for task in self.tasks:
            if unknown := set(task.depends_on) - known:
                raise ValueError(f"{task.id}: unknown dependencies {sorted(unknown)}")
            if task.id in task.depends_on:
                raise ValueError(f"{task.id}: task cannot depend on itself")

        visiting, visited = set(), set()
        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("depends_on must form a DAG")
            visiting.add(task_id)
            for dep in by_id[task_id].depends_on:
                visit(dep)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)

        task_sum = sum(t.est_duration_min for t in self.tasks)
        if task_sum != self.total_est_duration_min:
            # Absorb residual planner drift after nested task coerce.
            self.total_est_duration_min = task_sum
        return self


BLOCKING_SEVERITIES = frozenset({"blocker", "major"})


class CritiqueIssue(StrictModel):
    issue_id: str = Field(min_length=1)
    category: Literal[
        "relevance", "completeness", "consistency", "feasibility", "complexity", "security", "schema"
    ]
    severity: Literal["blocker", "major", "minor"]
    task_id: str | None = None
    message: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)

    @property
    def is_blocking(self) -> bool:
        return self.severity in BLOCKING_SEVERITIES


class PlanCritique(StrictModel):
    schema_version: Literal["plan-critique/0.1"]
    critique_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=1)
    critic_type: Literal["deterministic"] = "deterministic"
    verdict: Literal["approve", "revise"]
    issues: list[CritiqueIssue] = Field(default_factory=list)
    checked_at: datetime

    @field_validator("checked_at")
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def validate_verdict(self) -> "PlanCritique":
        blocking = any(i.is_blocking for i in self.issues)
        if self.verdict == "approve" and blocking:
            raise ValueError("approve is forbidden while blocker/major issues exist")
        if self.verdict == "revise" and not blocking:
            raise ValueError("revise requires at least one blocker/major issue")
        return self


_REPORTING_EXPORTS = frozenset({
    "ArtifactRef",
    "CriterionCheck",
    "ScientificCheck",
    "TaskResult",
    "artifact_name_from_location",
})


def __getattr__(name: str):
    """Lazy re-export of result contracts (avoids importing reporting at module load)."""
    if name in _REPORTING_EXPORTS:
        from CoScientist.experiments.reporting import models as reporting_models
        return getattr(reporting_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BLOCKING_SEVERITIES",
    "ArtifactRef",
    "CriterionCheck",
    "CritiqueIssue",
    "DataRef",
    "DesignAnalysisArtifact",
    "DesignBaseline",
    "DesignDataset",
    "DesignMetric",
    "ExecutionRoute",
    "ExpectedArtifact",
    "ExperimentPlan",
    "ExperimentTask",
    "HypothesisSpec",
    "MCPServerRef",
    "MCPToolRef",
    "PlanCritique",
    "ScientificCheck",
    "SuccessCriterion",
    "TaskDesign",
    "TaskResult",
    "artifact_name_from_location",
    "is_design_placeholder",
    "is_presigned_url",
    "lenient_planner_enabled",
    "reset_lenient_planner",
    "set_lenient_planner",
    "utc_now",
]
