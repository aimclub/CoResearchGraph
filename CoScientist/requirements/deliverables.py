"""Deliverable module: what had to be handed over, and what actually was.

A path, a JSON pointer, a SMILES file and a docking score are not substitutes
for a missing file, a table, a physical sample, or a shown biological property.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from CoScientist.requirements.models import OutcomeRecord, RequirementPart


def _location(artifact: dict[str, Any]) -> str:
    for key in ("path", "workspace_path", "external_url", "url", "uri"):
        if value := str(artifact.get(key) or "").strip():
            return value
    return ""


def artifact_satisfies(part: RequirementPart, artifact: dict[str, Any]) -> tuple[bool, str]:
    """Whether this output may count toward the ordered result."""
    if not artifact:
        return False, "no output recorded"
    kind = str(artifact.get("kind") or artifact.get("role") or "").lower()
    if kind in {"pointer", "manifest", "url_pointer"} or artifact.get("pointer_only"):
        return False, "a pointer is not the ordered table or file"
    if artifact.get("inline") and not part.physical_sample:
        content = artifact.get("content", artifact.get("text"))
        if isinstance(content, (str, list, dict)) and content:
            return True, ""
        return False, "inline output has no recorded content"
    location = _location(artifact)
    if not location and not artifact.get("inline"):
        return False, "the output has no location"
    if location and location.startswith(("s3://", "http://", "https://")):
        if artifact.get("content_verified") is not True:
            return False, "remote output content has not been verified"
    elif location:
        if not Path(location).is_file():
            return False, f"file is not available: {location}"
        try:
            with Path(location).open("rb") as stream:
                stream.read(1)
        except OSError:
            return False, f"file is not readable: {location}"
    if part.physical_sample:
        if not artifact.get("lab_trace"):
            return False, "a file is not a physical sample"
    return True, ""


def deliverable_status(
    part: RequirementPart,
    outcome: OutcomeRecord | None,
    *,
    volume: Any = None,
) -> tuple[str, str]:
    """Return met / partial / unmet and a reason."""
    if part.retired or not part.obligation:
        return "met", ""
    if outcome is None or outcome.infrastructure_error:
        return "unmet", "the result was not produced"
    if part.protocol_only:
        documents = []
        for artifact in (outcome.artifacts if outcome else []):
            kind = str(artifact.get("kind") or "").lower()
            if kind not in {"document", "protocol", "text"} and artifact.get("role") != "report":
                continue
            if artifact_satisfies(part, artifact)[0]:
                documents.append(artifact)
        if documents:
            if outcome.property_verified is False:
                return "partial", "protocol delivered; the requested property has not been verified"
            return "met", "protocol delivered; the property was not measured"
        return "unmet", "the protocol document was not delivered"
    if part.physical_sample:
        ok = [a for a in outcome.artifacts if artifact_satisfies(part, a)[0]]
        if not ok:
            return "unmet", "computational output is not the ordered physical sample"
    limit = "the requested property has not been verified" if outcome.property_verified is False else ""
    produced = len(set(outcome.produced_ids)) if outcome.produced_ids is not None else outcome.produced
    requested = None
    if requested is None and volume is not None and getattr(volume, "requested", None):
        quote = part.provenance.quote
        cap_quote = str(getattr(volume, "requested_quote", "") or "")
        if cap_quote and cap_quote.casefold() in quote.casefold():
            requested = int(volume.requested)
    files_ok = []
    missing = ""
    for artifact in outcome.artifacts:
        ok, why = artifact_satisfies(part, artifact)
        if ok:
            files_ok.append(artifact)
        elif not missing:
            missing = why
    if limit and files_ok:
        return "partial", limit
    if requested is not None:
        if produced is None:
            return ("partial" if files_ok else "unmet"), "produced count has not been recorded"
        got = produced
        if got >= requested and files_ok:
            return "met", ""
        if got > 0:
            return "partial", f"{got} of {requested}"
        return "unmet", missing or "the file does not meet the requested size"
    if files_ok:
        return "met", ""
    return "unmet", missing or "the ordered result is missing"
