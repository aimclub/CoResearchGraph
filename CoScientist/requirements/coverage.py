"""One projection of which tasks serve each question, hypothesis and deliverable.

Plan review, the plan card, the research graph and the result report all read
this. A task being ``done`` is not itself a closed requirement. A preparatory
step that shares a target with a later step does not close that target.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from CoScientist.requirements.execution import is_infrastructure_failure

_KINDS = {"question", "hypothesis", "deliverable"}
_HYPOTHESIS_ID = re.compile(r"H\d+")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _kind(row: Mapping[str, Any], ref: str) -> str:
    raw = _text(row.get("kind") or row.get("type")).lower()
    aliases = {
        "q": "question",
        "question": "question",
        "researchquestion": "question",
        "h": "hypothesis",
        "hypothesis": "hypothesis",
        "dl": "deliverable",
        "deliverable": "deliverable",
    }
    if raw in aliases:
        return aliases[raw]
    token = ref.upper()
    if token.startswith("DL"):
        return "deliverable"
    if _HYPOTHESIS_ID.fullmatch(token):
        return "hypothesis"
    if token.startswith("Q"):
        return "question"
    return "question"


def catalog(requirement_refs: Iterable[Any] | None = None) -> dict[str, dict[str, Any]]:
    """Known requirements keyed by id. Non-obligations stay in the catalog."""
    found: dict[str, dict[str, Any]] = {}
    for row in requirement_refs or []:
        if not isinstance(row, Mapping):
            continue
        ref = _text(row.get("id") or row.get("stable_id") or row.get("hypothesis_id"))
        if not ref:
            continue
        prior = found.get(ref)
        if prior and isinstance(prior.get("provenance"), Mapping) and not isinstance(row.get("provenance"), Mapping):
            # A compact graph alias must not replace the accepted catalog metadata.
            if row.get("hypothesis_id"):
                prior["hypothesis_id"] = row["hypothesis_id"]
            continue
        found[ref] = {
            **row,
            "id": ref,
            "kind": _kind(row, ref),
            "formulation": _text(row.get("formulation") or row.get("statement") or row.get("text")),
            "obligation": row.get("obligation") is not False,
            "expected": _text(row.get("expected") or row.get("expected_result") or row.get("formulation")),
            "completion": "; ".join(_text(c.get("text")) for c in row.get("criteria") or [] if c.get("obligation"))
                or _text(row.get("completion") or row.get("completion_condition")),
        }
    return found


def target_ids(design: Mapping[str, Any] | Any) -> list[str]:
    """``target_refs``, with the legacy hypothesis fields folded in once."""
    if not isinstance(design, Mapping):
        getter = getattr(design, "covered_target_ids", None)
        if callable(getter):
            return sorted(getter())
        design = {}
    ordered: list[str] = []
    raw = [
        *(design.get("target_refs") or []),
        design.get("hypothesis_ref"),
        *(design.get("also_tests") or []),
    ]
    for item in raw:
        text = _text(item)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def hypothesis_ids(design: Mapping[str, Any], requirement_refs: Iterable[Any] = ()) -> list[str]:
    """Resolve catalog ids to graph hypothesis ids; keep legacy H1 references."""
    known: dict[str, str] = {}
    for row in requirement_refs:
        if not isinstance(row, Mapping):
            continue
        graph_id = _text(row.get("hypothesis_id") or row.get("id"))
        if row.get("kind") == "hypothesis" or row.get("hypothesis_id"):
            for ref in (row.get("id"), row.get("stable_id"), graph_id):
                if ref:
                    known[str(ref)] = graph_id
    explicit = {_text(design.get("hypothesis_ref")), *(_text(x) for x in design.get("also_tests") or [])}
    found = []
    for ref in target_ids(design):
        hid = known.get(ref)
        if not hid and (ref in explicit or _HYPOTHESIS_ID.fullmatch(ref.upper())):
            hid = ref.upper() if _HYPOTHESIS_ID.fullmatch(ref.upper()) else ref
        if hid and hid not in found:
            found.append(hid)
    return found


def unknown_targets(
    tasks: Iterable[Any],
    known: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Target ids that are not in the catalog. Empty catalog means no check."""
    if not known:
        return []
    missing: list[tuple[str, str]] = []
    for task in tasks:
        task_id, design = _task_parts(task)
        for ref in target_ids(design):
            if ref not in known:
                missing.append((task_id, ref))
    return missing


def uncovered_obligations(
    tasks: Iterable[Any],
    known: Mapping[str, dict[str, Any]],
) -> list[str]:
    """Obligations with no non-optional task. Preparatory tasks may share one."""
    covered: set[str] = set()
    for task in tasks:
        _task_id, design, optional = _task_bits(task)
        if optional:
            continue
        covered.update(target_ids(design))
    return [
        ref for ref, row in known.items()
        if row.get("obligation") is not False and ref not in covered
    ]


def _task_parts(task: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(task, Mapping):
        design = task.get("design") or {}
        return _text(task.get("id")), design if isinstance(design, Mapping) else {}
    design = getattr(task, "design", None)
    dumped = design.model_dump(mode="json") if hasattr(design, "model_dump") else {}
    return _text(getattr(task, "id", "")), dumped


def _task_bits(task: Any) -> tuple[str, Mapping[str, Any], bool]:
    task_id, design = _task_parts(task)
    if isinstance(task, Mapping):
        optional = bool(task.get("optional"))
        depends = [str(item) for item in (task.get("depends_on") or [])]
    else:
        optional = bool(getattr(task, "optional", False))
        depends = [str(item) for item in (getattr(task, "depends_on", None) or [])]
    design = dict(design)
    design["_depends_on"] = depends
    return task_id, design, optional


def _closing_ids(linked: list[tuple[str, list[str]]]) -> set[str]:
    """A linked task is preparatory when another linked task depends on it."""
    ids = {task_id for task_id, _deps in linked}
    upstream: set[str] = set()
    changed = True
    while changed:
        changed = False
        for _task_id, deps in linked:
            for dep in deps:
                if dep in ids and dep not in upstream:
                    upstream.add(dep)
                    changed = True
    leaves = ids - upstream
    return leaves or ids


def closing_task_ids(tasks: Iterable[Any], ref: str) -> list[str]:
    linked: list[tuple[str, list[str]]] = []
    explicit: set[str] = set()
    supporting: set[str] = set()
    order: list[str] = []
    for task in tasks:
        task_id, design, optional = _task_bits(task)
        if optional or ref not in target_ids(design):
            continue
        roles = [link.get("role") for link in design.get("target_links") or []
                 if link.get("requirement_id") == ref]
        if roles:
            if "delivers" in roles:
                explicit.add(task_id)
            else:
                supporting.add(task_id)
        linked.append((task_id, list(design.get("_depends_on") or [])))
        order.append(task_id)
    leaves = _closing_ids(linked)
    return [task_id for task_id in order
            if task_id in explicit or (task_id in leaves and task_id not in supporting)]


def _result_rows(results: Iterable[Any], task_states: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Newest attempt of each current task revision; never prefer success."""
    grouped: dict[str, dict[str, Any]] = {}
    for item in results or []:
        if not isinstance(item, Mapping) or not item.get("task_id"):
            continue
        task_id = str(item["task_id"])
        state = (task_states or {}).get(task_id) or {}
        version = int(item.get("result_version") or 1)
        if version < int(state.get("operation_revision") or 0) + 1:
            continue
        prior = grouped.get(task_id)
        rank = lambda row: (int(row.get("result_version") or 1), int(row.get("attempt_no") or 0), str(row.get("finished_at") or ""))
        if prior is None or rank(item) >= rank(prior):
            grouped[task_id] = dict(item)
    return grouped


def _criterion_results(tasks: list[Any], results: Mapping[str, Any], ref: str) -> dict[str, bool | None]:
    """Join only explicit task-criterion → requirement-criterion references."""
    found: dict[str, list[bool | None]] = {}
    for task in tasks:
        task_id, _design = _task_parts(task)
        payload = task if isinstance(task, Mapping) else task.model_dump(mode="json")
        result = results.get(task_id) or {}
        checks = {c.get("criterion_id"): c for c in result.get("criteria_checks") or []}
        executed = result.get("status") in {"success", "partial"} and not is_infrastructure_failure(result)
        for criterion in payload.get("success_criteria") or []:
            if criterion.get("requirement_id") != ref or not criterion.get("requirement_criterion_id"):
                continue
            check = checks.get(criterion.get("criterion_id")) or {}
            found.setdefault(criterion["requirement_criterion_id"], []).append(check.get("passed") if executed else None)
    return {key: (False if False in values else True if all(v is True for v in values) else None)
            for key, values in found.items()}


def _failed_criteria(result: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    for check in result.get("criteria_checks") or []:
        if isinstance(check, Mapping) and check.get("passed") is False:
            failed.append(_text(check.get("criterion_id") or check.get("details")))
    return [item for item in failed if item]


def _grounds(result: Mapping[str, Any]) -> bool:
    """An answer needs an artifact or a recorded criterion, not a bare status."""
    if result.get("grounded") is not None:
        return bool(result["grounded"])
    if (result.get("outputs") or {}).get("grounded") is not None:
        return bool(result["outputs"]["grounded"])
    if result.get("artifacts"):
        return True
    for check in result.get("criteria_checks") or []:
        if isinstance(check, Mapping) and check.get("passed") is True and _text(check.get("details")):
            return True
    return False


def _scoped_result(result: Mapping[str, Any], ref: str, hypothesis_id: str | None = None) -> dict[str, Any]:
    """Use the explicitly addressed outcome without borrowing another part's output."""
    scoped = dict(result)
    outputs = result.get("outputs") or {}
    per_requirement = outputs.get("requirements")
    if isinstance(per_requirement, Mapping) and per_requirement:
        specific = per_requirement.get(ref) or {}
        for key in ("answer", "grounded", "answer_grounded", "verdict", "produced", "produced_ids", "property_verified", "limitation"):
            scoped.pop(key, None)
        scoped["outputs"] = dict(specific)
        scoped["summary"] = specific.get("summary", "")
        scoped["artifacts"] = [a for a in result.get("artifacts") or []
                               if a.get("artifact_id") in (specific.get("artifact_ids") or [])]
    scientific = scoped.get("scientific_check") or {}
    if scientific.get("hypothesis_ref") and scientific["hypothesis_ref"] not in {ref, hypothesis_id}:
        scoped.pop("scientific_check", None)
    return scoped


def _verdict(result: Mapping[str, Any]) -> str:
    outputs = result.get("outputs") if isinstance(result.get("outputs"), Mapping) else {}
    scientific = result.get("scientific_check") or {}
    raw = _text(
        result.get("verdict")
        or (outputs or {}).get("verdict")
        or (outputs or {}).get("hypothesis_verdict")
        or scientific.get("status")
    ).lower()
    if raw == "supported":
        return "confirmed"
    if raw in {"confirmed", "refuted", "inconclusive"}:
        return raw
    return ""


def assess_target(
    kind: str,
    outcomes: list[Mapping[str, Any]],
    requirement: Mapping[str, Any] | None = None,
    *, criteria_checks: dict[str, bool | None] | None = None,
) -> tuple[str, str, str, str]:
    """Return status, actual, grounds, debt.

    ``task.done`` is not an input. Infrastructure failure is not a verdict
    and does not refute a hypothesis or close a question.
    """
    if not outcomes:
        return "open", "", "", "no closing result yet"
    latest = _scoped_result(outcomes[-1], requirement["id"], requirement.get("hypothesis_id")) if requirement else dict(outcomes[-1])
    infra = bool(latest.get("infrastructure_error") or is_infrastructure_failure(latest))
    status = _text(latest.get("status")).lower()
    if infra or status not in {"success", "partial"}:
        return "open", "", "tool error is not a scientific result", "execution failed"
    outputs = latest.get("outputs") if isinstance(latest.get("outputs"), Mapping) else {}
    answer_grounded = latest.get("answer_grounded", outputs.get("answer_grounded"))
    canonical = requirement and isinstance(requirement.get("provenance"), Mapping)
    # Old catalog rows are adapted to the same assessor. Only their historical
    # answer-in-summary representation is accepted at this compatibility boundary.
    requirement = dict(requirement or {})
    requirement.setdefault("id", "legacy")
    requirement.setdefault("kind", kind)
    requirement.setdefault("formulation", "")
    requirement.setdefault("obligation", True)
    if not canonical:
        if kind == "hypothesis":
            requirement.setdefault("hypothesis_mode", "check")
        latest = {**latest, "answer": latest.get("answer") or outputs.get("answer") or latest.get("summary") or ""}
        latest.setdefault("grounded", _grounds(latest))
    from CoScientist.requirements.completion import assess_requirement
    from CoScientist.requirements.models import OutcomeRecord, RequirementPart, Volume
    from pydantic import ValidationError

    part = RequirementPart.model_validate(dict(requirement))
    artifacts = []
    for artifact in latest.get("artifacts") or []:
        if not isinstance(artifact, Mapping):
            continue
        artifact = dict(artifact)
        if artifact.get("bucket") and artifact.get("s3_key"):
            artifact.setdefault("uri", f"s3://{artifact['bucket']}/{artifact['s3_key']}")
        artifacts.append(artifact)
    actual = _text(latest.get("summary"))
    try:
        record = OutcomeRecord(
            part_id=part.id, executed=True,
            grounded=(bool(latest.get("grounded", outputs.get("grounded", False)))
                      if kind == "question" else _grounds(latest)),
            answer_grounded=answer_grounded,
            answer=_text(latest.get("answer") or outputs.get("answer") or (actual if kind != "question" else "")),
            verdict=_verdict(latest), artifacts=artifacts,
            produced=latest.get("produced", outputs.get("produced")),
            produced_ids=latest.get("produced_ids", outputs.get("produced_ids")),
            property_verified=latest.get("property_verified", outputs.get("property_verified")),
            limitation=_text(latest.get("limitation", outputs.get("limitation"))),
            criteria_checks=criteria_checks or {},
        )
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        return "open", actual, "", f"invalid outcome fields: {fields}"
    assessed, why = assess_requirement(
        part, record, Volume.model_validate(requirement.get("volume") or {}),
    )
    failed = _failed_criteria(latest)
    if assessed == "met" and (failed or status == "partial"):
        assessed, why = "partial", "criteria remain" if failed else "partial delivery"
    mapped = {"met": "met", "partial": "partial", "unmet": "open"}[assessed]
    grounded = record.grounded
    if kind == "question" and record.answer_grounded is not None:
        grounded = record.answer_grounded
    shown = (record.verdict if kind == "hypothesis" and part.hypothesis_mode == "check"
             else record.answer if kind == "question" else record.answer or actual)
    return mapped, shown, actual if grounded else "", why


def project_requirements(
    tasks: Iterable[Any],
    requirement_refs: Iterable[Any] | None = None,
    results: Iterable[Any] | None = None,
    *, task_states: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Coverage and fulfillment for every known requirement, in catalog order."""
    task_list = list(tasks)
    known = catalog(requirement_refs)
    if not known:
        for task in task_list:
            _task_id, design = _task_parts(task)
            for ref in target_ids(design):
                known.setdefault(ref, {
                    "id": ref,
                    "kind": _kind({}, ref),
                    "formulation": "",
                    "obligation": True,
                    "expected": "",
                    "completion": "",
                })
    grouped = _result_rows(results or [], task_states)
    rows: list[dict[str, Any]] = []
    task_payloads = {(_task_parts(t)[0]): (t if isinstance(t, Mapping) else t.model_dump(mode="json")) for t in task_list}
    for ref, meta in known.items():
        serving: list[dict[str, Any]] = []
        for task in task_list:
            task_id, design, optional = _task_bits(task)
            if ref not in target_ids(design):
                continue
            serving.append({"task_id": task_id, "optional": optional})
        closing = closing_task_ids(task_list, ref)
        checks = _criterion_results(task_list, grouped, ref)
        assessed = []
        closing_outcomes = []
        for tid in closing:
            outcome = dict(grouped.get(tid) or {})
            criteria = {c.get("criterion_id"): c for c in task_payloads[tid].get("success_criteria") or []}
            outcome["criteria_checks"] = [c for c in outcome.get("criteria_checks") or []
                if criteria.get(c.get("criterion_id"), {}).get("requirement_id") == ref
                or (not criteria.get(c.get("criterion_id"), {}).get("requirement_id")
                    and criteria.get(c.get("criterion_id"), {}).get("purpose") != "assessment")]
            closing_outcomes.append(outcome)
            assessed.append((tid, assess_target(meta["kind"], [outcome] if tid in grouped else [], meta,
                                                criteria_checks=checks)))
        statuses = [result[0] for _, result in assessed]
        status = "met" if statuses and all(s == "met" for s in statuses) else (
            "partial" if any(s in {"met", "partial"} for s in statuses) else "open")
        # A count can span tasks only when every delivery explicitly identifies
        # its items. Without identities, sums could count the same rows twice.
        combined = None
        if meta["kind"] == "deliverable" and len(closing) > 1 and isinstance(meta.get("provenance"), Mapping):
            scoped = [_scoped_result(outcome, ref, meta.get("hypothesis_id")) for outcome in closing_outcomes]
            item_sets = [(item.get("outputs") or {}).get("produced_ids") for item in scoped]
            if all(item.get("status") == "success" and not is_infrastructure_failure(item) for item in scoped) and all(
                isinstance(ids, list) and all(isinstance(value, str) and value.strip() for value in ids)
                for ids in item_sets
            ):
                from CoScientist.requirements.deliverables import artifact_satisfies
                from CoScientist.requirements.models import RequirementPart
                part = RequirementPart.model_validate(meta)
                if all(any(artifact_satisfies(part, art)[0] for art in item.get("artifacts") or []) for item in scoped):
                    ids = sorted({value for values in item_sets for value in values})
                    properties = [(item.get("outputs") or {}).get("property_verified") for item in scoped]
                    merged = {"status": "success", "summary": "\n".join(_text(item.get("summary")) for item in scoped),
                              "outputs": {"produced": len(ids), "produced_ids": ids,
                                          "property_verified": False if False in properties else True if all(p is True for p in properties) else None},
                              "artifacts": [art for item in scoped for art in item.get("artifacts") or []],
                              "criteria_checks": [check for item in scoped for check in item.get("criteria_checks") or []]}
                    combined = assess_target(meta["kind"], [merged], meta, criteria_checks=checks)
                    status = combined[0]
        verdicts = {result[1] for _, result in assessed if result[0] == "met" and result[1] in {"confirmed", "refuted"}}
        conflict = meta["kind"] == "hypothesis" and len(verdicts) > 1
        if conflict:
            status = "partial"
        actual = "\n".join(result[1] for _, result in assessed if result[1])
        grounds = "\n".join(result[2] for _, result in assessed if result[2])
        debt = "; ".join(f"{tid}: {result[3]}" for tid, result in assessed if result[3]) if assessed else "no closing task"
        # Keep the single-task reason stable for graph/completion consumers.
        if len(assessed) == 1:
            debt = assessed[0][1][3]
        if combined is not None:
            actual, grounds, debt = combined[1:]
        if conflict:
            debt = "conflicting scientific verdicts require reconciliation"
        artifacts = []
        chain = []
        for tid, result in assessed:
            outcome = grouped.get(tid) or {}
            current_artifacts = _scoped_result(outcome, ref, meta.get("hypothesis_id")).get("artifacts") or []
            if outcome.get("status") in {"success", "partial"}:
                artifacts.extend(current_artifacts)
            chain.append({
                "requirement": ref, "task_id": tid,
                "attempt_id": outcome.get("attempt_id"), "result_id": outcome.get("result_id"),
                "artifact_ids": [a.get("artifact_id") for a in current_artifacts], "status": result[0],
            })
        rows.append({
            "id": ref,
            "kind": meta["kind"],
            "formulation": meta["formulation"],
            "obligation": meta["obligation"],
            "tasks": [item["task_id"] for item in serving],
            "preparatory_tasks": [
                item["task_id"] for item in serving if item["task_id"] not in closing
            ],
            "closing_tasks": closing,
            "expected": meta["expected"],
            "completion": meta["completion"],
            "status": status,
            "actual": actual,
            "grounds": grounds,
            "debt": debt,
            "chain": chain,
            "criteria": [{**c, "passed": checks.get(c.get("id"))} for c in meta.get("criteria") or []],
            "artifacts": artifacts,
        })
    return rows


def artifact_location(artifact: Mapping[str, Any]) -> str:
    if artifact.get("session_artifact_id"):
        return "cos-artifact:" + str(artifact["session_artifact_id"])
    if artifact.get("bucket") and artifact.get("s3_key"):
        return f"s3://{artifact['bucket']}/{artifact['s3_key']}"
    return _text(artifact.get("external_url") or artifact.get("uri") or artifact.get("url")
                 or artifact.get("workspace_path") or artifact.get("path"))


def render_requirement_report(rows: Iterable[Mapping[str, Any]], lang: str = "en", *, planned: bool = False, markdown: bool = True) -> str:
    """The ordered catalog and its assessment, shared by plan, results and NIR."""
    from html import escape
    from urllib.parse import quote
    rows = list(rows)
    if not rows:
        return ""
    ru = str(lang or "en").lower().startswith("ru")
    kinds = {"question": "Вопрос" if ru else "Question", "hypothesis": "Гипотеза" if ru else "Hypothesis",
             "deliverable": "Результат" if ru else "Deliverable"}
    statuses = {"met": "выполнено", "partial": "частично", "open": "не выполнено"}
    lines = [("## " if markdown else "") + ("Требования" if ru else "Requirements")]
    for row in rows:
        ref = str(row.get("id") or "")
        if planned and markdown:
            lines.append(f'<a id="requirement-{escape(quote(ref, safe=""))}"></a>')
        shown_ref = f"`{ref}`" if markdown else ref
        lines += ["", ("### " if markdown else "") + f"{kinds.get(row.get('kind'), row.get('kind'))} {shown_ref} · {row.get('formulation') or ref}"]
        for criterion in row.get("criteria") or []:
            if criterion.get("technical"):
                continue
            check = "" if planned else ("✓ " if criterion.get("passed") is True else "✗ " if criterion.get("passed") is False else "? ")
            lines.append(f"- {check}{criterion.get('text')}")
        if planned:
            continue
        status = row.get("status") or "open"
        lines.append(f"{'Статус' if ru else 'status'}: {statuses.get(status, status) if ru else status}")
        actual_label = ("Ответ" if ru else "Answer") if row.get("kind") == "question" else ("Получено" if ru else "Result")
        if row.get("actual"):
            lines.append(f"{actual_label}: {row['actual']}")
        if row.get("grounds"):
            lines.append(f"{'Основание' if ru else 'Grounds'}: {row['grounds']}")
        for artifact in row.get("artifacts") or []:
            location = artifact_location(artifact)
            if location:
                name = str(artifact.get("name") or artifact.get("artifact_id") or "Artifact")
                lines.append(f"- [{name}](<{location}>)" if markdown else f"{name}: {location}")
        if row.get("debt"):
            lines.append(f"{'Не завершено' if ru else 'Remaining'}: {row['debt']}")
    return "\n".join(lines)


def projection_from_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime = state.get("experiment_runtime") or {}
    plan = runtime.get("plan") or {}
    context = state.get("experiment_context") or {}
    statement = state.get("normalized_statement") or (state.get("research_frame") or {}).get("normalized_statement")
    if isinstance(statement, Mapping):
        refs = [
            {**part, "volume": statement.get("volume") or {}}
            for part in statement.get("parts") or [] if not part.get("retired")
        ]
    else:
        refs = list(context.get("requirement_refs") or state.get("requirement_refs") or [])
    for row in context.get("hypothesis_refs") or []:
        if isinstance(row, Mapping):
            # catalog preserves canonical metadata and joins the graph alias.
            refs.append({**row, "kind": row.get("kind") or "hypothesis",
                         **({"obligation": False} if statement else {})})
    rows = project_requirements(
        plan.get("tasks") or [],
        refs,
        runtime.get("results") or state.get("experiment_task_results") or [],
        task_states=runtime.get("tasks"),
    )
    if isinstance(statement, Mapping) and statement.get("uncertain"):
        for row in rows:
            row.update(status="open", debt=statement.get("uncertainty") or "incomplete requirement catalog")
    return rows
