"""The NIR report pipeline: contract fidelity, asset rules, and the off-switch.

The expensive half of this feature is unreachable without a running normcontrol
MCP, so what is tested here is everything up to the network boundary: that a
document built from a real research graph satisfies the contract, that the
asset encoder refuses exactly what the server would refuse, and — the invariant
that matters most — that a run which did not ask for a NIR report behaves
exactly as it did before this existed.
"""
from __future__ import annotations

import base64
import io
import json
import re
import zlib
from pathlib import Path

import pytest

from CoScientist.reporting.nir import (
    assets, build, contract, hitl_form, register_check, slop_check,
)
from CoScientist.reporting.nir.evidence import NirEvidence


# ── fixtures ────────────────────────────────────────────────────────────────


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A real 8x8 PNG, written by hand so the test needs no image library."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            len(payload).to_bytes(4, "big")
            + tag
            + payload
            + zlib.crc32(tag + payload).to_bytes(4, "big")
        )

    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def evidence() -> NirEvidence:
    """A small but structurally complete research graph."""
    ev = NirEvidence(user_id="u", session_id="s")
    ev.original_request = "Оценить токсичность метаболитов H. sosnowskyi"
    ev.nodes = [
        {"id": "Q1", "type": "ResearchQuestion", "status": "open",
         "attrs": {"formulation": "Какой кластер наиболее токсичен",
                   "gap": "Нет стандартизированного профиля"}},
        {"id": "H1", "type": "Hypothesis", "status": "refuted",
         "attrs": {"formulation": "Фуранокумарины токсичнее прочих. Медиана LD50 ниже 50 мг/кг",
                   "rationale": "Литературные данные о фототоксичности"},
         "status_history": [{"from": "under_verification", "to": "refuted",
                             "source": "ValidatorAgent", "reason": "LD50 выше порога"}]},
        {"id": "E1", "type": "Evidence", "status": "obtained",
         "attrs": {"subtype": "computational", "content": "Медиана 638.0 мг/кг",
                   "measured_on": "predict_ld50 (CatBoost)"}},
        {"id": "CL1", "type": "Conclusion", "status": "approved",
         "attrs": {"synthesis": "H1 опровергнута",
                   "how_established": "Прогноз LD50 по 15 соединениям"}},
        {"id": "VM1", "type": "VerificationMethod", "status": "done",
         "attrs": {"method_type": "computational", "description": "Кластеризация и прогноз",
                   "literature_basis": "OECD QSAR"}},
    ]
    ev.edges = [{"type": "refutes", "from": "E1", "to": "H1"}]
    ev.agent_reports = [{"agent": "CoderAgent", "text": "Построена дендрограмма", "status": "success"}]
    return ev


@pytest.fixture
def prose() -> build.NirProse:
    return build.NirProse(
        research_title="Токсикологический профиль метаболитов",
        report_title="Отчёт о токсикологическом профилировании",
        abstract_text="Объектом исследования являются метаболиты. Цель — оценить токсичность.",
        keywords=["токсикология", "фуранокумарины", "LD50", "кластеризация", "QSAR"],
        introduction_paragraphs=["Борщевик вызывает фотохимические ожоги."],
        conclusion_paragraphs=["Гипотеза H1 опровергнута."],
    )


# ── the contract ────────────────────────────────────────────────────────────


def test_built_document_carries_every_required_field(evidence, prose):
    """A run with graph data and an empty form still satisfies the contract.

    Empty requisites are the realistic case: the operator may skip the form.
    Every `must` field has to be present anyway, as a visible placeholder.
    """
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)

    assert values["contract"] == {"id": contract.CONTRACT_ID, "version": contract.CONTRACT_VERSION}
    document = values["document"]
    for path in contract.REQUIRED_FIELDS:
        node = document
        for part in path.split("."):
            assert part in node, f"{path} missing from the built document"
            node = node[part]
        assert node not in ({}, [], ""), f"{path} is empty"


def test_unfilled_requisites_become_placeholders_not_inventions(evidence, prose):
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    title = values["document"]["title"]
    assert title["udc"] == contract.PLACEHOLDER
    assert title["registration_nioktr"] == contract.PLACEHOLDER


def test_stage_is_sent_only_for_an_interim_report(evidence, prose):
    """`title.stage` is conditional; sending it otherwise is an unknown_value."""
    outline = build.build_outline(evidence)

    final, _ = build.build_nir_values(
        evidence, build.NirRequisites(report_type=contract.REPORT_TYPE_FINAL), prose, outline
    )
    assert "stage" not in final["document"]["title"]

    interim, _ = build.build_nir_values(
        evidence, build.NirRequisites(report_type=contract.REPORT_TYPE_INTERIM), prose, outline
    )
    assert "stage" in interim["document"]["title"]


def test_identifiers_are_unique_and_well_formed(evidence, prose):
    outline = build.build_outline(evidence)
    values, problems = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)

    assert not [p for p in problems if "повторяется" in p or "шаблону" in p]
    seen = set()
    for section in values["document"]["sections"]:
        assert contract.SLUG_RE.match(section["id"])
        assert section["id"] not in seen
        seen.add(section["id"])
    for reference in values["document"]["references"]:
        assert contract.REFERENCE_ID_RE.match(reference["id"])


def test_blocks_carry_only_keys_the_closed_world_allows(evidence, prose):
    """No private bookkeeping key may survive into the request."""
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)

    def walk(node, path="values"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not (isinstance(key, str) and key.startswith("_")), f"{path}.{key} leaked"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(values)
    for section in values["document"]["sections"]:
        for block in section["blocks"]:
            allowed = contract.block_allowed_keys(block["type"])
            assert set(block) <= allowed, f"{block['type']} carries {set(block) - allowed}"


def test_the_repository_is_cited_so_a_reader_can_reach_the_code(evidence, prose):
    """The MCP has no LLM and never reads our repo, so the link has to be content."""
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    references = " ".join(r["text"] for r in values["document"]["references"])
    assert build.REPOSITORY_URL in references


def test_presigned_urls_never_enter_the_bibliography(evidence, prose):
    """A signed link expires; printing one as a source ages the document badly."""
    evidence.nodes.append({
        "id": "E2", "type": "Evidence", "status": "obtained",
        "attrs": {"subtype": "computational",
                  "content": "Рисунок http://minio:9000/b/k.png?X-Amz-Signature=deadbeef"},
    })
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    assert "X-Amz" not in json.dumps(values["document"]["references"], ensure_ascii=False)


def test_an_empty_graph_still_yields_a_renderable_document(prose):
    """A failed run must produce a report that says so, not a validator error."""
    empty = NirEvidence(user_id="u", session_id="s")
    outline = build.build_outline(empty)
    values, problems = build.build_nir_values(empty, build.NirRequisites(), prose, outline)
    assert values["document"]["sections"]
    assert values["document"]["references"]
    assert values["document"]["conclusion"]
    assert problems


# ── assets ──────────────────────────────────────────────────────────────────


def test_only_png_and_jpeg_survive_encoding(tmp_path: Path):
    good = tmp_path / "plot.png"
    good.write_bytes(_png_bytes())
    vector = tmp_path / "diagram.svg"
    vector.write_text("<svg/>", encoding="utf-8")
    liar = tmp_path / "broken.png"
    liar.write_bytes(b"not an image at all")

    bundle = assets.build_assets([good, vector, liar])
    assert list(bundle.assets) == ["plot.png"]
    assert len(bundle.dropped) == 2
    assert any("diagram.svg" in d for d in bundle.dropped)


def test_asset_names_are_rewritten_to_what_the_server_accepts(tmp_path: Path):
    """The collector names files after tools, so Cyrillic and spaces arrive."""
    source = tmp_path / "рисунок 1 (итог).png"
    source.write_bytes(_png_bytes())
    bundle = assets.build_assets([source])
    name = bundle.name_for(source)
    assert name and contract.ASSET_NAME_RE.match(name), name
    assert name.endswith(".png")


def test_colliding_names_stay_distinct(tmp_path: Path):
    first = tmp_path / "a" / "plot.png"
    second = tmp_path / "b" / "plot.png"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_bytes(_png_bytes())
    bundle = assets.build_assets([first, second])
    assert len(bundle.assets) == 2
    assert bundle.name_for(first) != bundle.name_for(second)


def test_the_budget_drops_figures_loudly(tmp_path: Path):
    """A silently missing illustration reads like one that never existed."""
    path = tmp_path / "big.png"
    path.write_bytes(_png_bytes(64, 64))
    bundle = assets.build_assets([path], budget_bytes=10)
    assert not bundle.assets
    assert any("big.png" in d for d in bundle.dropped)


def test_request_size_counts_cyrillic_as_utf8():
    """The server measures ensure_ascii=False, so escapes would understate it."""
    values = {"document": {"text": "я" * 100}}
    assert assets.request_size(values, {}) < 400


def test_figures_reference_an_encoded_asset_name(evidence, prose, tmp_path: Path):
    figure = tmp_path / "дендрограмма.png"
    figure.write_bytes(_png_bytes())
    evidence.figures = [figure]

    bundle = assets.build_assets(evidence.figures)
    outline = build.build_outline(evidence, bundle)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)

    paths = [
        block["path"]
        for section in values["document"]["sections"]
        for block in section["blocks"]
        if block["type"] == "figure"
    ]
    assert paths
    for path in paths:
        assert path in bundle.assets, "figure.path must be a key of assets"


def test_written_captions_replace_the_filename_fallback(evidence, prose, tmp_path: Path):
    figure = tmp_path / "run_sandbox_task_dendrogram.png"
    figure.write_bytes(_png_bytes())
    evidence.figures = [figure]
    bundle = assets.build_assets(evidence.figures)
    outline = build.build_outline(evidence, bundle)

    prose.figure_captions = {"fig-1": "Дендрограмма кластеризации метаболитов"}
    values, problems = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    titles = [
        block["title"]
        for section in values["document"]["sections"]
        for block in section["blocks"]
        if block["type"] == "figure"
    ]
    assert titles == ["Дендрограмма кластеризации метаболитов"]
    assert not [p for p in problems if "без подписи" in p]


# ── the operator's answer ───────────────────────────────────────────────────


def test_only_an_explicit_choice_turns_the_report_on():
    """Silence, a timeout and a dismissal must all mean no."""
    assert hitl_form.wants_nir(hitl_form.OPTION_NIR)
    assert not hitl_form.wants_nir(hitl_form.OPTION_SHORT)
    assert not hitl_form.wants_nir(None)
    assert not hitl_form.wants_nir("")
    assert not hitl_form.wants_nir("да")


def test_form_answers_survive_a_round_trip_through_state():
    answers = {
        hitl_form.FORM_BLOCK_TITLE: {"udc": "004.9", "report_type": "промежуточный",
                                     "page_count": "42"},
        hitl_form.FORM_BLOCK_PEOPLE: {"supervisor_name": "И.И. Иванов"},
    }
    parsed = hitl_form.parse_requisites(answers)
    assert parsed.udc == "004.9"
    assert parsed.report_type == contract.REPORT_TYPE_INTERIM
    assert parsed.page_count == 42

    restored = hitl_form.requisites_from_state(hitl_form.requisites_to_state(parsed))
    assert restored.udc == parsed.udc
    assert restored.page_count == parsed.page_count


def test_a_nonsense_page_count_is_dropped_not_guessed():
    parsed = hitl_form.parse_requisites({"b": {"page_count": "много"}})
    assert parsed.page_count is None


def test_performers_are_never_empty(evidence, prose):
    """`performers` is a must array with min_items 1; [] fails validation."""
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    performers = values["document"]["performers"]
    assert performers
    for performer in performers:
        assert performer["contributed_sections"]


def test_performer_lines_are_split_into_fields(evidence, prose):
    outline = build.build_outline(evidence)
    requisites = build.NirRequisites(
        performers_raw="Исполнитель | инженер | П.П. Петров | разделы 1-3\n"
                       "Консультант | доцент | С.С. Сидоров | раздел 4"
    )
    values, _ = build.build_nir_values(evidence, requisites, prose, outline)
    performers = values["document"]["performers"]
    assert len(performers) == 2
    assert performers[0]["name"] == "П.П. Петров"
    assert performers[1]["contributed_sections"] == ["раздел 4"]


# ── style ───────────────────────────────────────────────────────────────────


def test_slop_check_flags_the_tells_it_claims_to():
    warnings = slop_check.check_paragraphs([
        "В современном мире токсикология играет важную роль.",
        "Возможно, вероятно, потенциально это может быть связано.",
        "Надеюсь, это поможет разобраться.",
    ], "тест")
    codes = {w.code for w in warnings}
    assert {"formulaic-opener", "hedge-stack", "chat-register"} <= codes


# ── the off switch ──────────────────────────────────────────────────────────


def _aggregator(monkeypatch, *, nir_enabled: bool, url: str | None):
    from CoScientist.assembly import build_system, load_config
    from CoScientist.assembly.schema import resolve_config_path
    from CoScientist.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.nir, "enabled", nir_enabled)
    monkeypatch.setattr(settings.mcp, "normcontrol_url", url)
    system = build_system(load_config(resolve_config_path("system")))
    return system


@pytest.mark.parametrize("nir_enabled", [False, True], ids=["flag-off", "flag-on"])
def test_no_normcontrol_server_means_no_agent_at_all(monkeypatch, nir_enabled):
    """Without somewhere to submit the document, the feature does not exist.

    The flag is irrelevant here: a run with no normcontrol server must be the
    run this pipeline produced before the feature existed — no extra subordinate
    in the roster, no extra tool, and nothing about НИР in the prompt.
    """
    system = _aggregator(monkeypatch, nir_enabled=nir_enabled, url=None)
    aggregator = system.agent("ResultAggregatorAgent")

    names = {getattr(t, "name", type(t).__name__) for t in aggregator.tools}
    assert "NirReportAgent" not in names
    assert not {n for n in names if n.startswith("nir_report")}
    assert "НИР" not in aggregator.instruction
    assert "{nir_block?}" in aggregator.instruction, (
        "the placeholder stays unresolved until the callback writes it"
    )


def test_the_operator_switch_does_not_decide_attachment(monkeypatch):
    """With a server configured the agent is built whether the flag is on or not.

    Attachment happens once, when the tree is assembled at import; the switch
    lives in the web Settings modal and is flipped long afterwards, so a
    build-time gate on it could only ever read the value the process started
    with. The switch therefore gates the QUESTION, not the wiring —
    ``ask_nir_report`` reads it on every run.

    What must still hold with the flag off is the part a reader can see: the
    aggregator's prompt says nothing about НИР, because `{nir_block?}` is only
    ever written by the callback, and the tools refuse without an operator
    request (see ``test_the_tools_refuse_without_an_operator_request``).
    """
    for enabled in (False, True):
        system = _aggregator(
            monkeypatch, nir_enabled=enabled, url="http://localhost:8001/mcp"
        )
        aggregator = system.agent("ResultAggregatorAgent")
        assert "NirReportAgent" in {getattr(t, "name", "") for t in aggregator.tools}
        assert "НИР" not in aggregator.instruction
        assert "{nir_block?}" in aggregator.instruction

        writer = system.agent("NirReportAgent")
        assert {getattr(t, "name", "") for t in writer.tools} == {
            "nir_report_outline", "nir_report_draft", "nir_report_submit"
        }


def test_the_callback_disables_itself_when_nothing_can_answer(monkeypatch):
    """No HITL, no server or no flag: write the off state and never ask."""
    import asyncio

    from CoScientist.config import get_settings
    from CoScientist.reporting.nir.callback import make_ask_nir_report_callback

    class _Unreachable:
        async def handle_request(self, request):  # pragma: no cover - must not run
            raise AssertionError("the operator was asked when nobody could answer")

        async def notify(self, payload):  # pragma: no cover
            return None

    settings = get_settings()
    monkeypatch.setattr(settings.nir, "enabled", True)
    monkeypatch.setattr(settings.mcp, "normcontrol_url", "http://localhost:8001/mcp")
    monkeypatch.setattr(settings.web, "hitl_enabled", False)

    class _Ctx:
        agent_name = "ResultAggregatorAgent"
        state: dict = {}

    ctx = _Ctx()
    ctx.state = {}
    callback = make_ask_nir_report_callback(_Unreachable())
    assert asyncio.run(callback(ctx)) is None
    assert ctx.state[hitl_form.STATE_REQUEST_KEY] == {"enabled": False}
    assert ctx.state[hitl_form.STATE_BLOCK_KEY] == ""


def test_real_scientific_prose_passes_clean():
    """An em dash as a copula and dense numbers are correct Russian, not slop."""
    warnings = slop_check.check_paragraphs([
        "Прогнозные LD50 лежат в диапазоне 597,6-1065,9 мг/кг.",
        "Медиана кластера — 638,0 мг/кг против 3516,0 мг/кг у остальных метаболитов, "
        "что даёт отрыв 5,5 раза вместо требуемых десяти. Критерий CC1.2 не выполнен.",
        "Корреляция Spearman не рассчитывалась.",
        "Все семь кумаринов прогнозно кардиотоксичны; среди остальных восьми "
        "положительных прогнозов нет.",
        "Домен применимости покрывает выборку целиком: минимальное значение 42 процента.",
    ], "тест")
    assert not warnings, [w.render() for w in warnings]


# ── register: the study's notation must not reach the page ──────────────────


def test_an_identifier_leaks_only_when_the_graph_actually_used_it(evidence):
    """The check that keeps chemistry out of the findings.

    ``H1`` is a proton and ``T1`` a relaxation time. Flagging either on the
    pattern alone would make the whole category noise, so membership in this
    run's id set is what decides.
    """
    ids = register_check.graph_identifiers(evidence.nodes)
    assert "H1" in ids and "CL1" in ids

    found = register_check.check_text("Вывод следует из CL1.", "проба", ids)
    assert [w.code for w in found] == ["graph-id"]

    # No node called Z9 anywhere in this study.
    assert not register_check.check_text("Полоса Z9 в спектре.", "проба", ids)


def test_products_and_metrics_survive_the_code_identifier_check():
    """Naming RDKit is correct; naming a callable is not."""
    clean = "Использованы RDKit, CatBoost и дескрипторы ECFP4; показатель LD50 и RMSE."
    assert not register_check.check_text(clean, "проба", set())

    dirty = "Инструмент butina_clustering вернул результат."
    assert [w.code for w in register_check.check_text(dirty, "проба", set())] == [
        "code-identifier"
    ]


@pytest.mark.parametrize(
    "text,code",
    [
        ("Задача TASK-4 выполнена.", "plan-id"),
        ("Дендрограмма сохранена в dendrogram.png.", "file-name"),
        ("ValidatorAgent зафиксировал вердикт.", "agent-name"),
        ("Статус гипотезы — inconclusive.", "untranslated"),
    ],
)
def test_each_leak_category_is_caught(text, code):
    assert code in {w.code for w in register_check.check_text(text, "проба", set())}


def test_a_repeated_leak_is_reported_once(evidence):
    """A leaked id repeats a dozen times; the author needs to hear it once."""
    paragraphs = ["Гипотеза H1 опровергнута."] * 8
    found = register_check.check_document({"раздел": paragraphs}, evidence.nodes)
    assert len(found) == 1


# ── digest: what the author is shown decides what they can copy ─────────────


def test_the_digest_carries_no_identifier_the_report_must_not_print(evidence):
    """The root cause of the first report's ~160 leaks.

    The digest opened every hypothesis with the node id and passed tool names
    and English statuses through, so the author was quoting what it was handed.
    """
    outline = build.build_outline(evidence)
    shown = {plan.id: plan.digest for plan in outline.sections}
    shown["материалы"] = [m["text"] for m in outline.materials]

    leaks = register_check.check_document(shown, evidence.nodes)
    forbidden = {"graph-id", "plan-id", "code-identifier", "file-name", "agent-name"}
    assert not [w for w in leaks if w.code in forbidden], [w.render() for w in leaks]


def test_an_identifier_becomes_what_it_names(evidence):
    """Deleting it would leave a dangling preposition; naming it keeps the sentence."""
    labels = build._label_map(evidence)
    assert labels["H1"] == "первая гипотеза"
    scrubbed = build._scrub("Проверка основана на H1.", labels)
    assert "H1" not in scrubbed and "первая гипотеза" in scrubbed


def test_a_tracker_id_becomes_a_russian_numbered_row():
    assert build._tracker_word("OP-3") == "операция 3"
    assert build._tracker_word("TASK-4") == "задача 4"
    assert build._tracker_word("NOTATRACKER") == "NOTATRACKER"


def test_scrubbing_a_path_does_not_eat_cyrillic():
    """Python's ``\\w`` matches Cyrillic, so an earlier pattern read
    "корпус/база знаний" as a path and deleted half the phrase."""
    assert "база знаний" in build._scrub("Взято из корпус/база знаний.")
    assert "workspace" not in build._scrub("Файл лежит в /workspace/figures/out.png.")


def test_an_automatic_status_note_is_not_evidence(evidence):
    """A maintainer's own bookkeeping note is not a finding to report."""
    evidence.nodes.append({
        "id": "H9", "type": "Hypothesis", "status": "refuted",
        "attrs": {"formulation": "Проверяемое утверждение"},
        "status_history": [
            {"to": "under_verification", "source": "graph-maintainer",
             "reason": "auto: evidence attached"},
            {"to": "refuted", "source": "ValidatorAgent",
             "reason": "Измеренное значение выше порога"},
        ],
    })
    outline = build.build_outline(evidence)
    hypothesis_sections = [s for s in outline.sections if s.node_id == "H9"]
    assert hypothesis_sections
    lines = " ".join(hypothesis_sections[0].digest)
    assert "auto:" not in lines
    assert "Измеренное значение выше порога" in lines


# ── headings and proportion ─────────────────────────────────────────────────


def test_a_fallback_heading_never_carries_an_identifier(evidence):
    outline = build.build_outline(evidence)
    for plan in outline.sections:
        assert not re.search(r"\b[A-Z]{1,3}\d+\b", plan.title), plan.title


def test_a_heading_is_dropped_rather_than_cut_mid_clause():
    """The first table of contents carried an id plus a list severed mid-item."""
    long_claim = {
        "id": "H1", "type": "Hypothesis", "status": "refuted",
        "attrs": {"formulation": (
            "Кластер линейных фуранокумаринов (псорален, ксантотоксин, бергаптен, "
            "императорин, бергамоттин) окажется наиболее токсичным по медиане"
        )},
    }
    title = build._hypothesis_title(long_claim, 1)
    assert title == "Проверка первой гипотезы"


def test_a_heading_declines_correctly():
    """"Проверка первая гипотезы" is the kind of mistake a reader stops on."""
    short = {"id": "H1", "type": "Hypothesis", "status": "refuted",
             "attrs": {"formulation": "Токсичность выше нормы"}}
    assert build._hypothesis_title(short, 1).startswith("Проверка первой гипотезы")


def test_a_written_heading_replaces_the_planned_one(evidence, prose):
    outline = build.build_outline(evidence)
    section_id = outline.sections[1].id
    prose.section_titles = {section_id: "Прогноз острой токсичности фуранокумаринов"}
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    titles = [s["title"] for s in values["document"]["sections"]]
    assert "Прогноз острой токсичности фуранокумаринов" in titles


def test_agent_reports_are_shared_material_not_one_section(evidence):
    """They used to be appended to the results section, which put the whole
    study's substance in one place while other sections had nothing."""
    evidence.agent_reports = [
        {"agent": "CoderAgent", "text": "Построена дендрограмма по 15 соединениям.",
         "status": "success"},
    ]
    outline = build.build_outline(evidence)
    assert outline.materials
    assert "дендрограмма" in outline.materials[0]["text"]
    for plan in outline.sections:
        assert not any("CoderAgent" in line for line in plan.digest)


def test_every_section_reports_how_much_material_stands_behind_it(evidence):
    outline = build.build_outline(evidence)
    assert any(plan.evidence_chars() > 0 for plan in outline.sections)


# ── the appendix speaks Russian ─────────────────────────────────────────────


def test_the_appendix_names_nothing_in_the_schema_s_english(evidence, prose):
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    appendices = values["document"].get("appendices") or []
    rendered = json.dumps(appendices, ensure_ascii=False)
    for token in ("Hypothesis", "Evidence", "derived_from", "tested_by", "Conclusion"):
        assert token not in rendered, token


def test_the_appendix_ties_the_report_s_names_back_to_the_record(evidence, prose):
    """Traceability belongs in an appendix, not in the prose."""
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    tables = [
        block
        for appendix in values["document"].get("appendices") or []
        for block in appendix["blocks"]
        if block.get("type") == "table"
    ]
    trace = [t for t in tables if t["id"] == "graph-trace"]
    assert trace, "no correspondence table"
    assert ["Первая гипотеза", "H1", "опровергнута"] in trace[0]["rows"]


def test_scrubbing_a_path_leaves_a_url_alone():
    """The path pattern once matched "//github.com/aimclub/CoScientist" inside a
    URL and cut the repository citation down to the word "https"."""
    cited = "Исходный код: https://github.com/aimclub/CoScientist"
    assert build._scrub(cited) == cited
    assert "pubchem.ncbi.nlm.nih.gov" in build._scrub(
        "См. https://pubchem.ncbi.nlm.nih.gov/compound/10658"
    )
    assert "workspace" not in build._scrub("Файл лежит в /workspace/out.png тут")


def test_a_workspace_path_is_not_a_bibliographic_source(evidence, prose):
    """Agents write a path into source_ref as readily as a DOI, and the first
    bibliography listed seven of them."""
    evidence.nodes.append({
        "id": "E7", "type": "Evidence", "status": "obtained",
        "attrs": {"subtype": "literature",
                  "source_ref": r"D:\projects26\code_a\workspace\ld50_predictions.json"},
    })
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    references = " ".join(r["text"] for r in values["document"]["references"])
    assert "ld50_predictions" not in references
    assert "projects26" not in references
    # A real citation in the same field still gets through.
    assert build.REPOSITORY_URL in references


def test_a_numbered_source_is_never_empty(evidence, prose):
    """Scrubbing a path would leave a numbered row with nothing in it."""
    outline = build.build_outline(evidence)
    values, _ = build.build_nir_values(evidence, build.NirRequisites(), prose, outline)
    for reference in values["document"]["references"]:
        assert reference["text"].strip()
