"""Structural tests for the MICROFLUIDICS profile (agents/microfluidics.yaml).

These build the real reduced system (no LLM calls) and assert the invariants
of the microfluidics pipeline: ТЗ stage runs FIRST, the planner receives the
structured ТЗ, and the orchestrator can only delegate to ResearchAgent.

Run from the repo root:  pytest tests/unit/test_microfluidics_assembly.py -q
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.assembly import build_system, load_config  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config(resolve_config_path("microfluidics"))


@pytest.fixture(scope="module")
def system(config):
    return build_system(config)


# ── config shape ─────────────────────────────────────────────────────────────

def test_profile_resolves_by_name():
    path = resolve_config_path("microfluidics")
    assert path.name == "microfluidics.yaml" and path.exists()


def test_root_pipeline_runs_tz_first(config):
    """ТЗ agent runs BEFORE the planner, which runs before the orchestrator."""
    assert config.root.name == "InitAgent"
    assert config.root.children == ["TZAgent", "PlannerAgent", "OrchestratorAgent"]
    assert config.agent("TZAgent").children == ["TZSpecAgent", "TZQueryGenAgent"]


def test_everything_but_literature_analysis_is_stripped(config):
    """The profile keeps only ТЗ + planner + orchestrator + research."""
    assert set(config.agents) == {
        "InitAgent",
        "TZAgent",
        "TZSpecAgent",
        "TZQueryGenAgent",
        "PlannerAgent",
        "OrchestratorAgent",
        "ResearchAgent",
    }
    assert config.agent("OrchestratorAgent").subordinates == ["ResearchAgent"]


def test_tz_stage_outputs_feed_the_session_state(config):
    assert config.agent("TZSpecAgent").output_key == "structured_tz"
    assert config.agent("TZSpecAgent").output_schema == "structured_tz"
    assert config.agent("TZQueryGenAgent").output_key == "tz_literature_queries"
    assert config.agent("TZQueryGenAgent").output_schema == "tz_literature_queries"


# ── built system invariants ──────────────────────────────────────────────────

def test_system_builds_all_agents(config, system):
    for name in config.agents:
        assert system.agent(name).name == name


def test_planner_prompt_consumes_the_tz(system):
    """The planner receives the ТЗ and the derived literature queries via ADK
    state injections, and registers the plan with create_plan."""
    instruction = system.agent("PlannerAgent").instruction
    assert "{structured_tz?}" in instruction
    assert "{tz_literature_queries?}" in instruction
    assert "create_plan" in instruction
    # Plan steps may only reference the research agent.
    assert "ResearchAgent" in instruction
    for absent in ("TaskExecutorAgent", "CoderAgent", "HypothesesAgent", "MedicalAgent"):
        assert absent not in instruction


def test_orchestrator_prompt_carries_tz_context_and_research_routing(system):
    instruction = system.agent("OrchestratorAgent").instruction
    assert "{structured_tz?}" in instruction
    assert "{active_tasks}" in instruction
    assert "ResearchAgent" in instruction
    assert "query_en" in instruction  # the ТЗ-derived query must be passed on
    for absent in ("TaskExecutorAgent", "CoderAgent", "retrieve_tools"):
        assert absent not in instruction


def test_tz_spec_prompt_has_all_tz_blocks(system):
    from CoScientist.microfluidics.models import CANONICAL_BLOCKS

    instruction = system.agent("TZSpecAgent").instruction
    for title in CANONICAL_BLOCKS:
        assert title in instruction, f"ТЗ block «{title}» missing from the TZ prompt"
    assert "original_request" in instruction


def test_query_gen_prompt_reads_the_tz(system):
    instruction = system.agent("TZQueryGenAgent").instruction
    assert "{structured_tz?}" in instruction
    assert "query_en" in instruction
    assert "LIT-01" in instruction


def test_no_unfilled_placeholders(config, system):
    for name in config.agents:
        instruction = getattr(system.agent(name), "instruction", "") or ""
        assert "<<" not in instruction, f"{name}: unfilled placeholder in prompt"


def test_output_schemas_registered_and_valid():
    from CoScientist.assembly.registry import REGISTRY
    from CoScientist.microfluidics.models import LiteratureQueries, StructuredTZ

    assert REGISTRY.output_schema("structured_tz") is StructuredTZ
    assert REGISTRY.output_schema("tz_literature_queries") is LiteratureQueries

    tz = StructuredTZ.model_validate({
        "original_request": "test",
        "blocks": [{
            "title": "Критерии качества",
            "usage": "Планирование анализа",
            "fields": [
                {"name": "Минимальная чистота образца", "value": "80 %",
                 "status": "задано заказчиком"},
                {"name": "Допустимые примеси"},
            ],
        }],
    })
    block = tz.block("Критерии качества")
    assert block is not None and block.fields[0].is_set()
    assert block.fields[1].status == "не задано" and not block.fields[1].is_set()

    queries = LiteratureQueries.model_validate(
        {"queries": [{"id": "LIT-01", "task": "т", "query_en": "q", "extract": ["x"]}]}
    )
    assert queries.queries[0].id == "LIT-01"


def test_tz_document_renderer_matches_reference_structure():
    """The renderer must reproduce the sections of the reference document
    (Пример_уточненного_структурированного_ТЗ_для_агентов.md)."""
    from CoScientist.microfluidics.models import CANONICAL_BLOCKS, StructuredTZ
    from CoScientist.microfluidics.render import render_tz_document

    tz = StructuredTZ.model_validate({
        "original_request": "Нужен ПАВ для МУН в минерализованной воде.",
        "blocks": [
            {"title": "Тип задачи", "usage": "Определяет сценарий",
             "fields": [{"name": "Тип задачи", "value": "Разработка ПАВ",
                         "status": "задано заказчиком"}]},
            {"title": "Целевой продукт",
             "fields": [
                 {"name": "Функция продукта", "value": "ПАВ для МУН",
                  "status": "задано заказчиком"},
                 {"name": "CAS целевого вещества", "value": "Не задан",
                  "status": "не задано"},
                 {"name": "Требования к промывке", "value": "Определить позже",
                  "status": "рассчитывается агентом"},
             ]},
        ],
    })
    doc = render_tz_document(tz)

    # Sections of the reference format.
    assert "## Правила интерпретации полей" in doc
    assert "## Исходный запрос заказчика в свободной форме" in doc
    assert "> Нужен ПАВ для МУН" in doc
    assert "## Единая структура собранных данных" in doc
    assert "## Тип задачи" in doc and "## Целевой продукт" in doc
    assert "| Поле | Значение | Статус |" in doc
    assert "| Функция продукта | ПАВ для МУН | Задано заказчиком |" in doc
    assert "## Поля, которые остаются свободными или незаполненными" in doc
    assert "| Целевой продукт | CAS целевого вещества | Не задан |" in doc
    assert "## Проверка соответствия исходному опроснику" in doc
    # Missing canonical blocks are flagged, present ones counted.
    assert "| Ограничения по поставкам | Отсутствует | Блок не сформирован |" in doc
    assert "| Тип задачи | Есть | Заполнено 1 из 1 полей |" in doc
    assert doc.count("Отсутствует") == len(CANONICAL_BLOCKS) - 2


def test_json_sanitizer_extracts_payload_for_schema_validation():
    """The after_model sanitizer (ported from VibePAV's _extract_json) must
    strip fences/prose/trailing text so strict output_schema validation passes."""
    from CoScientist.agents.callbacks.json_output import _extract_json

    payload = {"queries": [{"id": "LIT-01", "task": "т", "query_en": "q", "extract": []}]}
    import json as _json
    clean = _json.dumps(payload, ensure_ascii=False)

    assert _extract_json(clean) == payload
    assert _extract_json(f"```json\n{clean}\n```") == payload
    assert _extract_json(f"Вот результат:\n{clean}\nНадеюсь, помог!") == payload
    assert _extract_json(clean + "\nERROR trailing text") == payload
    assert _extract_json("никакого json здесь нет") is None


def test_tz_agents_wire_the_json_sanitizer(config):
    for name in ("TZSpecAgent", "TZQueryGenAgent"):
        assert "sanitize_json_output" in config.agent(name).callbacks.after_model


def test_tz_agents_have_hitl_review_loops(config, system):
    """Both ТЗ stages (like the planner) are session agents with a human
    review loop; the loop engages only when HITL is globally enabled
    (HITL__ENABLED), so headless testing needs no config changes."""
    from CoScientist.config import get_settings
    from CoScientist.hitl.session_agent import SessionAgent

    for name in ("TZSpecAgent", "TZQueryGenAgent", "PlannerAgent"):
        cfg = config.agent(name)
        assert cfg.cls in ("custom:session", "custom:tz_session"), (
            f"{name} must be a session agent"
        )
        assert cfg.hitl is True, f"{name} must declare hitl"
        agent = system.agent(name)
        assert isinstance(agent, SessionAgent)
        assert agent.hitl_handler is not None, f"{name}: no HITL handler wired"


def test_tz_review_shows_rendered_document_and_publishes_it(config, system, tmp_path):
    """The human reviews the RENDERED ТЗ document, not raw JSON; once accepted
    the agent publishes it (file + chat event hook)."""
    from CoScientist.hitl.session_agent import SessionAgent
    from CoScientist.microfluidics.tz_agent import TZSessionAgent, persist_tz_document

    agent = system.agent("TZSpecAgent")
    assert isinstance(agent, TZSessionAgent)
    preview = agent._review_output({"original_request": "тест", "blocks": []})
    assert "## Правила интерпретации полей" in preview
    # A broken payload must degrade to raw text, never crash the review.
    assert agent._review_output("не json") == "не json"

    # The publication hook is overridden (the base SessionAgent emits nothing).
    assert TZSessionAgent._post_final_events is not SessionAgent._post_final_events

    # The persist helper renders and writes the document file.
    document, path = persist_tz_document(
        {"original_request": "тест", "blocks": []}, out_dir=tmp_path
    )
    assert "## Правила интерпретации полей" in document
    assert path is not None and path.read_text(encoding="utf-8") == document


def test_operator_questionnaire_built_from_open_fields():
    """The review questionnaire (ported from VibePAV) asks exactly about the
    blocks that still have «не задано» fields, with escape answers that let
    the pipeline continue («не знаю», «на усмотрение агента», Accept)."""
    from CoScientist.microfluidics.questionnaire import (
        build_questionnaire,
        render_questionnaire,
    )

    tz = {
        "original_request": "тест",
        "blocks": [
            {"title": "Тип задачи", "fields": [
                {"name": "Тип задачи", "value": "Разработка ПАВ",
                 "status": "задано заказчиком"},
            ]},
            {"title": "Целевой продукт", "fields": [
                {"name": "Функция продукта", "value": "ПАВ",
                 "status": "задано заказчиком"},
                {"name": "CAS целевого вещества"},
                {"name": "SMILES целевого вещества"},
            ]},
            {"title": "Ограничения по технологии", "fields": [
                {"name": "Требования к промывке", "value": "позже",
                 "status": "рассчитывается агентом"},
            ]},
        ],
    }
    qs = build_questionnaire(tz)
    by_block = {q.block: q for q in qs}

    # Fully-filled block → no question; «рассчитывается агентом» → not asked.
    assert "Тип задачи" not in by_block
    assert "Ограничения по технологии" not in by_block
    # A block with open fields is asked, naming the exact gaps.
    assert "CAS целевого вещества" in by_block["Целевой продукт"].open_fields
    # Missing canonical blocks are asked too: 16 - 2 excluded = 14 questions.
    assert len(qs) == 14
    assert [q.num for q in qs] == list(range(1, 15))
    # VibePAV originals survive verbatim where applicable.
    assert by_block["Аналитические методы"].text == "Какая аналитика доступна?"
    assert by_block["Масштаб результата"].hint == (
        "напр.: лабораторная рецептура и опытная партия"
    )

    rendered = render_questionnaire(qs)
    assert "## Уточняющие вопросы оператору" in rendered
    assert "не знаю" in rendered
    assert "на усмотрение агента" in rendered
    assert "Accept" in rendered
    assert "**Q1 ·" in rendered
    assert render_questionnaire([]) == ""

    # At review time the ТЗ may still be the RAW JSON STRING (state not yet
    # committed) — the questionnaire must handle it like the renderer does.
    import json as _json
    qs_from_str = build_questionnaire(_json.dumps(tz, ensure_ascii=False))
    assert [q.block for q in qs_from_str] == [q.block for q in qs]


def test_tz_review_shows_document_with_interview_note(system):
    """The document review shows the document + a note that the interview
    (one window per question) follows — NOT the inline questionnaire dump."""
    agent = system.agent("TZSpecAgent")
    preview = agent._review_output({"original_request": "т", "blocks": []})
    assert "## Правила интерпретации полей" in preview  # документ
    assert "## Уточняющие вопросы оператору" not in preview
    assert "по одному окну на вопрос" in preview
    assert "таких блоков: 16" in preview  # все 16 блоков пусты


def test_question_card_and_options():
    """Each interview window is self-contained: question, hint, open fields
    and the escape options that keep the pipeline moving."""
    from CoScientist.microfluidics.questionnaire import (
        OPT_AGENT,
        OPT_SKIP,
        OPT_STOP,
        QUESTION_OPTIONS,
        build_questionnaire,
        render_question_card,
    )

    assert list(QUESTION_OPTIONS) == [OPT_SKIP, OPT_AGENT, OPT_STOP]

    tz = {"original_request": "т", "blocks": [
        {"title": "Масштаб результата", "fields": [
            {"name": "Масштаб текущего результата", "value": "1 г",
             "status": "задано заказчиком"},
            {"name": "Масштаб следующей проверки"},
        ]},
    ]}
    questions = build_questionnaire(tz)
    q = next(x for x in questions if x.block == "Масштаб результата")
    card = render_question_card(q, 3, 7)
    assert "вопрос 3 из 7" in card
    assert "Масштаб результата" in card and q.text in card
    assert "Масштаб следующей проверки" in card  # именно незаполненное поле
    assert "Масштаб текущего результата" not in card  # заполненное не спрашиваем
    for opt in QUESTION_OPTIONS:
        assert opt in card


def test_default_profile_is_untouched():
    """Without $COSCIENTIST_CONFIG the default system.yaml still loads with the
    full agent roster."""
    default = load_config()
    assert "TZAgent" not in default.agents
    for name in ("CoderAgent", "TaskExecutorAgent", "MedicalAgent"):
        assert name in default.agents
