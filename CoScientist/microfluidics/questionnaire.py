"""Operator questionnaire for the ТЗ review, generated from the draft's gaps.

Ported from VibePAV's interactive questionnaire (``tz_agent.QUESTIONS``): the
question texts and example hints are taken from the original 12 questions and
extended to cover all 16 canonical blocks of the reference ТЗ document.

Unlike VibePAV (which interviewed the operator BEFORE synthesising the ТЗ),
here the questionnaire is built deterministically AFTER the draft: one
question per canonical block that still has fields with status «не задано»,
listing those fields. It is appended to the review document the operator sees
in HITL, with three ways to answer each question:

  * a concrete value              -> the field is filled («уточнено оператором»)
  * «не знаю» / skip              -> the field stays «не задано», the pipeline continues
  * «на усмотрение агента»        -> the agent fills a reasonable working value

Pressing Accept without answers continues with the open questions as-is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Union

from CoScientist.microfluidics.models import CANONICAL_BLOCKS, StructuredTZ

# Question text + example hint per canonical block. The first twelve entries
# follow VibePAV's QUESTIONS (merged where the reference document merged the
# blocks); the rest are derived from the reference ТЗ example.
QUESTION_BANK: dict[str, tuple[str, str]] = {
    "Тип задачи": (
        "Что именно нужно: отдельное вещество, рецептура, аналог продукта или "
        "состав с заданными свойствами?",
        "напр.: подбор ПАВ-содержащей рецептуры",
    ),
    "Целевой продукт": (
        "Есть ли предпочтительные классы веществ или целевой продукт "
        "(конкретное вещество, CAS/SMILES, торговый аналог)?",
        "напр.: амфотерные ПАВ, бетаины, сульфосукцинаты, ПИБ-эмульгаторы",
    ),
    "Область применения": (
        "Где будет применяться продукт и в каких условиях "
        "(минерализация, температура, ионы)?",
        "напр.: ХМУН/МУН, смачиватель; высокая минерализация, 60–90 °C, Ca²⁺/Mg²⁺",
    ),
    "Требуемые свойства": (
        "Какие свойства критичны и есть ли дополнительные требования "
        "(стабильность эмульсии, вязкость)?",
        "напр.: низкое IFT нефть/вода, солеустойчивость, термостабильность",
    ),
    "Критерии качества": (
        "Минимальные критерии успеха и качества образца "
        "(чистота, масса, подтверждение структуры)?",
        "напр.: 3–5 перспективных классов с данными; чистота ≥80 %, образец ≥1 г",
    ),
    "Масштаб результата": (
        "Какой масштаб результата нужен?",
        "напр.: лабораторная рецептура и опытная партия",
    ),
    "Ограничения по сырью": (
        "Ограничения по сырью?",
        "напр.: доступное в РФ сырьё — жирные кислоты/эфиры, амины, малеиновый ангидрид",
    ),
    "Ограничения по поставкам": (
        "Требования к поставкам сырья: география, срок поставки, число "
        "независимых поставщиков, минимальная партия?",
        "напр.: РФ, срок ≤45 дней, ≥2 поставщика, партия ≤1 кг",
    ),
    "Ограничения по себестоимости": (
        "Есть ли предельная себестоимость и нужен ли расчёт/сравнение "
        "маршрутов по стоимости сырья?",
        "напр.: сравнить маршруты по руб./кг без жёсткого порога отсечения",
    ),
    "Ограничения по технологии": (
        "Ограничения по технологии?",
        "напр.: проточная/микрофлюидная установка, умеренные температуры, "
        "без газофазных стадий; до 4 стадий",
    ),
    "Доступное оборудование": (
        "Какое оборудование доступно (тип установки, расходы, давление, "
        "температуры, материалы)?",
        "напр.: проточная установка 0,05–20 мл/мин, до 20 бар, –30…+200 °C, сталь 316L",
    ),
    "Аналитические методы": (
        "Какая аналитика доступна?",
        "напр.: тензиометрия, ККМ по проводимости, вязкость, стабильность "
        "эмульсий; ЯМР, ВЭЖХ",
    ),
    "Известные данные заказчика": (
        "Есть ли у заказчика статьи, патенты, отчёты, старые методики или "
        "данные о неудачных опытах?",
        "если есть — перечислите или приложите файлами",
    ),
    "Безопасность и регуляторика": (
        "Есть ли запрещённые вещества/растворители или особые регуляторные "
        "требования?",
        "напр.: исключить фосген, диазометан, бензол",
    ),
    "Приоритеты отбора": (
        "Приоритеты ранжирования вариантов?",
        "напр.: доступность сырья в РФ, простота технологии, стоимость",
    ),
    "Форма результата": (
        "Что должно быть итогом этапа?",
        "напр.: отчёт, таблица маршрутов, список кандидатов, условия для проверки",
    ),
}

# Button options of a single question window (web review). Selecting one is a
# complete answer — the pipeline can always continue with open questions.
OPT_SKIP = "Не знаю — пропустить"
OPT_AGENT = "На усмотрение агента"
OPT_STOP = "Завершить опрос (остальные пропустить)"
QUESTION_OPTIONS: tuple[str, ...] = (OPT_SKIP, OPT_AGENT, OPT_STOP)

_ANSWER_RULES = """\
Черновик ТЗ собран только из исходного запроса — ниже вопросы по блокам,
где остались незаполненные поля. Отвечать можно на любую часть вопросов:

- впишите ответы в поле правок в формате `Q1: <ответ>; Q4: <ответ>` и нажмите **Revise**;
- `Qn: не знаю` (или просто пропустите вопрос) — поле останется «не задано»,
  система продолжит работу без него;
- `Qn: на усмотрение агента` — агент подставит рабочее значение из отраслевого
  контекста (статус «уточнено оператором»);
- если уточнений нет — нажмите **Accept**: система продолжит с незакрытыми вопросами."""


@dataclass
class OperatorQuestion:
    """One questionnaire entry, bound to a canonical ТЗ block."""

    num: int
    block: str
    text: str
    hint: str = ""
    open_fields: List[str] = field(default_factory=list)

    @property
    def qid(self) -> str:
        return f"Q{self.num}"


def _coerce(tz: Union[StructuredTZ, dict, str]) -> StructuredTZ:
    if isinstance(tz, StructuredTZ):
        return tz
    if isinstance(tz, str):
        # At review time the ТЗ may still be the raw JSON text of the model
        # answer (the state commit happens only after approval).
        tz = json.loads(tz)
    return StructuredTZ.model_validate(tz)


def build_questionnaire(tz: Union[StructuredTZ, dict, str]) -> List[OperatorQuestion]:
    """One question per canonical block that still has «не задано» fields.

    Fields with status «рассчитывается агентом» are deliberately NOT asked —
    they are deferred to later pipeline stages, not missing customer input.
    """
    tz = _coerce(tz)
    questions: List[OperatorQuestion] = []
    for title in CANONICAL_BLOCKS:
        block = tz.block(title)
        if block is None:
            open_fields = []  # весь блок отсутствует — спросить целиком
        else:
            open_fields = [f.name for f in block.fields if f.status == "не задано"]
            if not open_fields:
                continue
        text, hint = QUESTION_BANK[title]
        questions.append(
            OperatorQuestion(
                num=len(questions) + 1,
                block=title,
                text=text,
                hint=hint,
                open_fields=open_fields,
            )
        )
    return questions


def render_questionnaire(questions: List[OperatorQuestion]) -> str:
    """Markdown section appended to the review document shown to the operator."""
    if not questions:
        return ""
    out: list[str] = ["---", "", "## Уточняющие вопросы оператору", "", _ANSWER_RULES, ""]
    for q in questions:
        out.append(f"**{q.qid} · {q.block}** — {q.text}")
        if q.hint:
            out.append(f"  _{q.hint}_")
        if q.open_fields:
            shown = ", ".join(q.open_fields[:6])
            more = f" (+{len(q.open_fields) - 6})" if len(q.open_fields) > 6 else ""
            out.append(f"  сейчас не задано: {shown}{more}")
        else:
            out.append("  блок пока не сформирован")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_question_card(q: OperatorQuestion, index: int, total: int) -> str:
    """One question as shown in its own HITL window."""
    lines = [
        f"Уточнение ТЗ — вопрос {index} из {total}",
        "",
        f"Блок: {q.block}",
        "",
        q.text,
    ]
    if q.hint:
        lines.append(f"({q.hint})")
    if q.open_fields:
        shown = ", ".join(q.open_fields[:6])
        more = f" (+{len(q.open_fields) - 6})" if len(q.open_fields) > 6 else ""
        lines += ["", f"Сейчас не задано: {shown}{more}"]
    else:
        lines += ["", "Блок пока не сформирован."]
    lines += [
        "",
        "Как ответить:",
        "- впишите ответ в поле и нажмите «Ответить»;",
        f"- «{OPT_SKIP}» — поля останутся «не задано», система продолжит без них;",
        f"- «{OPT_AGENT}» — агент подставит рабочее значение из отраслевого контекста;",
        f"- «{OPT_STOP}» — перейти к работе, пропустив оставшиеся вопросы.",
    ]
    return "\n".join(lines)


__all__ = [
    "OperatorQuestion",
    "OPT_AGENT",
    "OPT_SKIP",
    "OPT_STOP",
    "QUESTION_BANK",
    "QUESTION_OPTIONS",
    "build_questionnaire",
    "render_question_card",
    "render_questionnaire",
]
