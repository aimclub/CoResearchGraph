"""Техническое задание по ГОСТ 19.201-78, собранное из подтверждённой рамки.

Рамка описывает ПОСТАНОВКУ: вопрос, ограничения, бюджеты, инструменты,
критерии. ТЗ — это документ, который читает человек и по которому работу
принимают, поэтому у него другой набор разделов, заданный п. 1.4 стандарта:

    введение · основания для разработки · назначение разработки · требования к
    программе или программному изделию · требования к программной документации ·
    технико-экономические показатели · стадии и этапы разработки · порядок
    контроля и приёмки

Стандарт разрешает прямо: «в зависимости от особенностей программы или
программного изделия допускается уточнять содержание разделов, вводить новые
разделы или объединять отдельные из них» (п. 1.4). Этим и пользуемся —
исследование не программное изделие, и подразделы про маркировку, упаковку,
транспортирование и хранение (2.4.6, 2.4.7) к нему не применимы. Они не
выбрасываются молча: документ говорит, что они объединены и почему.

Сборка ДЕТЕРМИНИРОВАННАЯ. Каждый раздел собирается из полей рамки по
фиксированному правилу, и ничего сверх них не появляется: пустое поле даёт
«Не задано», а не правдоподобный текст. Связную прозу дописывает отдельный
проход модели (`tz_prose`), и ему на вход идут уже собранные значения — он
может переформулировать, но не может ввести факт, которого в рамке нет.

Переформулировать — не косметика, а половина работы. Рамку заполняют по
неформальному запросу («автоматизируй составление профиля», «собери данные»),
и та же речь, разложенная по разделам, документом не становится: ТЗ пишется
безлично и отглагольными существительными («требуется разработать», «сбор и
систематизация данных»), потому что по нему принимают работу. Поэтому модели
отдаётся ВСЁ, у чего есть содержание, — разделы, подразделы, формулировки задач
и само наименование темы, — а неизменным остаётся ровно то, что документ
утверждает сам: номера, заголовки, таблицы значений и оговорка о п. 1.4.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from CoScientist.context_init.models import ResearchFrame
from CoScientist.context_init.operations import OPS_FORM_BLOCK  # noqa: F401

#: Что печатается вместо значения, которого нет. Одна строка на весь документ —
#: читатель должен узнавать её с первого раза и не гадать, пусто ли это поле или
#: его забыли заполнить.
NOT_SET = "Не задано"

#: Заголовки разделов — дословно по п. 1.4 и п. 2 ГОСТ 19.201-78, кроме
#: четвёртого: «требования к программе или программному изделию» применительно к
#: исследованию названы тем, чем они являются. Стандарт это разрешает, а
#: заголовок «требования к программному изделию» над разделом о достоверности
#: измерений сбивал бы с толку сильнее, чем отступление от буквы.
SECTION_TITLES: Tuple[Tuple[str, str], ...] = (
    ("1", "Введение"),
    ("2", "Основания для разработки"),
    ("3", "Назначение разработки"),
    ("4", "Требования к результату исследования"),
    ("5", "Требования к отчётной документации"),
    ("6", "Технико-экономические показатели"),
    ("7", "Стадии и этапы разработки"),
    ("8", "Порядок контроля и приёмки"),
)

#: Подразделы четвёртого раздела. Номера сохранены от п. 2.4 стандарта, чтобы
#: соответствие читалось без сверки; 4.6 — то, во что объединены 2.4.6-2.4.8.
SUBSECTION_TITLES: Tuple[Tuple[str, str], ...] = (
    ("4.1", "Требования к составу выполняемых работ"),
    ("4.2", "Требования к достоверности результата"),
    ("4.3", "Условия проведения"),
    ("4.4", "Требования к составу и параметрам технических средств"),
    ("4.5", "Требования к исходным данным и совместимости"),
    ("4.6", "Специальные требования"),
)

#: Подраздел, в котором лежат задачи исследования. Их формулировки модель
#: переписывает отдельно от прозы: это строки таблицы, и в документе они должны
#: читаться отглагольными существительными («Сбор литературных данных»), а не
#: повелительным наклонением из запроса («Собери литературные данные»).
TASKS_NUMBER = "4.1"


class TZSection(BaseModel):
    """Один раздел ТЗ: номер, заголовок, текст и вложенные подразделы."""

    number: str = Field(description="Номер раздела по ГОСТ, напр. «4» или «4.1»")
    title: str = Field(description="Заголовок раздела")
    body: str = Field(default="", description="Текст раздела")
    rows: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Табличная часть: (что, значение). Пусто — таблицы нет.")
    subsections: List["TZSection"] = Field(default_factory=list)
    fixed: bool = Field(
        default=False,
        description="Текст раздела задан документом и модели не отдаётся.")

    def is_empty(self) -> bool:
        return (not self.rows and not self.subsections
                and self.body.strip() in ("", NOT_SET))

    def has_content(self) -> bool:
        """Есть ли в разделе что переформулировать — свой текст или таблица."""
        return bool(self.rows) or self.body.strip() not in ("", NOT_SET)


class TechnicalSpec(BaseModel):
    """Техническое задание целиком — то, из чего рендерятся .docx и .md."""

    topic: str = Field(default="", description="Наименование темы")
    topic_confirmed: bool = Field(
        default=False,
        description="Тему назвал заказчик — модель её не переписывает.")
    customer: str = Field(default="", description="Заказчик")
    basis: str = Field(default="", description="Основание для работы")
    original_request: str = Field(
        default="", description="Исходный запрос пользователя дословно")
    sections: List[TZSection] = Field(default_factory=list)

    def section(self, number: str) -> Optional[TZSection]:
        for s in self.sections:
            if s.number == number:
                return s
        return None

    def parts(self) -> List[TZSection]:
        """Разделы и подразделы одним списком, в порядке документа."""
        out: List[TZSection] = []
        for s in self.sections:
            out.append(s)
            out.extend(s.subsections)
        return out

    def part(self, number: str) -> Optional[TZSection]:
        """Раздел или подраздел по номеру: `part("4.1")` находит подраздел."""
        for p in self.parts():
            if p.number == number:
                return p
        return None

    def rewritable(self) -> List[TZSection]:
        """То, что отдаётся модели на переформулирование."""
        return [p for p in self.parts() if p.has_content() and not p.fixed]

    def unfilled(self) -> List[str]:
        """Номера разделов, которым рамка ничего не дала.

        Документ всё равно выпускается: «не задано» — это честный результат
        опроса, и оператор видит по нему, что дозаполнить.
        """
        out: List[str] = []
        for s in self.sections:
            for part in [s] + list(s.subsections):
                if part.is_empty():
                    out.append(part.number)
        return out


TZSection.model_rebuild()


# ── чтение рамки ────────────────────────────────────────────────────────────

def _fields(frame: ResearchFrame, title: str) -> Dict[str, str]:
    """Заполненные поля блока по его заголовку: {имя: значение}."""
    for b in frame.blocks:
        if b.title == title:
            return {f.name: f.value.strip() for f in b.set_fields()
                    if f.value and f.value.strip()}
    return {}


def _value(frame: ResearchFrame, title: str, name: str) -> str:
    return _fields(frame, title).get(name, "")


def _rows(frame: ResearchFrame, title: str,
          labels: Dict[str, str]) -> List[Tuple[str, str]]:
    """Табличная часть раздела: подпись поля и его значение, в порядке `labels`.

    Подписи берутся не из имён полей: «gpu_hours» в документе, который пойдёт
    заказчику, — это не название строки.
    """
    have = _fields(frame, title)
    return [(labels[name], have[name]) for name in labels if name in have]


def _joined(frame: ResearchFrame, titles: List[str]) -> str:
    """Тексты нескольких блоков подряд, каждый со своим заголовком.

    Для разделов, которые стандарт держит одним, а рамка разносит по нескольким
    блокам, — 4.6 собирает этику, стандарты, нормы и теоретические рамки.
    """
    parts: List[str] = []
    for title in titles:
        have = _fields(frame, title)
        said = "; ".join(v for v in have.values() if v)
        if said:
            parts.append(f"{title}: {said}.")
    return "\n".join(parts)


# ── сборка ──────────────────────────────────────────────────────────────────

_RESOURCE_LABELS = {
    "gpu_hours": "GPU-часы", "tokens": "Токены LLM", "money": "Денежный бюджет",
    "time": "Календарное время", "expert_hours": "Часы эксперта",
}
_TOOL_LABELS = {
    "computational": "Вычислительные", "laboratory": "Лабораторные",
    "analytical": "Аналитические", "informational": "Информационные",
}
_BASE_LABELS = {
    "datasets": "Наборы данных", "corpora": "Корпуса",
    "trusted_kb": "Доверенные базы знаний",
}
_CRITERIA_LABELS = {
    "threshold": "Порог подтверждения",
    "confirmations_needed": "Число подтверждений",
    "reproducibility": "Воспроизводимость",
}
_COST_LABELS = {
    "cost_rule": "Правило стоимости шага", "stop_rule": "Правило остановки",
    "expected_effect": "Ожидаемый эффект",
}


def _short(text: str, limit: int = 160) -> str:
    """Первая фраза длинного текста — целыми словами.

    Наименование темы печатается на титуле, и обрыв посреди слова («предскажу
    LD50 для мыш») выдаёт документ, собранный машиной, сильнее всего прочего.
    """
    said = " ".join((text or "").split())
    if len(said) <= limit:
        return said
    return said[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:—–-") + "…"


def spec_from_frame(frame: ResearchFrame,
                    original_request: str = "") -> TechnicalSpec:
    """Собрать ТЗ из подтверждённой рамки. Ничего не выдумывает."""
    frame = frame.normalized()
    q = _fields(frame, "Вопрос исследования")
    mode = _fields(frame, "Режим и завершение")
    basis = _fields(frame, "Основание и приёмка")

    def text(*parts: str) -> str:
        said = [p.strip() for p in parts if p and p.strip()]
        return " ".join(said) if said else NOT_SET

    intro = text(
        q.get("formulation", ""),
        f"Область применения: {q['domain']}." if q.get("domain") else "",
        f"Постановка: {q['target_setting']}." if q.get("target_setting") else "",
    )

    purpose = text(
        f"Функциональное назначение: получить ответ на вопрос исследования "
        f"средствами мультиагентной системы CoScientist."
        if q.get("formulation") else "",
        f"Закрываемый пробел в знаниях: {q['gap']}." if q.get("gap") else "",
        f"Режим применения ИИ: {mode['ai_application_model']}."
        if mode.get("ai_application_model") else "",
        f"Форма исследования: {q['research_form']}." if q.get("research_form") else "",
        f"Уровень готовности технологии: {q['trl']}." if q.get("trl") else "",
    )

    tasks = [(t.operation_id, t.statement) for t in frame.operations if t.statement]
    work = TZSection(
        number=TASKS_NUMBER, title=SUBSECTION_TITLES[0][1],
        body=(q.get("decomposition") or "").strip() or NOT_SET,
        rows=[(f"Задача {i}", statement) for i, (_id, statement)
              in enumerate(tasks, 1)],
    )

    sections = [
        TZSection(number="1", title=SECTION_TITLES[0][1], body=intro),
        TZSection(
            number="2", title=SECTION_TITLES[1][1],
            body=NOT_SET if not basis else "",
            rows=[(label, basis[name]) for name, label in (
                ("basis_document", "Документ-основание"),
                ("customer", "Заказчик"),
                ("topic_name", "Наименование темы"),
            ) if basis.get(name)],
        ),
        TZSection(number="3", title=SECTION_TITLES[2][1], body=purpose),
        TZSection(
            number="4", title=SECTION_TITLES[3][1],
            # Оговорка о соответствии стандарту — утверждение самого документа,
            # а не пересказ рамки. Модели не отдаётся: переписанная, она
            # перестанет означать то, что означает.
            fixed=True,
            body=("Раздел «Требования к программе или программному изделию» "
                  "ГОСТ 19.201-78, применённый к исследованию. Подразделы "
                  "2.4.6-2.4.8 стандарта (маркировка, упаковка, "
                  "транспортирование и хранение) к результату исследования не "
                  "применимы и объединены в 4.6 — п. 1.4 стандарта это "
                  "допускает."),
            subsections=[
                work,
                TZSection(number="4.2", title=SUBSECTION_TITLES[1][1],
                          rows=_rows(frame, "Условия подтверждения", _CRITERIA_LABELS),
                          body="" if _fields(frame, "Условия подтверждения") else NOT_SET),
                TZSection(number="4.3", title=SUBSECTION_TITLES[2][1],
                          rows=_rows(frame, "Ресурсы и бюджеты", _RESOURCE_LABELS),
                          body=_joined(frame, ["Роли участников"]) or NOT_SET),
                TZSection(number="4.4", title=SUBSECTION_TITLES[3][1],
                          rows=_rows(frame, "Инструменты", _TOOL_LABELS),
                          body="" if _fields(frame, "Инструменты") else NOT_SET),
                TZSection(number="4.5", title=SUBSECTION_TITLES[4][1],
                          rows=_rows(frame, "Эмпирическая база", _BASE_LABELS),
                          body="" if _fields(frame, "Эмпирическая база") else NOT_SET),
                TZSection(number="4.6", title=SUBSECTION_TITLES[5][1],
                          body=_joined(frame, [
                              "Этика и регуляторика", "Доменные стандарты",
                              "Методологические нормы", "Теоретические рамки",
                          ]) or NOT_SET),
            ],
        ),
        TZSection(number="5", title=SECTION_TITLES[4][1],
                  body=basis.get("deliverables") or NOT_SET),
        TZSection(number="6", title=SECTION_TITLES[5][1],
                  rows=_rows(frame, "Модель стоимости", _COST_LABELS),
                  body="" if _fields(frame, "Модель стоимости") else NOT_SET),
        TZSection(number="7", title=SECTION_TITLES[6][1],
                  body=basis.get("stages") or NOT_SET),
        TZSection(
            number="8", title=SECTION_TITLES[7][1],
            body=text(
                basis.get("acceptance", ""),
                f"Критерий завершения исследования: {mode['completion_criteria']}."
                if mode.get("completion_criteria") else "",
            ),
        ),
    ]

    named = (basis.get("topic_name") or "").strip()
    return TechnicalSpec(
        # Тему, названную заказчиком, берём дословно. Если её не называли,
        # ставим обрезанную формулировку вопроса — временно: наименование темы
        # предложит модель, а до неё лучше короткая фраза, чем 200 символов
        # чужой речи, оборванных посреди слова.
        topic=named or _short(q.get("formulation", "")) or NOT_SET,
        topic_confirmed=bool(named),
        customer=basis.get("customer") or NOT_SET,
        basis=basis.get("basis_document") or NOT_SET,
        original_request=(original_request or "").strip(),
        sections=sections,
    )


# ── проза от модели ─────────────────────────────────────────────────────────

def prose_request(spec: TechnicalSpec) -> str:
    """Черновик для модели: всё, что она вправе переформулировать, и не более.

    Отдаётся уже собранное ТЗ, а не рамка: модель видит ровно те факты, которые
    попадут в документ, и добавить ей нечего. Таблицы показываются, но помечены
    как неизменяемые — прозой они вводятся, а не пересказываются: перечисление,
    пересказанное словами, теряет числа.
    """
    lines: List[str] = []
    if spec.topic_confirmed:
        lines.append(f"НАИМЕНОВАНИЕ ТЕМЫ (задано заказчиком, не меняй): {spec.topic}")
    else:
        lines.append("НАИМЕНОВАНИЕ ТЕМЫ: заказчиком не задано — предложи его "
                     "сам, в поле topic.")
    if spec.original_request:
        lines += [
            "",
            "ИСХОДНЫЙ ЗАПРОС ЗАКАЗЧИКА. Это неформальная речь: в документ в "
            "таком виде она не попадает, но других сведений о работе нет, и "
            "содержание разделов берётся отсюда.",
            spec.original_request[:3000],
        ]

    tasks = spec.part(TASKS_NUMBER)
    if tasks and tasks.rows:
        lines += ["", "ЗАДАЧИ ИССЛЕДОВАНИЯ — перепиши формулировку каждой "
                      "(поле tasks), сохранив номер и предмет:"]
        lines += [f"  {i}. {statement}"
                  for i, (_key, statement) in enumerate(tasks.rows, 1)]

    for part in spec.rewritable():
        lines += ["", f"РАЗДЕЛ {part.number}. {part.title}"]
        body = part.body.strip()
        if body and body != NOT_SET:
            lines.append(body)
        if part.rows and part is tasks:
            # Задачи уже перечислены выше отдельным списком: продублировать их
            # здесь — значит позвать переписать их дважды и по-разному. Но и
            # промолчать нельзя: раздел с одним заголовком читается пустым, а
            # пустое модель заполняет выдумкой.
            lines.append("Содержание раздела — перечисленные выше задачи "
                         "исследования; текстом их обобщают, а не перечисляют.")
        elif part.rows:
            lines.append("Таблица раздела — печатается как есть; в тексте на "
                         "неё ссылаются, значения не переписывают:")
            lines += [f"  — {key}: {value}" for key, value in part.rows]
    return "\n".join(lines)


def apply_prose(spec: TechnicalSpec, prose: Dict[str, str]) -> TechnicalSpec:
    """Наложить переписанные абзацы на собранное ТЗ.

    Раздел принимает текст, только если ему есть что переформулировать — свой
    текст или таблица. Пустой раздел модель не заполняет ни при каких
    обстоятельствах: «Не задано» — это вопрос оператору, а правдоподобный абзац
    на его месте читается как обязательство, которого никто не брал.
    """
    allowed = {p.number: p for p in spec.rewritable()}
    for number, said in (prose or {}).items():
        part = allowed.get(str(number))
        text = str(said or "").strip()
        if part is not None and text:
            part.body = text
    return spec


def apply_tasks(spec: TechnicalSpec, tasks: Dict[int, str]) -> TechnicalSpec:
    """Заменить формулировки задач исследования на переписанные.

    По номеру, а не по порядку присланного: модель может вернуть их не все и не
    подряд, и задача, сместившаяся на строку, — это уже другая задача. Номер,
    которого в документе нет, игнорируется; пустая формулировка оставляет
    прежнюю.
    """
    section = spec.part(TASKS_NUMBER)
    if section is None:
        return spec
    for number, said in (tasks or {}).items():
        try:
            index = int(number) - 1
        except (TypeError, ValueError):
            continue
        text = str(said or "").strip()
        if text and 0 <= index < len(section.rows):
            section.rows[index] = (section.rows[index][0], text)
    return spec


def apply_topic(spec: TechnicalSpec, topic: str) -> TechnicalSpec:
    """Поставить наименование темы, предложенное моделью.

    Только если заказчик темы не называл: названная — это его формулировка, и
    переписывать её документ не вправе.
    """
    said = _short(str(topic or "").strip(), 200)
    if said and not spec.topic_confirmed:
        spec.topic = said
    return spec


__all__ = [
    "NOT_SET",
    "SECTION_TITLES",
    "SUBSECTION_TITLES",
    "TASKS_NUMBER",
    "TZSection",
    "TechnicalSpec",
    "apply_prose",
    "apply_tasks",
    "apply_topic",
    "prose_request",
    "spec_from_frame",
]
