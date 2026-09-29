"""The operator's NIR questions: generate at all, and with which requisites.

Two steps, in the shape ``hitl/pipeline_scope.py`` established: a SELECT to
decide, then an APPROVE carrying a ``form`` to collect what only a human knows.

Everything on this form is a title-page fact that a CoScientist run genuinely
does not have. There is no УДК in a research graph, no state registration
number, no prorector who signs the document, and no list of human performers —
the agents did the work, but ГОСТ's "список исполнителей" means people who
answer for it. Anything left blank becomes the contract's draft placeholder
``<...>``, which prints visibly and is what the normcontrol pass is for. A
fabricated registration number would pass silently and be worse.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from CoScientist.reporting.nir import contract
from CoScientist.reporting.nir.build import NirRequisites

STATE_REQUEST_KEY = "nir_report_request"
STATE_BLOCK_KEY = "nir_block"
STATE_DONE_KEY = "nir_report_asked"
STATE_RESULT_KEY = "nir_report"

FORM_BLOCK_TITLE = "Реквизиты отчёта о НИР"
FORM_BLOCK_PEOPLE = "Подписи и исполнители"

OPTION_SHORT = "Только краткий отчёт (как сейчас)"
OPTION_NIR = "Краткий отчёт + отчёт о НИР по ГОСТ 7.32-2017 (DOCX)"

SELECT_MESSAGE = (
    "Исследование завершено. Какой отчёт сформировать?\n\n"
    f"• «{OPTION_SHORT}» — итоговый Markdown с рисунками и таблицами.\n"
    f"• «{OPTION_NIR}» — дополнительно нормативный документ DOCX, собранный "
    "сервисом «Автонормоконтроль». Потребуется указать реквизиты титульного листа."
)

FORM_MESSAGE = "Заполните реквизиты титульного листа отчёта о НИР"

_INTRO = (
    "Эти сведения нельзя извлечь из хода исследования — их задаёт организация. "
    "Незаполненные поля попадут в документ как «<...>»: черновик соберётся, но "
    "нормоконтроль их отметит. Ничего не выдумывайте — видимый пробел лучше "
    "правдоподобной ошибки."
)

_PERFORMERS_PLACEHOLDER = (
    "По одному исполнителю в строке: роль | должность | И.О. Фамилия | вклад. "
    "Например: Исполнитель | инженер | П.П. Петров | разделы 1-3"
)


def _field(
    name: str,
    label_ru: str,
    label_en: str,
    placeholder: str = "",
    value: str = "",
    kind: str = "text",
) -> Dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "value": value,
        "status": "не задано" if not value else "задано заказчиком",
        "open": True,
        "label": {"ru": label_ru, "en": label_en},
        "placeholder": {"ru": placeholder, "en": placeholder},
    }


def choice_options() -> List[str]:
    return [OPTION_SHORT, OPTION_NIR]


def wants_nir(selected: Optional[str]) -> bool:
    """True only when the resolved selection is the full NIR option."""
    return str(selected or "").strip() == OPTION_NIR


def requisites_form() -> Dict[str, Any]:
    """The structured intake the web UI renders instead of a free-text review."""
    return {
        "kind": "nir_requisites",
        "title": "Реквизиты отчёта о НИР",
        "title_i18n": {"ru": "Реквизиты отчёта о НИР", "en": "NIR report requisites"},
        "intro": _INTRO,
        "intro_i18n": {"ru": _INTRO, "en": _INTRO},
        "message_i18n": {"ru": FORM_MESSAGE, "en": "Fill in the NIR title-page requisites"},
        "submit_label": "Сформировать отчёт",
        "allow_skip": True,
        "blocks": [
            {
                "title": FORM_BLOCK_TITLE,
                "usage": "титульный лист по ГОСТ 7.32-2017",
                "title_i18n": {"ru": FORM_BLOCK_TITLE, "en": "Report requisites"},
                "usage_i18n": {"ru": "титульный лист по ГОСТ 7.32-2017",
                               "en": "GOST 7.32-2017 title page"},
                "fields": [
                    _field("udc", "УДК", "UDC", "например 004.942:615.9"),
                    _field("registration_nioktr", "Рег. № НИОКТР", "NIOKTR number",
                           "например АААА-А26-126090300001-1"),
                    _field("registration_ikrbs", "Рег. № ИКРБС", "IKRBS number"),
                    _field("report_type", "Вид отчёта", "Report type",
                           "заключительный или промежуточный",
                           value=contract.REPORT_TYPE_FINAL),
                    _field("stage", "Номер этапа", "Stage",
                           "только для промежуточного отчёта"),
                    _field("program_code", "Шифр программы", "Programme code"),
                    _field("page_count", "Число страниц итогового документа",
                           "Page count",
                           "необязательно: если не знаете — оставьте пустым, "
                           "в реферате будет «___»"),
                ],
            },
            {
                "title": FORM_BLOCK_PEOPLE,
                "usage": "блок УТВЕРЖДАЮ, руководитель и список исполнителей",
                "title_i18n": {"ru": FORM_BLOCK_PEOPLE, "en": "Signatures and performers"},
                "usage_i18n": {"ru": "блок УТВЕРЖДАЮ, руководитель и список исполнителей",
                               "en": "Approval block, supervisor and performers"},
                "fields": [
                    _field("approval_position", "УТВЕРЖДАЮ — должность", "Approval position",
                           "например Проректор по научной работе"),
                    _field("approval_name", "УТВЕРЖДАЮ — И.О. Фамилия", "Approval name",
                           "например В.О. Никифоров"),
                    _field("approval_degree", "УТВЕРЖДАЮ — учёная степень", "Approval degree"),
                    _field("approval_academic_title", "УТВЕРЖДАЮ — учёное звание",
                           "Approval academic title"),
                    _field("approval_date", "Дата утверждения", "Approval date",
                           "ДД.ММ.ГГГГ"),
                    _field("supervisor_name", "Руководитель НИР — И.О. Фамилия",
                           "Supervisor name"),
                    _field("supervisor_position", "Руководитель НИР — должность",
                           "Supervisor position"),
                    _field("supervisor_degree", "Руководитель НИР — учёная степень",
                           "Supervisor degree"),
                    _field("supervisor_academic_title", "Руководитель НИР — учёное звание",
                           "Supervisor academic title"),
                    _field("performers_raw", "Список исполнителей", "Performers",
                           _PERFORMERS_PLACEHOLDER, kind="textarea"),
                ],
            },
        ],
    }


def flatten(form_values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse ``{block: {field: value}}`` — and a flat dict — into answers.

    Same shape as ``pipeline_scope.flatten_form_values``; duplicated rather
    than imported so this package does not depend on the pipeline-scope module
    for one three-line helper.
    """
    if not form_values:
        return {}
    answers: Dict[str, Any] = {}
    for key, value in form_values.items():
        if isinstance(value, dict):
            answers.update(value)
        else:
            answers[key] = value
    return answers


def _text(answers: Dict[str, Any], key: str) -> str:
    value = answers.get(key)
    return "" if value is None else str(value).strip()


def parse_requisites(form_values: Optional[Dict[str, Any]]) -> NirRequisites:
    """Build :class:`NirRequisites` from the operator's answers.

    Unknown or empty values stay empty here; turning them into placeholders is
    the builder's job, so there is exactly one place that decides what an
    unfilled field looks like in the document.
    """
    answers = flatten(form_values)

    report_type = _text(answers, "report_type").lower()
    if report_type not in contract.REPORT_TYPES:
        report_type = contract.REPORT_TYPE_FINAL

    page_count: Optional[int] = None
    raw_pages = _text(answers, "page_count")
    if raw_pages:
        try:
            parsed = int(raw_pages)
            page_count = parsed if parsed > 0 else None
        except ValueError:
            page_count = None

    return NirRequisites(
        udc=_text(answers, "udc"),
        registration_nioktr=_text(answers, "registration_nioktr"),
        registration_ikrbs=_text(answers, "registration_ikrbs"),
        report_type=report_type,
        stage=_text(answers, "stage"),
        program_code=_text(answers, "program_code"),
        approval_position=_text(answers, "approval_position"),
        approval_name=_text(answers, "approval_name"),
        approval_degree=_text(answers, "approval_degree"),
        approval_academic_title=_text(answers, "approval_academic_title"),
        approval_date=_text(answers, "approval_date"),
        supervisor_name=_text(answers, "supervisor_name"),
        supervisor_position=_text(answers, "supervisor_position"),
        supervisor_degree=_text(answers, "supervisor_degree"),
        supervisor_academic_title=_text(answers, "supervisor_academic_title"),
        performers_raw=_text(answers, "performers_raw"),
        page_count=page_count,
    )


def requisites_to_state(requisites: NirRequisites) -> Dict[str, Any]:
    """ADK state must hold plain JSON, not a dataclass."""
    return {
        "udc": requisites.udc,
        "registration_nioktr": requisites.registration_nioktr,
        "registration_ikrbs": requisites.registration_ikrbs,
        "report_type": requisites.report_type,
        "stage": requisites.stage,
        "program_code": requisites.program_code,
        "approval_position": requisites.approval_position,
        "approval_name": requisites.approval_name,
        "approval_degree": requisites.approval_degree,
        "approval_academic_title": requisites.approval_academic_title,
        "approval_date": requisites.approval_date,
        "supervisor_role": requisites.supervisor_role,
        "supervisor_name": requisites.supervisor_name,
        "supervisor_position": requisites.supervisor_position,
        "supervisor_degree": requisites.supervisor_degree,
        "supervisor_academic_title": requisites.supervisor_academic_title,
        "performers_raw": requisites.performers_raw,
        "page_count": requisites.page_count,
    }


def requisites_from_state(data: Optional[Dict[str, Any]]) -> NirRequisites:
    if not isinstance(data, dict):
        return NirRequisites()
    known = {f: data.get(f) for f in NirRequisites().__dict__ if f in data}
    return NirRequisites(**{k: v for k, v in known.items() if v is not None})


__all__ = [
    "STATE_REQUEST_KEY",
    "STATE_BLOCK_KEY",
    "STATE_DONE_KEY",
    "STATE_RESULT_KEY",
    "OPTION_SHORT",
    "OPTION_NIR",
    "SELECT_MESSAGE",
    "FORM_MESSAGE",
    "choice_options",
    "wants_nir",
    "requisites_form",
    "flatten",
    "parse_requisites",
    "requisites_to_state",
    "requisites_from_state",
]
