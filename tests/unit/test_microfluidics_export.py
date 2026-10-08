"""Export of the ТЗ + literature queries to shareable Markdown and HTML."""
from CoScientist.microfluidics.export import export_tz_and_queries
from CoScientist.microfluidics.render import (
    render_queries_markdown,
    render_tz_and_queries_html,
    render_tz_and_queries_markdown,
)

_TZ = {
    "original_request": "Нужен ПАВ для повышения нефтеотдачи",
    "blocks": [
        {
            "title": "Целевой продукт",
            "usage": "что синтезируем",
            "fields": [
                {"name": "Класс", "value": "анионный", "status": "задано заказчиком"},
                {"name": "CAS", "value": "", "status": "не задано"},
            ],
        }
    ],
}
_QUERIES = {
    "queries": [
        {
            "id": "LIT-01",
            "task": "Классы анионных ПАВ для ASP",
            "query_en": "anionic surfactants ASP flooding ultralow IFT",
            "extract": ["класс ПАВ", "IFT"],
        }
    ]
}


def test_render_queries_markdown_lists_each_query():
    md = render_queries_markdown(_QUERIES)
    assert "Целевые запросы к литературному агенту" in md
    assert "LIT-01" in md
    assert "anionic surfactants ASP flooding ultralow IFT" in md
    assert "класс ПАВ, IFT" in md


def test_render_queries_markdown_handles_empty():
    assert "не сформированы" in render_queries_markdown(None)
    assert "не сформированы" in render_queries_markdown({"queries": []})


def test_combined_markdown_has_both_tz_and_queries():
    md = render_tz_and_queries_markdown(_TZ, _QUERIES)
    assert "Целевой продукт" in md
    assert "Целевые запросы к литературному агенту" in md
    assert "LIT-01" in md


def test_html_is_self_contained_and_escapes_values():
    html = render_tz_and_queries_html(
        {"original_request": "<script>alert(1)</script>", "blocks": []}, _QUERIES
    )
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html  # styling is inlined, no external assets
    assert "http://" not in html and "https://" not in html  # fully self-contained
    assert "<script>alert(1)</script>" not in html  # the request text is escaped
    assert "&lt;script&gt;" in html
    assert "LIT-01" in html


def test_export_callback_writes_md_and_html(tmp_path, monkeypatch):
    import CoScientist.microfluidics.export as export_mod

    monkeypatch.setattr(export_mod, "TZ_DOCUMENTS_DIR", tmp_path / "docs")

    class _Ctx:
        state = {"structured_tz": _TZ, "tz_literature_queries": _QUERIES}

    result = export_mod.export_tz_and_queries(_Ctx())
    files = sorted(p.name for p in (tmp_path / "docs").glob("TZ_export_*"))
    assert len(files) == 2
    assert any(f.endswith(".md") for f in files)
    assert any(f.endswith(".html") for f in files)
    # the callback also reports the saved paths back into the chat
    assert result is not None
    assert "сохранены" in result.parts[0].text


def test_export_callback_noop_without_tz():
    class _Ctx:
        state = {}

    assert export_tz_and_queries(_Ctx()) is None
