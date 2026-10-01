from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "CoScientist" / "web" / "templates" / "graph.html"
SCRIPT = ROOT / "CoScientist" / "web" / "static" / "js" / "framing_panel.js"
STYLE = ROOT / "CoScientist" / "web" / "static" / "css" / "framing_panel.css"
APP = ROOT / "CoScientist" / "web" / "app.py"


def test_framing_card_uses_its_own_document_renderer():
    page = PAGE.read_text(encoding="utf-8")
    assert 'if (n.kind === "framing") return showFramingDetail(n);' in page
    assert '/static/js/framing_panel.js' in page
    assert '/static/css/framing_panel.css' in page


def test_framing_panel_does_not_render_graph_type_emoji_or_attachments_heading():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "R_ICON" not in script
    assert "graph.attach.title" not in script
    assert "Вложения" not in script
    assert "Документы не приложены" not in script
    assert "tz-document" in script


def test_framing_details_are_loaded_for_the_selected_study_only_on_open():
    script = SCRIPT.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    assert "graph/framing?study_id=" in script
    assert "framingStudyId() !== studyId" in script
    assert '@app.get("/api/users/{user_id}/sessions/{session_id}/graph/framing")' in app
    assert "framing_view_of(study_id)" in app


def test_long_fields_use_vertical_full_width_layout():
    style = STYLE.read_text(encoding="utf-8")
    assert ".tz-field-value" in style
    assert "white-space: pre-wrap" in style
    assert "overflow-wrap: anywhere" in style
    assert "grid-template-columns" not in style


def test_browser_also_filters_empty_fields_from_an_old_server_response():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "framingFieldHasValue" in script
    assert ".filter(framingFieldHasValue)" in script
    assert "section.fields.length || section.documents.length" in script
