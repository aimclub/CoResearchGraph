"""A preview is a marked slice. The next step still receives every row."""
from __future__ import annotations

from CoScientist.tools.fedot_artifact_handoff import (
    _table_from_records,
    format_upstream_inputs,
    project_tables_to_schema_args,
)


def test_b10_preview_does_not_drop_rows_from_the_next_step():
    records = [{"smiles": f"C{i}"} for i in range(25)]
    table = _table_from_records(records)
    assert table is not None
    assert table["total_rows"] == 25
    assert len(table["rows"]) == 25
    assert len(table["preview_rows"]) == 10
    assert table["truncated"] is True
    projected = project_tables_to_schema_args([table], ["smiles"])
    assert len(projected["smiles"]) == 25
    text = format_upstream_inputs(projected)
    assert "truncated=True" in text
    assert "preview" in text.lower()
    assert "25" in text
