from uuid import uuid4

import pytest

from CoScientist.cli import _configure_utf8_stdio, _run_graph, main
from CoScientist.graph.memory import get_knowledge_graph




def test_graph_cli_requires_complete_scope_pair():
    with pytest.raises(SystemExit):
        main(["graph", "show", "--user-id", "user-only"])


def test_cli_configures_scientific_unicode_output(monkeypatch):
    class Stream:
        def __init__(self):
            self.encodings = []

        def reconfigure(self, *, encoding):
            self.encodings.append(encoding)

    stdout = Stream()
    stderr = Stream()
    monkeypatch.setattr("CoScientist.cli.sys.stdout", stdout)
    monkeypatch.setattr("CoScientist.cli.sys.stderr", stderr)

    _configure_utf8_stdio()

    assert stdout.encodings == ["utf-8"]
    assert stderr.encodings == ["utf-8"]
