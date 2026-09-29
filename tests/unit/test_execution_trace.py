"""The execution graph regrouped as a chronological trace.

The call graph shows what is connected to what; these pin down what the trace
adds — one entry per user prompt, its calls in the order they ran, and the
input, output and duration of each.
"""
from CoScientist.graph.projection import turns


def _full():
    return {"nodes": [
        {"id": "goal:a", "kind": "goal", "turn_id": "a", "label": "first ask",
         "status": "success", "t_start": 100.0, "t_end": 140.0},
        {"id": "tool:2", "kind": "tool_call", "turn_id": "a", "label": "execute_bash",
         "executor_agent": "CoderAgent", "status": "failed", "input": "ls",
         "output": "boom", "t_start": 110.0, "t_end": 110.4},
        {"id": "tool:1", "kind": "tool_call", "turn_id": "a", "label": "tavily_search",
         "executor_agent": "ResearchAgent", "status": "success", "input": "{}",
         "output": "12 papers", "t_start": 101.0, "t_end": 103.5},
        {"id": "result:a", "kind": "result", "turn_id": "a", "output": "done",
         "t_start": 140.0, "t_end": 140.0},
        {"id": "goal:b", "kind": "goal", "turn_id": "b", "label": "second ask",
         "status": "running", "t_start": 200.0},
        {"id": "agent:CoderAgent", "kind": "agent", "status": "success"},
    ]}


def test_turns_are_separated_and_ordered_by_time():
    result = turns(_full())
    assert [t["turn_id"] for t in result["turns"]] == ["a", "b"]
    assert result["turns"][0]["prompt"] == "first ask"
    # Calls come back in the order they ran, not the order they were recorded.
    assert [c["tool"] for c in result["turns"][0]["calls"]] == [
        "tavily_search", "execute_bash"]


def test_a_call_carries_its_agent_arguments_result_and_duration():
    call = turns(_full())["turns"][0]["calls"][0]
    assert call["agent"] == "ResearchAgent"
    assert call["input"] == "{}" and call["output"] == "12 papers"
    assert call["status"] == "success" and call["duration"] == 2.5


def test_agent_nodes_are_not_turns():
    # One agent node serves the whole session, so it belongs to no single turn
    # and must not become one.
    assert all(t["turn_id"] != "agent:CoderAgent" for t in turns(_full())["turns"])


def test_snapshots_written_before_turn_id_still_group():
    legacy = {"nodes": [
        {"id": "goal:old", "kind": "goal", "label": "before the field existed",
         "status": "success", "t_start": 10.0, "t_end": 20.0},
        {"id": "result:old", "kind": "result", "output": "answer", "t_start": 20.0},
    ]}
    result = turns(legacy)
    assert result["count"] == 1
    assert result["turns"][0]["turn_id"] == "old"
    assert result["turns"][0]["answer"] == "answer"


def test_a_call_with_no_turn_is_kept_not_dropped():
    orphan = {"nodes": [{"id": "tool:x", "kind": "tool_call", "label": "t",
                         "status": "success", "t_start": 1.0}]}
    result = turns(orphan)
    assert result["count"] == 1 and result["turns"][0]["turn_id"] == "untagged"


def test_trace_view_is_served_and_grouped(tmp_path, monkeypatch):
    """The page and its data source exist and agree on the shape."""
    import time
    import uuid

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web"))
    from starlette.testclient import TestClient

    from CoScientist.graph.memory import get_knowledge_graph
    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        assert "Sessions" in client.get("/trace").text

        user = client.post(
            "/api/users", json={"nickname": f"t-{uuid.uuid4().hex[:8]}"}
        ).json()["user"]
        session = client.post(
            f"/api/users/{user['id']}/sessions", json={"title": "run"}
        ).json()["session"]

        graph = get_knowledge_graph(user_id=user["id"], session_id=session["id"])
        now = time.time()
        graph.add_node(id="goal:i1", kind="goal", turn_id="i1", label="do the thing",
                       status="success", t_start=now, t_end=now + 9)
        graph.add_node(id="tool:1", kind="tool_call", turn_id="i1", label="search",
                       executor_agent="ResearchAgent", status="success",
                       input="{}", output="hits", t_start=now + 1, t_end=now + 3)

        payload = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/graph",
            params={"view": "trace"},
        ).json()

    assert payload["count"] == 1
    turn = payload["turns"][0]
    assert turn["prompt"] == "do the thing"
    assert [c["tool"] for c in turn["calls"]] == ["search"]
    assert turn["calls"][0]["agent"] == "ResearchAgent"


def test_graph_opens_empty_and_grows_only_with_what_ran(tmp_path, monkeypatch):
    """No roster in the trace: an agent node exists because that agent acted."""
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    from CoScientist.graph.memory import KnowledgeGraph

    graph = KnowledgeGraph(run_id="execution")

    fresh = graph.full()
    assert [n["id"] for n in fresh["nodes"]] == ["system:root"]
    assert fresh["edges"] == []
    # The roster still resolves for prompts and for get_agents_info.
    assert len(graph.agents_info()) > 1

    graph.add_node(id="agent:CoderAgent", kind="agent", label="CoderAgent",
                   executor_agent="CoderAgent", status="success")
    acted = {n["id"] for n in graph.full()["nodes"] if n["kind"] == "agent"}
    assert acted == {"agent:CoderAgent"}


def test_legacy_snapshots_group_by_goal_and_by_time():
    """Older records carry no turn_id: namespaced ids and chronology recover it."""
    from CoScientist.graph.projection import turns as trace_by_turn

    full = {"nodes": [
        # Namespaced ids, the form used before turn_id existed.
        {"id": "goal:inv1", "kind": "goal", "label": "first", "t_start": 100.0,
         "t_end": 130.0, "status": "success"},
        {"id": "goal:inv1::tool:a", "kind": "tool_call", "label": "search",
         "t_start": 101.0, "t_end": 104.0},
        # No marker at all: it ran while the first request was open.
        {"id": "tool:loose", "kind": "tool_call", "label": "read",
         "t_start": 110.0, "t_end": 111.0},
        {"id": "goal:inv2", "kind": "goal", "label": "second", "t_start": 200.0,
         "t_end": 210.0, "status": "success"},
        {"id": "tool:later", "kind": "tool_call", "label": "write",
         "t_start": 205.0, "t_end": 206.0},
    ]}

    turns = trace_by_turn(full)["turns"]
    assert [t["prompt"] for t in turns] == ["first", "second"]
    assert [c["tool"] for c in turns[0]["calls"]] == ["search", "read"]
    assert [c["tool"] for c in turns[1]["calls"]] == ["write"]


def test_execution_tree_drops_the_roster_and_measures_depth():
    """request -> agent -> its calls -> answer, with nothing that never ran."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "system:root", "kind": "system", "label": "the system"},
        {"id": "agent:NeverCalled", "kind": "agent", "label": "NeverCalled"},
        {"id": "goal:i1", "kind": "goal", "label": "do it", "t_start": 10.0},
        {"id": "goal:i1::agent:Research", "kind": "agent_call",
         "executor_agent": "Research", "t_start": 11.0},
        {"id": "tool:a", "kind": "tool_call", "label": "search", "t_start": 12.0},
        {"id": "tool:b", "kind": "tool_call", "label": "extract", "t_start": 13.0},
        {"id": "result:i1", "kind": "result", "output": "done", "t_start": 20.0},
    ], "edges": [
        {"src": "system:root", "dst": "agent:NeverCalled", "type": "has_member"},
        {"src": "system:root", "dst": "goal:i1", "type": "caused_by"},
        {"src": "goal:i1", "dst": "goal:i1::agent:Research", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "tool:a", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "tool:b", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "result:i1", "type": "produced"},
    ]}

    tree = execution_tree(full, collapse_tools=False)
    ids = {n["id"] for n in tree["nodes"]}
    assert "agent:NeverCalled" not in ids, "an agent nothing called is roster"
    assert "system:root" not in ids, "the hub carries no information"
    # The calls are on the agent now, not beside it.
    assert not (ids & {"tool:a", "tool:b"})
    agent = next(n for n in tree["nodes"] if n["id"] == "goal:i1::agent:Research")
    assert [c["tool"] for c in agent["calls"]] == ["search", "extract"]

    level = {n["id"]: n["level"] for n in tree["nodes"]}
    assert level["goal:i1"] == 0
    assert level["goal:i1::agent:Research"] == 1
    # The answer ends the request, to the right of everything it did.
    assert level["result:i1"] > level["goal:i1::agent:Research"]


def _agent_with_calls():
    return {"nodes": [
        {"id": "goal:i1", "kind": "goal", "label": "do it", "t_start": 10.0,
         "t_end": 30.0},
        # Written by the last delegation of the session, not by this request.
        {"id": "agent:Research", "kind": "agent", "executor_agent": "Research",
         "status": "success", "input": "task: find papers",
         "output": "found 3", "t_start": 14.0, "t_end": 15.0},
        {"id": "tool:b", "kind": "tool_call", "label": "extract",
         "executor_agent": "Research", "status": "success", "input": "url: x",
         "output": "text", "t_start": 13.0, "t_end": 13.5,
         "output_files": ["s3://b/k"]},
        {"id": "tool:a", "kind": "tool_call", "label": "search",
         "executor_agent": "Research", "status": "failed", "input": "q: y",
         "output": "boom", "t_start": 12.0, "t_end": 12.25},
        {"id": "tool:orphan", "kind": "tool_call", "label": "stray",
         "t_start": 16.0},
        {"id": "result:i1", "kind": "result", "output": "done", "t_start": 20.0},
    ], "edges": [
        {"src": "goal:i1", "dst": "agent:Research", "type": "caused_by"},
        {"src": "agent:Research", "dst": "tool:a", "type": "caused_by"},
        {"src": "agent:Research", "dst": "tool:b", "type": "caused_by"},
        {"src": "goal:i1", "dst": "tool:orphan", "type": "caused_by"},
        {"src": "agent:Research", "dst": "result:i1", "type": "produced"},
    ]}


def test_tool_calls_fold_into_the_agent_that_made_them():
    """The canvas shows agents; what each one did is carried on its card."""
    from CoScientist.graph.projection import execution_tree

    tree = execution_tree(_agent_with_calls())
    by_id = {n["id"]: n for n in tree["nodes"]}
    assert "tool:a" not in by_id and "tool:b" not in by_id
    # A call with no agent to fold into is kept, not lost.
    assert "tool:orphan" in by_id

    calls = by_id["agent:Research"]["calls"]
    assert [c["tool"] for c in calls] == ["search", "extract"], "in the order run"
    first = calls[0]
    assert first["id"] == "tool:a" and first["status"] == "failed"
    assert first["input"] == "q: y" and first["output"] == "boom"
    assert first["duration"] == 0.25
    assert calls[1]["output_files"] == ["s3://b/k"]

    # No edge may name a card that is no longer drawn.
    ends = {end for e in tree["edges"] for end in (e["src"], e["dst"])}
    assert ends <= set(by_id)
    # The agent's own input and output are untouched.
    assert by_id["agent:Research"]["input"] == "task: find papers"
    assert by_id["agent:Research"]["output"] == "found 3"


def test_a_folded_agent_spans_its_calls():
    """A shared agent node carries whatever delegation wrote last; in this
    request it started no later than its first call and ended no earlier
    than its last, so the card's clock and duration describe this request."""
    from CoScientist.graph.projection import execution_tree

    agent = {n["id"]: n for n in execution_tree(_agent_with_calls())["nodes"]}["agent:Research"]
    assert agent["t_start"] == 12.0
    assert agent["t_end"] == 15.0, "its own later end stands"

    full = _agent_with_calls()
    node = next(n for n in full["nodes"] if n["id"] == "agent:Research")
    node["t_end"] = 12.1                       # ended before its own last call
    agent = {n["id"]: n for n in execution_tree(full)["nodes"]}["agent:Research"]
    assert agent["t_end"] == 13.5

    node["status"], node["t_end"] = "running", None
    agent = {n["id"]: n for n in execution_tree(full)["nodes"]}["agent:Research"]
    assert agent["t_end"] is None, "still running: no duration to print"


def test_a_folded_agent_keeps_its_place_in_time():
    """Folding leaves the picture request -> agents -> answer, in time order."""
    from CoScientist.graph.projection import execution_tree

    placed = {n["id"]: n for n in execution_tree(_agent_with_calls())["nodes"]}
    order = [n["id"] for n in sorted(placed.values(), key=lambda n: n["x"])]
    assert order == ["goal:i1", "agent:Research", "tool:orphan", "result:i1"]
    assert placed["agent:Research"]["row"] == 1


def test_nodes_are_placed_by_when_they_ran_and_who_ran_them():
    """x advances with the clock; a row belongs to an agent, not to a depth."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:i1", "kind": "goal", "label": "ask", "t_start": 0.0},
        {"id": "a:One", "kind": "agent_call", "executor_agent": "One", "t_start": 1.0},
        {"id": "t:1", "kind": "tool_call", "label": "first", "t_start": 2.0},
        {"id": "t:2", "kind": "tool_call", "label": "second", "t_start": 3.0},
        {"id": "a:Two", "kind": "agent_call", "executor_agent": "Two", "t_start": 4.0},
        {"id": "t:3", "kind": "tool_call", "label": "third", "t_start": 5.0},
    ], "edges": [
        {"src": "goal:i1", "dst": "a:One", "type": "caused_by"},
        {"src": "a:One", "dst": "t:1", "type": "caused_by"},
        {"src": "a:One", "dst": "t:2", "type": "caused_by"},
        {"src": "a:One", "dst": "a:Two", "type": "delegated_to"},
        {"src": "a:Two", "dst": "t:3", "type": "caused_by"},
    ]}

    placed = {n["id"]: n for n in execution_tree(full, collapse_tools=False)["nodes"]}

    # Reading left to right reads forward in time.
    order = sorted(placed.values(), key=lambda n: n["x"])
    assert [n["id"] for n in order] == ["goal:i1", "a:One", "a:Two"]
    # And no two cards in one lane can sit on top of each other.
    from CoScientist.graph.projection import _CARD_WIDTH
    # Only cards sharing a lane AND a sub-row can collide.
    by_cell = {}
    for node in placed.values():
        by_cell.setdefault((node["row"], node.get("sub_row", 0)), []).append(node["x"])
    for xs in by_cell.values():
        xs.sort()
        assert all(b - a >= _CARD_WIDTH for a, b in zip(xs, xs[1:]))

    # Each agent keeps its own calls, and keeps its own lane.
    assert [c["tool"] for c in placed["a:One"]["calls"]] == ["first", "second"]
    assert [c["tool"] for c in placed["a:Two"]["calls"]] == ["third"]
    assert placed["a:One"]["row"] != placed["a:Two"]["row"]
    assert placed["goal:i1"]["row"] == 0


def _two_requests_sharing_agents():
    """Two prompts served by the same agent nodes — the shape the plugin writes.

    An agent node is created once per session and reused, so its ``turn_id``
    names whichever request reached it first. Everything else is per-request.
    """
    return {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "first ask",
         "t_start": 100.0, "t_end": 150.0},
        {"id": "t:1", "kind": "tool_call", "turn_id": "one", "label": "search",
         "executor_agent": "Orchestrator", "t_start": 110.0, "t_end": 111.0},
        {"id": "t:2", "kind": "tool_call", "turn_id": "one", "label": "read",
         "executor_agent": "Research", "t_start": 120.0, "t_end": 121.0},

        {"id": "goal:2", "kind": "goal", "turn_id": "two", "label": "second ask",
         "t_start": 200.0, "t_end": 260.0},
        {"id": "t:3", "kind": "tool_call", "turn_id": "two", "label": "search",
         "executor_agent": "Orchestrator", "t_start": 210.0, "t_end": 211.0},
        {"id": "t:4", "kind": "tool_call", "turn_id": "two", "label": "read",
         "executor_agent": "Research", "t_start": 220.0, "t_end": 221.0},
        {"id": "res:2", "kind": "result", "turn_id": "two", "output": "done",
         "t_start": 260.0, "t_end": 260.0},

        # Written once, on the first request, and reused by the second.
        {"id": "a:Orchestrator", "kind": "agent", "turn_id": "one",
         "executor_agent": "Orchestrator", "t_start": 100.0},
        {"id": "a:Research", "kind": "agent", "turn_id": "one",
         "executor_agent": "Research", "t_start": 105.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:Orchestrator", "type": "caused_by"},
        {"src": "goal:2", "dst": "a:Orchestrator", "type": "caused_by"},
        {"src": "a:Orchestrator", "dst": "t:1", "type": "caused_by"},
        {"src": "a:Orchestrator", "dst": "t:3", "type": "caused_by"},
        {"src": "a:Orchestrator", "dst": "a:Research", "type": "delegated_to"},
        {"src": "a:Research", "dst": "t:2", "type": "caused_by"},
        {"src": "a:Research", "dst": "t:4", "type": "caused_by"},
        {"src": "a:Orchestrator", "dst": "res:2", "type": "produced"},
    ]}


def test_a_shared_agent_appears_in_every_request_it_served():
    """The reported bug: request 7 drawn as loose cards with no edges.

    Scoping an agent to the one request its ``turn_id`` names dropped it from
    all the others, and every edge that named it went with it.
    """
    from CoScientist.graph.projection import execution_tree

    full = _two_requests_sharing_agents()
    for turn_id in ("one", "two"):
        tree = execution_tree(full, turn_id)
        present = {n["id"] for n in tree["nodes"]}
        assert "a:Orchestrator" in present, turn_id
        assert "a:Research" in present, turn_id

        # Every node hangs off something: a trace, not a scatter of cards.
        touched = {end for e in tree["edges"] for end in (e["src"], e["dst"])}
        assert present - touched == set(), turn_id
        assert len(tree["edges"]) == len(present) - 1, turn_id


def test_a_request_keeps_only_its_own_calls():
    """Pulling shared agents in must not drag the other request's work along."""
    from CoScientist.graph.projection import execution_tree

    tree = execution_tree(_two_requests_sharing_agents(), "one",
                          collapse_tools=False)
    present = {n["id"] for n in tree["nodes"]}
    assert "goal:1" in present
    assert not present & {"goal:2", "res:2"}

    # The calls came with their agents; only this request's calls did.
    calls = {c["id"] for n in tree["nodes"] for c in (n.get("calls") or [])}
    assert calls == {"t:1", "t:2"}

    # Folded, the same request lists the same calls under their agents.
    folded = {n["id"]: n for n in execution_tree(_two_requests_sharing_agents(), "one")["nodes"]}
    assert [c["id"] for c in folded["a:Orchestrator"]["calls"]] == ["t:1"]
    assert [c["id"] for c in folded["a:Research"]["calls"]] == ["t:2"]


def test_an_agent_that_did_nothing_here_stays_out():
    """An agent joins a request by acting in it, not by existing."""
    from CoScientist.graph.projection import execution_tree

    full = _two_requests_sharing_agents()
    full["nodes"].append({"id": "a:Idle", "kind": "agent", "turn_id": "one",
                          "executor_agent": "Idle", "t_start": 106.0})
    full["edges"].append({"src": "a:Orchestrator", "dst": "a:Idle",
                          "type": "delegated_to"})

    assert "a:Idle" not in {n["id"] for n in execution_tree(full, "two")["nodes"]}


def test_a_call_keeps_everything_a_reader_needs_when_it_moves():
    """Folding must not cost the call its detail — that is the whole panel."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 10.0},
        {"id": "a:One", "kind": "agent_call", "turn_id": "one",
         "executor_agent": "One", "t_start": 11.0},
        {"id": "t:1", "kind": "tool_call", "turn_id": "one", "label": "search",
         "status": "failed", "input": "aspirin", "output": "boom",
         "t_start": 12.0, "t_end": 14.5},
    ], "edges": [
        {"src": "goal:1", "dst": "a:One", "type": "caused_by"},
        {"src": "a:One", "dst": "t:1", "type": "caused_by"},
    ]}

    agent = next(n for n in execution_tree(full, "one")["nodes"] if n["id"] == "a:One")
    call, = agent["calls"]

    assert {k: call[k] for k in
            ("id", "tool", "status", "input", "output", "t_start", "t_end", "duration")} == {
        "id": "t:1", "tool": "search", "status": "failed",
        "input": "aspirin", "output": "boom",
        "t_start": 12.0, "t_end": 14.5, "duration": 2.5}


def test_an_agent_reached_only_by_its_own_calls_survives():
    """It did work in this request, so it is history, not roster.

    The roster prune drops an agent nothing called. Reading that before the
    calls moved onto it dropped the agent AND took its calls with it.
    """
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 10.0},
        {"id": "a:Orphan", "kind": "agent", "turn_id": "one",
         "executor_agent": "Orphan", "t_start": 11.0},
        {"id": "t:1", "kind": "tool_call", "turn_id": "one", "label": "search",
         "t_start": 12.0},
    ], "edges": [
        {"src": "a:Orphan", "dst": "t:1", "type": "caused_by"},
    ]}

    nodes = {n["id"]: n for n in execution_tree(full, "one")["nodes"]}
    assert "a:Orphan" in nodes
    assert [c["tool"] for c in nodes["a:Orphan"]["calls"]] == ["search"]


def test_an_agent_is_timed_by_what_it_did_in_this_request():
    """A shared agent node carries the last time it ran ANYWHERE.

    Drawn on that clock the agent drifted away from the request it was in —
    days away, in a session used over a week — the layout stopped carrying any
    time at all, and the card showed an hour from another day.
    """
    from CoScientist.graph.projection import execution_tree

    later = 9_000.0          # when the agent was last touched, in a later request
    full = {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 100.0},
        {"id": "a:Orch", "kind": "agent", "turn_id": "one",
         "executor_agent": "Orch", "t_start": later},
        {"id": "a:Sub", "kind": "agent", "turn_id": "one",
         "executor_agent": "Sub", "t_start": later},
        {"id": "t:1", "kind": "tool_call", "turn_id": "one", "label": "search",
         "t_start": 130.0, "t_end": 140.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:Orch", "type": "caused_by"},
        {"src": "a:Orch", "dst": "a:Sub", "type": "delegated_to"},
        {"src": "a:Sub", "dst": "t:1", "type": "caused_by"},
    ]}

    nodes = {n["id"]: n for n in execution_tree(full, "one")["nodes"]}

    # The one that made the call is timed by the call, not by the week-old
    # stamp the shared node was carrying.
    assert nodes["a:Sub"]["t_start"] == 130.0
    assert nodes["a:Sub"]["t_end"] == 140.0
    # The one that only delegated has nothing of its own to measure, so it
    # falls back to the request rather than staying in another week.
    assert nodes["a:Orch"]["t_start"] == 100.0
    # Nothing is left sitting in another request's week.
    assert all(n["t_start"] < later for n in nodes.values() if n.get("t_start"))


def test_an_agent_with_nothing_measurable_falls_back_to_the_request():
    """Better at the request's own start than days away from it."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 100.0},
        {"id": "a:Quiet", "kind": "agent", "turn_id": "one",
         "executor_agent": "Quiet", "t_start": 9_000.0},
        # Something has to say when the request ended, or there is no window
        # to judge the agent against.
        {"id": "res:1", "kind": "result", "turn_id": "one", "output": "done",
         "t_start": 200.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:Quiet", "type": "caused_by"},
        {"src": "a:Quiet", "dst": "res:1", "type": "produced"},
    ]}

    agent = next(n for n in execution_tree(full, "one")["nodes"] if n["id"] == "a:Quiet")
    assert agent["t_start"] == 100.0


def _aggregator_over_two_requests():
    """The shape the plugin writes for a lifecycle stage.

    ResultAggregatorAgent is wired into the run, not delegated to, so nothing
    points at it — and it is one node for the whole session, so its calls come
    from every request it ever closed.
    """
    return {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "first", "t_start": 100.0},
        {"id": "a:Orch", "kind": "agent", "turn_id": "one",
         "executor_agent": "Orch", "t_start": 9_000.0},
        {"id": "goal:2", "kind": "goal", "turn_id": "two", "label": "second", "t_start": 500.0},

        # One node, no inbound edge, calls from both requests.
        {"id": "a:Agg", "kind": "agent", "executor_agent": "ResultAggregator",
         "t_start": 9_000.0},
        {"id": "t:1", "kind": "tool_call", "turn_id": "one", "label": "collect",
         "t_start": 120.0, "t_end": 130.0},
        {"id": "t:2", "kind": "tool_call", "turn_id": "two", "label": "collect",
         "t_start": 520.0, "t_end": 530.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:Orch", "type": "caused_by"},
        {"src": "a:Agg", "dst": "t:1", "type": "caused_by"},
        {"src": "a:Agg", "dst": "t:2", "type": "caused_by"},
    ]}


def test_a_lifecycle_stage_is_joined_to_the_request_it_ran_in():
    """Nothing delegates to it, so without this it floats beside the trace."""
    from CoScientist.graph.projection import execution_tree

    tree = execution_tree(_aggregator_over_two_requests(), "one")
    ids = {n["id"] for n in tree["nodes"]}
    touched = {end for e in tree["edges"] for end in (e["src"], e["dst"])}

    assert "a:Agg" in ids
    assert ids - touched == set(), "every node hangs off something"
    joined = [e for e in tree["edges"] if e["dst"] == "a:Agg"]
    assert joined and joined[0]["src"] == "goal:1"
    # Marked as "this also ran", not as a call the request made.
    assert joined[0].get("synthetic") is True


def test_a_shared_stage_shows_only_this_request_s_calls():
    """One node for the whole session must not pour every request into one card."""
    from CoScientist.graph.projection import execution_tree

    full = _aggregator_over_two_requests()
    first = next(n for n in execution_tree(full, "one")["nodes"] if n["id"] == "a:Agg")
    second = next(n for n in execution_tree(full, "two")["nodes"] if n["id"] == "a:Agg")

    assert [c["id"] for c in first["calls"]] == ["t:1"]
    assert [c["id"] for c in second["calls"]] == ["t:2"]
    # And each is timed by the request it is drawn in.
    assert first["t_start"] == 120.0 and second["t_start"] == 520.0
def test_agent_cards_are_numbered_stages_and_runs():
    """Read down the request: 1. Planner, 2. Critic, 3. Planner (run 2 of 2)."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:i", "kind": "goal", "label": "ask", "t_start": 0.0},
        {"id": "a:Planner@i", "kind": "agent", "executor_agent": "Planner", "t_start": 1.0},
        {"id": "a:Critic@i", "kind": "agent", "executor_agent": "Critic", "t_start": 2.0},
        {"id": "a:Planner@i#2", "kind": "agent", "executor_agent": "Planner", "t_start": 3.0},
    ], "edges": [
        {"src": "goal:i", "dst": "a:Planner@i", "type": "caused_by"},
        {"src": "goal:i", "dst": "a:Critic@i", "type": "caused_by"},
        {"src": "goal:i", "dst": "a:Planner@i#2", "type": "caused_by"},
    ]}
    by_id = {n["id"]: n for n in execution_tree(full)["nodes"]}
    assert (by_id["a:Planner@i"]["stage"], by_id["a:Planner@i"]["run"], by_id["a:Planner@i"]["runs"]) == (1, 1, 2)
    assert (by_id["a:Critic@i"]["stage"], by_id["a:Critic@i"]["run"], by_id["a:Critic@i"]["runs"]) == (2, 1, 1)
    assert (by_id["a:Planner@i#2"]["stage"], by_id["a:Planner@i#2"]["run"]) == (3, 2)
    # Runs of one agent share its lane, so the loop reads across one row.
    assert by_id["a:Planner@i"]["row"] == by_id["a:Planner@i#2"]["row"]


def test_an_agent_lists_the_files_and_links_it_produced():
    from CoScientist.graph.projection import execution_tree

    full = _agent_with_calls()
    agent = next(n for n in full["nodes"] if n["id"] == "agent:Research")
    agent["output"] = "Saved the figure to s3://b/fig.png, see https://x.y/report (done)."
    agent["output_files"] = ["s3://b/fig.png"]
    got = {n["id"]: n for n in execution_tree(full)["nodes"]}["agent:Research"]["artifacts"]
    assert [a["uri"] for a in got] == ["s3://b/k", "s3://b/fig.png", "https://x.y/report"]
    assert got[0] == {"uri": "s3://b/k", "kind": "file", "tool": "extract",
                      "href": "/api/artifact/b/k"}
    assert got[2]["kind"] == "link"


def test_a_file_artifact_is_something_the_reader_can_open(tmp_path, monkeypatch):
    """`uri` is the durable reference; `href` is where the anchor goes.

    A ``file`` artifact used to carry no href, and the panel drew it as plain
    text — so an ``s3://`` key sat there unopenable even though
    ``/api/artifact/`` had been serving exactly that for ages. A mirrored file
    resolves against the scope that is *reading*, which is what lets an
    imported session open its own artifacts.
    """
    from CoScientist.graph.projection import execution_tree
    from CoScientist.reporting import session_files

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    scope = ("u1", "s1")
    # The bytes have to exist: a reference to a file this session does not hold
    # is deliberately not offered as a link.
    record = session_files.put_bytes(scope, b"\x89PNG\r\n\x1a\nx", filename="fig.png")
    reference = session_files.ref(record["artifact_id"])

    full = _agent_with_calls()
    agent = next(n for n in full["nodes"] if n["id"] == "agent:Research")
    agent["output_files"] = ["s3://b/fig.png", reference]
    got = {n["id"]: n for n in
           execution_tree(full, scope=scope)["nodes"]}["agent:Research"]["artifacts"]
    by_uri = {a["uri"]: a for a in got}
    assert by_uri["s3://b/fig.png"]["href"] == "/api/artifact/b/fig.png"
    assert by_uri[reference]["href"] == (
        f"/api/users/u1/sessions/s1/artifacts/{record['artifact_id']}"
    )
    assert by_uri[reference]["kind"] == "file"


def test_a_truncated_link_is_not_offered_as_an_artifact():
    """`_short()` marks its cut with an ellipsis, and the scrape used to keep it.

    ``http://10.3…`` was listed beside real files, and told the reader something
    existed that could never be fetched.
    """
    from CoScientist.graph.projection import execution_tree

    full = _agent_with_calls()
    agent = next(n for n in full["nodes"] if n["id"] == "agent:Research")
    agent["output"] = "figure=http://10.3… and s3://b/real.png"
    agent["output_files"] = []
    got = {n["id"]: n for n in execution_tree(full)["nodes"]}["agent:Research"]["artifacts"]
    assert not [a for a in got if "…" in a["uri"]]
    assert "s3://b/real.png" in {a["uri"] for a in got}


def test_an_old_snapshot_borrows_task_and_report_for_its_agents():
    """Recorded before activations: the delegation's arguments and result sit
    on one node in the caller's request, the callee's calls on another in a
    request of its own. The reader opens the second and must still see both."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:outer", "kind": "goal", "turn_id": "outer", "label": "do it",
         "t_start": 0.0, "t_end": 50.0},
        {"id": "goal:outer::agent:Orch", "kind": "agent_call", "turn_id": "outer",
         "executor_agent": "Orch", "t_start": 1.0, "t_end": 49.0},
        # The delegation, in the caller's request, with the arguments and result.
        {"id": "goal:outer::agent:Coder", "kind": "agent_call", "turn_id": "outer",
         "executor_agent": "Coder", "input": "request: write it", "output": "wrote it",
         "t_start": 10.0, "t_end": 40.0},
        {"id": "result:outer", "kind": "result", "turn_id": "outer", "output": "all done",
         "t_start": 50.0, "t_end": 50.0},
        # The callee's own request: its goal repeats the task, its agent card is empty.
        {"id": "goal:inner", "kind": "goal", "turn_id": "inner", "label": "write it",
         "t_start": 10.5, "t_end": 39.5},
        {"id": "goal:inner::agent:Coder", "kind": "agent_call", "turn_id": "inner",
         "executor_agent": "Coder", "t_start": 11.0, "t_end": 39.0},
        {"id": "t:1", "kind": "tool_call", "turn_id": "inner", "label": "bash",
         "executor_agent": "Coder", "t_start": 12.0, "t_end": 13.0},
        {"id": "result:inner", "kind": "result", "turn_id": "inner", "output": "wrote it",
         "t_start": 39.5, "t_end": 39.5},
        # No delegation record anywhere: falls back to the request it served.
        {"id": "goal:solo", "kind": "goal", "turn_id": "solo", "label": "explore",
         "t_start": 60.0, "t_end": 70.0},
        {"id": "goal:solo::agent:Coder", "kind": "agent_call", "turn_id": "solo",
         "executor_agent": "Coder", "t_start": 61.0, "t_end": 69.0},
        {"id": "result:solo", "kind": "result", "turn_id": "solo", "output": "explored",
         "t_start": 70.0, "t_end": 70.0},
    ], "edges": [
        {"src": "goal:outer", "dst": "goal:outer::agent:Orch", "type": "caused_by"},
        {"src": "goal:outer::agent:Orch", "dst": "goal:outer::agent:Coder", "type": "delegated_to"},
        {"src": "goal:outer::agent:Orch", "dst": "result:outer", "type": "produced"},
        {"src": "goal:inner", "dst": "goal:inner::agent:Coder", "type": "caused_by"},
        {"src": "goal:inner::agent:Coder", "dst": "t:1", "type": "caused_by"},
        {"src": "goal:inner::agent:Coder", "dst": "result:inner", "type": "produced"},
        {"src": "goal:solo", "dst": "goal:solo::agent:Coder", "type": "caused_by"},
        {"src": "goal:solo::agent:Coder", "dst": "result:solo", "type": "produced"},
    ]}
    inner = {n["id"]: n for n in execution_tree(full, "inner")["nodes"]}["goal:inner::agent:Coder"]
    assert inner["input"] == "request: write it"
    assert inner["output"] == "wrote it"
    assert inner["io_source"] == "delegation"
    assert [c["tool"] for c in inner["calls"]] == ["bash"], "its own calls stay its own"

    solo = {n["id"]: n for n in execution_tree(full, "solo")["nodes"]}["goal:solo::agent:Coder"]
    assert solo["input"] == "explore" and solo["output"] == "explored"
    assert solo["io_source"] == "request"

    # The delegation node itself already says what it says.
    outer = {n["id"]: n for n in execution_tree(full, "outer")["nodes"]}["goal:outer::agent:Coder"]
    assert outer["input"] == "request: write it" and "io_source" not in outer


def test_work_that_ran_before_any_request_is_not_lost():
    """Stranding a call loses it from every view at once.

    A node that starts before every goal used to resolve to a request nobody
    can open. One stored session showed 35 of its 130 calls that way, and the
    evidence linking to the other 95 led to a page with nothing on it.
    """
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        # The goal is stamped later than the work recorded under it.
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 500.0},
        {"id": "a:One", "kind": "agent", "turn_id": "one",
         "executor_agent": "One", "t_start": 100.0},
        {"id": "t:early", "kind": "tool_call", "label": "search", "t_start": 110.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:One", "type": "caused_by"},
        {"src": "a:One", "dst": "t:early", "type": "caused_by"},
    ]}

    tree = execution_tree(full)
    assert [t["turn_id"] for t in tree["turns"]] == ["one"]
    calls = [c["id"] for n in execution_tree(full, "one")["nodes"]
             for c in (n.get("calls") or [])]
    assert calls == ["t:early"]


def _a_run_through_its_phases():
    """A request that frames, searches, computes and then writes up."""
    return {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 0.0},
        {"id": "a:Orch", "kind": "agent", "turn_id": "one",
         "executor_agent": "OrchestratorAgent", "t_start": 1.0},
        {"id": "a:Hyp", "kind": "agent", "turn_id": "one",
         "executor_agent": "HypothesesAgent", "t_start": 2.0},
        {"id": "a:Res", "kind": "agent", "turn_id": "one",
         "executor_agent": "ResearchAgent", "t_start": 3.0},
        {"id": "a:Coder", "kind": "agent", "turn_id": "one",
         "executor_agent": "CoderAgent", "t_start": 4.0},
        {"id": "a:Agg", "kind": "agent", "turn_id": "one",
         "executor_agent": "ResultAggregatorAgent", "t_start": 5.0},
    ], "edges": [
        {"src": "goal:1", "dst": "a:Orch", "type": "caused_by"},
        {"src": "a:Orch", "dst": "a:Hyp", "type": "delegated_to"},
        {"src": "a:Orch", "dst": "a:Res", "type": "delegated_to"},
        {"src": "a:Orch", "dst": "a:Coder", "type": "delegated_to"},
        {"src": "a:Orch", "dst": "a:Agg", "type": "delegated_to"},
    ]}


def test_a_request_reads_as_the_stretches_it_went_through():
    from CoScientist.graph.projection import execution_tree

    bands = execution_tree(_a_run_through_its_phases(), "one")["phases"]

    assert [b["phase"] for b in bands] == ["framing", "research", "experiment", "report"]
    assert [b["label"] for b in bands] == [
        "постановка", "поиск и данные", "эксперимент", "отчёт"]
    # Bands run left to right and never overlap.
    for earlier, later in zip(bands, bands[1:]):
        assert earlier["x1"] <= later["x0"], (earlier, later)


def test_the_orchestrator_does_not_get_a_band_of_its_own():
    """It conducts every stretch; a band for it would cut the run into slivers."""
    from CoScientist.graph.projection import execution_tree

    bands = execution_tree(_a_run_through_its_phases(), "one")["phases"]

    assert "orchestrator" not in {b["phase"] for b in bands}
    # It ran first and introduced the framing, so that is the band it sits in.
    assert bands[0]["agents"] == 2


def test_a_run_of_agents_nobody_classified_gets_no_bands():
    """Better nothing than a band that means whatever was left over."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "goal:1", "kind": "goal", "turn_id": "one", "label": "ask", "t_start": 0.0},
        {"id": "a:X", "kind": "agent", "turn_id": "one", "executor_agent": "Mystery",
         "t_start": 1.0},
    ], "edges": [{"src": "goal:1", "dst": "a:X", "type": "caused_by"}]}

    assert execution_tree(full, "one")["phases"] == []


def test_mcp_endpoint_links_are_not_artifacts():
    """An agent that echoes the tool server it called produced nothing there."""
    from CoScientist.graph.projection import _artifacts_of

    agent = {"output": (
        "Uploaded through http://10.32.1.114:7338/mcp and the figure is at "
        "http://10.32.1.114:9000/agent-vault/ephemeral/u/s/fig.png?X-Amz-Signature=ab12. "
        "See also s3://agent-vault/ephemeral/u/s/table.csv"
    )}

    artifacts = _artifacts_of(agent)

    uris = [a["uri"] for a in artifacts]
    assert not any(uri.rstrip("/").endswith("/mcp") for uri in uris)
    assert any(uri.endswith("table.csv") for uri in uris)
    assert any("fig.png" in uri for uri in uris)


def test_a_refused_write_is_drawn_as_a_failure():
    """A transactional write that saved NOTHING used to be drawn as a success.

    `research_commit` answers a rejected commit with {"ok": false, "errors":
    [...]} and carries neither `status` nor `error`, so the log said the call
    went fine. In one real session three of ten commits were refused that way:
    whole steps never reached the graph, and nothing anywhere said so.
    """
    from CoScientist.graph.emitter import _is_error as emitter_is_error
    from CoScientist.graph.plugin import _is_error as plugin_is_error

    refused = {"ok": False, "errors": ["nodes[0]: missing attrs.subtype"]}
    for is_error in (emitter_is_error, plugin_is_error):
        assert is_error(refused) is True
        assert is_error({"ok": True, "message": "committed 2 node(s)"}) is False
        # A tool that simply has no `ok` key is not a failure — which is why
        # this tests `is False` and not a falsy `ok`.
        assert is_error({"result": "success", "plan": []}) is False
        assert is_error({"status": "error"}) is True
        assert is_error("plain text") is False

def test_a_recorded_plan_shows_up_in_the_trace():
    """The Experiment Module records its plan as a `decision` node.

    A trace that listed only tool calls showed a run executing a plan the reader
    could not see. A decision belongs with the calls: something an agent did in
    the turn, with a start, an end and a verdict.
    """
    full = {"nodes": [
        {"id": "goal:a", "kind": "goal", "turn_id": "a", "label": "run the experiment",
         "status": "success", "t_start": 100.0, "t_end": 140.0},
        {"id": "plan:P1@r1:agent:x", "kind": "decision", "turn_id": "a",
         "label": "plan rev 1 - 3 tasks - 45 min",
         "executor_agent": "ExperimentPlannerAgent", "status": "success",
         "verdict": "approved",
         "input": {"kind": "experiment_plan", "task_count": 3},
         "output": "plan rev 1 - 3 tasks - 45 min", "t_start": 105.0, "t_end": 118.0},
        {"id": "tool:1", "kind": "tool_call", "turn_id": "a", "label": "start_task",
         "executor_agent": "ExperimentExecutorAgent", "status": "success",
         "t_start": 120.0, "t_end": 120.5},
    ]}

    calls = turns(full)["turns"][0]["calls"]
    assert [c["tool"] for c in calls] == ["plan rev 1 - 3 tasks - 45 min", "start_task"]
    plan = calls[0]
    assert plan["agent"] == "ExperimentPlannerAgent"
    assert plan["input"]["kind"] == "experiment_plan"
    assert plan["duration"] == 13.0


def test_the_experiment_module_reads_as_one_stretch():
    """Its stages share the experiment band instead of scattering the module
    across three of them."""
    from CoScientist.graph.projection import _PHASE_OF_AGENT

    for name in (
        "ExperimentModuleAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
        "ToolRetrieverAgent",
        "ToolReranker",
    ):
        assert _PHASE_OF_AGENT[name] == "experiment", name
