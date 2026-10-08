"""Lifecycle coverage for the session-aware execution-graph plugin."""

import asyncio
from types import SimpleNamespace

from google.genai import types

from CoScientist.graph import plugin as plugin_module
from CoScientist.graph.plugin import GraphMemoryPlugin
from CoScientist.graph.session_scope import session_key


class _RecordingGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, **node):
        self.nodes[node["id"]] = dict(node)

    def add_edge(self, src, dst, type="caused_by"):
        self.edges.append((src, dst, type))

    def set_status(self, node_id, **updates):
        self.nodes[node_id].update(updates)

    def full(self):
        return {
            "nodes": [dict({"id": node_id}, **node) for node_id, node in self.nodes.items()],
            "edges": [
                {"src": src, "dst": dst, "type": edge_type}
                for src, dst, edge_type in self.edges
            ],
        }


def _contexts(*, user_id: str, session_id: str, invocation_id: str):
    state = {}
    session = SimpleNamespace(
        user_id=user_id,
        id=session_id,
        state=state,
    )
    invocation = SimpleNamespace(
        invocation_id=invocation_id,
        session=session,
        agent=SimpleNamespace(name="OrchestratorAgent"),
    )
    tool = SimpleNamespace(
        state=state,
        agent_name="OrchestratorAgent",
        function_call_id="call-1",
        _invocation_context=invocation,
    )
    return invocation, tool


def _final_event(text: str):
    return SimpleNamespace(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        is_final_response=lambda: True,
    )


def _install_graph_resolver(monkeypatch):
    graphs = {}

    def resolve(context):
        return graphs.setdefault(session_key(context), _RecordingGraph())

    monkeypatch.setattr(plugin_module, "get_knowledge_graph", resolve)
    monkeypatch.setattr(plugin_module, "_agent_names", lambda: set())
    # Pin the agent topology: these scenarios drive OrchestratorAgent directly,
    # and reading root/parents out of system.yaml made the assertions depend on
    # the deployed hierarchy — wrapping the orchestrator in a composite parent
    # legitimately adds that parent's node and broke the exact-set checks.
    monkeypatch.setattr(plugin_module, "_system_root", lambda: "OrchestratorAgent")
    monkeypatch.setattr(plugin_module, "_composite_parents", lambda: {})
    monkeypatch.setenv("KG_SEMANTIC_ENABLED", "0")
    return graphs


def test_run_state_survives_until_after_run_and_builds_one_goal_tree(monkeypatch):
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        invocation, tool_context = _contexts(
            user_id="user-a",
            session_id="session-a",
            invocation_id="inv-a",
        )
        tool = SimpleNamespace(name="search_tool")

        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(
                role="user",
                parts=[types.Part(text="Find candidate molecules")],
            ),
        )
        run_key = plugin._run_key(invocation)
        assert run_key in plugin._runs

        await plugin.before_tool_callback(
            tool=tool,
            tool_args={"query": "candidate molecules"},
            tool_context=tool_context,
        )
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={"query": "candidate molecules"},
            tool_context=tool_context,
            result={"status": "success", "items": ["A"]},
        )
        await plugin.on_event_callback(
            invocation_context=invocation,
            event=_final_event("Candidate A is the best match."),
        )

        graph = graphs[("user-a", "session-a")]
        goal_id = "goal:inv-a"
        # One node per activation: the agent, the request it served, and
        # (from the second run on) which run in that request this was.
        agent_id = "agent:OrchestratorAgent@inv-a"
        tool_id = "tool:call-1"
        result_id = "result:inv-a"

        assert "goal:pending" not in graph.nodes
        assert set(graph.nodes) == {goal_id, agent_id, tool_id, result_id}
        assert graph.nodes[goal_id]["status"] == "success"
        assert graph.nodes[agent_id]["status"] == "success"
        assert graph.nodes[tool_id]["status"] == "success"
        assert (goal_id, agent_id, "caused_by") in graph.edges
        assert (agent_id, tool_id, "caused_by") in graph.edges
        assert (agent_id, result_id, "produced") in graph.edges
        assert run_key in plugin._runs

        await plugin.after_run_callback(invocation_context=invocation)
        assert run_key not in plugin._runs

    asyncio.run(scenario())


def test_parallel_invocations_keep_independent_state_until_each_run_ends(monkeypatch):
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        first_invocation, first_tool_context = _contexts(
            user_id="user-a",
            session_id="session-a",
            invocation_id="inv-a",
        )
        second_invocation, second_tool_context = _contexts(
            user_id="user-b",
            session_id="session-b",
            invocation_id="inv-b",
        )
        tool = SimpleNamespace(name="shared_tool")

        await asyncio.gather(
            plugin.on_user_message_callback(
                invocation_context=first_invocation,
                user_message=types.Content(
                    role="user",
                    parts=[types.Part(text="First research")],
                ),
            ),
            plugin.on_user_message_callback(
                invocation_context=second_invocation,
                user_message=types.Content(
                    role="user",
                    parts=[types.Part(text="Second research")],
                ),
            ),
        )
        await asyncio.gather(
            plugin.before_tool_callback(
                tool=tool,
                tool_args={"run": "first"},
                tool_context=first_tool_context,
            ),
            plugin.before_tool_callback(
                tool=tool,
                tool_args={"run": "second"},
                tool_context=second_tool_context,
            ),
        )

        first_key = plugin._run_key(first_invocation)
        second_key = plugin._run_key(second_invocation)
        assert first_key != second_key
        assert plugin._runs[first_key].goal_id == "goal:inv-a"
        assert plugin._runs[first_key].goal_text == "First research"
        assert plugin._runs[second_key].goal_id == "goal:inv-b"
        assert plugin._runs[second_key].goal_text == "Second research"

        await plugin.on_event_callback(
            invocation_context=first_invocation,
            event=_final_event("First result"),
        )
        await plugin.after_run_callback(invocation_context=first_invocation)
        assert first_key not in plugin._runs
        assert second_key in plugin._runs

        await plugin.on_event_callback(
            invocation_context=second_invocation,
            event=_final_event("Second result"),
        )
        await plugin.after_run_callback(invocation_context=second_invocation)
        assert plugin._runs == {}

        first_nodes = set(graphs[("user-a", "session-a")].nodes)
        second_nodes = set(graphs[("user-b", "session-b")].nodes)
        assert "goal:inv-a" in first_nodes
        assert "result:inv-a" in first_nodes
        assert not any("inv-b" in node_id for node_id in first_nodes)
        assert "goal:inv-b" in second_nodes
        assert "result:inv-b" in second_nodes
        assert not any("inv-a" in node_id for node_id in second_nodes)

    asyncio.run(scenario())


def test_run_without_final_response_is_marked_interrupted(monkeypatch):
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        invocation, tool_context = _contexts(
            user_id="user-stop",
            session_id="session-stop",
            invocation_id="inv-stop",
        )
        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(
                role="user",
                parts=[types.Part(text="Long running research")],
            ),
        )
        await plugin.before_tool_callback(
            tool=SimpleNamespace(name="slow_tool"),
            tool_args={},
            tool_context=tool_context,
        )

        await plugin.after_run_callback(invocation_context=invocation)

        graph = graphs[("user-stop", "session-stop")]
        assert graph.nodes["goal:inv-stop"]["status"] == "interrupted"
        assert graph.nodes["agent:OrchestratorAgent@inv-stop"]["status"] == "interrupted"
        assert graph.nodes["tool:call-1"]["status"] == "interrupted"
        assert plugin._runs == {}

    asyncio.run(scenario())




# ── activations: every run of an agent is a stage of its own ─────────────────

def _agent(name, parent=None, children=()):
    node = SimpleNamespace(name=name, parent_agent=parent, sub_agents=list(children))
    for child in children:
        child.parent_agent = node
    return node


def _callback_context(invocation, name):
    """What ADK hands to before/after_agent_callback for one agent's run."""
    return SimpleNamespace(
        state=invocation.session.state,
        agent_name=name,
        invocation_id=invocation.invocation_id,
        _invocation_context=invocation,
    )


def _delegation_contexts(parent_invocation, callee, call_id, child_invocation_id):
    """A callee runs in an invocation of its own, on the same public session."""
    child = SimpleNamespace(
        invocation_id=child_invocation_id,
        session=parent_invocation.session,
        agent=SimpleNamespace(name=callee),
    )
    tool = SimpleNamespace(
        state=parent_invocation.session.state,
        agent_name=callee,
        function_call_id=call_id,
        _invocation_context=child,
    )
    return child, tool


def test_each_delegation_is_its_own_activation_with_its_own_calls(monkeypatch):
    """Planner → Critic → Planner is three cards, in order, not two merged ones."""
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        monkeypatch.setattr(plugin_module, "_agent_names", lambda: {"CoderAgent"})
        plugin = GraphMemoryPlugin()
        invocation, orchestrator_tool = _contexts(
            user_id="u", session_id="s", invocation_id="inv-1")
        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(role="user", parts=[types.Part(text="build it")]))
        await plugin.before_agent_callback(
            agent=_agent("OrchestratorAgent"),
            callback_context=_callback_context(invocation, "OrchestratorAgent"))

        coder = SimpleNamespace(name="CoderAgent")
        for run, task in enumerate(("write the code", "fix the tests"), start=1):
            orchestrator_tool.function_call_id = f"deleg-{run}"
            await plugin.before_tool_callback(
                tool=coder, tool_args={"request": task}, tool_context=orchestrator_tool)
            child, coder_tool = _delegation_contexts(
                invocation, "CoderAgent", f"bash-{run}", f"inv-coder-{run}")
            await plugin.before_agent_callback(
                agent=_agent("CoderAgent"),
                callback_context=_callback_context(child, "CoderAgent"))
            bash = SimpleNamespace(name="execute_bash")
            await plugin.before_tool_callback(
                tool=bash, tool_args={"cmd": f"make {run}"}, tool_context=coder_tool)
            await plugin.after_tool_callback(
                tool=bash, tool_args={"cmd": f"make {run}"}, tool_context=coder_tool,
                result={"status": "success"})
            await plugin.on_event_callback(
                invocation_context=child, event=_final_event(f"done with {task}"))
            await plugin.after_agent_callback(
                agent=_agent("CoderAgent"),
                callback_context=_callback_context(child, "CoderAgent"))
            await plugin.after_run_callback(invocation_context=child)
            await plugin.after_tool_callback(
                tool=coder, tool_args={"request": task}, tool_context=orchestrator_tool,
                result=f"done with {task}")

        await plugin.on_event_callback(
            invocation_context=invocation, event=_final_event("all built"))
        await plugin.after_run_callback(invocation_context=invocation)

        graph = graphs[("u", "s")]
        first, second = "agent:CoderAgent@inv-1", "agent:CoderAgent@inv-1#2"
        assert {first, second} <= set(graph.nodes)
        assert graph.nodes[first]["input"] == "request: write the code"
        assert graph.nodes[first]["output"] == "done with write the code"
        assert graph.nodes[second]["input"] == "request: fix the tests"
        assert graph.nodes[second]["output"] == "done with fix the tests"
        assert graph.nodes[first]["turn_id"] == graph.nodes[second]["turn_id"] == "inv-1"
        # Each run's tool call hangs off that run, not off a shared agent node.
        assert (first, "tool:bash-1", "caused_by") in graph.edges
        assert (second, "tool:bash-2", "caused_by") in graph.edges
        # And each run was delegated by the orchestrator's activation.
        orch = "agent:OrchestratorAgent@inv-1"
        assert (orch, first, "delegated_to") in graph.edges
        assert (orch, second, "delegated_to") in graph.edges
        assert graph.nodes[second]["status"] == "success"
        assert graph.nodes["goal:inv-1"]["status"] == "success"
        assert graph.nodes["result:inv-1"]["output"] == "all built"

    asyncio.run(scenario())


def test_pipeline_wrapper_opens_the_goal_and_the_last_stage_answers(monkeypatch):
    """The Runner's root is a SequentialAgent around the system root; the
    request still opens, every stage hangs off it in order, a composite's
    child hangs under the composite, and the report written by the post-stage
    is the answer the user got."""
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        invocation, orchestrator_tool = _contexts(
            user_id="u", session_id="s", invocation_id="inv-p")
        orchestrator = _agent("OrchestratorAgent")
        planner = _agent("PlannerAgent")
        planning = _agent("PlanningPipelineAgent", children=(planner,))
        aggregator = _agent("ResultAggregatorAgent")
        wrapper = _agent("ResearchPipeline", children=(planning, orchestrator, aggregator))
        invocation.agent = wrapper

        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(role="user", parts=[types.Part(text="study X")]))
        assert "goal:inv-p" in graphs[("u", "s")].nodes, "the wrapper is the request"

        cc = lambda name: _callback_context(invocation, name)  # noqa: E731
        await plugin.before_agent_callback(agent=wrapper, callback_context=cc("ResearchPipeline"))
        await plugin.before_agent_callback(agent=planning, callback_context=cc("PlanningPipelineAgent"))
        await plugin.before_agent_callback(agent=planner, callback_context=cc("PlannerAgent"))
        await plugin.on_event_callback(
            invocation_context=invocation,
            event=SimpleNamespace(author="PlannerAgent", **vars(_final_event("the plan"))))
        await plugin.after_agent_callback(agent=planner, callback_context=cc("PlannerAgent"))
        await plugin.after_agent_callback(agent=planning, callback_context=cc("PlanningPipelineAgent"))
        await plugin.before_agent_callback(agent=orchestrator, callback_context=cc("OrchestratorAgent"))
        await plugin.on_event_callback(
            invocation_context=invocation,
            event=SimpleNamespace(author="OrchestratorAgent", **vars(_final_event("did the work"))))
        await plugin.after_agent_callback(agent=orchestrator, callback_context=cc("OrchestratorAgent"))
        await plugin.before_agent_callback(agent=aggregator, callback_context=cc("ResultAggregatorAgent"))
        await plugin.on_event_callback(
            invocation_context=invocation,
            event=SimpleNamespace(author="ResultAggregatorAgent", **vars(_final_event("# Report"))))
        await plugin.after_agent_callback(agent=aggregator, callback_context=cc("ResultAggregatorAgent"))
        await plugin.after_agent_callback(agent=wrapper, callback_context=cc("ResearchPipeline"))
        await plugin.after_run_callback(invocation_context=invocation)

        graph = graphs[("u", "s")]
        assert not any("ResearchPipeline" in node_id for node_id in graph.nodes), \
            "the wrapper is a fixture, not a stage"
        goal, plan_stage = "goal:inv-p", "agent:PlanningPipelineAgent@inv-p"
        orch, agg = "agent:OrchestratorAgent@inv-p", "agent:ResultAggregatorAgent@inv-p"
        assert (goal, plan_stage, "caused_by") in graph.edges
        assert (goal, orch, "caused_by") in graph.edges
        assert (goal, agg, "caused_by") in graph.edges
        assert (plan_stage, "agent:PlannerAgent@inv-p", "delegated_to") in graph.edges
        # Every agent's final words are its report, on its own activation.
        assert graph.nodes["agent:PlannerAgent@inv-p"]["output"] == "the plan"
        assert graph.nodes[orch]["output"] == "did the work"
        assert graph.nodes[agg]["output"] == "# Report"
        # The answer is what the request's top level last said.
        assert graph.nodes["result:inv-p"]["output"] == "# Report"
        assert graph.nodes["result:inv-p"]["executor_agent"] == "ResultAggregatorAgent"
        assert graph.nodes[goal]["status"] == "success"
        assert all(graph.nodes[n]["status"] == "success" for n in (plan_stage, orch, agg))

    asyncio.run(scenario())


def test_a_composite_child_run_twice_is_two_stages(monkeypatch):
    """A loop over a sequential composite: each pass of each child is a card."""
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        invocation, _ = _contexts(user_id="u", session_id="s", invocation_id="inv-l")
        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(role="user", parts=[types.Part(text="loop")]))
        cc = lambda name: _callback_context(invocation, name)  # noqa: E731
        critic = _agent("CriticAgent")
        loop = _agent("ReviewLoop", children=(critic,))
        await plugin.before_agent_callback(agent=loop, callback_context=cc("ReviewLoop"))
        for _ in range(2):
            await plugin.before_agent_callback(agent=critic, callback_context=cc("CriticAgent"))
            # A second pass in the same invocation opens a new activation.
            plugin._state(invocation).agent_node.pop("CriticAgent", None)
        graph = graphs[("u", "s")]
        assert ("agent:ReviewLoop@inv-l", "agent:CriticAgent@inv-l", "delegated_to") in graph.edges
        assert ("agent:ReviewLoop@inv-l", "agent:CriticAgent@inv-l#2", "delegated_to") in graph.edges

    asyncio.run(scenario())


def test_the_runs_answer_is_recorded_whole(monkeypatch):
    """It was the most tightly cut thing in the graph, at 600 characters.

    A tool result is allowed twenty thousand; the answer the reader actually
    came for — the report — lost its findings a paragraph in.
    """
    async def scenario():
        graphs = _install_graph_resolver(monkeypatch)
        plugin = GraphMemoryPlugin()
        invocation, _ = _contexts(
            user_id="user-r", session_id="session-r", invocation_id="inv-r",
        )
        await plugin.on_user_message_callback(
            invocation_context=invocation,
            user_message=types.Content(role="user", parts=[types.Part(text="write it up")]),
        )
        answer = "Результаты. " + ("влияние УФ на токсичность фурокумаринов. " * 200)
        await plugin.on_event_callback(
            invocation_context=invocation, event=_final_event(answer),
        )

        graph = graphs[("user-r", "session-r")]
        result = next(n for n in graph.full()["nodes"] if n["kind"] == "result")
        assert len(answer) > 5000, "the fixture has to be longer than the old cap"
        assert result["output"].startswith("Результаты.")
        assert len(result["output"]) > 5000
        # The card's label is still a label, not the whole report.
        assert len(result["label"]) <= 260

    asyncio.run(scenario())
