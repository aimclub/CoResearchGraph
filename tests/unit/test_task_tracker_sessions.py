from types import SimpleNamespace

from CoScientist.agents.callbacks.tool_callbacks import before_get_task
from CoScientist.tools.task_tracker import TaskTrackerToolset


def _context(agent_name="OrchestratorAgent"):
    return SimpleNamespace(state={}, agent_name=agent_name)


def _plan(title, assignee="ResearchAgent"):
    return [{
        "title": title,
        "description": f"Investigate {title}",
        "assignee": assignee,
    }]


def test_task_plans_are_isolated_by_context():
    tracker = TaskTrackerToolset()
    first = _context()
    second = _context()

    tracker.create_plan(_plan("alpha"), first)
    tracker.create_plan(_plan("beta"), second)

    assert first.state["active_tasks"][0]["title"] == "alpha"
    assert second.state["active_tasks"][0]["title"] == "beta"

    tracker.update_task_status("TASK-1", "DONE", first, notes="complete")
    assert first.state["active_tasks"][0]["status"] == "DONE"
    assert second.state["active_tasks"][0]["status"] == "TODO"


def test_get_active_tasks_reads_only_current_session_state():
    tracker = TaskTrackerToolset()
    context = _context("ResearchAgent")
    tracker.create_plan(_plan("session-only"), context)

    result = tracker.get_active_tasks(context)

    assert [task["title"] for task in result["tasks"]] == ["session-only"]
    assert result["tasks"][0]["description"] == "Investigate session-only"


def test_coder_task_stays_behind_the_task_it_depends_on():
    """The planner's own order and parent_id survive the CoderAgent merge."""
    tracker = TaskTrackerToolset()
    context = _context()

    tracker.create_plan([
        {"title": "Evaluate compatibility", "description": "Check labels",
         "assignee": "HypothesesAgent"},
        {"title": "Train model", "description": "Train detector",
         "assignee": "CoderAgent", "parent_id": "TASK-1"},
    ], context)

    tasks = context.state["active_tasks"]
    assert [t["title"] for t in tasks] == ["Evaluate compatibility", "Train model"]
    assert [t["id"] for t in tasks] == ["TASK-1", "TASK-2"]
    assert tasks[1]["parent_id"] == "TASK-1"


def test_shuffled_tasks_are_linearised_by_their_dependencies():
    tracker = TaskTrackerToolset()
    context = _context()

    result = tracker.create_plan([
        {"id": "B", "title": "second", "description": "d", "parent_id": "A",
         "assignee": "ResearchAgent"},
        {"id": "C", "title": "third", "description": "d", "parent_id": "B",
         "assignee": "HypothesesAgent"},
        {"id": "A", "title": "first", "description": "d",
         "assignee": "ResearchAgent"},
    ], context)

    tasks = context.state["active_tasks"]
    assert [t["title"] for t in tasks] == ["first", "second", "third"]
    assert [t["id"] for t in tasks] == ["TASK-1", "TASK-2", "TASK-3"]
    assert [t["parent_id"] for t in tasks] == [None, "TASK-1", "TASK-2"]
    assert [t["id"] for t in result["plan"]] == ["TASK-1", "TASK-2", "TASK-3"]


def test_independent_tasks_keep_the_order_they_were_registered_in():
    tracker = TaskTrackerToolset()
    context = _context()

    tracker.create_plan([
        {"title": "LIT-01", "description": "d", "assignee": "ResearchAgent"},
        {"title": "LIT-02", "description": "d", "assignee": "ResearchAgent"},
        {"title": "LIT-03", "description": "d", "assignee": "ResearchAgent"},
    ], context)

    assert [t["title"] for t in context.state["active_tasks"]] == [
        "LIT-01", "LIT-02", "LIT-03"
    ]


def test_merged_coder_task_inherits_every_dependency(monkeypatch):
    from CoScientist.config import get_settings
    monkeypatch.setattr(get_settings().web, "merge_tasks_enabled", True)
    tracker = TaskTrackerToolset()
    context = _context()

    tracker.create_plan([
        {"id": "T1", "title": "code A", "description": "a",
         "assignee": "CoderAgent"},
        {"id": "T2", "title": "research", "description": "r",
         "assignee": "ResearchAgent"},
        {"id": "T3", "title": "code B", "description": "b",
         "assignee": "CoderAgent", "parent_id": "T2"},
    ], context)

    tasks = context.state["active_tasks"]
    # The merged coder task now waits for the research it absorbed a link to.
    assert [t["title"] for t in tasks] == ["research", "code A - code B"]
    assert tasks[1]["parent_id"] == "TASK-1"
    assert tasks[1]["description"] == "a - b"


def test_dependency_on_a_dropped_orchestrator_task_is_rewired():
    tracker = TaskTrackerToolset()
    context = _context()

    tracker.create_plan([
        {"id": "T1", "title": "report", "description": "d", "parent_id": "T3",
         "assignee": "OrchestratorAgent"},
        {"id": "T2", "title": "publish", "description": "d", "parent_id": "T1",
         "assignee": "ResearchAgent"},
        {"id": "T3", "title": "research", "description": "d",
         "assignee": "ResearchAgent"},
    ], context)

    tasks = context.state["active_tasks"]
    assert [t["title"] for t in tasks] == ["research", "publish"]
    assert tasks[1]["parent_id"] == "TASK-1"


def test_self_reference_and_cycles_do_not_break_the_plan():
    tracker = TaskTrackerToolset()
    context = _context()

    self_ref = tracker.create_plan([
        {"id": "T1", "title": "alone", "description": "d", "parent_id": "T1",
         "assignee": "ResearchAgent"},
    ], context)
    assert context.state["active_tasks"][0]["parent_id"] is None
    assert self_ref["warnings"]

    cycle = tracker.create_plan([
        {"id": "T1", "title": "a", "description": "d", "parent_id": "T2",
         "assignee": "ResearchAgent"},
        {"id": "T2", "title": "b", "description": "d", "parent_id": "T1",
         "assignee": "ResearchAgent"},
    ], context)
    assert [t["title"] for t in context.state["active_tasks"]] == ["a", "b"]
    assert cycle["warnings"]


def test_before_get_task_initializes_list_without_overwriting_plan():
    empty = _context()
    before_get_task(empty)
    assert empty.state["active_tasks"] == []

    existing = _context()
    tasks = [{"id": "TASK-1", "title": "keep me", "assignee": "OrchestratorAgent"}]
    existing.state["active_tasks"] = tasks
    before_get_task(existing)
    assert existing.state["active_tasks"] == [{"id": "TASK-1", "title": "keep me", "assignee": "OrchestratorAgent"}]


def test_task_context_omits_descriptions_for_other_assignees():
    from CoScientist.tools.task_tracker import clean_tasks_for_agent

    tasks = [
        {
            "id": "TASK-1",
            "title": "Public Dataset Compatibility",
            "description": "Long detailed description for HypothesesAgent",
            "assignee": "HypothesesAgent",
            "status": "TODO",
            "parent_id": None,
            "notes": "",
            "created_at": "2026-07-31T16:00:00",
            "updated_at": "2026-07-31T16:00:00",
        },
        {
            "id": "TASK-2",
            "title": "Train YOLO Model",
            "description": "Long detailed description for TaskExecutorAgent",
            "assignee": "TaskExecutorAgent",
            "status": "TODO",
            "parent_id": "TASK-1",
            "notes": "",
            "created_at": "2026-07-31T16:00:00",
            "updated_at": "2026-07-31T16:00:00",
        },
    ]

    # For TaskExecutorAgent: TASK-1 has NO description; TASK-2 HAS description
    for_executor = clean_tasks_for_agent(tasks, "TaskExecutorAgent")
    assert "description" not in for_executor[0]
    assert "created_at" not in for_executor[0]
    assert "updated_at" not in for_executor[0]
    assert "notes" not in for_executor[0]
    assert for_executor[1]["description"] == "Long detailed description for TaskExecutorAgent"

    # For HypothesesAgent: TASK-1 HAS description; TASK-2 has NO description
    for_hypotheses = clean_tasks_for_agent(tasks, "HypothesesAgent")
    assert for_hypotheses[0]["description"] == "Long detailed description for HypothesesAgent"
    assert "description" not in for_hypotheses[1]

    # For OrchestratorAgent: BOTH have descriptions
    for_orchestrator = clean_tasks_for_agent(tasks, "OrchestratorAgent")
    assert for_orchestrator[0]["description"] == "Long detailed description for HypothesesAgent"
    assert for_orchestrator[1]["description"] == "Long detailed description for TaskExecutorAgent"


def test_before_get_task_filters_active_tasks_per_agent():
    tracker = TaskTrackerToolset()
    ctx = _context("OrchestratorAgent")

    tracker.create_plan([
        {"title": "Task A", "description": "Desc A", "assignee": "HypothesesAgent"},
        {"title": "Task B", "description": "Desc B", "assignee": "TaskExecutorAgent", "parent_id": "TASK-1"},
    ], ctx)

    # Invoke before_get_task for HypothesesAgent
    hypo_ctx = _context("HypothesesAgent")
    hypo_ctx.state = ctx.state
    before_get_task(hypo_ctx)

    assert hypo_ctx.state["active_tasks"][0]["description"] == "Desc A"
    assert "description" not in hypo_ctx.state["active_tasks"][1]

    # Invoke before_get_task for TaskExecutorAgent: master descriptions preserved!
    exec_ctx = _context("TaskExecutorAgent")
    exec_ctx.state = ctx.state
    before_get_task(exec_ctx)

    assert "description" not in exec_ctx.state["active_tasks"][0]
    assert exec_ctx.state["active_tasks"][1]["description"] == "Desc B"


def test_create_plan_and_create_task_preserves_description_and_notes():
    tracker = TaskTrackerToolset()
    ctx = _context("PlannerAgent")

    res = tracker.create_plan([
        {"title": "Task 1", "description": "Full details for task 1", "notes": "Initial note", "assignee": "HypothesesAgent"},
        {"title": "Task 2", "description": "Full details for task 2", "assignee": "TaskExecutorAgent", "parent_id": "TASK-1"},
    ], ctx)

    # create_plan response must include description and notes for web UI / callers
    assert res["plan"][0]["description"] == "Full details for task 1"
    assert res["plan"][0]["notes"] == "Initial note"
    assert res["plan"][1]["description"] == "Full details for task 2"

    # _master_active_tasks must retain full description and notes
    master = ctx.state.get("_master_active_tasks")
    assert master is not None
    assert master[0]["description"] == "Full details for task 1"
    assert master[0]["notes"] == "Initial note"

    # create_task should also populate _master_active_tasks
    t_res = tracker.create_task("Task 3", "Description 3", ctx, assignee="HypothesesAgent")
    assert t_res["result"] == "success"
    master_after = ctx.state.get("_master_active_tasks")
    assert len(master_after) == 3
    assert master_after[2]["description"] == "Description 3"


