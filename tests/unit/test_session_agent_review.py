from CoScientist.hitl.session_agent import render_task_plan


def test_render_task_plan_contains_only_authoritative_registered_tasks():
    tasks = [{
        "id": "TASK-1",
        "title": "Generate both inhibitor sets",
        "description": "Use generate_mols and rank each target separately.",
        "assignee": "TaskExecutorAgent",
        "status": "TODO",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }]

    review = render_task_plan(tasks)

    assert "Generate both inhibitor sets" in review
    assert "TaskExecutorAgent" in review
    assert "generate_mols" in review
    assert "created_at" not in review
    assert "status" not in review
