"""Task Tracker infrastructure and tools for managing and tracking tasks."""
from google.adk.tools.tool_context import ToolContext
from typing import Any, Dict, List, Optional
from datetime import datetime

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

class TaskTrackerToolset(BaseToolset):
    """Stateless task tools backed by the current ADK session state.

    The toolset itself is shared by all agents, but task data is not.  Keeping
    ``active_tasks`` in ``ToolContext.state`` makes the ADK session the single
    source of truth and prevents plans from leaking between users/sessions.
    """

    def __init__(self, prefix: str = None):
        super().__init__(tool_name_prefix=prefix)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        return [
            #FunctionTool(self.create_task),
            FunctionTool(self.update_task_status),
            FunctionTool(self.get_active_tasks),
        ]

    async def close(self) -> None:
        pass

    def create_plan(self, tasks: List[Dict[str, Any]], tool_context: ToolContext) -> Dict[str, Any]:

        """Replace ALL tasks with a new plan provided by the planner agent.

        The stored plan is EXECUTED IN LIST ORDER, so the tasks are linearised
        here: every task is placed after the tasks it depends on, whatever order
        the arguments arrive in.

        Each task in the list should have:
          - title (str)
          - description (str)
          - assignee (str)
          - id (str, optional): id other tasks can reference. When omitted, the
            n-th task in THIS call can be referenced as "TASK-<n>".
          - parent_id (str or None, optional): id of the task that must be
            finished before this one starts. Use it for every task that consumes
            another task's result; leave it null for independent tasks. A task
            may never reference itself.
          - notes (str, optional)

        Ids are renumbered TASK-1..TASK-N in final execution order, and the
        returned "plan" lists that order — check it before finishing your turn.
        """
        if not isinstance(tasks, list):
            return {"result": "error", "message": "'tasks' must be a list of task definitions."}

        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return {"result": "error", "message": f"Task at index {i} must be a dictionary."}
            if "title" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'title'."}
            if "description" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'description'."}
            if "assignee" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'assignee'."}

        warnings: List[str] = []

        # References are resolved against the ids the planner could see when it
        # wrote the call (explicit "id", or "TASK-<n>" by argument position).
        # Final ids are handed out only after ordering, so renumbering can never
        # make a task steal an id another task was pointing at.
        source_ids = self._source_ids(tasks)

        # source id -> id of the task that absorbed it (None once it is gone):
        # dependents of a dropped/merged task follow the chain to a real task.
        redirect: Dict[str, Optional[str]] = {}
        by_source_id: Dict[str, Dict[str, Any]] = {}
        coder_source_id = None

        for sid, t in zip(source_ids, tasks):
            if t.get("assignee") == "OrchestratorAgent":
                # The orchestrator reports after the plan runs; its dependents
                # inherit whatever this task itself was waiting for.
                redirect[sid] = self._first_dependency(t)
                continue

            if self._should_merge_tasks() and t.get("assignee") in ("CoderAgent", "TaskExecutorAgent") and coder_source_id is not None:
                coder_task = by_source_id[coder_source_id]
                coder_task["title"] += f" - {t.get('title')}"
                coder_task["description"] += f" - {t.get('description')}"
                coder_task["_deps"].update(self._dependencies(t))

                current_note = t.get("notes", "")
                if current_note:
                    if coder_task["notes"]:
                        coder_task["notes"] += f" - {current_note}"
                    else:
                        coder_task["notes"] = current_note

                redirect[sid] = coder_source_id
                continue

            task = {
                "title": t.get("title"),
                "description": t.get("description"),
                "assignee": t.get("assignee"),
                "notes": t.get("notes", ""),
                "_deps": self._dependencies(t),
            }
            by_source_id[sid] = task
            if t.get("assignee") in ("CoderAgent", "TaskExecutorAgent"):
                coder_source_id = sid

        if not by_source_id:
            return {"result": "error", "message": "The plan contains no executable tasks."}

        # Planner order is the tie-breaker between tasks that are equally ready,
        # so a plan without dependencies keeps the order it was written in.
        position = {sid: i for i, sid in enumerate(by_source_id)}
        deps: Dict[str, set] = {}
        for sid, task in by_source_id.items():
            resolved = set()
            for ref in task.pop("_deps"):
                target = self._resolve(ref, redirect)
                if target is None:
                    continue
                if target == sid:
                    warnings.append(
                        f"Task '{task['title']}' depended on itself — link dropped."
                    )
                elif target in by_source_id:
                    resolved.add(target)
                else:
                    warnings.append(
                        f"Unknown parent_id '{ref}' on task '{task['title']}' — link dropped."
                    )
            deps[sid] = resolved

        ordered_ids = self._topological_order(deps, position, warnings)

        final_ids = {sid: f"TASK-{i + 1}" for i, sid in enumerate(ordered_ids)}
        new_tasks = []
        for sid in ordered_ids:
            task = by_source_id[sid]
            # A single predecessor is stored, and it is the LAST dependency to
            # run: that is the task this one actually waits on.
            parents = sorted(deps[sid], key=ordered_ids.index)
            new_tasks.append({
                "id": final_ids[sid],
                "title": task["title"],
                "description": task["description"],
                "assignee": task["assignee"],
                "status": "TODO",
                "parent_id": final_ids[parents[-1]] if parents else None,
                "notes": task["notes"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            })

        tool_context.state["_master_active_tasks"] = new_tasks
        current_agent = getattr(tool_context, "agent_name", None)
        tool_context.state["active_tasks"] = clean_tasks_for_agent(new_tasks, current_agent)

        result = {
            "result": "success",
            "message": f"Plan created with {len(new_tasks)} tasks, ordered for execution.",
            "plan": [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "description": t.get("description", ""),
                    "notes": t.get("notes", ""),
                    "assignee": t["assignee"],
                    "parent_id": t["parent_id"],
                }
                for t in new_tasks
            ],
        }
        if warnings:
            result["warnings"] = warnings
        return result

    @staticmethod
    def _source_ids(tasks: List[Dict[str, Any]]) -> List[str]:
        """Id each incoming task by what the planner may have referenced.

        Explicit ids win; the rest answer to their positional "TASK-<n>" name,
        which is what a planner that omitted ids means by that reference.
        """
        explicit = {
            str(t.get("id")) for t in tasks if t.get("id")
        }
        source_ids = []
        taken = set()
        for i, t in enumerate(tasks):
            sid = str(t.get("id")) if t.get("id") else f"TASK-{i + 1}"
            if sid in taken or (not t.get("id") and sid in explicit):
                # Collision: this task keeps a private id nothing can reference.
                sid = f"#{i}"
            source_ids.append(sid)
            taken.add(sid)
        return source_ids

    @staticmethod
    def _should_merge_tasks() -> bool:
        """Check whether consecutive executor tasks should be merged."""
        try:
            from CoScientist.config import get_settings
            return get_settings().web.merge_tasks_enabled
        except Exception:
            return True  # safe default — merge as before

    @staticmethod
    def _dependencies(task: Dict[str, Any]) -> set:
        """Ids a task declares as prerequisites (`parent_id` or `depends_on`)."""
        refs = set()
        for key in ("parent_id", "depends_on"):
            value = task.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
            elif isinstance(value, (list, tuple, set)):
                refs.update(str(v) for v in value if v)
        return refs

    @classmethod
    def _first_dependency(cls, task: Dict[str, Any]) -> Optional[str]:
        refs = sorted(cls._dependencies(task))
        return refs[0] if refs else None

    @staticmethod
    def _resolve(ref: str, redirect: Dict[str, Optional[str]]) -> Optional[str]:
        """Follow a reference through merged/dropped tasks to a surviving one."""
        seen = set()
        while ref in redirect and ref not in seen:
            seen.add(ref)
            ref = redirect[ref]
            if ref is None:
                return None
        return ref

    @staticmethod
    def _topological_order(deps: Dict[str, set], position: Dict[str, int],
                           warnings: List[str]) -> List[str]:
        """Sort so every task follows its prerequisites, ties by planner order."""
        remaining = {sid: set(d) for sid, d in deps.items()}
        ordered: List[str] = []
        done: set = set()

        while remaining:
            ready = sorted(
                (sid for sid, d in remaining.items() if not d - done),
                key=lambda sid: position[sid],
            )
            if not ready:
                # A dependency cycle: cut it at the earliest task the planner
                # wrote, rather than refusing the whole plan.
                stuck = min(remaining, key=lambda sid: position[sid])
                warnings.append(
                    f"Circular dependency around '{stuck}' — its links were dropped."
                )
                remaining[stuck] = set()
                continue
            for sid in ready:
                ordered.append(sid)
                done.add(sid)
                del remaining[sid]

        return ordered


    def create_task(
        self,
        title: str,
        description: str,
        tool_context: ToolContext,
        assignee: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new task in the task tracker.
        
        Args:
            title: The title of the task.
            description: Detailed description of what needs to be done.
            assignee: The agent or sub-system responsible for this task.
            
        Returns:
            A dictionary with the task ID and current state.
        """
        master_tasks = list(
            tool_context.state.get("_master_active_tasks")
            or tool_context.state.get("active_tasks", [])
        )
        task_id = f"TASK-{len(master_tasks) + 1}"
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "status": "TODO",
            "assignee": assignee or "unassigned",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "notes": "",
            "parent_id": None
        }
        master_tasks.append(task)
        tool_context.state["_master_active_tasks"] = master_tasks
        current_agent = getattr(tool_context, "agent_name", None)
        tool_context.state["active_tasks"] = clean_tasks_for_agent(master_tasks, current_agent)
        return {"result": "success", "task": task}

    def update_task_status(
        self,
        task_id: str,
        status: str,
        tool_context: ToolContext,
        notes: str = None
    ) -> Dict[str, Any]:

        """Update the status of a specific task.
        Args:
            task_id: The ID of the task to update (e.g. TASK-1).
            status: The new status (e.g., TODO, IN_PROGRESS, DONE).
            tool_context: ToolContext provided by ADK.
            notes: Optional notes or updates about the task.
        Returns:
            A dictionary indicating success or failure.
        """
        master_tasks = list(
            tool_context.state.get("_master_active_tasks")
            or tool_context.state.get("active_tasks", [])
        )
        found_task = None
        for task in master_tasks:
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                if notes:
                    task["notes"] = task.get("notes", "") + (
                        f"\n[{datetime.now().isoformat()}] {notes}"
                    )
                found_task = task
                break

        if found_task:
            tool_context.state["_master_active_tasks"] = master_tasks
            current_agent = getattr(tool_context, "agent_name", None)
            tool_context.state["active_tasks"] = clean_tasks_for_agent(master_tasks, current_agent)
            return {"result": "success", "task": found_task}

        return {"result": "error", "message": f"Task {task_id} not found."}

    def get_active_tasks(self, tool_context: ToolContext) -> Dict[str, Any]:
        """Get a list of all tracked tasks.
            Returns:
                A dictionary containing the matching tasks.
        """
        master_tasks = (
            tool_context.state.get("_master_active_tasks")
            or tool_context.state.get("active_tasks", [])
        )
        current_agent = getattr(tool_context, "agent_name", None)
        cleaned_tasks = clean_tasks_for_agent(master_tasks, current_agent)
        return {"tasks": cleaned_tasks}


def clean_tasks_for_agent(
    tasks: List[Dict[str, Any]], current_agent: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Clean and filter tasks based on the accessing agent's role.

    - Internal timestamps (created_at, updated_at) are omitted.
    - Empty notes are omitted.
    - OrchestratorAgent, PlannerAgent, or unspecified agents (None) retain full task descriptions.
    - Worker agents retain full descriptions and notes ONLY for tasks assigned to current_agent.
    """
    if not isinstance(tasks, list):
        return []

    cleaned_tasks = []
    is_management = current_agent in ("OrchestratorAgent", "PlannerAgent", None)

    for task in tasks:
        if not isinstance(task, dict):
            continue
        cleaned = {k: v for k, v in task.items() if k not in ("created_at", "updated_at")}

        if not cleaned.get("notes"):
            cleaned.pop("notes", None)

        is_assigned = task.get("assignee") == current_agent

        if not is_management and not is_assigned:
            cleaned.pop("description", None)
            cleaned.pop("notes", None)

        cleaned_tasks.append(cleaned)

    return cleaned_tasks

# A global tool definition is safe: all mutable data lives in ToolContext.state.
task_tracker_instance = TaskTrackerToolset()

def get_task_tracker_tools() -> list:
    tools = [
        #FunctionTool(task_tracker_instance.create_task),
        FunctionTool(task_tracker_instance.update_task_status),
        FunctionTool(task_tracker_instance.get_active_tasks)
    ]
    return tools

def create_plan_tool():
    return FunctionTool(task_tracker_instance.create_plan)
