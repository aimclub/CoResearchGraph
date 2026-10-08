"""Agent callbacks: tool/reranker, critic, medical, research, and report-language callbacks.

Re-exported here so callers can use `from CoScientist.agents.callbacks import
<name>` regardless of which submodule defines it.
"""
from CoScientist.agents.callbacks.critic import (
    make_plan_critique,
    make_post_action_critique,
    make_pre_action_critique,
)
from CoScientist.agents.callbacks.med_callbacks import (
    before_model_modifier,
    med_agent_before_model,
)
from CoScientist.agents.callbacks.json_output import sanitize_json_output
from CoScientist.agents.callbacks.link_registry import (
    expand_link_refs,
    redact_link_urls,
    register_tool_result_links,
    resolve_link_refs,
    user_links,
)
from CoScientist.agents.callbacks.report_language import inject_report_language
from CoScientist.agents.callbacks.research_callbacks import (
    cleanup_uploaded_papers,
    ensure_local_papers_uploaded,
    papers_agent_before_model,
)
from CoScientist.agents.callbacks.tool_callbacks import (
    after_fullset_reranker_agent,
    after_tool_reranker_agent,
    before_get_task,
    before_tool_reranker_model,
    shortlist_reranker_tools,
    capture_mcp_artifacts,
    inject_dataset_context,
    inject_fedot_candidates,
    inject_graph_root,
    make_plan_registration_guard,
    make_unknown_tool_guard,
    inject_original_query,
    print_research_agent_tool_call,
    redirect_when_no_tools,
)

__all__ = [
    "make_pre_action_critique",
    "make_post_action_critique",
    "make_plan_critique",
    "before_model_modifier",
    "med_agent_before_model",
    "papers_agent_before_model",
    "ensure_local_papers_uploaded",
    "cleanup_uploaded_papers",
    "before_tool_reranker_model",
    "shortlist_reranker_tools",
    "after_tool_reranker_agent",
    "after_fullset_reranker_agent",
    "print_research_agent_tool_call",
    "capture_mcp_artifacts",
    "redirect_when_no_tools",
    "make_unknown_tool_guard",
    "make_plan_registration_guard",
    "inject_original_query",
    "before_get_task",
    "inject_graph_root",
    "inject_dataset_context",
    "inject_fedot_candidates",
    "inject_report_language",
    "sanitize_json_output",
    "user_links",
    "redact_link_urls",
    "resolve_link_refs",
    "register_tool_result_links",
    "expand_link_refs",
]
