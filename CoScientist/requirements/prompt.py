"""Production prompt and schema for statement parsing inside context init.

The model fills this schema. ``normalize_statement`` then checks quotes and
kinds. Callers must not replace the model output with a hand-written parse
when they are evaluating the prompt.
"""
from __future__ import annotations

STATEMENT_PROMPT = """\
Split the user request into the parts that are actually asked for.
Return a statement draft with parts, conditions, volume, clarification and planning_notes.

Kinds, and only these kinds:
- question: an unknown that needs an answer (a number, an explanation, a comparison).
- hypothesis: a claim the user asked to test, or hypotheses the user asked you only to formulate.
- deliverable: a thing the user asked to receive (file, candidate set, protocol, sample).

Rules:
- Do not invent a hypothesis so that a pipeline can start. Zero hypotheses is valid.
- A property of a result ("binds to X", "novel", "in CSV") stays on that result.
  It is a hypothesis only when the user asked to check a claim.
- Do not turn an input, a target, a method step, a tool restriction, a budget/cap,
  or a format instruction into its own part. Instructions about how to execute
  the research are not claims to test. Keep them as conditions or criteria on
  the outcomes they constrain; never invent a hypothesis about obeying them.
- Do not turn "if N > 30" into the fact that N > 30. Put conditions only in the quote, not as new parts.
- Put conditional instructions in conditions=[{"text": "...", "quote": "verbatim complete clause"}].
- Set hypothesis_mode="check" or "formulate" explicitly for every hypothesis.
  Other kinds must have hypothesis_mode=null. Do not infer a check from the word hypothesis.
  Checking a claim is complete with a grounded confirmation OR refutation. Its criteria
  constrain the quality or format of the check, never require that the claim be true.
- Set physical_sample=true only for an ordered physical result, protocol_only=true
  only for an ordered protocol document. Both flags default to false and cannot both be true.
  protocol_only means the document itself is sufficient: it is false when the requested
  protocol must contain validated or optimal conditions. Preserve these requested
  properties as explicit criteria; a proposed procedure does not establish them.
- Put counts in volume: requested, run_cap, input_count (integers or null).
  Each count needs its own verbatim requested_quote, run_cap_quote or input_count_quote
  containing the number. Set limit_scope for the run cap. A conditional threshold
  is never a requested count; input size is never an output count. Leave produced null.
  volume describes the whole requested result, not the quantity of one substep in a mixed request.
  Keep substep quantities in that part's criteria. The numeric volume fields require a source
  quote containing the literal digits; quantities written in words stay verbatim in criteria,
  with threshold=null. Do not invent numeric spellings or turn a conditional cap into a request.
- Attach answer format instructions as criteria on the answer they constrain.
- For each deliverable declare its acceptance in criteria, including the properties
  the result must satisfy. Give each criterion a stable id within its part. A filename or execution success alone is not acceptance.
- Every user part needs source="user_request" and a quote copied verbatim from the request.
- Preserve the user's language and the direction of the question; do not turn a comparison into an assumption that one option is better.
- Criteria with origin="agent_method" or "technical" are implementation choices, never user obligations.
- A number that is not in that quote is not a user threshold. Leave it out.
- Do not mark source as accepted. You cannot accept your own proposal.
- clarification is null when you can identify the requested outcome. If a requested part
  cannot be identified, provide clarification={"question": "specific question to the user",
  "quote": "verbatim ambiguous fragment"} and parts=[]. These are mutually exclusive
  responses: a complete requested catalog, or a question needed before that catalog can be built.
- Put unspecified methods, quantities, ranking rules, reagents and thresholds in planning_notes.
  These are choices for the plan, not reasons for clarification. A selection based on future
  measurements is a dependency, not an ambiguous request. Do not demand the answer before
  the research starts. Do not invent numbers to fill empty volume fields: use null.
- Example: "Find materials resistant to heat" identifies an outcome; clarification=null.
  The method of measuring resistance and the number of materials belong to planning_notes.
  "Choose the best group after measuring it" is a dependent step; an explicit ranking
  method belongs in the plan. "Do that instead" without a referent needs clarification.
- Do not create a part per noun.

Schema of one part:
{"id": "", "kind": "question|hypothesis|deliverable", "formulation": "",
 "role": "user_obligation", "source": "user_request", "quote": "",
 "hypothesis_mode": null, "physical_sample": false, "protocol_only": false,
 "criteria": [{"id": "", "text": "", "origin": "user|agent_method|technical", "quote": "", "threshold": null}]}
"""


def statement_schema() -> dict:
    from CoScientist.requirements.models import StatementDraft

    return StatementDraft.model_json_schema()


__all__ = ["STATEMENT_PROMPT", "statement_schema"]
