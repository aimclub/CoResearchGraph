"""Agent configs must name models by alias, not by provider.

``model: main`` / ``model: coder`` resolve through ``LLM__MAIN_MODEL`` /
``LLM__CODER_MODEL``, which is where the deployment's provider is configured —
OpenRouter here. A literal litellm string in a profile (``openai/<id>``,
``anthropic/<id>``, …) bypasses that entirely and calls the named provider
directly with whatever key litellm happens to hold.

experiments.yaml pinned ``openai/gemini-3.7-flash`` on five agents, so the whole
profile called OpenAI no matter what .env said, and every run died on

    litellm.APIError: OpenAIException - Country, region, or territory not supported

which reads like a network problem and is a one-word config mistake.

A literal is still allowed for a model a profile genuinely has to pin — it just
has to carry the provider the deployment actually reaches, so this asks for the
alias or for an explicit exception below.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from CoScientist.assembly.schema import profile_paths

# The configs in CoScientist/agents/ and the case packages' own profiles.
PROFILES = profile_paths()

#: The aliases the assembler resolves through settings (assembly/assembler.py::
#: _resolve_model). Anything else is a literal passed straight to litellm.
#: ``nir`` resolves through ``LLM__NIR_MODEL`` and falls back to the main model
#: when unset, so it obeys the same rule: the deployment picks the provider.
ALIASES = {"main", "coder", "nir"}

#: Literals a profile is allowed to pin, with the reason. Empty on purpose: add
#: an entry only for a model the deployment can actually reach.
ALLOWED_LITERALS: dict[str, str] = {}

MODEL_LINE = re.compile(r"^\s*model:\s*(\S+)\s*$", re.M)


def _model_refs(profile: Path) -> dict[str, str]:
    """agent name -> model ref, for every agent in the profile."""
    raw = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
    refs = {}
    default = (raw.get("defaults") or {}).get("model")
    if default:
        refs["defaults"] = str(default)
    for name, cfg in (raw.get("agents") or {}).items():
        if isinstance(cfg, dict) and cfg.get("model"):
            refs[name] = str(cfg["model"])
    return refs


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_models_are_named_by_alias_not_by_provider(profile: Path):
    offenders = {
        name: ref
        for name, ref in _model_refs(profile).items()
        if ref not in ALIASES and ref not in ALLOWED_LITERALS
    }

    assert not offenders, (
        f"{profile.name} pins provider-specific models: {offenders}. "
        "Use `main`/`coder` so LLM__MAIN_MODEL / LLM__CODER_MODEL decide the "
        "provider, or add the literal to ALLOWED_LITERALS with a reason."
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_no_model_line_escapes_the_parsed_check(profile: Path):
    """A commented-out or oddly-nested `model:` would slip past the YAML walk
    above; count them so the check cannot quietly stop covering the file."""
    in_yaml = set(_model_refs(profile).values())
    in_text = {
        m for m in MODEL_LINE.findall(profile.read_text(encoding="utf-8"))
        if not m.startswith("[")  # `before_model: [...]` and friends
    }

    assert in_text <= in_yaml | ALIASES, (
        f"{profile.name} has a `model:` line the config walk does not see: "
        f"{sorted(in_text - (in_yaml | ALIASES))}"
    )
