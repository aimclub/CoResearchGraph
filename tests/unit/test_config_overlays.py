"""``extends:`` — a config variant is an overlay, not a copy.

A copy of system.yaml starts drifting from the original the day it is made, so
a profile inherits and overrides only what it names. The merge has to be
faithful in three ways: per-agent fields merge (flipping ``enabled`` must not
erase the agent's model or prompt), sections the overlay declares win, and a
cycle is an error rather than a hang.
"""

import pytest

from CoScientist.assembly.schema import load_config

_BASE = """
defaults:
  model: main
agents:
  Root:
    class: llm
    description: the root
    root: true
    model: coder
    subordinates: [Helper]
  Helper:
    class: llm
    description: a helper
    prompt: helper_prompt
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_an_overlay_flips_one_field_and_keeps_the_rest(tmp_path):
    base = _write(tmp_path, "base.yaml", _BASE)
    overlay = _write(
        tmp_path, "profile.yaml", f"extends: {base}\nagents:\n  Helper:\n    enabled: false\n"
    )

    config = load_config(overlay)

    assert not config.agent("Helper").is_enabled()
    assert config.agent("Helper").prompt == "helper_prompt"  # not erased
    assert config.agent("Root").model == "coder"  # untouched agents survive


def test_an_overlay_can_add_an_agent(tmp_path):
    base = _write(tmp_path, "base.yaml", _BASE)
    overlay = _write(
        tmp_path,
        "profile.yaml",
        f"extends: {base}\nagents:\n  Extra:\n    class: llm\n    description: extra\n",
    )

    assert load_config(overlay).agent("Extra").description == "extra"


def test_defaults_merge_shallowly(tmp_path):
    base = _write(tmp_path, "base.yaml", _BASE)
    overlay = _write(
        tmp_path, "profile.yaml", f"extends: {base}\ndefaults:\n  model: coder\n"
    )

    assert load_config(overlay).defaults.model == "coder"


def test_an_overlay_keeps_its_own_top_level_section(tmp_path):
    """defaults and agents merge specially; every other section must still pass
    through, or an overlay silently inherits the base's lifecycle stages."""
    base = _write(tmp_path, "base.yaml", _BASE + "pipeline:\n  pre: [Helper]\n")
    overlay = _write(tmp_path, "profile.yaml", f"extends: {base}\npipeline:\n  pre: []\n")

    assert load_config(base).pipeline.pre == ["Helper"]
    assert load_config(overlay).pipeline.pre == []


def test_a_config_without_extends_is_unaffected(tmp_path):
    base = _write(tmp_path, "base.yaml", _BASE)

    assert load_config(base).agent("Root").model == "coder"


def test_an_extends_cycle_is_an_error(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"extends: {b}\nagents: {{}}\n", encoding="utf-8")
    b.write_text(f"extends: {a}\nagents: {{}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        load_config(a)
