"""The FEDOT.MAS compatibility shim has to survive an upstream signature change.

Twice now a drifted signature cost a whole batch run: upstream added a
keyword-only ``autonomous`` to ``build`` and the routing builders, ``BaseMAS.
build_app`` started calling ``self.build(config, autonomous=...)``, and the
override raised TypeError on every FEDOT task. uv.lock still pins the commit
without that parameter, so the shim must work in BOTH directions.
"""
from __future__ import annotations

import inspect

from CoScientist.tools.fedot_mas_patch import (
    PatchedMAS,
    PatchedMAW,
    _autonomous_kwargs,
)


def _new_style(config, *, autonomous: bool = True):
    return (config, autonomous)


def _old_style(config, registry=None, worker_models=None):
    return config


def test_the_keyword_is_passed_only_to_callees_that_accept_it():
    assert _autonomous_kwargs(_new_style, True) == {"autonomous": True}
    assert _autonomous_kwargs(_new_style, False) == {"autonomous": False}
    assert _autonomous_kwargs(_old_style, True) == {}


def test_a_non_introspectable_callee_does_not_raise():
    """C callables have no signature; the shim must degrade, not explode."""
    assert _autonomous_kwargs(len, True) == {}


def test_both_overrides_accept_what_build_app_passes():
    """BaseMAS.build_app calls self.build(config, autonomous=...)."""
    for cls in (PatchedMAS, PatchedMAW):
        params = inspect.signature(cls.build).parameters
        assert "autonomous" in params, f"{cls.__name__}.build lost the keyword"
        assert params["autonomous"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["autonomous"].default is True


def test_the_overrides_still_match_the_installed_base_class():
    """A drifted base signature is the failure this module exists to absorb."""
    from fedotmas import MAS, MAW

    for patched, base in ((PatchedMAS, MAS), (PatchedMAW, MAW)):
        base_params = set(inspect.signature(base.build).parameters)
        patched_params = set(inspect.signature(patched.build).parameters)
        missing = base_params - patched_params
        assert not missing, f"{patched.__name__}.build does not accept {sorted(missing)}"
