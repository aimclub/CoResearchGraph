"""setup.sh is a transcript of what worked during the live build. Replayed cold
on a clean image, two habits of that live session break it, so the commands are
normalised as they are written rather than patched by whoever replays them.
"""

from _codegen_loader import load_codegen

cg = load_codegen()
portable_command = cg.portable_command
render_setup_sh = cg.render_setup_sh


def test_uv_venv_is_seeded_so_a_later_pip_call_resolves():
    """`uv venv` installs no pip, but transcripts go on to call .venv/bin/pip."""
    assert portable_command("uv venv .venv") == "uv venv --seed .venv"


def test_an_already_seeded_venv_is_left_alone():
    assert portable_command("uv venv --seed .venv") == "uv venv --seed .venv"


def test_a_relative_cd_is_anchored_to_the_work_dir():
    """Otherwise it resolves against wherever an earlier cd left us."""
    assert portable_command("cd .alembic/repo/repos") == "cd /work/.alembic/repo/repos"


def test_a_chained_cd_is_anchored_too():
    assert (
        portable_command("source .venv/bin/activate && cd .alembic/repo")
        == "source .venv/bin/activate && cd /work/.alembic/repo"
    )


def test_an_absolute_path_is_not_touched():
    assert portable_command("cd /work/.alembic/repo") == "cd /work/.alembic/repo"


def test_ordinary_commands_pass_through():
    assert portable_command("pip install -e .") == "pip install -e ."


def test_the_rendered_script_carries_the_fixes():
    script = render_setup_sh(["uv venv .venv", "cd .alembic/repo && pip install -e ."])

    assert "uv venv --seed .venv" in script
    assert "cd /work/.alembic/repo" in script
    assert script.startswith("#!/usr/bin/env bash")


def test_an_empty_transcript_still_renders():
    assert "no environment commands" in render_setup_sh([])
