"""Task inputs are linked into the tool's working tree, not copied — and the
tree has to stay writable.

Inputs are bind-mounted read-only and can run to tens of gigabytes, so they are
referenced by symlink. But a bare directory symlink sends every write beside
the inputs onto the read-only volume, and tools routinely write scratch files
there. Directories are therefore recreated for real and only the files linked.
"""

from pathlib import Path

from CoScientist.alembic.staging import (
    link_writable_tree,
    stage_task_inputs,
    task_mounts,
)


def _task(**mounts):
    return {"example": {"mount": dict(mounts)}}


def test_a_directory_input_can_be_written_into(tmp_path):
    data, work = tmp_path / "data", tmp_path / "input"
    (data / "slides").mkdir(parents=True)
    (data / "slides" / "a.svs").write_text("wsi", encoding="utf-8")

    stage_task_inputs([_task(slides="slides")], data, work)

    staged = work / "slides"
    assert staged.is_dir() and not staged.is_symlink()  # a real, writable dir
    assert (staged / "a.svs").is_symlink()  # the bulk data is not copied
    (staged / "scratch.tmp").write_text("ok", encoding="utf-8")  # would fail on ro


def test_nested_directories_are_mirrored(tmp_path):
    data, work = tmp_path / "data", tmp_path / "input"
    (data / "ds" / "train").mkdir(parents=True)
    (data / "ds" / "train" / "x.npy").write_bytes(b"0")

    stage_task_inputs([_task(ds="ds")], data, work)

    assert (work / "ds" / "train").is_dir()
    assert (work / "ds" / "train" / "x.npy").is_symlink()


def test_a_single_file_is_linked(tmp_path):
    data, work = tmp_path / "data", tmp_path / "input"
    data.mkdir()
    (data / "ct.nii").write_text("scan", encoding="utf-8")

    stage_task_inputs([_task(**{"ct.nii": "ct.nii"})], data, work)

    assert (work / "ct.nii").is_symlink()
    assert (work / "ct.nii").read_text(encoding="utf-8") == "scan"


def test_a_symlinked_subdirectory_is_not_walked_into(tmp_path):
    """Following it could lead back out of the tree, or round a loop."""
    data, work = tmp_path / "data", tmp_path / "input"
    (data / "ds").mkdir(parents=True)
    (data / "ds" / "self").symlink_to(data / "ds", target_is_directory=True)

    link_writable_tree(data / "ds", work / "ds")

    assert (work / "ds" / "self").is_symlink()


def test_a_missing_source_is_reported_and_skipped(tmp_path):
    data, work = tmp_path / "data", tmp_path / "input"
    data.mkdir()
    said = []

    stage_task_inputs([_task(gone="gone")], data, work, warn=said.append)

    assert any("mount source missing" in m for m in said)
    assert not (work / "gone").exists()


def test_no_data_mount_is_reported_once(tmp_path):
    said = []

    stage_task_inputs([_task(slides="slides")], tmp_path / "absent", tmp_path, warn=said.append)

    assert said and "no /mount/data" in said[0]


def test_already_staged_inputs_are_left_alone(tmp_path):
    data, work = tmp_path / "data", tmp_path / "input"
    data.mkdir()
    (data / "ct.nii").write_text("new", encoding="utf-8")
    work.mkdir()
    (work / "ct.nii").write_text("kept", encoding="utf-8")

    stage_task_inputs([_task(**{"ct.nii": "ct.nii"})], data, work)

    assert (work / "ct.nii").read_text(encoding="utf-8") == "kept"


def test_mounts_are_collected_from_example_and_test_cases():
    task = {
        "example": {"mount": {"a": "a"}},
        "test_cases": {"one": {"mount": {"b": "b"}}},
    }

    assert sorted(task_mounts(task)) == [("a", "a"), ("b", "b")]
