from CoScientist.tools.coder_tools.coder_tools import CoderToolset


def test_normalization_preserves_the_complete_dataset_and_diagnostics():
    stdout = "molecule_id,SMILES,score\n" + "\n".join(
        f"M-{i},CCO,{i}" for i in range(1000)
    )
    stderr = "\n".join(f"diagnostic {i}" for i in range(1000))
    result = CoderToolset._normalize(
        {"stdout": stdout, "stderr": stderr, "exit_code": 0}
    )
    assert result["stdout"] == stdout
    assert result["stderr"] == stderr
