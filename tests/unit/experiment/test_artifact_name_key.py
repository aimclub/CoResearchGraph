from CoScientist.experiments.runtime.shared import artifact_name_key


def test_hex_suffix_does_not_merge_distinct_artifact_names():
    assert artifact_name_key("measurements-89ee24ce.json") != artifact_name_key(
        "measurements.json"
    )


def test_unrelated_stems_stay_distinct():
    assert artifact_name_key("predictions.csv") != artifact_name_key(
        "measurements.json"
    )
