import pytest

from support.train_command_model import split_by_speaker


def sample_rows():
    return [
        {"speaker_id": "speaker_01", "command_id": "light_on"},
        {"speaker_id": "speaker_02", "command_id": "light_on"},
        {"speaker_id": "speaker_03", "command_id": "light_on"},
    ]


def test_split_by_speaker_has_no_identity_overlap():
    train, validation, test = split_by_speaker(
        sample_rows(),
        val_speakers={"speaker_02"},
        test_speakers={"speaker_03"},
    )

    assert {row["speaker_id"] for row in train} == {"speaker_01"}
    assert {row["speaker_id"] for row in validation} == {"speaker_02"}
    assert {row["speaker_id"] for row in test} == {"speaker_03"}


def test_split_by_speaker_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_by_speaker(
            sample_rows(),
            val_speakers={"speaker_03"},
            test_speakers={"speaker_03"},
        )
