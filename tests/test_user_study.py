"""Phase 6 tests: SUS scoring and user-study pack generation."""

import pytest

from procure.user_study import score_trust, sus_score, write_pack


def test_sus_score_bounds_and_known_value() -> None:
    assert sus_score([5, 1] * 5) == 100.0
    assert sus_score([1, 5] * 5) == 0.0
    assert sus_score([3] * 10) == 50.0
    with pytest.raises(ValueError):
        sus_score([3] * 9)


def test_trust_and_pack(tmp_path) -> None:
    assert score_trust({"trust": [4, 5, 3]}) == {"trust": 4.0}
    out = write_pack(tmp_path)
    for f in ["scenarios.md", "sus_questionnaire.md", "trust_questionnaire.md", "responses_template.csv", "procedure.md"]:
        assert (out / f).exists()
    assert (out / "scenarios.md").read_text().count("## S") == 5
