"""Phase 0 smoke tests: package imports and configuration is complete."""

from procure import config


def test_settings_have_required_sections() -> None:
    s = config.load_settings()
    for key in ["seed", "paths", "split", "models", "optimise", "explain", "api"]:
        assert key in s
    assert s["seed"] == 42


def test_split_years_are_disjoint_and_ordered() -> None:
    sp = config.load_settings()["split"]
    assert max(sp["train_years"]) < min(sp["val_years"]) < min(sp["test_years"])


def test_rules_and_geo_load() -> None:
    assert {r["id"] for r in config.load_rules_config()["rules"]} >= {"R1", "R2", "R3", "R4", "R5"}
    assert set(config.load_geo()["lgas"]) == set(config.load_settings()["region"]["states"])
