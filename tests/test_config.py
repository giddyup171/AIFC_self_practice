import pytest
import yaml

from src.config import ConfigError, load_config

from .conftest import ROOT


def test_default_config_is_valid():
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert cfg["weak_students"]["score_threshold"] == 50
    assert set(cfg["models"]) == {"linear_regression", "random_forest", "gradient_boosting"}
    assert "bag_color" in cfg["features"]["excluded"]


def test_fast_profile_extends_and_overrides_base():
    cfg = load_config(ROOT / "configs" / "fast.yaml")
    assert cfg["data"]["sample_rows"] == 4000
    assert cfg["weak_students"]["f_beta"] == 2                         # inherited
    rf = cfg["models"]["random_forest"]
    assert rf["search_space"]["n_estimators"]["high"] == 60             # nested override
    assert "max_depth" in rf["search_space"]                            # nested value kept
    assert cfg["artifacts"]["report_dir"] is None


@pytest.mark.parametrize("changes, message", [
    ({"data": {"test_size": 0}}, "test_size"),
    ({"weak_students": {"score_threshold": 150}}, "score_threshold"),
    ({"weak_students": {"f_beta": 0}}, "f_beta"),
    ({"weak_students": {"cutoff_grid": {"start": 70, "stop": 40, "step": 1}}}, "cutoff_grid"),
    ({"data": {"valid_age_range": [17, 14]}}, "valid_age_range"),
    ({"cv": {"tuning_folds": 1}}, "tuning_folds"),
    ({"models": {"svm": {"n_iter": 1}}}, "Unknown model"),
    ({"models": {"random_forest": {"search_space": {"max_depth": {"dist": "normal"}}}}}, "dist"),
    ({"features": {"numeric": ["CCA"]}}, "both categorical and numeric"),
    ({"features": {"numeric": ["bag_color"]}}, "both used and excluded"),
    ({"features": {"numeric": ["final_test"]}}, "target"),
])
def test_invalid_values_are_rejected(write_config, changes, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(changes))


def test_missing_file_and_circular_extends(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": "b.yaml"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": "a.yaml"}))
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")
