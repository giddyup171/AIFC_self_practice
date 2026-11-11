"""End-to-end: train on the fixture, then validate and score through the CLI."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import main
from src.artifacts import latest_run_dir, load_model
from src.config import deep_merge, load_config
from src.predict import score
from src.train import run_training

from .conftest import FIXTURE


@pytest.fixture(scope="module")
def trained(tmp_path_factory, base_config_dict):
    tmp = tmp_path_factory.mktemp("run")
    cfg_dict = deep_merge(base_config_dict, {
        "data": {"path": str(FIXTURE)},
        "cv": {"tuning_folds": 2, "evaluation_folds": 2},
        "models": {
            "linear_regression": {"n_iter": 2},
            "random_forest": {"n_iter": 1, "search_space": {"n_estimators": {"dist": "randint", "low": 10, "high": 20}}},
            "gradient_boosting": {"n_iter": 1, "search_space": {"max_iter": {"dist": "randint", "low": 20, "high": 40}}},
        },
        "evaluation": {"bootstrap_samples": 20, "importance_sample": 200, "importance_repeats": 1},
        "artifacts": {"dir": str(tmp / "artifacts"), "report_dir": str(tmp / "report")},
        "logging": {"level": "WARNING"},
    })
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_config(path)
    return cfg, run_training(cfg, run_id="test-run"), path


def test_run_writes_expected_artifacts(trained):
    _, art, _ = trained
    for rel in ["config.yaml", "run.log", "data_report.json", "model.joblib", "model_card.json",
                "metrics/cv_summary.csv", "metrics/test_results.csv", "metrics/selection.csv",
                "metrics/error_by_score_band.csv", "predictions/oof_train.csv", "predictions/test.csv",
                "models/gradient_boosting.joblib", "plots/predicted_vs_actual.png"]:
        assert (art.dir / rel).exists(), rel
    assert latest_run_dir(art.root) == art.dir


def test_summary_export_excludes_models(trained):
    cfg, _, _ = trained
    rep = Path(cfg["artifacts"]["report_dir"])
    assert (rep / "model_card.json").exists() and (rep / "plots" / "cv_comparison.png").exists()
    assert not list(rep.rglob("*.joblib")) and not (rep / "predictions").exists()


def test_model_card_consistent_with_selection(trained):
    _, art, _ = trained
    model, card = load_model(art.dir)
    sel = pd.read_csv(art.dir / "metrics" / "selection.csv", index_col=0)
    assert sel.loc[card["model_name"], "selected"]
    assert card["flag_cutoff"] == pytest.approx(sel.loc[card["model_name"], "cutoff"])
    assert "bag_color" in card["features"]["excluded"]
    assert len(card["training_data"]["sha256"]) == 64 and hasattr(model, "predict")


def test_metrics_are_sane(trained):
    _, art, _ = trained
    res = pd.read_csv(art.dir / "metrics" / "test_results.csv", index_col=0)
    assert (res["rmse"] > 0).all() and (res["rmse"] < 20).all()
    assert res["weak_recall"].between(0, 1).all()
    report = json.loads((art.dir / "data_report.json").read_text())
    assert report["cleaning"]["duplicate_records_merged"] > 0 and report["cleaning"]["conflicting_values"] == {}


def test_scoring_sorts_and_flags(trained):
    cfg, art, _ = trained
    new = pd.read_csv(FIXTURE).head(40).drop(columns="final_test")
    new.loc[new.index[0], "attendance_rate"] = None
    new.loc[new.index[1], "mode_of_transport"] = "bicycle"
    out = score(new, art.dir, cfg["data"]["valid_age_range"])
    assert list(out.columns) == ["student_id", "predicted_score", "flag_for_support"]
    assert out.predicted_score.is_monotonic_increasing and out.predicted_score.between(0, 100).all()
    assert out.student_id.is_unique


def test_scoring_batch_with_entirely_missing_columns(trained):
    cfg, art, _ = trained
    new = pd.read_csv(FIXTURE).head(10).drop(columns="final_test")
    new["CCA"] = float("nan")
    new["attendance_rate"] = float("nan")
    out = score(new, art.dir, cfg["data"]["valid_age_range"])
    assert len(out) == new.student_id.nunique()


def test_cli_predict_and_exit_codes(trained, tmp_path):
    _, art, cfg_path = trained
    inp, outp = tmp_path / "new.csv", tmp_path / "scored.csv"
    pd.read_csv(FIXTURE).head(20).to_csv(inp, index=False)
    assert main.main(["--config", str(cfg_path), "predict", "--input", str(inp), "--output", str(outp),
                      "--run-dir", str(art.dir)]) == 0
    assert outp.exists()
    bad = tmp_path / "bad.csv"
    pd.read_csv(FIXTURE).head(5).drop(columns=["sleep_time"]).to_csv(bad, index=False)
    assert main.main(["--config", str(cfg_path), "predict", "--input", str(bad), "--output",
                      str(tmp_path / "o.csv"), "--run-dir", str(art.dir)]) == 1
    assert main.main(["--config", str(cfg_path), "validate"]) == 0
    assert main.main(["--config", str(cfg_path), "validate", "--input", str(bad)]) == 1


def test_cli_bad_config_exit_code(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\n")
    assert main.main(["--config", str(p), "validate"]) == 2
