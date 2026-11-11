import numpy as np
import pandas as pd
import pytest

from src.evaluation import best_cutoff, regression_metrics, select_model, weak_metrics


def test_regression_metrics_known_values():
    m = regression_metrics([50, 60, 70, 80], [52, 58, 70, 90])
    assert m["mae"] == pytest.approx(3.5)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 0 + 100) / 4))
    assert m["within_5_marks"] == 0.75
    assert m["bias"] == pytest.approx(2.5)


def test_weak_metrics_counts():
    y = np.array([40, 45, 55, 70])          # first two are weak (< 50)
    p = np.array([48, 60, 52, 75])          # flag if predicted < 53 -> flags 40 and 55
    m = weak_metrics(y, p, threshold=50, cutoff=53, beta=2)
    assert (m["caught"], m["missed"], m["false_alarms"]) == (1, 1, 1)
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def test_cutoff_above_threshold_when_recall_matters(base_config_dict):
    rng = np.random.default_rng(0)
    y = rng.normal(67, 14, 5000).clip(30, 100)
    p = y + rng.normal(0, 6, 5000)
    cutoff, table = best_cutoff(y, p, base_config_dict)
    assert cutoff > 50                       # F2 favours catching weak students: flag a bit above 50
    assert table["recall"].is_monotonic_increasing


def test_selection_prefers_lowest_rmse_and_breaks_ties(base_config_dict):
    rng = np.random.default_rng(1)
    y = rng.normal(67, 14, 3000).clip(30, 100)
    oof = {"a": y + rng.normal(0, 5, 3000), "b": y + rng.normal(0, 5, 3000), "c": y + rng.normal(0, 9, 3000)}
    cv = pd.DataFrame({"rmse_mean": {"a": 5.00, "b": 5.03, "c": 9.0}})
    chosen, cutoffs, table = select_model(oof, y, cv, base_config_dict)
    assert chosen in {"a", "b"}                                   # c is far worse
    assert table.loc["a", "tied_for_best"] and table.loc["b", "tied_for_best"]
    assert not table.loc["c", "tied_for_best"]
    f2 = table["oof_f2"]
    assert f2[chosen] == f2[["a", "b"]].max()                     # tie broken by F2
    assert set(cutoffs) == set(oof)
