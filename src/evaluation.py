"""Regression metrics, weak-student identification, statistical comparison and model selection."""

from __future__ import annotations

import logging
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, fbeta_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Regression metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": float(mean_squared_error(y, p) ** 0.5), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "median_ae": float(np.median(np.abs(y - p))),
            "within_5_marks": float(np.mean(np.abs(y - p) <= 5)), "bias": float(np.mean(p - y))}


# --------------------------------------------------------------------------- #
# Weak-student identification
# --------------------------------------------------------------------------- #
def cutoff_grid(cfg: dict) -> np.ndarray:
    g = cfg["weak_students"]["cutoff_grid"]
    return np.arange(g["start"], g["stop"] + 1e-9, g["step"])


def weak_metrics(y, p, threshold: float, cutoff: float, beta: float) -> dict[str, float]:
    """Treat 'actual < threshold' as the positive class and flag students whose prediction < cutoff."""
    actual = np.asarray(y) < threshold
    flagged = np.asarray(p) < cutoff
    return {"cutoff": float(cutoff), "flagged_share": float(flagged.mean()),
            "recall": float(recall_score(actual, flagged, zero_division=0)),
            "precision": float(precision_score(actual, flagged, zero_division=0)),
            f"f{beta:g}": float(fbeta_score(actual, flagged, beta=beta, zero_division=0)),
            # ranking quality: lower predicted score = higher risk
            "pr_auc": float(average_precision_score(actual, -np.asarray(p))),
            "caught": int((actual & flagged).sum()), "missed": int((actual & ~flagged).sum()),
            "false_alarms": int((~actual & flagged).sum())}


def best_cutoff(y, p, cfg: dict) -> tuple[float, pd.DataFrame]:
    """Prediction cut-off that maximises F-beta (chosen on training out-of-fold predictions)."""
    ws = cfg["weak_students"]
    rows = [weak_metrics(y, p, ws["score_threshold"], c, ws["f_beta"]) for c in cutoff_grid(cfg)]
    table = pd.DataFrame(rows)
    key = f"f{ws['f_beta']:g}"
    return float(table.loc[table[key].idxmax(), "cutoff"]), table


# --------------------------------------------------------------------------- #
# Cross-validation on identical folds
# --------------------------------------------------------------------------- #
def cross_validate(pipelines: dict, X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int
                   ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Paired CV: every model sees the same folds. Returns fold metrics (incl. training RMSE, to diagnose
    over-fitting) and out-of-fold predictions."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed + 1).split(X))
    rows, oof = [], {}
    for name, pipe in pipelines.items():
        oof[name] = np.zeros(len(y))
        for k, (tr, va) in enumerate(folds):
            t0 = time.perf_counter()
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            fit_s = time.perf_counter() - t0
            p = m.predict(X.iloc[va])
            oof[name][va] = p
            train_rmse = mean_squared_error(y.iloc[tr], m.predict(X.iloc[tr])) ** 0.5
            rows.append({"model": name, "fold": k, **regression_metrics(y.iloc[va], p),
                         "train_rmse": train_rmse, "fit_seconds": fit_s})
        logger.info("CV %-18s RMSE %.3f", name, np.mean([r["rmse"] for r in rows if r["model"] == name]))
    return pd.DataFrame(rows), oof


def summarise_cv(folds: pd.DataFrame) -> pd.DataFrame:
    s = folds.drop(columns="fold").groupby("model").agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["overfit_gap"] = s["rmse_mean"] - s["train_rmse_mean"]
    return s.sort_values("rmse_mean")


def paired_cv_tests(folds: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    piv = folds.pivot(index="fold", columns="model", values=metric)
    rows = []
    for a, b in combinations(piv.columns, 2):
        d = piv[a] - piv[b]
        rows.append({"model_a": a, "model_b": b, f"mean_{metric}_diff_a_minus_b": d.mean(),
                     "a_better_folds": f"{int((d < 0).sum())}/{len(d)}",
                     "paired_t_p_value": stats.ttest_rel(piv[a], piv[b]).pvalue})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Selection (training out-of-fold predictions only)
# --------------------------------------------------------------------------- #
def select_model(oof: dict, y_train, cv_summary: pd.DataFrame, cfg: dict) -> tuple[str, dict, pd.DataFrame]:
    ws = cfg["weak_students"]
    key = f"f{ws['f_beta']:g}"
    rows, cutoffs = [], {}
    for name, p in oof.items():
        c, _ = best_cutoff(y_train, p, cfg)
        cutoffs[name] = c
        w = weak_metrics(y_train, p, ws["score_threshold"], c, ws["f_beta"])
        rows.append({"model": name, "cv_rmse": cv_summary.loc[name, "rmse_mean"], "cutoff": c,
                     f"oof_{key}": w[key], "oof_recall": w["recall"], "oof_precision": w["precision"]})
    table = pd.DataFrame(rows).set_index("model").sort_values("cv_rmse")
    best = table["cv_rmse"].min()
    tied = table[table["cv_rmse"] <= best * (1 + cfg["selection"]["tie_tolerance"])]
    chosen = tied[f"oof_{key}"].idxmax()
    table["tied_for_best"] = table.index.isin(tied.index)
    table["selected"] = table.index == chosen
    others = [m for m in tied.index if m != chosen]
    if others:
        logger.info("Selected %s: within %.0f%% of the best CV RMSE together with %s; tie broken by OOF %s",
                    chosen, 100 * cfg["selection"]["tie_tolerance"], others, key.upper())
    else:
        logger.info("Selected %s: lowest CV RMSE", chosen)
    return chosen, cutoffs, table


# --------------------------------------------------------------------------- #
# Test set
# --------------------------------------------------------------------------- #
def evaluate_test(preds: dict, cutoffs: dict, y_test, cfg: dict, fitted: dict, X_test) -> pd.DataFrame:
    ws = cfg["weak_students"]
    y = np.asarray(y_test, float)
    weak = y < ws["score_threshold"]
    rows = {}
    for name, p in preds.items():
        t0 = time.perf_counter(); fitted[name].predict(X_test)
        ms = (time.perf_counter() - t0) * 1000 / len(X_test) * 1000
        low = regression_metrics(y[weak], p[weak])
        rows[name] = {**regression_metrics(y, p), "rmse_weak_students": low["rmse"],
                      "bias_weak_students": low["bias"],
                      **{f"weak_{k}": v for k, v in weak_metrics(y, p, ws["score_threshold"], cutoffs[name],
                                                                 ws["f_beta"]).items()},
                      "predict_ms_per_1000": ms}
    return pd.DataFrame(rows).T


def bootstrap_compare(preds: dict, cutoffs: dict, y_test, cfg: dict) -> pd.DataFrame:
    """Bootstrap 95% CIs for test differences in RMSE and weak-student recall."""
    ws = cfg["weak_students"]
    y = np.asarray(y_test, float)
    rng = np.random.default_rng(cfg["seed"])
    idx = [rng.integers(0, len(y), len(y)) for _ in range(cfg["evaluation"]["bootstrap_samples"])]
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rows = []
    for a, b in combinations(preds, 2):
        pa, pb = preds[a], preds[b]
        d_rmse = np.array([rmse(y[i], pa[i]) - rmse(y[i], pb[i]) for i in idx])
        weak = y < ws["score_threshold"]
        fa, fb = pa < cutoffs[a], pb < cutoffs[b]
        d_rec = np.array([(fa[i] & weak[i]).sum() / max(weak[i].sum(), 1) - (fb[i] & weak[i]).sum() / max(weak[i].sum(), 1)
                          for i in idx])
        rows.append({"model_a": a, "model_b": b, "rmse_diff_a_minus_b": d_rmse.mean(),
                     "rmse_ci_low": np.percentile(d_rmse, 2.5), "rmse_ci_high": np.percentile(d_rmse, 97.5),
                     "weak_recall_diff_a_minus_b": d_rec.mean(),
                     "recall_ci_low": np.percentile(d_rec, 2.5), "recall_ci_high": np.percentile(d_rec, 97.5)})
    return pd.DataFrame(rows)


def error_by_score_band(y_test, preds: dict) -> pd.DataFrame:
    y = pd.Series(np.asarray(y_test, float))
    bands = pd.cut(y, [0, 45, 50, 60, 70, 80, 101], right=False,
                   labels=["<45", "45-49", "50-59", "60-69", "70-79", "80+"])
    out = {"n": y.groupby(bands, observed=False).size()}
    for name, p in preds.items():
        err = pd.Series(np.asarray(p) - y.to_numpy())
        out[f"{name}_mae"] = err.abs().groupby(bands, observed=False).mean()
        out[f"{name}_bias"] = err.groupby(bands, observed=False).mean()
    return pd.DataFrame(out)


def importance(fitted: dict, X_test: pd.DataFrame, y_test, cfg: dict) -> pd.DataFrame:
    ev = cfg["evaluation"]
    sample = X_test.sample(min(ev["importance_sample"], len(X_test)), random_state=cfg["seed"])
    yt = pd.Series(np.asarray(y_test), index=X_test.index).loc[sample.index]
    out = {}
    for name, m in fitted.items():
        r = permutation_importance(m, sample, yt, scoring="neg_root_mean_squared_error",
                                   n_repeats=ev["importance_repeats"], random_state=cfg["seed"], n_jobs=-1)
        out[name] = pd.Series(r.importances_mean, index=sample.columns)  # RMSE increase in marks
    first = next(iter(out))
    return pd.DataFrame(out).sort_values(first, ascending=False)
