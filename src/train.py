"""End-to-end training pipeline: data -> features -> tune -> compare -> select -> artifacts."""

from __future__ import annotations

import logging
import time

import pandas as pd
from sklearn.base import clone

from . import evaluation as ev
from . import plots
from .artifacts import RunArtifacts, environment
from .data import load_and_clean, split
from .features import add_features
from .logging_utils import setup_logging
from .models import DISPLAY_NAMES, enabled_models, tune

logger = logging.getLogger(__name__)


def run_training(cfg: dict, run_id: str | None = None) -> RunArtifacts:
    art = RunArtifacts(cfg["artifacts"]["dir"], run_id)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"), art.log_file)
    logger.info("Run %s | config %s | output %s", art.run_id, cfg.get("_source"), art.dir)
    art.save_config(cfg)
    t_start = time.perf_counter()
    ws = cfg["weak_students"]

    # 1. Data
    df, data_report = load_and_clean(cfg)
    df = add_features(df)
    features = cfg["features"]["categorical"] + cfg["features"]["numeric"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Configured features not found after feature engineering: {missing}")
    X_train, X_test, y_train, y_test = split(df, features, cfg["data"]["test_size"], cfg["seed"],
                                             ws["score_threshold"])
    data_report["split"] = {"train_rows": len(X_train), "test_rows": len(X_test),
                            "train_weak_share": float((y_train < ws["score_threshold"]).mean()),
                            "test_weak_share": float((y_test < ws["score_threshold"]).mean())}
    art.save_json("data_report.json", data_report)
    logger.info("Split: train %s / test %s | weak students (< %g) %.1f%%", f"{len(X_train):,}",
                f"{len(X_test):,}", ws["score_threshold"], 100 * (y_train < ws["score_threshold"]).mean())

    # 2. Tune
    tuned = {name: tune(name, cfg, X_train, y_train) for name in enabled_models(cfg)}
    art.save_table("tuning", pd.DataFrame([{"model": t.name, "inner_cv_rmse": t.inner_cv_rmse,
                                            "tuning_seconds": t.seconds, "best_params": t.best_params}
                                           for t in tuned.values()]).set_index("model"))
    for t in tuned.values():
        t.trials.to_csv(art.path("metrics", f"tuning_trials_{t.name}.csv"), index=False)

    # 3. Paired CV
    pipes = {n: t.pipeline for n, t in tuned.items()}
    folds, oof = ev.cross_validate(pipes, X_train, y_train, cfg["cv"]["evaluation_folds"], cfg["seed"])
    cv_summary = ev.summarise_cv(folds)
    art.save_table("cv_folds", folds, index=False)
    art.save_table("cv_summary", cv_summary)
    art.save_table("cv_paired_tests", ev.paired_cv_tests(folds), index=False)
    y_tr = y_train.to_numpy()
    art.save_predictions("oof_train", pd.DataFrame({"final_test": y_tr, **oof}))
    plots.cv_comparison(folds, art.path("plots", "cv_comparison.png"))

    # 4. Selection and weak-student cut-offs (training OOF only)
    chosen, cutoffs, selection = ev.select_model(oof, y_tr, cv_summary, cfg)
    art.save_table("selection", selection)
    plots.weak_tradeoff({n: ev.best_cutoff(y_tr, p, cfg)[1] for n, p in oof.items()}, cutoffs, ws["f_beta"],
                        art.path("plots", "weak_student_tradeoff.png"))

    # 5. Test set (touched once)
    fitted = {n: clone(p).fit(X_train, y_train) for n, p in pipes.items()}
    preds = {n: m.predict(X_test) for n, m in fitted.items()}
    y_te = y_test.to_numpy()
    art.save_predictions("test", pd.DataFrame({"final_test": y_te, **preds}))
    test = ev.evaluate_test(preds, cutoffs, y_te, cfg, fitted, X_test)
    art.save_table("test_results", test)
    art.save_table("test_bootstrap_comparisons", ev.bootstrap_compare(preds, cutoffs, y_te, cfg), index=False)
    art.save_table("error_by_score_band", ev.error_by_score_band(y_te, preds))
    imp = ev.importance(fitted, X_test, y_te, cfg)
    art.save_table("permutation_importance", imp)
    plots.predicted_vs_actual(y_te, preds, ws["score_threshold"], cutoffs, art.path("plots", "predicted_vs_actual.png"))
    plots.importance(imp, art.path("plots", "permutation_importance.png"))

    # 6. Models and model card
    for n, m in fitted.items():
        art.save_model(n, m)
    art.save_model("model", fitted[chosen])
    t = test.loc[chosen]
    key = f"f{ws['f_beta']:g}"
    card = {
        "model_name": chosen, "display_name": DISPLAY_NAMES.get(chosen, chosen), "run_id": art.run_id,
        "task": "Regression: predicted O-level mathematics score (0-100); flag students predicted below a cut-off",
        "features": {"categorical": cfg["features"]["categorical"], "numeric": cfg["features"]["numeric"],
                     "excluded": cfg["features"].get("excluded", {})},
        "weak_student_definition": f"actual score < {ws['score_threshold']}",
        "flag_cutoff": cutoffs[chosen],
        "cutoff_basis": f"Maximises F{ws['f_beta']:g} (recall weighted {ws['f_beta']:g}x) on out-of-fold training predictions",
        "selection_rule": (f"Lowest mean CV RMSE; models within {cfg['selection']['tie_tolerance']:.0%} are tied and the "
                           f"tie is broken by out-of-fold F{ws['f_beta']:g} for flagging weak students"),
        "hyperparameters": tuned[chosen].best_params,
        "cv": {"rmse_mean": cv_summary.loc[chosen, "rmse_mean"], "rmse_std": cv_summary.loc[chosen, "rmse_std"],
               "r2_mean": cv_summary.loc[chosen, "r2_mean"]},
        "test": {k: float(t[k]) for k in ["rmse", "mae", "r2", "within_5_marks", "rmse_weak_students",
                                          "weak_recall", "weak_precision", f"weak_{key}"]},
        "training_data": {"path": cfg["data"]["path"], "sha256": data_report["data_sha256"],
                          "train_rows": len(X_train), "sample_rows": cfg["data"].get("sample_rows")},
        "intended_use": "Identify students likely to score below a credit pass so they can be offered support early.",
        "limitations": ["Predictions are estimates with a typical error of a few marks; use them to prioritise "
                        "support, not to label students.",
                        "Trained on one school's cohort; re-validate before using elsewhere.",
                        "Some attributes (e.g. gender) are used as predictors; review fairness before deployment."],
        "environment": environment(),
    }
    art.save_json("model_card.json", card)
    art.mark_latest()
    if cfg["artifacts"].get("report_dir"):
        art.export_summary(cfg["artifacts"]["report_dir"])

    view = pd.DataFrame({"cv_rmse": cv_summary["rmse_mean"], "test_rmse": test["rmse"], "test_mae": test["mae"],
                         "test_r2": test["r2"], "cutoff": test["weak_cutoff"], "weak_recall": test["weak_recall"],
                         "weak_precision": test["weak_precision"]}).astype(float)
    logger.info("Results\n%s", view.round(3).to_string())
    logger.info("Selected model: %s (flag if predicted < %.1f) | run finished in %.0fs | artifacts in %s",
                chosen, cutoffs[chosen], time.perf_counter() - t_start, art.dir)
    return art
