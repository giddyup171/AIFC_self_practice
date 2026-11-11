"""Plots saved with every run (non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .models import DISPLAY_NAMES  # noqa: E402

COLORS = {"linear_regression": "#4C72B0", "random_forest": "#DD8452", "gradient_boosting": "#55A868"}


def _label(n):
    return DISPLAY_NAMES.get(n, n)


def predicted_vs_actual(y, preds: dict, threshold: float, cutoffs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.asarray(y)
    for ax, (n, p) in zip(axes, preds.items()):
        ax.scatter(y, p, s=5, alpha=0.3, color=COLORS.get(n))
        ax.plot([30, 100], [30, 100], "k:", lw=1)
        ax.axvline(threshold, color="red", lw=1, ls="--", label=f"weak: actual < {threshold:g}")
        ax.axhline(cutoffs[n], color="grey", lw=1, ls="--", label=f"flag: predicted < {cutoffs[n]:g}")
        rmse = np.sqrt(np.mean((y - p) ** 2))
        ax.set(title=f"{_label(n)} (RMSE {rmse:.2f})", xlabel="Actual score")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Predicted score")
    fig.suptitle("Predicted vs actual (test set); bottom-left quadrant = weak students correctly flagged")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def cv_comparison(folds: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    s = folds.groupby("model")[["rmse", "train_rmse"]].agg(["mean", "std"])
    order = s[("rmse", "mean")].sort_values(ascending=False).index
    y = np.arange(len(order))
    axes[0].barh(y + 0.2, s.loc[order, ("rmse", "mean")], 0.4, xerr=s.loc[order, ("rmse", "std")],
                 color=[COLORS.get(m) for m in order], label="validation", capsize=3)
    axes[0].barh(y - 0.2, s.loc[order, ("train_rmse", "mean")], 0.4, color="#BBBBBB", label="training")
    axes[0].set_yticks(y, [_label(m) for m in order])
    axes[0].set(xlabel="RMSE (marks), lower is better", title="Cross-validated RMSE: training vs validation")
    axes[0].legend(fontsize=8)
    piv = folds.pivot(index="fold", columns="model", values="rmse")
    for m in piv.columns:
        axes[1].plot(piv.index, piv[m], "o-", color=COLORS.get(m), label=_label(m))
    axes[1].set(xlabel="Fold", ylabel="Validation RMSE", title="Per-fold RMSE (identical folds)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def weak_tradeoff(tables: dict, chosen_cutoffs: dict, beta: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    key = f"f{beta:g}"
    for n, t in tables.items():
        axes[0].plot(t["recall"], t["precision"], color=COLORS.get(n), label=_label(n))
        axes[1].plot(t["cutoff"], t[key], color=COLORS.get(n), label=_label(n))
        axes[1].axvline(chosen_cutoffs[n], color=COLORS.get(n), ls=":", lw=1)
    axes[0].set(xlabel="Recall (share of weak students flagged)", ylabel="Precision",
                title="Weak-student flagging trade-off (training OOF)")
    axes[1].set(xlabel="Flag if predicted score below", ylabel=key.upper(),
                title=f"{key.upper()} vs cut-off (dotted = chosen)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def importance(imp: pd.DataFrame, path: Path) -> None:
    ax = imp.rename(columns=_label).plot.barh(figsize=(9, 6), color=[COLORS.get(c) for c in imp.columns])
    ax.invert_yaxis()
    ax.set(xlabel="Increase in RMSE (marks) when the feature is shuffled", title="Permutation importance (test sample)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
