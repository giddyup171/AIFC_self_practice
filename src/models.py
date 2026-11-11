"""Model definitions and hyperparameter tuning, driven by the config.

* linear_regression   Ridge regression; the search may add cubic splines to the numeric features
* random_forest       bagging of deep, independent regression trees
* gradient_boosting   boosting of shallow trees fitted sequentially to residuals
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, make_spline

logger = logging.getLogger(__name__)

ESTIMATORS = {
    "linear_regression": (Ridge, True),        # (class, needs scaling)
    "random_forest": (RandomForestRegressor, False),
    "gradient_boosting": (HistGradientBoostingRegressor, False),
}
DISPLAY_NAMES = {"linear_regression": "Linear (Ridge)", "random_forest": "Random Forest",
                 "gradient_boosting": "Gradient Boosting"}
SPLINE_STEP = "prep__num__splines"


def to_distribution(spec: dict):
    d = spec["dist"]
    if d == "loguniform":
        return loguniform(spec["low"], spec["high"])
    if d == "uniform":
        return uniform(spec["low"], spec["high"] - spec["low"])
    if d == "randint":
        return randint(spec["low"], spec["high"])
    if d == "choice":
        return list(spec["values"])
    raise ValueError(f"Unknown distribution {d!r}")


def build_pipeline(name: str, cfg: dict) -> Pipeline:
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown model {name!r}")
    cls, scale = ESTIMATORS[name]
    params = dict(cfg["models"][name].get("params") or {})
    if "random_state" in cls().get_params():
        params.setdefault("random_state", cfg["seed"])
    # The linear model always gets a spline slot; the search decides whether to use it ("passthrough" = off).
    prep = build_preprocessor(cfg["features"]["categorical"], cfg["features"]["numeric"], scale=scale,
                              splines=(name == "linear_regression"))
    return Pipeline([("prep", prep), ("model", cls(**params))])


def search_space(name: str, cfg: dict) -> dict:
    space_cfg = dict(cfg["models"][name].get("search_space") or {})
    space = {}
    if name == "linear_regression":
        use = space_cfg.pop("splines", {"dist": "choice", "values": [False]})
        knots = space_cfg.pop("n_knots", {"dist": "randint", "low": 5, "high": 6})
        options = []
        if False in use.get("values", [False]):
            options.append("passthrough")
        if True in use.get("values", []):
            options += [make_spline(k) for k in range(knots["low"], knots["high"])]
        space[SPLINE_STEP] = options
    space.update({f"model__{p}": to_distribution(s) for p, s in space_cfg.items()})
    return space


def enabled_models(cfg: dict) -> list[str]:
    return [m for m, spec in cfg["models"].items() if spec.get("enabled", True)]


@dataclass
class TunedModel:
    name: str
    pipeline: Pipeline
    best_params: dict
    inner_cv_rmse: float
    seconds: float
    trials: pd.DataFrame = field(repr=False)


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k == SPLINE_STEP:
            out["splines"] = "none" if isinstance(v, str) else f"{v.n_knots} knots"
            continue
        k = k.replace("model__", "")
        if isinstance(v, (float, np.floating)):
            v = float(f"{v:.4g}")
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


def tune(name: str, cfg: dict, X: pd.DataFrame, y: pd.Series) -> TunedModel:
    """Randomised search minimising RMSE on K-fold inner CV (training data only)."""
    spec = cfg["models"][name]
    n_jobs = 1 if spec.get("params", {}).get("n_jobs") not in (None, 1) else -1
    search = RandomizedSearchCV(
        build_pipeline(name, cfg), search_space(name, cfg), n_iter=spec["n_iter"],
        scoring="neg_root_mean_squared_error",
        cv=KFold(cfg["cv"]["tuning_folds"], shuffle=True, random_state=cfg["seed"]),
        random_state=cfg["seed"], n_jobs=n_jobs, refit=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    secs = time.perf_counter() - t0
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    trials = pd.DataFrame({"rank": res["rank_test_score"].to_numpy(),
                           "mean_rmse": -res["mean_test_score"].to_numpy(),
                           "std_rmse": res["std_test_score"].to_numpy(),
                           "params": [_clean_params(p) for p in res["params"]]})
    best = _clean_params(search.best_params_)
    logger.info("Tuned %-18s %3d trials in %5.0fs | inner CV RMSE %.3f | %s", name, spec["n_iter"], secs,
                -search.best_score_, best)
    return TunedModel(name, search.best_estimator_, best, float(-search.best_score_), secs, trials)
