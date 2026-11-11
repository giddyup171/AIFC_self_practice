"""Load, merge and validate the YAML configuration.

A profile may ``extends:`` another file; its values are deep-merged on top.
``load_config`` returns a plain dict after validation, so the rest of the code
can trust that values exist and are in range.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DISTRIBUTIONS = {"loguniform", "uniform", "randint", "choice"}
MODEL_NAMES = {"linear_regression", "random_forest", "gradient_boosting"}


class ConfigError(ValueError):
    """Raised when the configuration is missing a value or has an invalid one."""


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular 'extends' involving {path}")
    seen.add(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(_read(path.parent / parent, seen), cfg)
    return cfg


def _require(cfg: dict, dotted: str):
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Missing required config value: {dotted}")
        node = node[part]
    return node


def _in_range(name: str, v, low, high, inclusive=True):
    ok = isinstance(v, (int, float)) and ((low <= v <= high) if inclusive else (low < v < high))
    if not ok:
        raise ConfigError(f"{name} must be between {low} and {high}, got {v!r}")


def validate_config(cfg: dict) -> dict:
    for key in ["seed", "data.path", "data.target", "data.test_size", "data.valid_age_range",
                "features.categorical", "features.numeric", "cv.tuning_folds", "cv.evaluation_folds", "models",
                "weak_students.score_threshold", "weak_students.f_beta", "weak_students.cutoff_grid",
                "selection.tie_tolerance", "artifacts.dir"]:
        _require(cfg, key)

    _in_range("data.test_size", cfg["data"]["test_size"], 0, 1, inclusive=False)
    _in_range("selection.tie_tolerance", cfg["selection"]["tie_tolerance"], 0, 1)
    _in_range("weak_students.score_threshold", cfg["weak_students"]["score_threshold"], 0, 100)
    if not isinstance(cfg["weak_students"]["f_beta"], (int, float)) or cfg["weak_students"]["f_beta"] <= 0:
        raise ConfigError("weak_students.f_beta must be a positive number")
    g = cfg["weak_students"]["cutoff_grid"]
    if not (0 <= g["start"] < g["stop"] <= 100) or g["step"] <= 0:
        raise ConfigError("weak_students.cutoff_grid needs 0 <= start < stop <= 100 and step > 0")
    age = cfg["data"]["valid_age_range"]
    if not (isinstance(age, list) and len(age) == 2 and age[0] < age[1]):
        raise ConfigError("data.valid_age_range must be [low, high] with low < high")
    sample = cfg["data"].get("sample_rows")
    if sample is not None and (not isinstance(sample, int) or sample < 500):
        raise ConfigError("data.sample_rows must be null or an integer >= 500")
    for k in ["tuning_folds", "evaluation_folds"]:
        if not isinstance(cfg["cv"][k], int) or cfg["cv"][k] < 2:
            raise ConfigError(f"cv.{k} must be an integer >= 2")

    cat, num = set(cfg["features"]["categorical"]), set(cfg["features"]["numeric"])
    if cat & num:
        raise ConfigError(f"Features listed as both categorical and numeric: {sorted(cat & num)}")
    excluded = set((cfg["features"].get("excluded") or {}).keys())
    if excluded & (cat | num):
        raise ConfigError(f"Features both used and excluded: {sorted(excluded & (cat | num))}")
    if cfg["data"]["target"] in cat | num:
        raise ConfigError("The target cannot be used as a feature")

    unknown = set(cfg["models"]) - MODEL_NAMES
    if unknown:
        raise ConfigError(f"Unknown model(s) {sorted(unknown)}; choose from {sorted(MODEL_NAMES)}")
    if not [m for m, s in cfg["models"].items() if s.get("enabled", True)]:
        raise ConfigError("At least one model must be enabled")
    for name, spec in cfg["models"].items():
        if not isinstance(spec.get("n_iter", 0), int) or spec.get("n_iter", 0) < 1:
            raise ConfigError(f"models.{name}.n_iter must be an integer >= 1")
        for p, d in (spec.get("search_space") or {}).items():
            if d.get("dist") not in DISTRIBUTIONS:
                raise ConfigError(f"models.{name}.search_space.{p}: dist must be one of {sorted(DISTRIBUTIONS)}")
            if d["dist"] == "choice" and not d.get("values"):
                raise ConfigError(f"models.{name}.search_space.{p}: 'choice' needs a non-empty 'values' list")
            if d["dist"] != "choice" and not (d.get("low") is not None and d.get("high") is not None
                                              and d["low"] < d["high"]):
                raise ConfigError(f"models.{name}.search_space.{p}: needs low < high")
    return cfg


def load_config(path: str | Path) -> dict:
    """Read a YAML profile (following ``extends``) and validate it."""
    path = Path(path)
    cfg = _read(path, set())
    cfg["_source"] = str(path)
    return validate_config(cfg)
