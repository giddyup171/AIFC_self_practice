"""Loading, cleaning (including merging duplicate student records) and splitting."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import TARGET, validate_clean, validate_raw

logger = logging.getLogger(__name__)

# Columns allowed to differ between duplicate records of one student without being a conflict.
NOISE_COLUMNS = ["index", "bag_color"]


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", f"{len(df):,}", df.shape[1], path)
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _to_object(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), np.nan)


def merge_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse repeated records of the same student into one.

    Verified on the training data: the 900 duplicated student_ids agree on every attribute
    except bag_color (random) and, in some pairs, one copy has a missing final_test or
    attendance_rate. Taking the first non-missing value per column therefore loses nothing.
    Any genuine conflict in another column is counted and logged.
    """
    id_col = "student_id"
    dup_mask = df[id_col].duplicated(keep=False)
    report = {"duplicate_records_merged": int(df[id_col].duplicated().sum())}
    check = [c for c in df.columns if c not in NOISE_COLUMNS + [id_col]]
    if dup_mask.any():
        g = df[dup_mask].groupby(id_col)
        conflicts = {c: int((g[c].nunique(dropna=True) > 1).sum()) for c in check}
        conflicts = {c: n for c, n in conflicts.items() if n}
        report["conflicting_values"] = conflicts
        if conflicts:
            logger.warning("Duplicate records disagree on %s; keeping the first record's value", conflicts)
    merged = df.groupby(id_col, sort=False, as_index=False).first()
    return merged[df.columns], report


def clean(df: pd.DataFrame, valid_age_range=(14, 17), require_target: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fix the known quality issues. Returns (clean data, report of what was changed)."""
    report: dict = {"rows_in": len(df)}
    df, dup_report = merge_duplicates(df.copy())
    report.update(dup_report)

    if require_target:
        before = len(df)
        df = df.dropna(subset=[TARGET])
        report["dropped_missing_target"] = before - len(df)

    cca = _text(df["CCA"]).str.title()
    report["CCA_missing_set_to_None"] = int(cca.isna().sum())
    report["CCA_recased"] = int((_text(df["CCA"]) != cca).fillna(False).sum())
    df["CCA"] = _to_object(cca.fillna("None"))

    tuition = _text(df["tuition"])
    report["tuition_expanded"] = int(tuition.isin(["Y", "N"]).sum())
    df["tuition"] = _to_object(tuition.replace({"Y": "Yes", "N": "No"}))

    for c in ["direct_admission", "learning_style", "gender", "mode_of_transport"]:
        df[c] = _to_object(_text(df[c]))

    age = pd.to_numeric(df["age"], errors="coerce")
    low, high = valid_age_range
    typo = age.isin([low - 10 + k for k in range(high - low + 1)]) & ~age.between(low, high)
    fixed = age.where(~typo, age + 10)
    invalid = fixed.notna() & ~fixed.between(low, high)
    report["age_typo_fixed (+10)"] = int(typo.sum())
    report["age_invalid_set_missing"] = int(invalid.sum())
    df["age"] = fixed.where(~invalid)

    for c in ["number_of_siblings", "n_male", "n_female", "hours_per_week", "attendance_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["sleep_time", "wake_time"]:
        df[c] = _to_object(_text(df[c]))
    if require_target:
        df[TARGET] = df[TARGET].astype(float)

    df = df.drop(columns=[c for c in NOISE_COLUMNS if c in df.columns])
    report["rows_out"] = len(df)
    validate_clean(df, require_target=require_target)
    return df.reset_index(drop=True), report


def load_and_clean(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = load_raw(cfg["data"]["path"])
    raw_report = validate_raw(raw, require_target=True).raise_if_failed()
    for k, v in raw_report.warnings.items():
        logger.info("Data quality: %-42s %s", k, f"{v:,}")
    df, clean_report = clean(raw, tuple(cfg["data"]["valid_age_range"]), require_target=True)
    n = cfg["data"].get("sample_rows")
    if n and n < len(df):
        df = df.sample(n, random_state=cfg["seed"]).reset_index(drop=True)
        logger.warning("Using a random sample of %s students (data.sample_rows)", f"{n:,}")
    return df, {"raw_validation": raw_report.to_dict(), "cleaning": clean_report,
                "data_sha256": file_sha256(cfg["data"]["path"])}


def split(df: pd.DataFrame, features: list[str], test_size: float, seed: int, weak_threshold: float):
    """Train/test split stratified on weak (< threshold) vs not, so both sets have the same share of
    the students the school cares most about."""
    X, y = df[features], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=(y < weak_threshold))
