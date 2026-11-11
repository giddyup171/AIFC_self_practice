"""Schema validation for raw input and for cleaned data.

* ``validate_raw`` checks data as it arrives (training file or students to score).
  Structural problems are fatal errors; known quality problems that cleaning
  repairs are counted as warnings, so every run records how dirty its input was.
* ``validate_clean`` checks the output of ``data.clean``; any failure is a bug in
  the cleaning code, so every check is fatal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "final_test"
TIME_PATTERN = r"^\s*([01]?\d|2[0-3]):[0-5]\d\s*$"

RAW_COLUMNS = {
    "student_id": {"kind": "id"},
    "number_of_siblings": {"kind": "number", "min": 0, "max": 10},
    "direct_admission": {"kind": "category", "allowed": ["Yes", "No"]},
    "CCA": {"kind": "category", "allowed": ["Sports", "Clubs", "Arts", "None"], "case_insensitive": True,
            "missing_ok": True},
    "learning_style": {"kind": "category", "allowed": ["Visual", "Auditory"]},
    "gender": {"kind": "category", "allowed": ["Male", "Female"]},
    "tuition": {"kind": "category", "allowed": ["Yes", "No", "Y", "N"]},
    "n_male": {"kind": "number", "min": 0, "max": 60},
    "n_female": {"kind": "number", "min": 0, "max": 60},
    "age": {"kind": "number", "min": 14, "max": 17},
    "hours_per_week": {"kind": "number", "min": 0, "max": 60},
    "attendance_rate": {"kind": "number", "min": 0, "max": 100},
    "sleep_time": {"kind": "time"},
    "wake_time": {"kind": "time"},
    "mode_of_transport": {"kind": "category", "allowed": ["public transport", "private transport", "walk"]},
}
OPTIONAL_COLUMNS = ["index", "bag_color"]


class SchemaError(ValueError):
    """Raised when data cannot be used safely."""


@dataclass
class ValidationReport:
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"rows": self.rows, "ok": self.ok, "errors": self.errors, "warnings": self.warnings}

    def raise_if_failed(self) -> "ValidationReport":
        if self.errors:
            raise SchemaError("Input failed schema validation: " + "; ".join(self.errors))
        return self


def validate_raw(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    rep = ValidationReport(rows=len(df))
    if df.empty:
        rep.errors.append("input has no rows")
        return rep
    required = list(RAW_COLUMNS) + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")
        return rep

    dup = df["student_id"].duplicated()
    if dup.any():
        rep.warnings["duplicate_student_records (merged)"] = int(dup.sum())

    if require_target:
        t = pd.to_numeric(df[TARGET], errors="coerce")
        if (df[TARGET].notna() & t.isna()).any():
            rep.errors.append(f"{TARGET} has non-numeric values")
        out = t.notna() & ((t < 0) | (t > 100))
        if out.any():
            rep.errors.append(f"{TARGET} has {int(out.sum())} values outside 0-100")
        if t.isna().any():
            rep.warnings[f"{TARGET}_missing (rows dropped)"] = int(t.isna().sum())
        if t.notna().sum() < 10:
            rep.errors.append(f"fewer than 10 rows with a {TARGET} value; cannot train")

    for col, spec in RAW_COLUMNS.items():
        if spec["kind"] == "id":
            if df[col].isna().any():
                rep.errors.append(f"{col} has {int(df[col].isna().sum())} missing values")
            continue
        s = df[col]
        if s.isna().any():
            key = f"{col}_missing" + (" (means no CCA)" if spec.get("missing_ok") else "")
            rep.warnings[key] = int(s.isna().sum())
        s = s.dropna()
        kind = spec["kind"]
        if kind == "category":
            txt = s.astype(str).str.strip()
            allowed = spec["allowed"]
            if spec.get("case_insensitive"):
                odd = int((~txt.isin(allowed) & txt.str.title().isin(allowed)).sum())
                if odd:
                    rep.warnings[f"{col}_inconsistent_case"] = odd
                txt = txt.str.title()
            if col == "tuition":
                short = int(txt.isin(["Y", "N"]).sum())
                if short:
                    rep.warnings["tuition_abbreviated (Y/N)"] = short
            unknown = int((~txt.isin(allowed)).sum())
            if unknown:
                rep.warnings[f"{col}_unknown_category"] = unknown
        elif kind == "number":
            v = pd.to_numeric(s, errors="coerce")
            if v.isna().any():
                rep.errors.append(f"{col} has {int(v.isna().sum())} non-numeric values")
            out = int(((v < spec["min"]) | (v > spec["max"])).sum())
            if out:
                rep.warnings[f"{col}_out_of_range"] = out
        elif kind == "time":
            bad = int((~s.astype(str).str.match(TIME_PATTERN)).sum())
            if bad:
                rep.errors.append(f"{col} has {bad} values not in H:MM 24-hour format")
    if (pd.to_numeric(df["n_male"], errors="coerce") + pd.to_numeric(df["n_female"], errors="coerce") == 0).any():
        rep.errors.append("some classes have n_male + n_female = 0")
    return rep


def validate_clean(df: pd.DataFrame, require_target: bool = True) -> None:
    problems = []
    if require_target and (TARGET not in df or df[TARGET].isna().any() or not df[TARGET].between(0, 100).all()):
        problems.append(f"{TARGET} must be present and within 0-100")
    if df["student_id"].duplicated().any():
        problems.append("student_id must be unique after merging duplicates")
    if not df["CCA"].isin(["Sports", "Clubs", "Arts", "None"]).all():
        problems.append("CCA must be one of Sports/Clubs/Arts/None")
    if not df["tuition"].dropna().isin(["Yes", "No"]).all():
        problems.append("tuition must be Yes/No")
    a = df["age"].dropna()
    if ((a < 14) | (a > 17)).any():
        problems.append("age outside the valid range after cleaning")
    for col in ["sleep_time", "wake_time"]:
        if not df[col].dropna().astype(str).str.match(TIME_PATTERN).all():
            problems.append(f"{col} not in H:MM format")
    if problems:
        raise SchemaError("Cleaned data failed validation: " + "; ".join(problems))
    logger.debug("Cleaned data passed schema validation (%d rows)", len(df))
