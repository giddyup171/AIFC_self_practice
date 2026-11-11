"""Feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def to_minutes(times: pd.Series) -> pd.Series:
    parts = times.astype("string").str.strip().str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features.

    sleep_hours           wake - sleep; bedtimes before noon are taken as after midnight (01:00 -> 25:00)
    sleep_after_midnight  1 if the student goes to bed at or after 00:00
    class_size            n_male + n_female
    male_ratio            n_male / class_size
    """
    df = df.copy()
    sleep = to_minutes(df["sleep_time"])
    wake = to_minutes(df["wake_time"])
    sleep = sleep.where(sleep >= 12 * 60, sleep + 24 * 60)
    df["sleep_hours"] = (wake + 24 * 60 - sleep) / 60
    df["sleep_after_midnight"] = (sleep >= 24 * 60).astype(float).where(sleep.notna())
    df["class_size"] = df["n_male"] + df["n_female"]
    df["male_ratio"] = (df["n_male"] / df["class_size"].replace(0, np.nan))
    return df


def make_spline(n_knots: int) -> SplineTransformer:
    """Cubic B-splines with evenly spaced knots (quantile knots fail on discrete features such as siblings)."""
    return SplineTransformer(n_knots=n_knots, degree=3, knots="uniform", extrapolation="constant")


def build_preprocessor(categorical: list[str], numeric: list[str], scale: bool,
                       splines: bool = False, n_knots: int = 5) -> ColumnTransformer:
    """Categorical: most frequent imputation, one-hot (unknown levels ignored).
    Numeric: median imputation with missing indicators; optional cubic splines (lets a linear model bend);
    scaling for linear models. Fitted inside each model pipeline, so it is re-fitted in every CV fold.
    """
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if splines:
        steps.append(("splines", make_spline(n_knots)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("cat", cat, categorical), ("num", Pipeline(steps), numeric)])
