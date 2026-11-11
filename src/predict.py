"""Score new students with a saved model."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .artifacts import load_model
from .data import clean
from .features import add_features
from .schema import TARGET, validate_raw

logger = logging.getLogger(__name__)


def score(raw: pd.DataFrame, run_dir: str | Path, valid_age_range=(14, 17)) -> pd.DataFrame:
    """Validate, clean and score raw student records (same format as the training file, target optional).

    Returns student_id, predicted score and a 0/1 flag for students predicted below the model's cut-off,
    sorted from lowest predicted score (highest priority for support).
    """
    model, card = load_model(run_dir)
    report = validate_raw(raw, require_target=False).raise_if_failed()
    for k, v in report.warnings.items():
        logger.info("Input quality: %-42s %s", k, f"{v:,}")
    df, _ = clean(raw.drop(columns=[TARGET], errors="ignore"), tuple(valid_age_range), require_target=False)
    df = add_features(df)
    features = card["features"]["categorical"] + card["features"]["numeric"]
    pred = model.predict(df[features]).clip(0, 100)
    out = pd.DataFrame({"student_id": df["student_id"].to_numpy(), "predicted_score": pred.round(1),
                        "flag_for_support": (pred < card["flag_cutoff"]).astype(int)})
    out = out.sort_values("predicted_score").reset_index(drop=True)
    logger.info("Scored %s students with %s (run %s); %s flagged (predicted < %.1f)", f"{len(out):,}",
                card["model_name"], card["run_id"], f"{out.flag_for_support.sum():,}", card["flag_cutoff"])
    return out
