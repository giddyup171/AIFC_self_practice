import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_students.csv"


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """~1,600 real records, including duplicate students and the original quality problems."""
    return pd.read_csv(FIXTURE)


@pytest.fixture
def raw(raw_sample) -> pd.DataFrame:
    return raw_sample.copy()


@pytest.fixture(scope="session")
def base_config_dict() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def write_config(tmp_path, base_config_dict):
    def _write(changes: dict | None = None, name: str = "cfg.yaml") -> Path:
        from src.config import deep_merge
        cfg = deep_merge(copy.deepcopy(base_config_dict), changes or {})
        p = tmp_path / name
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return p
    return _write


def student(**overrides) -> pd.DataFrame:
    row = {"index": 0, "number_of_siblings": 1, "direct_admission": "No", "CCA": "Sports",
           "learning_style": "Visual", "student_id": "AAA111", "gender": "Female", "tuition": "Yes",
           "final_test": 70.0, "n_male": 10.0, "n_female": 10.0, "age": 15.0, "hours_per_week": 8.0,
           "attendance_rate": 95.0, "sleep_time": "22:00", "wake_time": "6:00",
           "mode_of_transport": "walk", "bag_color": "red"}
    row.update(overrides)
    return pd.DataFrame([row])
