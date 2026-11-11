import numpy as np
import pandas as pd
import pytest

from src.data import clean, merge_duplicates, split
from src.features import add_features, build_preprocessor

from .conftest import student


def test_cleaning_fixes_known_issues(raw):
    df, rep = clean(raw)
    assert df.student_id.is_unique
    assert set(df.CCA) <= {"Sports", "Clubs", "Arts", "None"}
    assert set(df.tuition) <= {"Yes", "No"}
    assert df.age.dropna().between(14, 17).all()
    assert df.final_test.notna().all()
    assert "bag_color" not in df.columns and "index" not in df.columns
    assert rep["duplicate_records_merged"] == int(raw.student_id.duplicated().sum())
    assert rep["conflicting_values"] == {}


def test_duplicates_merge_keeps_non_missing_values():
    a = student(final_test=np.nan, attendance_rate=90.0, bag_color="red")
    b = student(index=1, final_test=64.0, attendance_rate=np.nan, bag_color="blue")
    merged, rep = merge_duplicates(pd.concat([a, b], ignore_index=True))
    assert len(merged) == 1
    assert merged.final_test.iloc[0] == 64.0 and merged.attendance_rate.iloc[0] == 90.0
    assert rep["conflicting_values"] == {}       # bag colour differences are expected noise


def test_duplicates_with_real_conflict_are_reported():
    a, b = student(hours_per_week=5.0), student(index=1, hours_per_week=15.0)
    _, rep = merge_duplicates(pd.concat([a, b], ignore_index=True))
    assert rep["conflicting_values"] == {"hours_per_week": 1}


@pytest.mark.parametrize("raw_value, expected", [("CLUBS", "Clubs"), ("NONE", "None"), (np.nan, "None"),
                                                 ("arts", "Arts")])
def test_cca_normalised_and_missing_means_none(raw_value, expected):
    df, _ = clean(student(CCA=raw_value))
    assert df.CCA.iloc[0] == expected


@pytest.mark.parametrize("raw_age, expected", [(5.0, 15.0), (6.0, 16.0), (15.0, 15.0), (-5.0, np.nan)])
def test_age_typos_fixed_and_invalid_set_missing(raw_age, expected):
    df, _ = clean(student(age=raw_age))
    got = df.age.iloc[0]
    assert (np.isnan(got) and np.isnan(expected)) or got == expected


def test_tuition_abbreviations():
    assert clean(student(tuition="Y"))[0].tuition.iloc[0] == "Yes"
    assert clean(student(tuition="N"))[0].tuition.iloc[0] == "No"


@pytest.mark.parametrize("sleep, wake, hours, after_midnight", [("22:00", "6:00", 8.0, 0), ("21:30", "5:30", 8.0, 0),
                                                                ("1:00", "6:00", 5.0, 1), ("0:00", "7:00", 7.0, 1),
                                                                ("23:30", "6:30", 7.0, 0)])
def test_sleep_features_cross_midnight(sleep, wake, hours, after_midnight):
    df = add_features(clean(student(sleep_time=sleep, wake_time=wake))[0])
    assert df.sleep_hours.iloc[0] == hours and df.sleep_after_midnight.iloc[0] == after_midnight


def test_class_features():
    df = add_features(clean(student(n_male=15.0, n_female=5.0))[0])
    assert df.class_size.iloc[0] == 20 and df.male_ratio.iloc[0] == pytest.approx(0.75)


def test_preprocessor_handles_missing_and_unseen_values(raw, base_config_dict):
    cat, num = base_config_dict["features"]["categorical"], base_config_dict["features"]["numeric"]
    df = add_features(clean(raw)[0])
    for splines in (False, True):
        prep = build_preprocessor(cat, num, scale=True, splines=splines).fit(df[cat + num])
        new = df[cat + num].head(3).copy()
        new.loc[new.index[0], "mode_of_transport"] = "bicycle"
        new.loc[new.index[1], "attendance_rate"] = np.nan
        out = prep.transform(new)
        assert out.shape[0] == 3 and np.isfinite(out).all()


def test_split_stratifies_weak_students(raw, base_config_dict):
    df = add_features(clean(raw)[0])
    feats = base_config_dict["features"]["categorical"] + base_config_dict["features"]["numeric"]
    X_tr, X_te, y_tr, y_te = split(df, feats, 0.2, 42, 50)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs((y_tr < 50).mean() - (y_te < 50).mean()) < 0.01
