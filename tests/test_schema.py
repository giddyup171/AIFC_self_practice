import pytest

from src.data import clean
from src.schema import SchemaError, validate_clean, validate_raw


def test_real_sample_passes_with_quality_warnings(raw):
    rep = validate_raw(raw)
    assert rep.ok, rep.errors
    assert rep.warnings["duplicate_student_records (merged)"] == int(raw.student_id.duplicated().sum())
    assert rep.warnings["tuition_abbreviated (Y/N)"] == int(raw.tuition.isin(["Y", "N"]).sum())
    assert rep.warnings["CCA_missing (means no CCA)"] == int(raw.CCA.isna().sum())


def test_missing_column_is_fatal(raw):
    rep = validate_raw(raw.drop(columns=["hours_per_week"]))
    assert not rep.ok and "hours_per_week" in rep.errors[0]
    with pytest.raises(SchemaError):
        rep.raise_if_failed()


def test_target_optional_for_scoring(raw):
    assert validate_raw(raw.drop(columns=["final_test"]), require_target=False).ok
    assert not validate_raw(raw.drop(columns=["final_test"]), require_target=True).ok


@pytest.mark.parametrize("column, value, message", [
    ("final_test", 140.0, "outside 0-100"),
    ("sleep_time", "late", "H:MM"),
    ("wake_time", "25:00", "H:MM"),
    ("attendance_rate", "high", "non-numeric"),
])
def test_bad_values_are_fatal(raw, column, value, message):
    bad = raw.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[bad.index[0], column] = value
    assert any(message in e for e in validate_raw(bad).errors)


def test_empty_class_and_missing_id_are_fatal(raw):
    bad = raw.copy(); bad.loc[bad.index[0], ["n_male", "n_female"]] = 0
    assert any("n_male + n_female = 0" in e for e in validate_raw(bad).errors)
    bad = raw.copy(); bad.loc[bad.index[0], "student_id"] = None
    assert any("student_id" in e for e in validate_raw(bad).errors)


def test_unknown_category_is_a_warning(raw):
    bad = raw.copy(); bad.loc[bad.index[:2], "mode_of_transport"] = "bicycle"
    rep = validate_raw(bad)
    assert rep.ok and rep.warnings["mode_of_transport_unknown_category"] == 2


def test_validate_clean_catches_cleaning_bugs(raw):
    df, _ = clean(raw)
    validate_clean(df)
    for col, value, msg in [("tuition", "Y", "tuition"), ("CCA", "CLUBS", "CCA"), ("age", 5, "age"),
                            ("final_test", None, "final_test")]:
        broken = df.copy(); broken[col] = broken[col].astype(object); broken.loc[0, col] = value
        with pytest.raises(SchemaError, match=msg):
            validate_clean(broken)
    dup = df.copy(); dup.loc[1, "student_id"] = dup.loc[0, "student_id"]
    with pytest.raises(SchemaError, match="unique"):
        validate_clean(dup)
