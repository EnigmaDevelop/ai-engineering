"""Unit tests for cip.evals.golden_set — loader validation and ambiguous-row handling."""

import pandas as pd
import pytest

from cip.evals.golden_set import load_golden_set


def write_csv(tmp_path, rows: list[dict]):
    path = tmp_path / "golden.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def base_row(**overrides) -> dict:
    row = {
        "complaint_id": 1,
        "product": "Credit card",
        "narrative": "text",
        "churn_signal": 1,
        "ambiguous": False,
    }
    row.update(overrides)
    return row


def test_loads_valid_set(tmp_path):
    path = write_csv(tmp_path, [base_row(), base_row(complaint_id=2, churn_signal=0)])
    df = load_golden_set(path)
    assert len(df) == 2
    assert df["churn_signal"].tolist() == [1, 0]
    assert df.attrs["n_ambiguous_dropped"] == 0


def test_drops_ambiguous_rows(tmp_path):
    path = write_csv(
        tmp_path,
        [base_row(), base_row(complaint_id=2, churn_signal=None, ambiguous=True)],
    )
    df = load_golden_set(path)
    assert len(df) == 1
    assert df.attrs["n_ambiguous_dropped"] == 1


def test_rejects_missing_columns(tmp_path):
    path = write_csv(tmp_path, [{"complaint_id": 1, "narrative": "text"}])
    with pytest.raises(ValueError, match="missing required columns"):
        load_golden_set(path)


def test_rejects_unlabeled_rows(tmp_path):
    path = write_csv(tmp_path, [base_row(), base_row(complaint_id=2, churn_signal=None)])
    with pytest.raises(ValueError, match="unlabeled"):
        load_golden_set(path)


def test_rejects_out_of_range_labels(tmp_path):
    path = write_csv(tmp_path, [base_row(churn_signal=2)])
    with pytest.raises(ValueError, match="must be 0 or 1"):
        load_golden_set(path)
