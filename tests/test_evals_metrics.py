"""Unit tests for cip.evals.metrics — the first real test suite in this repo."""

from cip.evals.metrics import score_binary


def test_perfect_predictions():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    result = score_binary(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 3]]


def test_all_wrong_predictions():
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]
    result = score_binary(y_true, y_pred)
    assert result["accuracy"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_confusion_matrix_layout():
    # 1 true negative, 1 false positive, 1 false negative, 2 true positives
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    result = score_binary(y_true, y_pred)
    assert result["confusion_matrix"] == [[1, 1], [1, 2]]


def test_zero_division_does_not_raise():
    # model predicts all-negative: precision is undefined (0 predicted positives),
    # must return 0.0 rather than raising
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    result = score_binary(y_true, y_pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
