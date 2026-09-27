import math

from ensemble.evaluation.metrics import average_precision_at_k, map_at_k, ndcg_at_k, recall_at_k, relative_lift


def test_ap_perfect_and_empty():
    assert average_precision_at_k([1, 2], {1, 2}) == 1.0
    assert average_precision_at_k([3, 4], {1}) == 0.0


def test_ap_rewards_early_hits():
    early = average_precision_at_k([1, 9, 9], {1})
    late = average_precision_at_k([9, 9, 1], {1})
    assert early == 1.0 and math.isclose(late, 1 / 3)


def test_ap_normalises_by_min_truth_k():
    # 20 true items, 12 hits in 12 slots → AP = 1
    assert average_precision_at_k(list(range(12)), set(range(20)), k=12) == 1.0


def test_ap_ignores_duplicate_predictions():
    assert average_precision_at_k([1, 1, 2], {1, 2}) < 1.0


def test_map_and_recall():
    preds = {"a": [1, 2], "b": [5]}
    truth = {"a": {1}, "b": {6, 7}}
    assert map_at_k(preds, truth) == 0.5
    assert recall_at_k(preds, truth) == 1 / 3


def test_ndcg_and_lift():
    assert ndcg_at_k([1], {1}, 5) == 1.0
    assert relative_lift(0.12, 0.10) == pytest_approx(0.2)


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
