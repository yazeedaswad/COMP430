from pill_counting.evaluation.counting import compute_counting_metrics


def test_compute_counting_metrics() -> None:
    metrics = compute_counting_metrics([3, 5, 2], [3, 4, 2])

    assert metrics.mae == 1 / 3
    assert metrics.mse == 1 / 3
    assert metrics.exact_count_accuracy == 2 / 3

