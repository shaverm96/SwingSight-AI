from swingsight.metrics import compute_swing_metrics


def test_compute_swing_metrics_returns_keys():
    metrics = compute_swing_metrics([], {})
    assert "spine_angle_deg" in metrics
    assert "balance_score" in metrics
