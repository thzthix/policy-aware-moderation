import numpy as np
import pytest
import torch

from scripts.evaluate_toxicity_model import (
    _calculate_metrics_by_threshold,
    _calculate_pos_weight,
)


def test_calculate_metrics_by_threshold_compares_same_scores() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.float32)
    scores = np.asarray([0.1, 0.4, 0.8, 0.6], dtype=np.float32)

    metrics = _calculate_metrics_by_threshold(labels, scores, [0.5, 0.3])

    assert metrics[0.5]["precision"] == pytest.approx(0.5)
    assert metrics[0.5]["recall"] == pytest.approx(0.5)
    assert metrics[0.3]["precision"] == pytest.approx(2 / 3)
    assert metrics[0.3]["recall"] == pytest.approx(1.0)


def test_calculate_pos_weight_uses_negative_positive_ratio() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.float32)

    pos_weight = _calculate_pos_weight(labels)

    assert isinstance(pos_weight, torch.Tensor)
    assert pos_weight.item() == pytest.approx(3.0)
