import pytest
import torch

from src.moderation.model import GRUModel


def test_gru_model_returns_batch_output_shape() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=2)
    inputs = torch.randn(4, 6, 3, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (4, 1)
    assert outputs.dtype == torch.float32


def test_gru_model_handles_single_sequence_length() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    inputs = torch.randn(4, 1, 3, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (4, 1)


def test_gru_model_handles_single_batch() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    inputs = torch.randn(1, 6, 3, dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (1, 1)


def test_gru_model_rejects_wrong_input_size() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    inputs = torch.randn(2, 4, 2, dtype=torch.float32)

    with pytest.raises(RuntimeError):
        model(inputs)


def test_gru_model_rejects_non_float32_input() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    inputs = torch.ones(2, 4, 3, dtype=torch.float64)

    with pytest.raises(TypeError):
        model(inputs)


def test_gru_model_accepts_sentence_features() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1, feature_size=4)
    inputs = torch.randn(2, 4, 3, dtype=torch.float32)
    sentence_features = torch.randn(2, 4, dtype=torch.float32)

    outputs = model(inputs, sentence_features=sentence_features)

    assert outputs.shape == (2, 1)


def test_gru_model_requires_sentence_features_when_configured() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1, feature_size=4)
    inputs = torch.randn(2, 4, 3, dtype=torch.float32)

    with pytest.raises(ValueError):
        model(inputs)


def test_gru_model_rejects_sentence_features_for_baseline_model() -> None:
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    inputs = torch.randn(2, 4, 3, dtype=torch.float32)
    sentence_features = torch.randn(2, 4, dtype=torch.float32)

    with pytest.raises(ValueError):
        model(inputs, sentence_features=sentence_features)
