import numpy as np
import pytest

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors


class MockWord2VecModel:
    def __init__(self) -> None:
        self.vectors = {
            "hello": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "world": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "OOV": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        }
        self.key_to_index = {key: index for index, key in enumerate(self.vectors)}
        self.vector_size = 3

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]


class MockFastTextModel:
    def __init__(self) -> None:
        self.key_to_index = {"hello": 0}
        self.vector_size = 3

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "hello":
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

        return np.array([0.5, -0.5, 1.5], dtype=np.float32)


def test_tokens_to_sequence_vectors_returns_sequence_shape() -> None:
    model = MockWord2VecModel()

    vectors = tokens_to_sequence_vectors(["hello", "world"], model)

    assert vectors.shape == (2, model.vector_size)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[0], model["hello"])
    np.testing.assert_array_equal(vectors[1], model["world"])


def test_tokens_to_sequence_vectors_uses_oov_vector() -> None:
    model = MockWord2VecModel()

    vectors = tokens_to_sequence_vectors(["hello", "unknown"], model)

    assert vectors.shape == (2, model.vector_size)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[1], model["OOV"])


def test_tokens_to_sequence_vectors_handles_oov_only_without_nan() -> None:
    model = MockWord2VecModel()

    vectors = tokens_to_sequence_vectors(["unknown", "missing"], model)

    assert vectors.shape == (2, model.vector_size)
    assert vectors.dtype == np.float32
    assert not np.isnan(vectors).any()
    np.testing.assert_array_equal(vectors[0], model["OOV"])
    np.testing.assert_array_equal(vectors[1], model["OOV"])


def test_tokens_to_sequence_vectors_returns_zero_vector_for_empty_tokens() -> None:
    model = MockWord2VecModel()

    vectors = tokens_to_sequence_vectors([], model)

    assert vectors.shape == (1, model.vector_size)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(
        vectors,
        np.zeros((1, model.vector_size), dtype=np.float32),
    )


def test_tokens_to_sequence_vectors_uses_subword_vector_without_oov_token() -> None:
    model = MockFastTextModel()

    vectors = tokens_to_sequence_vectors(["unknown"], model)

    assert vectors.shape == (1, model.vector_size)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[0], np.array([0.5, -0.5, 1.5], dtype=np.float32))


def test_pad_sequence_vectors_returns_float32_batch() -> None:
    first_sequence = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second_sequence = np.array([[5.0, 6.0]], dtype=np.float32)

    batch = pad_sequence_vectors([first_sequence, second_sequence], vector_size=2)

    assert batch.shape == (2, 2, 2)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0], first_sequence)
    np.testing.assert_array_equal(batch[1, 0], second_sequence[0])


def test_pad_sequence_vectors_adds_padding_to_the_end() -> None:
    sequence = np.array([[1.0, 2.0]], dtype=np.float32)
    longer_sequence = np.array(
        [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=np.float32,
    )

    batch = pad_sequence_vectors([sequence, longer_sequence], vector_size=2)

    np.testing.assert_array_equal(batch[0, 0], sequence[0])
    np.testing.assert_array_equal(batch[0, 1:], np.zeros((2, 2), dtype=np.float32))


def test_pad_sequence_vectors_rejects_empty_sequence_list() -> None:
    with pytest.raises(ValueError):
        pad_sequence_vectors([], vector_size=3)
