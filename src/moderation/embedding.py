import numpy as np


def tokens_to_sequence_vectors(
    tokens: list[str],
    model,
    oov_token: str = "OOV",
) -> np.ndarray:
    """토큰 목록을 벡터 시퀀스로 변환한다."""
    vector_size = model.vector_size

    if not tokens:
        return np.zeros((1, vector_size), dtype=np.float32)

    vectors = [_get_token_vector(token, model, oov_token) for token in tokens]
    sequence_vectors = np.vstack(vectors)
    return np.nan_to_num(sequence_vectors, nan=0.0).astype(np.float32, copy=False)


def pad_sequence_vectors(sequences: list[np.ndarray], vector_size: int) -> np.ndarray:
    """벡터 시퀀스 목록을 같은 길이로 패딩한다."""
    if not sequences:
        raise ValueError("sequences는 비어 있을 수 없습니다.")

    max_sequence_length = max(sequence.shape[0] for sequence in sequences)
    batch_vectors = np.zeros(
        (len(sequences), max_sequence_length, vector_size),
        dtype=np.float32,
    )

    for index, sequence in enumerate(sequences):
        _validate_sequence(sequence, vector_size)
        sequence_length = sequence.shape[0]
        batch_vectors[index, :sequence_length, :] = np.nan_to_num(
            sequence,
            nan=0.0,
        ).astype(np.float32, copy=False)

    return batch_vectors


def _get_token_vector(token: str, model, oov_token: str) -> np.ndarray:
    vector_key = token if token in model.key_to_index else oov_token
    if vector_key not in model.key_to_index:
        raise ValueError("임베딩 모델에 OOV 토큰 벡터가 없습니다.")

    vector = np.asarray(model[vector_key], dtype=np.float32)
    if vector.shape != (model.vector_size,):
        raise ValueError("임베딩 벡터 크기가 model.vector_size와 다릅니다.")

    return np.nan_to_num(vector, nan=0.0).astype(np.float32, copy=False)


def _validate_sequence(sequence: np.ndarray, vector_size: int) -> None:
    if sequence.ndim != 2:
        raise ValueError("sequence는 2차원 배열이어야 합니다.")

    if sequence.shape[1] != vector_size:
        raise ValueError("sequence 벡터 크기가 vector_size와 다릅니다.")
