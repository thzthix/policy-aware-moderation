from __future__ import annotations

import torch
from torch import nn


class GRUModel(nn.Module):
    """시퀀스 벡터를 입력받아 독성 점수를 예측한다."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        feature_size: int = 0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size + feature_size, output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        sentence_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.dtype != torch.float32:
            raise TypeError("inputs는 torch.float32여야 합니다.")

        if sentence_features is not None and sentence_features.dtype != torch.float32:
            raise TypeError("sentence_features는 torch.float32여야 합니다.")

        if self.feature_size == 0 and sentence_features is not None:
            raise ValueError("이 모델은 sentence_features를 사용하지 않습니다.")

        if self.feature_size > 0 and sentence_features is None:
            raise ValueError("sentence_features가 필요합니다.")

        if sentence_features is not None:
            expected_shape = (inputs.shape[0], self.feature_size)
            if sentence_features.shape != expected_shape:
                raise ValueError("sentence_features 크기가 올바르지 않습니다.")

        _, hidden = self.gru(inputs)
        last_hidden_state = hidden[-1]
        if sentence_features is not None:
            last_hidden_state = torch.cat([last_hidden_state, sentence_features], dim=1)

        return self.fc(last_hidden_state)
