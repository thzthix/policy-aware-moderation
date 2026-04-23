import torch
from torch import nn


class GRUModel(nn.Module):
    """시퀀스 벡터를 입력받아 독성 점수를 예측한다."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.float32:
            raise TypeError("inputs는 torch.float32여야 합니다.")

        output, hidden = self.gru(inputs)
        last_hidden_state = hidden[-1]
        return self.fc(last_hidden_state)
