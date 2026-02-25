from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class StormLSTM(nn.Module):
    """
    Single-layer LSTM that ingests a short temporal sequence of
    feature vectors (CNN features + entropy + drift rate + duration)
    and outputs a geomagnetic storm probability via sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        sequence: (batch, time, features)
        """
        output, (hn, _cn) = self.lstm(sequence)
        last_hidden = hn[-1]  # (batch, hidden_dim)
        logits = self.fc(last_hidden)
        prob = self.activation(logits)
        return prob.squeeze(-1)


def build_storm_lstm(input_dim: int) -> StormLSTM:
    return StormLSTM(input_dim=input_dim)

