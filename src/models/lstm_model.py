import torch
from torch import nn


class LSTMPredictor(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        x = x.reshape(b * s, f)
        x = self.bn(x)
        x = x.reshape(b, s, f)
        out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden).squeeze(-1)
