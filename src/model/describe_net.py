from __future__ import annotations

import torch
from src.config import conf


class DescribeNetWISDM(torch.nn.Module):
    def __init__(
        self, input_channels=3, hidden_dim=64, output_dim=len(conf.subject_ids)
    ):
        super().__init__()

        lstm_hidden = 128

        self.conv1 = torch.nn.Conv1d(
            input_channels, hidden_dim, kernel_size=5, stride=1, padding=2
        )
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(
            hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2
        )
        self.norm2 = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv1d(
            hidden_dim * 2, lstm_hidden, kernel_size=5, stride=1, padding=2
        )
        self.norm3 = torch.nn.BatchNorm1d(lstm_hidden)
        self.relu3 = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=False,
            dropout=0.1,
        )

        self.L1 = torch.nn.Linear(lstm_hidden, hidden_dim)
        self.norm4 = torch.nn.LayerNorm(hidden_dim)
        self.relu4 = torch.nn.ReLU()

        self.L2 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm5 = torch.nn.LayerNorm(hidden_dim * 2)
        self.relu5 = torch.nn.ReLU()

        self.L3 = torch.nn.Linear(hidden_dim * 2, output_dim)

        self.to(conf.device)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))

        # (B, C, T) -> (B, T, C) for LSTM
        x = x.transpose(1, 2)
        seq, _ = self.lstm(x)
        x = seq[:, -1, :]

        x = self.relu4(self.norm4(self.L1(x)))
        x = self.relu5(self.norm5(self.L2(x)))
        return self.L3(x)
