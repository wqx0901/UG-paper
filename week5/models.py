"""
手写 MLP 与 LSTM 模型（继承 nn.Module）
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机：用于序列到标量/向量的回归"""

    def __init__(self, input_dim, hidden_dims=(64, 32), output_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len) -> flatten 或 (batch, input_dim)
        out = self.encoder(x)
        return self.fc_out(out)


class LSTM(nn.Module):
    """LSTM 序列模型：用于时序预测"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_dim=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        mult = 2 if bidirectional else 1
        self.fc_out = nn.Linear(hidden_size * mult, output_dim)

    def forward(self, x):
        # x: (batch, seq_len) 或 (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # 取最后时间步
        out = out[:, -1, :]
        return self.fc_out(out)
