"""Variable-length 1-D models for raw waveform classification."""

from __future__ import annotations

import torch
from torch import nn


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.se(self.main(x)) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = 5, use_se: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            ResidualBlock1D(32, 32, use_se=use_se),
            ResidualBlock1D(32, 64, stride=2, use_se=use_se),
            ResidualBlock1D(64, 64, use_se=use_se),
            ResidualBlock1D(64, 128, stride=2, use_se=use_se),
            ResidualBlock1D(128, 128, use_se=use_se),
            ResidualBlock1D(128, 256, stride=2, use_se=use_se),
            ResidualBlock1D(256, 256, use_se=use_se),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


class RawCNN1D(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=9,
                        stride=2,
                        padding=4,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, num_classes: int = 5) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "cnn1d":
        return RawCNN1D(num_classes)
    if normalized == "resnet1d":
        return ResNet1D(num_classes, use_se=False)
    if normalized in {"resnet_se", "resnet1d_se", "resnet_se1d"}:
        return ResNet1D(num_classes, use_se=True)
    raise ValueError(f"Unknown model: {name}")

