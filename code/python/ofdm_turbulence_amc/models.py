"""1-D residual AMC model with optional SE attention."""

from __future__ import annotations

import torch
from torch import nn


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_se: bool):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 9, stride=stride, padding=4, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SE1D(out_channels) if use_se else nn.Identity()
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


class AMCResNetSE(nn.Module):
    def __init__(self, num_classes: int = 6, use_se: bool = True):
        super().__init__()
        self.backbone = AMCResNetBackbone(2, use_se=use_se)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class GAFCNN(nn.Module):
    """2-D CNN for GAF images."""

    def __init__(self, num_classes: int = 6, in_channels: int = 4):
        super().__init__()
        layers = []
        current_channels = in_channels
        for out_channels in (32, 64, 128, 192):
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels, out_channels, 3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            current_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class IQAmplitudePhaseNet(nn.Module):
    """Dual-branch 1-D model: raw IQ branch + amplitude/phase branch."""

    def __init__(self, num_classes: int = 6, use_se: bool = True):
        super().__init__()
        self.iq_branch = AMCResNetBackbone(2, use_se=use_se)
        self.ap_branch = AMCResNetBackbone(2, use_se=use_se)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iq = self.iq_branch(x[:, :2, :])
        ap = self.ap_branch(x[:, 2:, :])
        return self.classifier(torch.cat([iq, ap], dim=1))


class MultiScaleAMCNet(nn.Module):
    """Single-branch 1-D model with parallel multi-scale stem kernels."""

    def __init__(self, num_classes: int = 6, in_channels: int = 2, use_se: bool = True):
        super().__init__()
        self.stem = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, 16, kernel, stride=4, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True),
                )
                for kernel in (7, 15, 31, 73)
            ]
        )
        # Large-kernel branch output 32 channels to compensate for more aggressive downsampling
        self.stem_large = nn.Sequential(
            nn.Conv1d(in_channels, 24, 73, stride=2, padding=36, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 32, 37, stride=2, padding=18, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.stem_proj = nn.Sequential(
            nn.Conv1d(96, 48, 1, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(48, 64, 1, use_se),
            ResidualBlock(64, 64, 1, use_se),
            ResidualBlock(64, 128, 2, use_se),
            ResidualBlock(128, 128, 1, use_se),
            ResidualBlock(128, 256, 2, use_se),
            ResidualBlock(256, 256, 1, use_se),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.stem], dim=1)
        large_branch = self.stem_large(x)
        # Resample large_branch to match multi_scale length (need to align dimensions)
        if large_branch.shape[-1] != multi_scale.shape[-1]:
            large_branch = torch.nn.functional.interpolate(
                large_branch, size=multi_scale.shape[-1], mode='nearest'
            )
        features = torch.cat([multi_scale, large_branch], dim=1)
        features = self.stem_proj(self.pool(features))
        return self.classifier(self.backbone(features))


class AMCResNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, use_se: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32, 1, use_se),
            ResidualBlock(32, 64, 2, use_se),
            ResidualBlock(64, 64, 1, use_se),
            ResidualBlock(64, 128, 2, use_se),
            ResidualBlock(128, 128, 1, use_se),
            ResidualBlock(128, 256, 2, use_se),
            ResidualBlock(256, 256, 1, use_se),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
