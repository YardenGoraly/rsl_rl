# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            padding_mode="replicate",
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class HeightScanEncoder(nn.Module):
    """CNN encoder for the height-scan observation."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        out_features: int,
        normalize: bool = True,
        height_min: float = -1.0,
        height_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = ResidualBlock(1, 4, stride=2, dilation=1)
        self.block2 = ResidualBlock(4, 8, stride=2, dilation=2)
        self.block3 = ResidualBlock(8, 16, stride=2, dilation=3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(16, out_features)
        self.out_features = out_features
        
        # Normalization parameters
        self.normalize = normalize
        self.height_min = height_min  # Minimum expected height scan value
        self.height_max = height_max  # Maximum expected height scan value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize height scan to [-1, 1] range
        # For height scans with range [height_min, height_max], normalize to [-1, 1]:
        # normalized = 2 * (x - height_min) / (height_max - height_min) - 1
        if self.normalize:
            # Clamp to expected range first to handle outliers
            x_clamped = torch.clamp(x, min=self.height_min, max=self.height_max)
            # Normalize to [-1, 1]
            range_size = self.height_max - self.height_min
            if range_size > 0:
                x_normalized = 2.0 * (x_clamped - self.height_min) / range_size - 1.0
            else:
                x_normalized = x_clamped  # Avoid division by zero
            x = x_normalized
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class PositionEncoder(nn.Module):
    """1D CNN encoder for past position history.
    
    Processes past positions as a temporal sequence using 1D convolutions.
    Input: flattened position history [batch, num_positions * 3] where each position is (x, y, z)
    Output: encoded features [batch, out_features]
    """

    def __init__(self, num_positions: int, out_features: int, normalize: bool = True) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.out_features = out_features
        self.normalize = normalize
        
        # Process x, y, and z coordinates as separate channels (3 channels: x, y, z)
        # Architecture: multiple 1D conv layers with increasing channels
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1, padding_mode="replicate", bias=True)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, padding_mode="replicate", bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode="replicate", bias=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Global max pooling to get fixed-size representation
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Final MLP layer
        self.fc = nn.Linear(64, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened position history [batch, num_positions * 3]
        Returns:
            Encoded features [batch, out_features]
        """
        # Reshape from [batch, num_positions * 3] to [batch, 3, num_positions]
        # Treating x, y, and z as separate channels for the 1D conv
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, self.num_positions)
        
        # Optional normalization (similar to height scan)
        if self.normalize:
            # Normalize each channel independently to [-1, 1]
            # Assuming positions are in reasonable range (e.g., [-10, 10] meters)
            x = torch.clamp(x, min=-10.0, max=10.0)
            x = x / 10.0  # Normalize to [-1, 1]
        
        # 1D CNN layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        # Global max pooling: [batch, 64, num_positions] -> [batch, 64, 1]
        out = self.maxpool(out)
        
        # Flatten and final FC layer: [batch, 64, 1] -> [batch, 64] -> [batch, out_features]
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

