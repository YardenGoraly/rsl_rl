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
        
        # Normalization parameters - register as buffers to ensure correct device placement
        self.normalize = normalize
        self.register_buffer("height_min", torch.tensor(height_min, dtype=torch.float32))
        self.register_buffer("height_max", torch.tensor(height_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize height scan to [-1, 1] range
        # For height scans with range [height_min, height_max], normalize to [-1, 1]:
        # normalized = 2 * (x - height_min) / (height_max - height_min) - 1
        if self.normalize:
            # Check for NaN/Inf values that could break gradients
            # Replace NaN/Inf with the midpoint of the expected range to prevent gradient issues
            if torch.any(~torch.isfinite(x)):
                midpoint = (self.height_min + self.height_max) / 2.0
                x = torch.where(torch.isfinite(x), x, midpoint)
            
            # Clamp to expected range first to handle outliers
            x_clamped = torch.clamp(x, min=self.height_min, max=self.height_max)
            # Normalize to [-1, 1]
            range_size = self.height_max - self.height_min
            # Use a small epsilon to avoid numerical instability with very small ranges
            eps = 1e-8
            if range_size > eps:
                x_normalized = 2.0 * (x_clamped - self.height_min) / range_size - 1.0
            else:
                # If range is too small, just center around zero
                x_normalized = x_clamped - self.height_min
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


class SpatialSelfAttention(nn.Module):
    """Self-attention layer for spatial feature maps.
    
    Enriches each spatial feature with global context by computing attention weights
    across spatial dimensions. Given a feature map F_t ∈ R^(C × H × W), produces
    a refined feature map F'_t with the same dimensions.
    
    Args:
        channels: Number of channels C in the feature map
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [batch, C, H, W]
        Returns:
            Refined feature map [batch, C, H, W]
        """
        batch_size, C, H, W = x.shape
        
        # Reshape to [batch, H*W, C] for attention computation
        x_flat = x.view(batch_size, C, H * W).permute(0, 2, 1)  # [batch, H*W, C]
        
        # Store residual
        residual = x_flat
        
        # Layer norm
        x_flat = self.norm(x_flat)
        
        # Compute Q, K, V
        qkv = self.qkv(x_flat)  # [batch, H*W, 3*C]
        qkv = qkv.reshape(batch_size, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [batch, num_heads, H*W, H*W]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [batch, num_heads, H*W, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, H * W, C)  # [batch, H*W, C]
        
        # Project and add residual
        out = self.proj(out)
        out = self.dropout(out)
        out = out + residual
        
        # Reshape back to [batch, C, H, W]
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)
        
        return out


class SpatialCrossAttention(nn.Module):
    """Cross-attention layer for compressing spatial feature maps.
    
    Uses a query derived from proprioceptive state and goal position to compress
    a 2D feature map F'_t ∈ R^(C × H × W) into a 1D representation F̂_t ∈ R^(C × 1).
    
    Args:
        feature_channels: Number of channels C in the feature map
        query_dim: Dimension of the query vector (proprioceptive + goal features)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, feature_channels: int, query_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_channels = feature_channels
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        
        assert feature_channels % num_heads == 0, \
            f"feature_channels ({feature_channels}) must be divisible by num_heads ({num_heads})"
        
        # Project query to match feature channels
        self.query_proj = nn.Linear(query_dim, feature_channels)
        
        # Linear projections for K, V from features
        self.kv = nn.Linear(feature_channels, feature_channels * 2, bias=False)
        
        # Output projection
        self.proj = nn.Linear(feature_channels, feature_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature map F'_t [batch, C, H, W]
            query: Query vector from proprioceptive state and goal [batch, query_dim]
        Returns:
            Compressed feature F̂_t [batch, C, 1]
        """
        batch_size, C, H, W = features.shape
        
        # Reshape features to [batch, H*W, C]
        features_flat = features.view(batch_size, C, H * W).permute(0, 2, 1)  # [batch, H*W, C]
        
        # Project query: [batch, query_dim] -> [batch, 1, C]
        q = self.query_proj(query).unsqueeze(1)  # [batch, 1, C]
        
        # Reshape query for multi-head attention
        q = q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, 1, head_dim]
        
        # Compute K, V from features
        kv = self.kv(features_flat)  # [batch, H*W, 2*C]
        kv = kv.reshape(batch_size, H * W, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, batch, num_heads, H*W, head_dim]
        k, v = kv[0], kv[1]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [batch, num_heads, 1, H*W]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [batch, num_heads, 1, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, 1, C)  # [batch, 1, C]
        
        # Project
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape to [batch, C, 1] for consistency with feature map format
        out = out.permute(0, 2, 1)  # [batch, C, 1]
        
        return out


class AttentionFeatureCompressor(nn.Module):
    """Complete attention-based feature compression module.
    
    Combines self-attention and cross-attention layers to process and compress
    spatial feature maps according to the paper's architecture.
    
    Args:
        feature_channels: Number of channels C in the input feature map
        query_dim: Dimension of the query vector (proprioceptive + goal features)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, feature_channels: int, query_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attention = SpatialSelfAttention(feature_channels, num_heads, dropout)
        self.cross_attention = SpatialCrossAttention(feature_channels, query_dim, num_heads, dropout)
        
    def forward(self, features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input feature map F_t [batch, C, H, W]
            query: Query vector [batch, query_dim]
        Returns:
            Compressed feature F̂_t [batch, C, 1]
        """
        # Self-attention: enrich spatial features with global context
        features_refined = self.self_attention(features)  # [batch, C, H, W]
        
        # Cross-attention: compress to 1D using query
        features_compressed = self.cross_attention(features_refined, query)  # [batch, C, 1]
        
        return features_compressed

