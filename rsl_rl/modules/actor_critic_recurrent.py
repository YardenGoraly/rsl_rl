# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch.nn as nn
import torch

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


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


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        self.num_height_scan_points = 651
        self.height_scan_size = (6, 4)
        self.height_scan_resolution = 0.2
        self.height_scan_x = int(round(self.height_scan_size[0] / self.height_scan_resolution)) + 1
        self.height_scan_y = int(round(self.height_scan_size[1] / self.height_scan_resolution)) + 1

        # Height scan normalization parameters
        # Should match your height_scan_clipped clip_height range (default: (-1.0, 0.5))
        # height_scan_normalize = kwargs.pop("height_scan_normalize", True)
        # height_scan_min = kwargs.pop("height_scan_min", -2.0)
        # height_scan_max = kwargs.pop("height_scan_max", 2.0)
        
        # self.CNN_encoder = HeightScanEncoder(
        #     input_shape=(self.height_scan_x, self.height_scan_y),
        #     out_features=34,
        #     normalize=height_scan_normalize,
        #     height_min=height_scan_min,
        #     height_max=height_scan_max,
        # )
        # self.encoder_out_dim = self.CNN_encoder.out_features

        # # Past positions encoder parameters
        # # Calculate from config: max_distance=10.0, interval=0.2 -> num_positions = int(10.0/0.2) + 1 = 51
        # num_past_positions = kwargs.pop("num_past_positions", 20) 
        # past_positions_out_features = kwargs.pop("past_positions_out_features", 16)  # Encoded feature size
        # past_positions_normalize = kwargs.pop("past_positions_normalize", True)
        # self.num_past_positions = num_past_positions
        # self.num_past_position_points = num_past_positions * 3  # Each position is (x, y, z)
        
        # self.position_encoder = PositionEncoder(
        #     num_positions=num_past_positions,
        #     out_features=past_positions_out_features,
        #     normalize=past_positions_normalize,
        # )
        # self.position_encoder_out_dim = self.position_encoder.out_features

        # # Calculate encoded observation dimensions
        # # Remove: height_scan (651) + past_positions (num_past_positions * 2)
        # # Add: CNN features (34) + position encoder features (past_positions_out_features)
        # encoded_actor_obs_dim = (
        #     num_actor_obs 
        #     - self.num_height_scan_points 
        #     - self.num_past_position_points
        #     + self.encoder_out_dim 
        #     + self.position_encoder_out_dim
        # )
        # encoded_critic_obs_dim = (
        #     num_critic_obs 
        #     - self.num_height_scan_points 
        #     - self.num_past_position_points
        #     + self.encoder_out_dim 
        #     + self.position_encoder_out_dim
        # )

        # self.memory_a = Memory(encoded_actor_obs_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        # self.memory_c = Memory(encoded_critic_obs_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        # observations = self.encode_observations(observations)
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        # observations = self.encode_observations(observations)
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # critic_observations = self.encode_observations(critic_observations)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def encode_observations(self, observations):
        """Encode height-scan and past positions using CNNs while keeping the rest intact.

        The height-scan segment (starting at index 34) is reshaped into a 2-D grid and passed through
        `HeightScanEncoder`. The past positions segment (after height-scan) is reshaped and passed
        through `PositionEncoder`. Both encoded features are spliced back into the observation vector.
        Time-major inputs are flattened to batch-major for the CNNs and reshaped afterward.
        """
        original_shape = observations.shape
        # Deal with end of episodes when observation dimension is 3
        if observations.dim() == 3:
            time_steps, batch_size, obs_dim = original_shape
            observations = observations.reshape(time_steps * batch_size, obs_dim)

        # Extract height-scan portion and reshape for 2D CNN
        height_scan_start_idx = 34
        height_scan_end_idx = height_scan_start_idx + self.num_height_scan_points
        CNN_input = (
            observations[:, height_scan_start_idx : height_scan_end_idx]
            .clone()
            .reshape(observations.shape[0], 1, self.height_scan_x, self.height_scan_y)
        )

        # Pass through height-scan CNN encoder
        encoded_height_scan = self.CNN_encoder(CNN_input)

        # Extract past positions portion (comes after height-scan)
        past_positions_start_idx = height_scan_end_idx
        past_positions_end_idx = past_positions_start_idx + self.num_past_position_points
        past_positions_input = observations[:, past_positions_start_idx : past_positions_end_idx].clone()

        # Pass through position encoder (1D CNN)
        encoded_past_positions = self.position_encoder(past_positions_input)

        # Concatenate: [before_height_scan, encoded_height_scan, encoded_past_positions, after_past_positions]
        observations = torch.cat(
            [
                observations[:, :height_scan_start_idx].clone(),  # Observations before height scan
                encoded_height_scan,  # Encoded height scan features
                encoded_past_positions,  # Encoded past positions features
                observations[:, past_positions_end_idx:].clone(),  # Observations after past positions
            ],
            dim=1,
        )

        if len(original_shape) == 3:
            observations = observations.reshape(time_steps, batch_size, observations.shape[-1])
        return observations
