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


class HeightScanEncoder(nn.Module):
    """CNN encoder for the height-scan observation."""

    def __init__(self, input_shape: tuple[int, int], out_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, dilation=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, dilation=3, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.ReLU(inplace=True)

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=13, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.flat_dim = 3
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.flat_dim, out_features)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = out1 + self.shortcut1(x)
        out2 = self.relu1(out1)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = out2 + self.shortcut2(out1)
        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = out3 + self.shortcut3(out2)
        out3 = self.relu3(out3)
        out3 = self.pool(out3)
        out3 = torch.flatten(out3, 1)
        out3 = self.fc(out3)
        return out3


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

        self.num_height_scan_points = 2501
        self.height_scan_size = (6, 4)
        self.height_scan_resolution = 0.1
        self.height_scan_x = int(round(self.height_scan_size[0] / self.height_scan_resolution)) + 1
        self.height_scan_y = int(round(self.height_scan_size[1] / self.height_scan_resolution)) + 1

        self.CNN_encoder = HeightScanEncoder(
            input_shape=(self.height_scan_x, self.height_scan_y),
            out_features=34,
        )
        self.encoder_out_dim = self.CNN_encoder.out_features

        encoded_actor_obs_dim = num_actor_obs - self.num_height_scan_points + self.encoder_out_dim
        encoded_critic_obs_dim = num_critic_obs - self.num_height_scan_points + self.encoder_out_dim

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
        """Replace the flattened height-scan portion with CNN features while keeping the rest intact.

        The segment starting at index 34 is reshaped into a 2-D grid, passed through
        `HeightScanEncoder`, and the resulting features are spliced back into the
        observation vector. Time-major inputs are flattened to batch-major for the
        CNN and reshaped afterward.
        """
        original_shape = observations.shape
        # Deal with end of episodes when observation dimension is 3
        if observations.dim() == 3:
            time_steps, batch_size, obs_dim = original_shape
            observations = observations.reshape(time_steps * batch_size, obs_dim)

        # Extract height-scan portion and reshape for CNN
        CNN_input = (
            observations[:, 34 : 34 + self.num_height_scan_points]
            .clone()
            .reshape(observations.shape[0], 1, self.height_scan_x, self.height_scan_y)
        )

        # Pass through CNN encoder
        recurrent_height_scan_input = self.CNN_encoder(CNN_input)

        # Concatenate with original observations
        observations = torch.cat(
            [
                observations[:, :34].clone(),
                recurrent_height_scan_input,
                observations[:, 34 + self.num_height_scan_points :].clone(),
            ],
            dim=1,
        )

        if len(original_shape) == 3:
            observations = observations.reshape(time_steps, batch_size, observations.shape[-1])
        return observations
