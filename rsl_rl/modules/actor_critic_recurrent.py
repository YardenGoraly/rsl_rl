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

        self.conv2 = nn.Conv2d(1, 3, kernel_size=3, dilation=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )
        # self.shortcut = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=True),
        #     nn.BatchNorm2d(1),
        # )
        self.flat_dim = 3
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.flat_dim, out_features)
        self.out_features = out_features

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        original_input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.shortcut(original_input)
        x = self.relu2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def _conv_output_dim(
        size: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> int:
        return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


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
        # self.actor_obs_proj = self._build_projection(encoded_actor_obs_dim, num_actor_obs)
        # self.critic_obs_proj = self._build_projection(encoded_critic_obs_dim, num_critic_obs)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        # observations = self.encode_observations(observations)
        # projected_actor_obs = self.actor_obs_proj(observations)
        # input_a = self.memory_a(projected_actor_obs, masks, hidden_states)
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        # observations = self.encode_observations(observations)
        # projected_actor_obs = self.actor_obs_proj(observations)
        # input_a = self.memory_a(projected_actor_obs)
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # critic_observations = self.encode_observations(critic_observations)
        # projected_critic_obs = self.critic_obs_proj(critic_observations)
        # input_c = self.memory_c(projected_critic_obs, masks, hidden_states)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def encode_observations(self, observations):
        original_shape = observations.shape
        if observations.dim() == 3:
            time_steps, batch_size, obs_dim = original_shape
            observations = observations.reshape(time_steps * batch_size, obs_dim)

        CNN_input = (
            observations[:, 34 : 34 + self.num_height_scan_points]
            .clone()
            .reshape(observations.shape[0], 1, self.height_scan_x, self.height_scan_y)
        )
        recurrent_height_scan_input = self.CNN_encoder(CNN_input)
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

    @staticmethod
    def _build_projection(in_dim: int, out_dim: int) -> nn.Module:
        if in_dim == out_dim:
            return nn.Identity()
        return nn.Linear(in_dim, out_dim)
