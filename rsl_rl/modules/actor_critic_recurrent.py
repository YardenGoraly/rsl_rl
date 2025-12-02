# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch.nn as nn
import torch

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import (
    Memory,
    ResidualBlock,
    HeightScanEncoder,
    PositionEncoder,
    AttentionFeatureCompressor,
)
from rsl_rl.utils import resolve_nn_activation


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

        # Attention-based encoding parameters
        self.use_attention_encoding = kwargs.pop("use_attention_encoding", False)
        self.use_position_encoding = kwargs.pop("use_position_encoding", False)
        self.use_height_scan_encoding = kwargs.pop("use_height_scan_encoding", True)
        if self.use_attention_encoding:
            # Observation indices (configurable via kwargs)
            self.height_scan_start_idx = kwargs.pop("height_scan_start_idx", 34)
            self.proprioception_start_idx = kwargs.pop("proprioception_start_idx", 0)
            self.proprioception_dim = kwargs.pop("proprioception_dim", 30)  # base_lin_vel(3) + base_ang_vel(3) + joint_pos(12) + joint_vel(12)
            self.goal_dim = kwargs.pop("goal_dim", 4)  # goal_commands (x, y, z, yaw)
            self.goal_start_idx = kwargs.pop("goal_start_idx", None)  # Will be computed if None
            self.past_positions_start_idx = kwargs.pop("past_positions_start_idx", None)  # Will be computed if None
            self.num_past_positions = kwargs.pop("num_past_positions", 20)
            self.num_past_position_points = self.num_past_positions * 3  # Each position is (x, y, z)
            
            # Attention parameters
            self.attention_feature_channels = kwargs.pop("attention_feature_channels", 32)
            self.attention_num_heads = kwargs.pop("attention_num_heads", 1)
            self.attention_dropout = kwargs.pop("attention_dropout", 0.1)
            
            # Compute goal and past positions indices if not provided
            if self.goal_start_idx is None:
                # Goal comes after proprioception
                self.goal_start_idx = self.proprioception_start_idx + self.proprioception_dim
            if self.past_positions_start_idx is None:
                # Past positions come after height scan
                self.past_positions_start_idx = self.height_scan_start_idx + self.num_height_scan_points
            
            # Query dimension: proprioception + goal + past positions
            self.query_dim = self.proprioception_dim + self.goal_dim + self.num_past_position_points
            
            # Simple conv layer to convert height scan from 1 channel to feature_channels
            self.height_scan_to_features = nn.Sequential(
                nn.Conv2d(1, self.attention_feature_channels, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(self.attention_feature_channels),
                nn.ReLU(inplace=True),
            )
            
            # Attention compressor
            self.attention_compressor = AttentionFeatureCompressor(
                feature_channels=self.attention_feature_channels,
                query_dim=self.query_dim,
                num_heads=self.attention_num_heads,
                dropout=self.attention_dropout,
            )
            
            # Output dimension after compression: feature_channels (flattened from [C, 1])
            self.attention_output_dim = self.attention_feature_channels

        # Height scan normalization parameters
        # Should match your height_scan_clipped clip_height range (default: (-1.0, 0.5))
        height_scan_normalize = kwargs.pop("height_scan_normalize", True)
        height_scan_min = kwargs.pop("height_scan_min", -2.0)
        height_scan_max = kwargs.pop("height_scan_max", 2.0)
        
        self.CNN_encoder = HeightScanEncoder(
            input_shape=(self.height_scan_x, self.height_scan_y),
            out_features=64,
            normalize=height_scan_normalize,
            height_min=height_scan_min,
            height_max=height_scan_max,
        )
        self.encoder_out_dim = self.CNN_encoder.out_features

        # Past positions encoder parameters
        # Calculate from config: max_distance=10.0, interval=0.2 -> num_positions = int(10.0/0.2) + 1 = 51
        num_past_positions = kwargs.pop("num_past_positions", 20) 
        past_positions_out_features = kwargs.pop("past_positions_out_features", 16)  # Encoded feature size
        past_positions_normalize = kwargs.pop("past_positions_normalize", True)
        self.num_past_positions = num_past_positions
        self.num_past_position_points = num_past_positions * 3  # Each position is (x, y, z)
        
        self.position_encoder = PositionEncoder(
            num_positions=num_past_positions,
            out_features=past_positions_out_features,
            normalize=past_positions_normalize,
        )
        self.position_encoder_out_dim = self.position_encoder.out_features

        # Calculate encoded observation dimensions
        # Remove: height_scan (651) + past_positions (num_past_positions * 2)
        # Add: CNN features (34) + position encoder features (past_positions_out_features)
        if self.use_height_scan_encoding:
            encoded_actor_obs_dim = (
                num_actor_obs 
                - self.num_height_scan_points 
                + self.encoder_out_dim
            )
            encoded_critic_obs_dim = (
                num_critic_obs 
                - self.num_height_scan_points 
                + self.encoder_out_dim
            )
        else:
            encoded_actor_obs_dim = num_actor_obs
            encoded_critic_obs_dim = num_critic_obs
        if self.use_position_encoding and self.use_height_scan_encoding:
            encoded_actor_obs_dim = (
                num_actor_obs 
                - self.num_height_scan_points 
                - self.num_past_position_points
                + self.encoder_out_dim 
                + self.position_encoder_out_dim
            )
            encoded_critic_obs_dim = (
                num_critic_obs 
                - self.num_height_scan_points 
                - self.num_past_position_points
                + self.encoder_out_dim 
                + self.position_encoder_out_dim
            )

        # Calculate encoded observation dimensions
        if self.use_attention_encoding:
            # With attention encoding:
            # - Remove: height_scan (num_height_scan_points)
            # - Add: compressed attention features (attention_output_dim = attention_feature_channels)
            # - Keep: past_positions (they're used in query but also kept in output)
            encoded_actor_obs_dim = (
                num_actor_obs 
                - self.num_height_scan_points 
                + self.attention_output_dim
            )
            encoded_critic_obs_dim = (
                num_critic_obs 
                - self.num_height_scan_points 
                + self.attention_output_dim
            )

        self.memory_a = Memory(encoded_actor_obs_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(encoded_critic_obs_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        if self.use_height_scan_encoding:
            observations = self.encode_observations(observations)
        if self.use_attention_encoding:
            observations = self.encode_observations_with_attention(observations)
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        if self.use_height_scan_encoding:
            observations = self.encode_observations(observations)
        if self.use_attention_encoding:
            observations = self.encode_observations_with_attention(observations)
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if self.use_height_scan_encoding:
            critic_observations = self.encode_observations(critic_observations)
        if self.use_attention_encoding:
            critic_observations = self.encode_observations_with_attention(critic_observations)
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

        if self.use_position_encoding:
            # Extract past positions portion (comes after height-scan)
            past_positions_start_idx = height_scan_end_idx
            past_positions_end_idx = past_positions_start_idx + self.num_past_position_points
            past_positions_input = observations[:, past_positions_start_idx : past_positions_end_idx].clone()

            # Pass through position encoder (1D CNN)
            encoded_past_positions = self.position_encoder(past_positions_input)
        else:
            past_positions_start_idx = height_scan_end_idx
            past_positions_end_idx = past_positions_start_idx + self.num_past_position_points
            encoded_past_positions = observations[:, past_positions_start_idx : past_positions_end_idx].clone()

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

    def encode_observations_with_attention(self, observations):
        """Encode height-scan using spatial and cross-attention while keeping the rest intact.
        
        Applies self-attention and cross-attention to the height map. The query for cross-attention
        is constructed from proprioception observations, goal point, and raw past positions.
        
        The height-scan segment is reshaped into a 2-D grid, converted to feature channels,
        processed through attention layers, and compressed to a 1D representation that replaces
        the original height scan in the observation vector.
        
        Args:
            observations: Observation tensor [batch, obs_dim] or [time_steps, batch, obs_dim]
        
        Returns:
            Encoded observations with height scan replaced by attention-compressed features
        """
        if not self.use_attention_encoding:
            raise RuntimeError("Attention encoding is not enabled. Set use_attention_encoding=True in __init__")
        
        original_shape = observations.shape
        # Deal with time-major inputs (flatten to batch-major for processing)
        if observations.dim() == 3:
            time_steps, batch_size, obs_dim = original_shape
            observations = observations.reshape(time_steps * batch_size, obs_dim)
            batch_size_actual = time_steps * batch_size
        else:
            time_steps = None
            batch_size_actual = observations.shape[0]
        
        # Extract height-scan and reshape to 2D feature map [batch, 1, H, W]
        height_scan_end_idx = self.height_scan_start_idx + self.num_height_scan_points
        height_scan_flat = observations[:, self.height_scan_start_idx : height_scan_end_idx].clone()
        height_scan_2d = height_scan_flat.reshape(batch_size_actual, 1, self.height_scan_x, self.height_scan_y)
        
        # Convert to feature channels [batch, C, H, W]
        height_features = self.height_scan_to_features(height_scan_2d)  # [batch_size_actual, C, H, W]
        
        # Extract query components: proprioception + goal + past positions
        proprioception = observations[:, self.proprioception_start_idx : self.proprioception_start_idx + self.proprioception_dim].clone()
        
        goal_end_idx = self.goal_start_idx + self.goal_dim
        goal = observations[:, self.goal_start_idx : goal_end_idx].clone()
        
        past_positions_end_idx = self.past_positions_start_idx + self.num_past_position_points
        past_positions = observations[:, self.past_positions_start_idx : past_positions_end_idx].clone()
        
        # Concatenate to form query [batch, query_dim]
        query = torch.cat([proprioception, goal, past_positions], dim=1)
        
        # Apply attention: [batch, C, H, W] -> [batch, C, 1]
        compressed_features = self.attention_compressor(height_features, query)
        
        # Flatten compressed features: [batch, C, 1] -> [batch, C]
        compressed_features_flat = compressed_features.squeeze(-1)
        
        # Replace height scan with compressed features
        # Structure: [before_height_scan, compressed_features, past_positions, after_past_positions]
        #TODO: do the prop observations need to be concated back here?
        observations_encoded = torch.cat(
            [
                observations[:, :self.height_scan_start_idx].clone(),  # Before height scan
                compressed_features_flat,  # Compressed attention features (replaces height scan)
                past_positions,  # Keep past positions in output
                observations[:, past_positions_end_idx:].clone(),  # After past positions
            ],
            dim=1,
        )
        
        # Reshape back to original format if needed
        if time_steps is not None:
            observations_encoded = observations_encoded.reshape(time_steps, batch_size, observations_encoded.shape[-1])
        
        return observations_encoded
