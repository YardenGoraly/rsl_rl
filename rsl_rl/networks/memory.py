# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class SRUCell(nn.Module):
    """SRU-LSTM cell following paper definition: st = Wxs*xt + bs, gt = tanh(st ⊙ (Wxg*xt + Whg*ht-1 + bg))"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Spatial transformation: st = Wxs*xt + bs
        self.W_xs = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_s = nn.Parameter(torch.Tensor(hidden_size))
        
        # Input gate components
        self.W_xi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forget gate components
        self.W_xf = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # Cell gate components (SRU modification): gt = tanh(st ⊙ (Wxg*xt + Whg*ht-1 + bg))
        self.W_xg = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output gate components
        self.W_xo = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x, hidden):
        """SRU-LSTM forward: st = Wxs*xt + bs, gt = tanh(st ⊙ (Wxg*xt + Whg*ht-1 + bg)), then standard LSTM updates"""
        h_prev, c_prev = hidden
        
        # Spatial transformation: st = Wxs*xt + bs
        # x: [batch_size, input_size], W_xs: [hidden_size, input_size]
        # Result: [batch_size, hidden_size]
        s_t = x @ self.W_xs.T + self.b_s
        
        # Standard LSTM gates
        # x: [batch_size, input_size], W_xi: [hidden_size, input_size]
        # h_prev: [batch_size, hidden_size], W_hi: [hidden_size, hidden_size]
        # Result: [batch_size, hidden_size]
        i_t = torch.sigmoid(x @ self.W_xi.T + h_prev @ self.W_hi.T + self.b_i)  # Input gate
        f_t = torch.sigmoid(x @ self.W_xf.T + h_prev @ self.W_hf.T + self.b_f)  # Forget gate
        o_t = torch.sigmoid(x @ self.W_xo.T + h_prev @ self.W_ho.T + self.b_o)  # Output gate
        
        # SRU-modified cell gate: gt = tanh(st ⊙ (Wxg*xt + Whg*ht-1 + bg))
        g_t = torch.tanh(s_t * (x @ self.W_xg.T + h_prev @ self.W_hg.T + self.b_g))

        r_t = i_t * (1 - (1 - f_t)**2) + (1 - i_t) * (f_t**2)
        
        # Cell and hidden state updates (standard LSTM)
        c_t = r_t * c_prev + (1 - r_t) * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t


class SRU(nn.Module):
    """Multi-layer SRU compatible with PyTorch RNN interface."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            SRUCell(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        """input: [seq_len, batch, input_size], hidden: tuple of (h, c) or None"""
        seq_len, batch_size = input.shape[0], input.shape[1]   #TODO: check if this is correct for memory module
        
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=input.device, dtype=input.dtype) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=input.device, dtype=input.dtype) 
                 for _ in range(self.num_layers)]
        else:
            # Handle both tuple (h, c) and single tensor (for backward compatibility)
            if isinstance(hidden, tuple):
                h = [hidden[0][i] for i in range(self.num_layers)]
                c = [hidden[1][i] for i in range(self.num_layers)]
            else:
                # Single tensor: assume it's c, initialize h from c
                c = [hidden[i] for i in range(self.num_layers)]
                h = [hidden[i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            x_t = input[t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, (h[i], c[i]))
                x_t = h[i]
            outputs.append(x_t)
        
        # Return tuple (h, c) like LSTM for proper state management
        return torch.stack(outputs, dim=0).squeeze(1), (torch.stack(h, dim=0), torch.stack(c, dim=0))


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        if type.lower() == "sru":
            self.rnn = SRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
            self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode: needs saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference/distillation mode: uses hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM or SRU
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:  # GRU
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM or SRU
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:  # GRU
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM or SRU
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:  # GRU
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()
