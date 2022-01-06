# Slot-attention module base on https://github.com/lucidrains/slot-attention

import torch
from torch import nn
from torch.nn import init
from torch_scatter import scatter


class SlotAttentionRef(nn.Module):
    # pylint: disable=too-many-locals

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        batch_size, _, n_input_features = inputs.shape

        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(batch_size, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=inputs.device)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, n_input_features),
                slots_prev.reshape(-1, n_input_features),
            )

            slots = slots.reshape(batch_size, -1, n_input_features)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttention(nn.Module):
    # pylint: disable=too-many-locals

    def __init__(
        self,
        num_slots,
        dim,
        d_in=None,
        d_out=None,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
    ):
        super().__init__()

        if d_in is None:
            d_in = dim

        if d_out is None:
            d_out = dim

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, d_out))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, d_out))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(d_out, dim)
        self.to_k = nn.Linear(d_in, dim)
        self.to_v = nn.Linear(d_in, d_out)

        self.gru = nn.GRUCell(num_slots * d_out, num_slots * d_out)

        self.mlp = nn.Sequential(
            nn.Linear(d_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out),
        )

        self.norm_input = nn.LayerNorm(d_in)
        self.norm_slots = nn.LayerNorm(d_out)
        self.norm_pre_ff = nn.LayerNorm(d_out)

    def forward(self, inputs):
        x, _, batch = inputs

        x = x.unsqueeze(1)
        batch_size = torch.max(batch) + 1

        n_s = self.num_slots

        mu = self.slots_mu.expand(batch_size, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=x.device)

        x = self.norm_input(x)
        k = self.to_k(x)
        v = self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            q_scatter = q[batch]
            dots = torch.matmul(k, q_scatter.transpose(2, 1)) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            W = attn / scatter(attn, batch, dim=0)[batch]

            updates = scatter(W.transpose(2, 1) * v, batch, dim=0)

            slots = self.gru(
                updates.reshape(batch_size, -1), slots_prev.reshape(batch_size, -1)
            )

            slots = slots.reshape(batch_size, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
