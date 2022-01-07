# Slot-attention module base on https://github.com/lucidrains/slot-attention

import torch
from torch import nn
from torch.nn import init
from torch_scatter import scatter

from .base import BaseModule


class SlotAttentionRef(BaseModule):
    # pylint: disable=too-many-locals,no-self-use,unused-argument

    def __init__(
        self,
        num_slots,
        dim,
        d_in=None,
        d_out=None,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
        fixed_slots=False,
        final_act=True,
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

        self.final_act = final_act
        self.fixed_slots = fixed_slots
        self.out_channels = num_slots * d_out

    def _get_slots(self, batch_size):
        n_s = self.num_slots

        mu = self.slots_mu.expand(batch_size, n_s, -1)
        slots = mu

        if not self.fixed_slots:
            sigma = self.slots_logsigma.exp().expand(batch_size, n_s, -1)
            slots = slots + sigma * torch.randn(mu.shape, device=self.device)

        return slots

    def _slots_update(self, updates, slots_prev, batch_size):
        slots = self.gru(
            updates.reshape(batch_size, -1),
            slots_prev.reshape(batch_size, -1),
        )

        slots = slots.reshape(batch_size, self.num_slots, -1)
        return slots + self.mlp(self.norm_pre_ff(slots))

    def _get_M(self, k, q, batch):
        return torch.matmul(k, q.transpose(2, 1)) * self.scale

    def _get_attn(self, M):
        return M.softmax(dim=-1) + self.eps

    def _get_W(self, attn, batch):
        return attn / attn.sum(dim=1, keepdim=True)

    def _get_updates(self, W, v, batch):
        return torch.matmul(W.transpose(2, 1), v)

    def _prep_inputs(self, inputs):
        x = inputs
        assert x.dim() == 3
        batch_size = x.shape[0]

        return x, None, batch_size

    def forward(self, inputs):
        x, batch, batch_size = self._prep_inputs(inputs)

        slots = self._get_slots(batch_size)

        x = self.norm_input(x)
        k = self.to_k(x)
        v = self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            M = self._get_M(k, q, batch)

            attn = self._get_attn(M)
            W = self._get_W(attn, batch)
            updates = self._get_updates(W, v, batch)

            slots = self._slots_update(updates, slots_prev, batch_size)

        if self.final_act:
            return torch.relu(slots)

        return slots


class SlotAttention(SlotAttentionRef):
    def _get_M(self, k, q, batch):
        q_scatter = q[batch]
        return torch.matmul(k, q_scatter.transpose(2, 1)) * self.scale

    def _get_W(self, attn, batch):
        return attn / scatter(attn, batch, dim=0)[batch]

    def _get_updates(self, W, v, batch):
        return scatter(W.transpose(2, 1) * v, batch, dim=0)

    def _prep_inputs(self, inputs):
        x, _, batch = inputs
        assert x.dim() == 2

        x = x.unsqueeze(1)
        batch_size = torch.max(batch) + 1

        return x, batch, batch_size
