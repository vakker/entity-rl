# pylint: disable=protected-access,too-many-locals

import pytest
import torch
from torch import nn

from spg_experiments.models import slot_attention

TEST_BATCH_SIZE = 16
TEST_D0 = 10
TEST_D1 = 20
TEST_D2 = 15
TEST_N = 5
TEST_K = 7


@pytest.fixture(name="even_batch")
def fixture_even_batch():
    x = torch.randn(TEST_BATCH_SIZE, TEST_N, TEST_D0)
    return x


@pytest.fixture(name="uneven_batch")
def fixture_uneven_batch():
    x = [torch.randn(1, N, TEST_D0) for N in torch.randint(1, 10, (TEST_BATCH_SIZE,))]
    return x


def compare_outputs(out_ref, out_new):
    assert len(out_ref) == len(out_new)

    for sample_ref, sample_new in zip(out_ref, out_new):
        tensors_equal = torch.norm(sample_ref - sample_new).detach() == pytest.approx(
            0, abs=2.0e-06
        )
        assert tensors_equal

    return True


def to_pyg(batch):
    batch_idx = []
    pyg_batch = []
    for i, sample in enumerate(batch):
        if sample.dim() == 3:
            sample = sample.squeeze(0)

        pyg_batch.append(sample)
        batch_idx.append(torch.tensor([i] * len(sample)))

    pyg_batch = torch.cat(pyg_batch)
    batch_idx = torch.cat(batch_idx)
    return pyg_batch, batch_idx


def from_pyg(pyg_batch, batch_idx):
    x = []
    for i in range(torch.max(batch_idx) + 1):
        x.append(pyg_batch[batch_idx == i])

    return x


def pyg_wrapper(fcn, x):
    x, batch = to_pyg(x)
    return from_pyg(fcn(x), batch)


def layers_list():
    layer_norm = nn.LayerNorm(TEST_D0)
    lin_layer_1 = nn.Linear(TEST_D0, TEST_D1)
    seq_layer = nn.Sequential(
        nn.Linear(TEST_D0, TEST_D1),
        nn.ReLU(),
        nn.Linear(TEST_D1, TEST_D0),
    )

    torch.manual_seed(0)
    sa_ref1 = slot_attention.SlotAttentionRef(TEST_K, TEST_D0, fixed_slots=True)
    torch.manual_seed(0)
    sa_new1 = slot_attention.SlotAttention(TEST_K, TEST_D0, fixed_slots=True)
    torch.manual_seed(0)
    sa_new2 = slot_attention.SlotAttention(TEST_K, TEST_D0, fixed_slots=True)

    def sa_wrapper(sa, x):
        x, batch = to_pyg(x)
        inputs = (x, None, batch)
        return sa(inputs)

    def sa_ref_get_M(sa, x):
        batch_size = x.shape[0]
        slots = sa._get_slots(batch_size)

        k = sa.to_k(x)
        q = sa.to_q(slots)
        M = sa._get_M(k, q)
        return M

    def sa_ref_get_attn(sa, x):
        M = sa_ref_get_M(sa, x)
        attn = sa._get_attn(M)
        return attn

    def sa_ref_get_W(sa, x):
        attn = sa_ref_get_attn(sa, x)
        return sa._get_W(attn)

    def sa_ref_get_updates(sa, x):
        v = sa.to_v(x)
        W = sa_ref_get_W(sa, x)
        return sa._get_updates(W, v)

    def _sa_new_get_M(sa, x):
        x, batch = to_pyg(x)
        x = x.unsqueeze(1)
        batch_size = torch.max(batch) + 1

        slots = sa._get_slots(batch_size)
        k = sa.to_k(x)
        q = sa.to_q(slots)
        M = sa._get_M(k, q, batch)
        return M, batch

    def sa_new_get_M(sa, x):
        M, batch = _sa_new_get_M(sa, x)
        return from_pyg(M.squeeze(), batch)

    def _sa_new_get_attn(sa, x):
        M, batch = _sa_new_get_M(sa, x)
        attn = sa._get_attn(M)
        return attn, batch

    def sa_new_get_attn(sa, x):
        attn, batch = _sa_new_get_attn(sa, x)
        return from_pyg(attn.squeeze(), batch)

    def _sa_new_get_W(sa, x):
        attn, batch = _sa_new_get_attn(sa, x)
        W = sa._get_W(attn, batch)
        return W, batch

    def sa_new_get_W(sa, x):
        W, batch = _sa_new_get_W(sa, x)
        return from_pyg(W.squeeze(), batch)

    def sa_new_get_updates(sa, inputs):
        x, _ = to_pyg(inputs)
        x = x.unsqueeze(1)
        v = sa.to_v(x)
        W, batch = _sa_new_get_W(sa, inputs)
        updates = sa._get_updates(W, v, batch)
        return updates

    return [
        [lambda x: x, lambda x: from_pyg(*to_pyg(x))],
        [nn.Identity(), lambda x: pyg_wrapper(nn.Identity(), x)],
        [layer_norm, lambda x: pyg_wrapper(layer_norm, x)],
        [lin_layer_1, lambda x: pyg_wrapper(lin_layer_1, x)],
        [seq_layer, lambda x: pyg_wrapper(seq_layer, x)],
        [lambda x: sa_wrapper(sa_new1, x), lambda x: sa_wrapper(sa_new2, x)],
        [sa_ref1.norm_input, lambda x: pyg_wrapper(sa_new1.norm_input, x)],
        [sa_ref1.to_v, lambda x: pyg_wrapper(sa_new1.to_v, x)],
        [sa_ref1.to_k, lambda x: pyg_wrapper(sa_new1.to_k, x)],
        [sa_ref1.to_q, lambda x: pyg_wrapper(sa_new1.to_q, x)],
        [
            lambda x: sa_ref_get_M(sa_ref1, x),
            lambda x: sa_new_get_M(sa_new1, x),
        ],
        [
            lambda x: sa_ref_get_attn(sa_ref1, x),
            lambda x: sa_new_get_attn(sa_new1, x),
        ],
        [
            lambda x: sa_ref_get_W(sa_ref1, x),
            lambda x: sa_new_get_W(sa_new1, x),
        ],
        [
            lambda x: sa_ref_get_updates(sa_ref1, x),
            lambda x: sa_new_get_updates(sa_new1, x),
        ],
        [sa_ref1, lambda x: sa_wrapper(sa_new1, x)],
    ]


@pytest.mark.parametrize("layers", layers_list())
def test_even_batch(even_batch, layers):
    layer_ref, layer_new = layers

    out_ref = layer_ref(even_batch)
    out_new = layer_new(even_batch)

    assert compare_outputs(out_ref, out_new)


@pytest.mark.parametrize("layers", layers_list())
def test_uneven_batch(uneven_batch, layers):
    layer_ref, layer_new = layers

    out_ref = [layer_ref(b) for b in uneven_batch]
    out_new = layer_new(uneven_batch)

    assert compare_outputs(out_ref, out_new)
