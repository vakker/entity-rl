from .attn import AttnNetwork
from .cnn1d import Cnn1DNetwork
from .fc import FcNetwork
from .gnn import GnnNetwork

__all__ = ["FcNetwork", "Cnn1DNetwork", "GnnNetwork", "AttnNetwork"]


def sum_params(module):
    s = 0
    for p in module.parameters():
        s += p.sum()
    return s.item()
