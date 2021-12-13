from .attn import AttnNetwork
from .cnn import CnnNetwork
from .fc import FcNetwork
from .gnn import GnnNetwork

__all__ = ["FcNetwork", "CnnNetwork", "GnnNetwork", "AttnNetwork"]


def sum_params(module):
    s = 0
    for p in module.parameters():
        s += p.sum()
    return s.item()
