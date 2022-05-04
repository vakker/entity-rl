from .attn import AttnPolicy
from .cnn1d import Cnn1DPolicy
from .fc import FcPolicy
from .gnn import GnnPolicy
from .space_policy import SpaceGnnPolicy

__all__ = ["FcPolicy", "Cnn1DPolicy", "GnnPolicy", "AttnPolicy", "SpaceGnnPolicy"]


def sum_params(module):
    s = 0
    for p in module.parameters():
        s += p.sum()
    return s.item()
