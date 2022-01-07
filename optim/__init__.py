from torch.optim import Adam
from torch.optim.lbfgs import LBFGS
from .radam import RAdam


OPTIMIZER_MAP = {
    "adam": Adam,
    "radam": RAdam,
    "lbfgs": LBFGS,
}


def get_optimizer_class(optimizer_name):
    name = optimizer_name.lower()
    return OPTIMIZER_MAP[name]
