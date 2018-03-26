import torch
from torch.autograd import Variable


def tolist(*args):
    return [[i] if type(i) not in [tuple, list] else i for i in args]