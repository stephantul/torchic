import contextlib
from typing import Iterator

import torch
from torch import nn


@contextlib.contextmanager
def evaluation(module: nn.Module) -> Iterator:
    prev_training = module.training
    module.train(False)
    yield
    module.train(prev_training)


def get_device(module: nn.Module) -> torch.device:
    all_devices = set([x.device for x in module.parameters()])
    if len(all_devices) > 1:
        raise ValueError("More than 1 device detected.")

    return next(iter(all_devices))
