import contextlib
from typing import Dict, Iterator, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_test_split(
    X: torch.Tensor, y: torch.LongTensor, test_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    split = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, X_val, y_train, y_val


def create_dataloader(
    X: torch.Tensor, y: torch.LongTensor, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


@torch.no_grad()
def calculate_accuracy(X: torch.Tensor, y: torch.Tensor) -> float:
    return (X.argmax(1) == y).float().mean().item()


def format_history(history: Dict[str, List[float]]) -> str:
    s = []

    for key, values in history.items():
        s.append(f"{key}: {values[-1]:.5f}")

    return " ".join(s)


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
