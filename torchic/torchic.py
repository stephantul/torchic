from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchic.utilities import (
    calculate_accuracy,
    create_dataloaders,
    evaluation,
    format_history,
    get_device,
    get_dtype,
)

AnyArray = Union[np.ndarray, torch.Tensor]


class Torchic(nn.Module):
    def __init__(
        self, n_features: int, n_classes: int, learning_rate: float = 1e-3
    ) -> None:
        super().__init__()
        self.model = self.create_model(n_features, n_classes)
        self.optimizer = self.create_optimizer(learning_rate)
        self.criterion = self.create_criterion()

        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_params: Optional[Dict[str, torch.Tensor]] = None

    def create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=learning_rate)

    def create_model(self, n_features: int, n_classes: int) -> nn.Module:
        n_hidden = int(min(n_features, 90_000) ** 0.5)
        act = nn.ReLU()
        lin_in = nn.Linear(n_features, n_hidden)
        lin_out = nn.Linear(n_hidden, n_classes)
        lnorm = nn.LayerNorm(n_hidden)
        model = nn.Sequential(lin_in, act, lnorm, lin_out)

        return model

    def create_criterion(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return get_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def _fit_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self(X_batch)
        loss = self.criterion(logits, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _eval_batch(
        self, X_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Tuple[float, float]:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self(X_batch)
        loss = self.criterion(logits, y_batch)

        accuracy = calculate_accuracy(logits, y_batch)

        return loss.item(), accuracy

    def _epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        pbar = tqdm(total=len(train_dataloader))

        history_string = format_history(self.history)

        for X_, y_ in train_dataloader:
            loss = self._fit_batch(X_, y_)
            pbar.set_description(
                desc=f"train loss: {loss:.5f} {history_string}",
                refresh=True,
            )
            pbar.update(1)

        avg_loss = 0.0
        avg_acc = 0.0
        with evaluation(self):
            for X_, y_ in val_dataloader:
                batch_loss, batch_acc = self._eval_batch(X_, y_)
                avg_loss += batch_loss
                avg_acc += batch_acc

        self.history["val_loss"].append(avg_loss / len(val_dataloader))
        self.history["val_acc"].append(avg_acc / len(val_dataloader))

        pbar.close()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.LongTensor,
        batch_size: int = 128,
        max_epochs: int = 10_000,
        early_stopping: int = 5,
    ) -> Torchic:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        X = X.type(self.dtype)

        train_dataloader, val_dataloader = create_dataloaders(X, y, batch_size)

        wrong_epochs = 0

        for epoch in range(max_epochs):
            self._epoch(train_dataloader, val_dataloader)

            val_loss = self.history["val_loss"]

            if epoch > 0:
                *rest, current_loss = val_loss
                lowest_loss = min(rest)

                if current_loss < lowest_loss:
                    wrong_epochs = 0
                    self.best_params = self.model.state_dict()
                else:
                    wrong_epochs += 1
                    if wrong_epochs > early_stopping:
                        break

        if self.best_params is not None:
            self.model.load_state_dict(self.best_params)
        self.best_params = None

        self.train(False)

        return self

    def predict(self, X: torch.Tensor, batch_size: int = 1024) -> AnyArray:
        return self.predict_proba(X, batch_size).argmax(1)

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, batch_size: int = 1024) -> AnyArray:
        was_numpy = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            was_numpy = True

        X = X.type(self.dtype)

        dataloader: DataLoader = DataLoader(
            TensorDataset(X), batch_size=batch_size, shuffle=False
        )

        predictions = []
        with evaluation(self):
            for (X_batch,) in dataloader:
                batch = X_batch.to(self.device)
                predictions.append(torch.softmax(self(batch), dim=1).cpu())

        predictions_concatenated = torch.cat(predictions)

        return (
            predictions_concatenated.numpy() if was_numpy else predictions_concatenated
        )
