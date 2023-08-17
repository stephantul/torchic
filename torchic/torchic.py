from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchic.utilities import (
    calculate_accuracy,
    create_dataloader,
    evaluation,
    format_history,
    get_device,
)


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
        model = nn.Linear(n_features, n_classes)
        nn.init.kaiming_normal_(model.weight)
        lnorm = nn.BatchNorm1d(n_features)
        model = nn.Sequential(lnorm, model)

        return model

    def create_criterion(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return get_device(self)

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
        self, X_batch: torch.Tensor, y_batch: torch.LongTensor
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

        train_dataloader, val_dataloader = create_dataloader(X, y, batch_size)

        wrong_epochs = 0

        for _ in range(max_epochs):
            self._epoch(train_dataloader, val_dataloader)

            if val_loss := self.history["val_loss"]:
                lowest_loss = min(val_loss[:-1])
                current_loss = val_loss[-1]
                if current_loss < lowest_loss:
                    wrong_epochs = 0
                    self.best_params = self.model.state_dict()
                else:
                    wrong_epochs += 1
                    if wrong_epochs > early_stopping:
                        break

        self.model.load_state_dict(self.best_params)
        self.best_params = None

        return self

    def predict(self, X: torch.Tensor, batch_size: int = 1024) -> torch.LongTensor:
        return self.predict_proba(X, batch_size).argmax(1)

    def predict_proba(
        self, X: torch.Tensor, batch_size: int = 1024
    ) -> torch.LongTensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)

        predictions = []
        with evaluation(self):
            for batch in dataloader:
                batch = batch.to(self.device)
                predictions.append(torch.softmax(self(batch), dim=1))

        return torch.cat(predictions)
