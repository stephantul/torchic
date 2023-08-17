from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchic.utilities import evaluation, get_device


class Torchic(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 10_000,
        early_stopping: int = 5,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.model = nn.Linear(n_features, n_classes)
        nn.init.kaiming_normal_(self.model.weight)

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping

        self.validation_losses: List[float] = []

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
    def _eval_batch(self, X_batch: torch.Tensor, y_batch: torch.LongTensor) -> float:
        logits = self(X_batch)
        loss = self.criterion(logits, y_batch)

        return loss.item()

    def _epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        total_batches = len(train_dataloader) // self.batch_size
        pbar = tqdm(total=total_batches)

        for X_, y_ in train_dataloader:
            loss = self._fit_batch(X_, y_)
            pbar.set_description(desc=f"{loss:.5f}", refresh=True)
            pbar.update(1)

        avg_loss = 0.0
        with evaluation(self):
            for X_, y_ in val_dataloader:
                avg_loss += self._eval_batch(X_, y_)

        self.validation_losses.append(avg_loss)

        pbar.close()

    def fit(self, X: torch.Tensor, y: torch.LongTensor) -> Torchic:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        split = int(len(X) * 0.9)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        wrong_epochs = 0

        for epoch in range(self.max_epochs):
            self._epoch(train_dataloader, val_dataloader)

            if epoch > 0:
                lowest_loss = min(self.validation_losses[:-1])
                current_loss = self.validation_losses[-1]
                if current_loss < lowest_loss:
                    wrong_epochs = 0
                else:
                    wrong_epochs += 1
                    if wrong_epochs > self.early_stopping:
                        break

        return self
