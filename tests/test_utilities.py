import unittest

import torch
from torch import nn

from torchic.utilities import (
    calculate_accuracy,
    create_dataloaders,
    evaluation,
    format_history,
    get_device,
    train_test_split,
)


class TestUtilities(unittest.TestCase):
    def test_get_device(self) -> None:
        lin1 = nn.Linear(10, 10)
        device = get_device(lin1)

        self.assertEqual(device.type, "cpu")

    def test_get_device_mixed(self) -> None:
        lin1 = nn.Linear(10, 10).to("meta")
        lin2 = nn.Linear(10, 10)

        model = nn.Sequential(lin1, lin2)

        with self.assertRaises(ValueError):
            get_device(model)

    def test_train_test_split(self) -> None:
        X = torch.randn(100, 3)
        y = torch.arange(100)
        X_train, X_val, y_train, y_val = train_test_split(X, y, 0.1)

        self.assertEqual(len(X_train) + len(X_val), len(X))
        self.assertEqual(len(y_train) + len(y_val), len(y))
        self.assertEqual(set(y_train.tolist() + y_val.tolist()), set(y.tolist()))

    def test_create_dataloader(self) -> None:
        X = torch.randn(100, 3)
        y = torch.arange(100)

        dl, _ = create_dataloaders(X, y, 10)

        self.assertEqual(len(dl), 9)
        self.assertEqual([len(x[0]) for x in list(dl)], ([10] * 9))

    def test_calculate_accuracy(self) -> None:
        x = torch.zeros(10, 1)
        y = torch.ones(10)

        self.assertEqual(calculate_accuracy(x, y), 0.0)

        x = torch.zeros(10, 1)
        y = torch.zeros(10)

        self.assertEqual(calculate_accuracy(x, y), 1.0)

        x = torch.zeros(10, 1)
        y = torch.cat([torch.zeros(5), torch.ones(5)])

        self.assertEqual(calculate_accuracy(x, y), 0.5)

    def test_format_history_empty(self) -> None:
        result = format_history({})
        self.assertEqual(result, "")

        result = format_history({"history": [0, 1, 2]})
        self.assertEqual(result, "history: 2.00000")

        result = format_history({"history": [0, 1, 2], "shmistory": [-1, 2, 0.33]})
        self.assertEqual(result, "history: 2.00000 shmistory: 0.33000")

    def test_evaluation(self) -> None:
        module = nn.Linear(10, 10)

        self.assertTrue(module.training)
        with evaluation(module):
            self.assertFalse(module.training)
        self.assertTrue(module.training)

        module.train(False)
        self.assertFalse(module.training)
        with evaluation(module):
            self.assertFalse(module.training)
        self.assertFalse(module.training)
