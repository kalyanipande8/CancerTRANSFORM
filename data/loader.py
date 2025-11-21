"""Dataset loaders and dataloader helpers.

This module provides a lightweight CSV loader expected to contain columns `f0`..`f6` and `label`.
If no path is provided, a synthetic dataset is generated for quick experiments.
"""

from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(path: Optional[str] = None, n_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load clinical dataset as tensors.

    Args:
        path: optional path to CSV with columns `f0`..`f6` and `label`.
        n_samples: when path is None, generate this many synthetic samples.

    Returns:
        X: FloatTensor shape [n, 7]
        y: LongTensor shape [n]
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        X = df[[f"f{i}" for i in range(7)]].values.astype(float)
        y = df["label"].values.astype(int)
    else:
        X = np.random.randn(n_samples, 7).astype(float)
        logits = X[:, :3].sum(axis=1) - X[:, 3:5].sum(axis=1)
        y = (logits + 0.1 * np.random.randn(n_samples) > 0).astype(int)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def create_dataloaders(X, y, batch_size: int = 64, shuffle: bool = True):
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader
