import torch


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Simple per-feature zero-mean unit-variance normalization."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)
