"""Small script demonstrating how to run inference and plot attention weights per feature.

Run:
    python notebooks/explainability.py --data data/sample.csv

This is a lightweight replacement for a Jupyter notebook for quick runs.
"""

import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from data.loader import load_dataset, create_dataloaders
from models.clinical_attention import ClinicalAttentionModel
from utils.preprocess import normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.data)
    X = normalize(X)
    loader = create_dataloaders(X, y, batch_size=16, shuffle=False)
    model = ClinicalAttentionModel()
    # If checkpoint exists, load it
    try:
        model.load_state_dict(torch.load('checkpoints/model_checkpoint.pt', map_location='cpu'))
        print('Loaded checkpoint')
    except Exception:
        print('No checkpoint found; running with untrained model')

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits, attn = model(xb)
            scores = torch.softmax(logits, dim=1)[:, 1].numpy()
            attn_np = attn.numpy()
            # Plot attention for first sample in batch
            plt.bar(range(attn_np.shape[1]), attn_np[0])
            plt.xlabel('Feature index (f0..f6)')
            plt.ylabel('Attention weight')
            plt.title('Attention per feature (sample 0)')
            plt.show()
            break


if __name__ == '__main__':
    main()
