#!/usr/bin/env python3
"""
Small runnable example: Clinical attention model over 7 features.
Usage:
  python run.py --mode train --epochs 5
  python run.py --mode eval

This is a minimal example. Replace the `load_dataset` function to load real clinical CSVs.
"""

import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.loader import load_dataset, create_dataloaders
from models.clinical_attention import ClinicalAttentionModel
from utils.preprocess import normalize

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Use loader.load_dataset when available; kept here for backward compatibility



# Model implementation moved to `models/clinical_attention.py` and imported above


def train(model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / (len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 1)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.numpy().tolist())

        acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        print(f"Epoch {ep}/{epochs} — loss: {avg_loss:.4f} — val_acc: {acc:.4f}")

    return model


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    attn_sample = None
    with torch.no_grad():
        for xb, yb in loader:
            logits, attn = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(yb.numpy().tolist())
            if attn_sample is None:
                attn_sample = attn[0].cpu().numpy()
    acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
    print(f"Eval accuracy: {acc:.4f}")
    if attn_sample is not None:
        print("Sample attention weights (per feature):", attn_sample)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--data", default=None, help="Path to CSV with f0..f6 and label columns")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.data)
    # normalize
    X = normalize(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    train_loader = create_dataloaders(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloaders(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    model = ClinicalAttentionModel()

    if args.mode == "train":
        model = train(model, train_loader, val_loader, epochs=args.epochs)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("checkpoints", "model_checkpoint.pt"))
        print("Model saved to checkpoints/model_checkpoint.pt")
    else:
        # Eval mode (loads checkpoint if exists)
        ck = os.path.join("checkpoints", "model_checkpoint.pt")
        if os.path.exists(ck):
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            print("Loaded", ck)
        evaluate(model, val_loader)


if __name__ == "__main__":
    main()
