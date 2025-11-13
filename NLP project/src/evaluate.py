"""
Evaluate a trained checkpoint on the IMDb test split.

Usage example:
--------------
python src/evaluate.py \
  --checkpoint checkpoints/bilstm_relu_adam_L100.pt \
  --architecture bilstm \
  --activation relu \
  --seq_len 100 \
  --batch_size 32 \
  --device cpu

Notes:
- Expects processed artifacts in data/processed/ (created by src/preprocess.py).
- The checkpoint should be saved from a model built with the same hyperparameters.
- If you didn’t save checkpoints yet, run training with saving enabled (you can
  modify train.py to torch.save(model.state_dict(), "checkpoints/<name>.pt")).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Ensure we can import sibling modules when called as: python src/evaluate.py
HERE = os.path.dirname(__file__)
sys.path.append(HERE)

from models import make_model
from utils import make_loader, evaluate_probs, set_seed


def load_processed():
    Xtr = np.load("data/processed/train_sequences.npy", allow_pickle=True).tolist()
    Xte = np.load("data/processed/test_sequences.npy", allow_pickle=True).tolist()
    ytr = np.load("data/processed/train_labels.npy")
    yte = np.load("data/processed/test_labels.npy")
    vocab = json.load(open("data/processed/vocab.json"))
    return Xtr, ytr, Xte, yte, len(vocab)


def evaluate(checkpoint: str,
             architecture: str,
             activation: str,
             seq_len: int,
             batch_size: int,
             device: str,
             dropout: float = 0.5,
             emb_dim: int = 100,
             hidden_size: int = 64,
             n_layers: int = 2):

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Load data & vocab size
    _, _, Xte, yte, vocab_size = load_processed()

    # Data loader
    test_loader = make_loader(Xte, yte, batch_size=batch_size, max_len=seq_len, shuffle=False)

    # Build model to match checkpoint hyperparams
    model = make_model(
        architecture=architecture,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation
    )

    # Load weights
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()

    # Collect logits on test
    ys, ylogits = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            ys.append(yb.numpy())
            ylogits.append(logits.cpu())

    y_true = np.vstack(ys).ravel()
    y_logits = torch.vstack(ylogits)

    acc, f1 = evaluate_logits(y_true, y_logits)
    print(f"[EVAL] arch={architecture} act={activation} seq_len={seq_len} | "
          f"Accuracy={acc:.4f} F1={f1:.4f}")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RNN/LSTM/BiLSTM checkpoint on IMDb.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt or .pth state_dict file.")
    parser.add_argument("--architecture", type=str, choices=["rnn", "lstm", "bilstm"], required=True)
    parser.add_argument("--activation", type=str, choices=["relu","tanh","sigmoid"], default="relu")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    evaluate(
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        activation=args.activation,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        dropout=args.dropout,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers
    )


if __name__ == "__main__":
    main()
