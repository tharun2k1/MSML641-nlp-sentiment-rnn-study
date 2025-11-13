import os
import sys
import json
import time
import uuid
import argparse
import itertools
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# Ensure we can import sibling modules when called as: python src/train.py
HERE = os.path.dirname(__file__)
sys.path.append(HERE)

from models import make_model

# ------------------------- Helpers -------------------------

def set_seed(seed: int = 42):
    import os, random, numpy as np, torch
    # Python / NumPy / PyTorch RNGs
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Stronger determinism (safe on CPU; cuDNN flags are no-ops on CPU)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _pad_truncate(batch, max_len: int):
    # batch: list of (seq, label)
    seqs, labels = zip(*batch)
    out = []
    for s in seqs:
        if len(s) >= max_len:
            out.append(s[:max_len])
        else:
            out.append(s + [0] * (max_len - len(s)))
    x = torch.tensor(out, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return x, y

class _SeqDs(torch.utils.data.Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.seqs[idx], int(self.labels[idx])

def make_loader(seqs, labels, batch_size: int, max_len: int, shuffle: bool):
    ds = _SeqDs(seqs, labels)
    collate = lambda batch: _pad_truncate(batch, max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

def evaluate_probs(y_true_np: np.ndarray, y_prob_np: np.ndarray):
    """Compute Accuracy and macro-F1 from probabilities (threshold 0.5)."""
    y_pred = (y_prob_np >= 0.5).astype(int)
    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average="macro")
    return acc, f1

def load_processed():
    Xtr = np.load("data/processed/train_sequences.npy", allow_pickle=True).tolist()
    Xte = np.load("data/processed/test_sequences.npy", allow_pickle=True).tolist()
    ytr = np.load("data/processed/train_labels.npy")
    yte = np.load("data/processed/test_labels.npy")
    vocab = json.load(open("data/processed/vocab.json"))
    return Xtr, ytr, Xte, yte, len(vocab)

def ensure_preprocessed(csv_path: str, vocab_size: int = 10000):
    """Run preprocess only if processed artifacts are missing."""
    needed = [
        "data/processed/vocab.json",
        "data/processed/train_sequences.npy",
        "data/processed/train_labels.npy",
        "data/processed/test_sequences.npy",
        "data/processed/test_labels.npy",
    ]
    if all(os.path.exists(p) for p in needed):
        print("[SKIP] Preprocess: artifacts already exist in data/processed/")
        return
    print("[RUN] Preprocess: building processed artifacts...")
    cmd = [
        sys.executable, os.path.join(HERE, "preprocess.py"),
        "--csv_path", csv_path,
        "--vocab_size", str(vocab_size),
    ]
    subprocess.check_call(cmd)

# ------------------------- Training -------------------------

def train_one(model, train_loader, val_loader, device, epochs, optimizer_name, lr, clip_norm):
    """
    Assumes the model's final layer returns PROBABILITIES in [0,1] (explicit Sigmoid).
    Uses BCELoss.
    """
    model.to(device)
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    criterion = nn.BCELoss()
    loss_history = []

    for _ in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            probs = model(xb)          # model outputs probabilities due to explicit Sigmoid
            loss = criterion(probs, yb)
            loss.backward()

            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            opt.step()
            batch_losses.append(loss.item())
        loss_history.append(float(np.mean(batch_losses)))

    # Evaluate on val (test) loader using probabilities
    model.eval()
    ys, youts = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs = model(xb)          # probabilities in [0,1]
            ys.append(yb.numpy())
            youts.append(probs.cpu())
    y_true = np.vstack(ys).ravel()
    y_prob = torch.vstack(youts).numpy().ravel()
    acc, f1 = evaluate_probs(y_true, y_prob)
    return acc, f1, loss_history

# ------------------------- Grid Runner -------------------------

def run_grid(args):
    set_seed(42)

    Xtr, ytr, Xte, yte, vocab_size = load_processed()

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    metrics_path = "results/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("model,activation,optimizer,seq_len,grad_clip,accuracy,f1,epoch_time_s\n")

    device = torch.device(args.device)

    for arch, act, opt_name, seqlen, clip in itertools.product(
        args.architectures, args.activations, args.optimizers, args.seq_lens, args.clip
    ):
        # Data loaders (pad/truncate at seqlen)
        train_loader = make_loader(Xtr, ytr, batch_size=args.batch_size, max_len=seqlen, shuffle=True)
        test_loader  = make_loader(Xte, yte, batch_size=args.batch_size, max_len=seqlen, shuffle=False)

        # Model (must have explicit Sigmoid in models.py final layer)
        model = make_model(
            architecture=arch,
            vocab_size=vocab_size,
            emb_dim=100,
            hidden_size=64,
            n_layers=2,
            dropout=args.dropout,
            activation=act
        )

        run_id = f"{arch}_{act}_{opt_name}_L{seqlen}_clip{clip}_{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        acc, f1, loss_hist = train_one(
            model, train_loader, test_loader, device,
            epochs=args.epochs, optimizer_name=opt_name, lr=args.lr, clip_norm=clip
        )
        avg_epoch_time = (time.time() - t0) / max(1, args.epochs)

        # Log row
        with open(metrics_path, "a") as f:
            f.write(f"{arch},{act},{opt_name},{seqlen},{clip},{acc:.4f},{f1:.4f},{avg_epoch_time:.2f}\n")

        # Save loss history for plotting
        with open(os.path.join("results", "logs", f"{run_id}_loss.json"), "w") as jf:
            json.dump({"loss": loss_hist}, jf, indent=2)

        print(f"[DONE] {run_id} | acc={acc:.4f} f1={f1:.4f} epoch_time={avg_epoch_time:.2f}s")

# ------------------------- Run-all (no new file) -------------------------

def _make_ns(architectures, activations, optimizers, seq_lens, clip,
             epochs, batch_size, dropout, lr, device):
    return SimpleNamespace(
        architectures=architectures,
        activations=activations,
        optimizers=optimizers,
        seq_lens=seq_lens,
        clip=clip,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        lr=lr,
        device=device
    )

def run_all_sweeps(epochs, batch_size, dropout, lr, device):
    """
    36 UNIQUE experiments (no duplicates), each sweep varies exactly one factor.
      - Optimizers @ seq_len in {25,50,100}    (fix: arch=LSTM, act=ReLU, clip=0)
      - Architectures @ seq_len in {25,50,100} (fix: act=Tanh, opt=Adam, clip=0)
      - Activations @ seq_len in {25,50,100}   (fix: arch=BiLSTM, opt=Adam, clip=0)
      - Grad clipping @ seq_len in {25,50,100} (fix: arch=LSTM, act=ReLU, opt=Adam; clip in {0.5,1.0})
      - Sequence lengths once {25,50,100}      (fix: arch=RNN, act=ReLU, opt=RMSprop, clip=0)
    Totals: 9 + 9 + 9 + 6 + 3 = 36 unique.
    """
    seq_contexts = [25, 50, 100]

    # 1) Optimizers (3) × seq_len (3) = 9
    for L in seq_contexts:
        print(f"\n=== Sweep: Optimizers @ seq_len={L} (fix arch=LSTM, act=ReLU, clip=0) ===")
        run_grid(_make_ns(["lstm"], ["relu"], ["adam", "sgd", "rmsprop"], [L], [0.0],
                          epochs, batch_size, dropout, lr, device))

    # 2) Architectures (3) × seq_len (3) = 9 (fix act=Tanh to avoid dupes with #1)
    for L in seq_contexts:
        print(f"\n=== Sweep: Architectures @ seq_len={L} (fix act=Tanh, opt=Adam, clip=0) ===")
        run_grid(_make_ns(["rnn", "lstm", "bilstm"], ["tanh"], ["adam"], [L], [0.0],
                          epochs, batch_size, dropout, lr, device))

    # 3) Activations (3) × seq_len (3) = 9 (fix arch=BiLSTM to avoid dupes with #1/#2)
    for L in seq_contexts:
        print(f"\n=== Sweep: Activations @ seq_len={L} (fix arch=BiLSTM, opt=Adam, clip=0) ===")
        run_grid(_make_ns(["bilstm"], ["sigmoid", "relu", "tanh"], ["adam"], [L], [0.0],
                          epochs, batch_size, dropout, lr, device))

    # 4) Gradient clipping (compare off vs on) × seq_len (3) = 6
    for L in seq_contexts:
        print(f"\n=== Sweep: Grad Clipping @ seq_len={L} (fix arch=LSTM, act=ReLU, opt=Adam) ===")
        run_grid(_make_ns(["lstm"], ["relu"], ["adam"], [L], [0.0, 1.0],
                        epochs, batch_size, dropout, lr, device))


    # 5) Sequence lengths once (3) = 3 (fix a baseline not used above)
    print("\n=== Sweep: Sequence Lengths (fix arch=RNN, act=ReLU, opt=RMSprop, clip=0) ===")
    run_grid(_make_ns(["rnn"], ["relu"], ["rmsprop"], [25, 50, 100], [0.0],
                      epochs, batch_size, dropout, lr, device))


def run_plots():
    try:
        import plot_results
        print("\n=== Plots: Accuracy/F1 vs SeqLen, Loss (best vs worst) ===")
        plot_results.main()
    except Exception as e:
        print(f"[WARN] Could not generate plots automatically: {e}")

# ------------------------- CLI -------------------------

def main():
    p = argparse.ArgumentParser(description="Train grid for IMDb sentiment with RNN/LSTM/BiLSTM (explicit sigmoid + BCELoss).")
    # normal grid args
    p.add_argument("--architectures", nargs="+", default=["rnn", "lstm", "bilstm"])
    p.add_argument("--activations",  nargs="+", default=["relu", "tanh", "sigmoid"])
    p.add_argument("--optimizers",   nargs="+", default=["adam", "sgd", "rmsprop"])
    p.add_argument("--seq_lens",     nargs="+", type=int, default=[25, 50, 100])
    p.add_argument("--clip",         nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--epochs",       type=int, default=None)  # None here; we set defaults below
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--device",       type=str, default="cpu", help="cpu or cuda")

    # run-all extras
    p.add_argument("--run_all",      type=int, default=0, help="1 = run all required sweeps and plots.")
    p.add_argument("--csv_path",     type=str, default="data/IMDB Dataset.csv",
                   help="Path to raw IMDb CSV (used for auto-preprocess).")
    p.add_argument("--vocab_size",   type=int, default=10000)
    p.add_argument("--skip_preprocess", action="store_true",
                   help="If set, do not run preprocess even if artifacts missing (use existing).")

    args = p.parse_args()

    if args.run_all:
        # preprocess if missing
        if not args.skip_preprocess:
            ensure_preprocessed(csv_path=args.csv_path, vocab_size=args.vocab_size)
        else:
            print("[WARN] Skipping preprocess by user request.")

        # default to 5 epochs for run_all unless user overrides
        ep = args.epochs if args.epochs is not None else 5

        run_all_sweeps(
            epochs=ep,
            batch_size=args.batch_size,
            dropout=args.dropout,
            lr=args.lr,
            device=args.device
        )
        run_plots()
        print("\n[ALL DONE] See results/metrics.csv and results/plots/*.png")
        return

    # ---- normal single/grid run path ----
    # keep original default of 3 epochs if user didn't pass anything
    if args.epochs is None:
        args.epochs = 3

    run_grid(args)

if __name__ == "__main__":
    main()
