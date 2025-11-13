import json
import random
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ------------------------ Reproducibility ------------------------

SEED = 42

def set_seed(seed: int = SEED):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------ Dataset & Loader ------------------------

class SequenceDataset(Dataset):
    """Holds pre-encoded variable-length sequences and binary labels."""
    def __init__(self, sequences: List[List[int]], labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], int(self.labels[idx])


def _pad_and_truncate(batch, max_len: int):
    """
    Collate function: pads/truncates to max_len and returns tensors.
    batch: List[Tuple[List[int], int]]
    """
    seqs, labels = zip(*batch)
    processed = []
    for s in seqs:
        if len(s) >= max_len:
            processed.append(s[:max_len])
        else:
            processed.append(s + [0] * (max_len - len(s)))  # 0 is PAD
    x = torch.tensor(processed, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return x, y


def make_loader(
    seqs: List[List[int]],
    labels: np.ndarray,
    batch_size: int,
    max_len: int,
    shuffle: bool = False
) -> DataLoader:
    ds = SequenceDataset(seqs, labels)
    collate = lambda batch: _pad_and_truncate(batch, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# ------------------------ Metrics & Timers ------------------------

def evaluate_logits(y_true_np: np.ndarray, y_logits_t: torch.Tensor) -> Tuple[float, float]:
    """
    Convert logits -> probabilities -> predictions and compute metrics.
    Returns (accuracy, macro-F1).
    """
    y_prob = torch.sigmoid(y_logits_t).detach().cpu().numpy().ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average="macro")
    return acc, f1


def epoch_timer():
    """Simple wall-clock timer for epochs."""
    start = time.time()
    def end():
        return time.time() - start
    return end


# ------------------------ I/O helpers ------------------------

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
    
def evaluate_probs(y_true_np: np.ndarray, y_prob_np: np.ndarray):
    y_pred = (y_prob_np >= 0.5).astype(int)
    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average="macro")
    return acc, f1

