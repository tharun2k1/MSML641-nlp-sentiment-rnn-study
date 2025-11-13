import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List


def clean_text(t: str) -> str:
    """Lowercase, remove punctuation/special chars, squeeze spaces."""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(texts: List[str]) -> List[List[str]]:
    """Tokenize using NLTK word_tokenize."""
    return [word_tokenize(t) for t in texts]


def build_vocab(token_lists: List[List[str]], vocab_size: int = 10000):
    """
    Build vocab of size `vocab_size`, reserving:
      0: <PAD>, 1: <OOV>
    Returns stoi (token->id) and itos (id->token).
    """
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    most_common = counter.most_common(vocab_size - 2)  # reserve 0/1
    stoi = {"<PAD>": 0, "<OOV>": 1}
    for i, (w, _) in enumerate(most_common, start=2):
        stoi[w] = i
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def encode(token_lists: List[List[str]], stoi: dict) -> List[List[int]]:
    """Map tokens to ids using stoi with <OOV> fallback."""
    oov = stoi["<OOV>"]
    return [[stoi.get(tok, oov) for tok in toks] for toks in token_lists]


def main(args):
    os.makedirs("data/processed", exist_ok=True)

    # Read CSV
    df = pd.read_csv(args.csv_path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must have columns: 'review', 'sentiment'")

    # Deterministic 25k/25k split
    if len(df) < 50000:
        print(f"[WARN] CSV has {len(df)} rows; expected 50,000. Proceeding anyway.")
    train_df = df.iloc[:25000].copy()
    test_df = df.iloc[25000:50000].copy()

    # Clean
    train_texts = [clean_text(t) for t in train_df["review"].astype(str).tolist()]
    test_texts = [clean_text(t) for t in test_df["review"].astype(str).tolist()]

    # Ensure NLTK punkt exists
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Tokenize
    train_tokens = tokenize(train_texts)
    test_tokens = tokenize(test_texts)

    # Vocab from train only
    stoi, itos = build_vocab(train_tokens, vocab_size=args.vocab_size)

    # Encode
    train_seqs = encode(train_tokens, stoi)
    test_seqs = encode(test_tokens, stoi)

    # Labels -> 0/1
    lab_map = {"negative": 0, "positive": 1}
    y_train = train_df["sentiment"].str.lower().map(lab_map).astype(int).to_numpy()
    y_test = test_df["sentiment"].str.lower().map(lab_map).astype(int).to_numpy()

    # Save artifacts
    with open("data/processed/vocab.json", "w") as f:
        json.dump(stoi, f)

    np.save("data/processed/train_sequences.npy", np.array(train_seqs, dtype=object), allow_pickle=True)
    np.save("data/processed/test_sequences.npy", np.array(test_seqs, dtype=object), allow_pickle=True)
    np.save("data/processed/train_labels.npy", y_train)
    np.save("data/processed/test_labels.npy", y_test)

    avg_len = int(np.mean([len(s) for s in train_seqs])) if len(train_seqs) else 0
    print("[OK] Saved artifacts to data/processed/")
    print(f"Vocab size: {len(stoi)} | Avg train length: {avg_len} tokens | "
          f"Train: {len(y_train)} | Test: {len(y_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/IMDB Dataset.csv",
                        help="Path to IMDb CSV with columns: review,sentiment")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()
    main(args)
