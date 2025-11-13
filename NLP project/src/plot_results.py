import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = "results/metrics.csv"

def plot_acc_f1_vs_seqlen(df: pd.DataFrame):
    # Aggregate across other factors by mean to visualize the trend vs sequence length
    agg = df.groupby("seq_len")[["accuracy", "f1"]].mean().reset_index()

    # Accuracy vs Sequence Length
    plt.figure()
    plt.plot(agg["seq_len"], agg["accuracy"], marker="o", label="Accuracy")
    plt.plot(agg["seq_len"], agg["f1"], marker="o", label="F1 (macro)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Score")
    plt.title("Accuracy & F1 vs Sequence Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/accuracy_f1_vs_seq_len.png", dpi=160)
    plt.close()

def plot_loss_best_worst(df: pd.DataFrame):
    # Best & worst by F1
    best = df.iloc[df["f1"].idxmax()]
    worst = df.iloc[df["f1"].idxmin()]

    def find_loss_json(row):
        # Try to match the run pattern used in train.py when naming files
        pattern = f"{row.model}_{row.activation}_{row.optimizer}_L{row.seq_len}_clip{row.grad_clip}"
        for p in glob.glob("results/logs/*_loss.json"):
            if pattern in p:
                return p
        # fallback to any file if exact match fails
        any_jsons = glob.glob("results/logs/*_loss.json")
        return any_jsons[0] if any_jsons else None

    best_json = find_loss_json(best)
    worst_json = find_loss_json(worst)

    if best_json and worst_json:
        with open(best_json) as f:
            b = json.load(f).get("loss", [])
        with open(worst_json) as f:
            w = json.load(f).get("loss", [])

        if b and w:
            plt.figure()
            plt.plot(range(1, len(b) + 1), b, marker="o", label=f"Best (F1={best.f1:.3f})")
            plt.plot(range(1, len(w) + 1), w, marker="o", label=f"Worst (F1={worst.f1:.3f})")
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss: Best vs Worst Models")
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/plots/loss_best_worst.png", dpi=160)
            plt.close()
        else:
            print("[WARN] Loss arrays empty in JSON files.")
    else:
        print("[WARN] Could not locate loss JSON files to plot best/worst.")

def main():
    if not os.path.exists(METRICS):
        raise SystemExit("results/metrics.csv not found. Run training first.")

    os.makedirs("results/plots", exist_ok=True)

    # Expect columns:
    # model,activation,optimizer,seq_len,grad_clip,accuracy,f1,epoch_time_s
    df = pd.read_csv(METRICS)
    required_cols = {"model","activation","optimizer","seq_len","grad_clip","accuracy","f1","epoch_time_s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"metrics.csv missing columns: {missing}")

    plot_acc_f1_vs_seqlen(df)
    plot_loss_best_worst(df)
    print("[OK] Plots saved in results/plots/")

if __name__ == "__main__":
    main()
