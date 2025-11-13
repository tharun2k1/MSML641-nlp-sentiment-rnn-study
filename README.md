# Comparative Analysis of RNN Architectures for Sentiment Classification (IMDb)

This repository implements a **controlled comparative study** of RNN architectures for binary sentiment classification on the **IMDb 50k** dataset.  
You can reproduce all results (metrics table + plots) with **one command**.

---

## 1) Setup

### Requirements
- **Python** 3.9–3.11 (tested on 3.9.6)
- macOS / Linux / Windows
- CPU or Apple **MPS** / NVIDIA CUDA (optional)

# Clone the repository
git clone https://github.com/tharun2k1/MSML641-nlp-sentiment-rnn-study.git
cd MSML641-nlp-sentiment-rnn-study

### Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows PowerShell
```

### Install dependencies
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### NLTK tokenizer data
Download NLTK tokenizer models (required for nltk.word_tokenize)

```bash
python3 - <<'PY'
import nltk
# punkt is enough for most installs, but some environments now need punkt_tab as well.
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.download(pkg)
    except Exception as e:
        print(f"Could not download {pkg}: {e}")
PY
```

---

## 2) Data

Download the Dataset from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the IMDb CSV at:
```
data/IMDB Dataset.csv
```
> If the file name or path differs, pass `--csv_path` to `src/train.py`.

On first run, the code creates processed artifacts here:
```
data/processed/
  ├── vocab.json
  ├── train_sequences.npy
  ├── train_labels.npy
  ├── test_sequences.npy
  └── test_labels.npy
```

Preprocessing details:
- lowercase, strip punctuation/special chars
- Tokenize with NLTK
- Build 10k-word vocabulary (train only) and added <pad>=0, <unk>.
- Map tokens to ids
- pad/truncate to **25 / 50 / 100** tokens (tested in experiments)

---

## 3) Command to run the entire project

This command runs the **factor‑at‑a‑time** sweeps (36 experiments), writes a single CSV, and produces plots.

```bash
python3 src/train.py --run_all 1
```
Optional flags:
```bash
# Force CPU or choose device
python3 src/train.py --run_all 1 --device cpu

# Custom epochs (default 5 for run_all)
python3 src/train.py --run_all 1 --epochs 5

# Custom data path / vocab size
python3 src/train.py --run_all 1 --csv_path "data/IMDB Dataset.csv" --vocab_size 10000
```

Outputs:
```
results/
  ├── metrics.csv           # aggregated results table
  ├── logs/                 # JSON loss logs per run
  ├── plots/
  │   ├── acc_f1_vs_seq_len.png
  │   └── loss_best_vs_worst.png
  └── hardware.json         # hardware snapshot
```

Generate a hardware snapshot (for the report) without retraining:
```bash
python3 src/hardware_info.py
```

---

## 4) Run selected experiments 

You can also run a smaller grid manually. Examples:

**Change only the optimizer (fix arch=LSTM, act=ReLU, seq_len=50, no clip):**
```bash
python3 src/train.py --architectures lstm --activations relu --seq_lens 50 \
  --clip 0 --optimizers adam sgd rmsprop --epochs 5 --device cpu
```

**Change only the sequence length (fix arch=LSTM, act=ReLU, opt=Adam):**
```bash
python3 src/train.py --architectures lstm --activations relu --optimizers adam \
  --seq_lens 25 50 100 --clip 0 --epochs 5 --device cpu
```

**Architectures (RNN/LSTM/BiLSTM) at seq_len=50, fixed Adam/ReLU:**
```bash
python3 src/train.py --architectures rnn lstm bilstm --activations relu \
  --optimizers adam --seq_lens 50 --clip 0 --epochs 5 --device cpu
```

**Make plots from the current `results/metrics.csv`:**
```bash
python3 src/plot_results.py
```

---

## 5) Expected runtime 

On an 8‑core Apple Silicon laptop (CPU):  

- **Per epoch** ≈ **5–6 s** (seq_len=25), **~10 s** (50), **~20 s** (100) for LSTM (batch 32).  

- Full `--run_all 1` (~36 runs × 5 epochs by default) typically completes within **40 minutes** on CPU.

Your runtime may vary by hardware, Python/PyTorch build, and background load.

---

## 6) Project structure

```
├── data/
│   └── IMDB Dataset.csv              # (you supply)
├── src/
│   ├── preprocess.py                 # build vocab + processed splits
│   ├── models.py                     # RNN / LSTM / BiLSTM variants
│   ├── train.py                      # grid runner + --run_all orchestrator
│   ├── evaluate.py                   # utilities to score predictions
│   ├── utils.py                      # common helpers
│   ├── plot_results.py               # figures for the report
│   └── hardware_info.py              # environment snapshot (JSON)
├── results/
│   ├── metrics.csv
│   ├── plots/
│   └── hardware.json
├── requirements.txt
└── README.md
```

---

## 7) Reproducibility

Ensures reproducible results across runs. Same splits, shuffles, and inits.

- Seeds fixed in `src/train.py` (`set_seed(42)`):
  ```python
  import random, numpy as np, torch, os
  os.environ["PYTHONHASHSEED"] = "42"
  random.seed(42); np.random.seed(42); torch.manual_seed(42)
  torch.use_deterministic_algorithms(True)
  ```


---

## 8) Troubleshooting

- **NLTK error: `LookupError: punkt_tab`**  
  Re‑run the NLTK download cell above (ensure the venv is active).

- **Slow runs**  
  Lower `--epochs` for quick smoke tests or run single sweeps (Section 4).

- **Different IMDb path**  
  Pass `--csv_path` to `src/train.py` and `src/preprocess.py`.

---


