# HWRS640 – Assignment 4: Streamflow Prediction with Sequence Models

Daily streamflow prediction from Daymet meteorological forcings and static basin attributes using LSTM and Transformer models trained on the [minicamels](https://github.com/BennettHydroLab/minicamels) dataset (50 CAMELS-US basins, WY1981–WY2010).

---

## Repository structure

```
.
├── main.py                  # CLI entry point
├── src/
│   ├── data.py              # dataset loading, preprocessing, DataLoaders
│   ├── model.py             # LSTM and Transformer model definitions
│   ├── train.py             # training / validation loops, checkpointing
│   ├── utils.py             # NSE, KGE, RMSE, MAE, percent-bias metrics
│   └── visualization.py     # all plotting functions
├── notebooks/
│   └── 01_eda.ipynb         # exploratory data analysis
├── outputs/                 # checkpoints, normalizers, plots (git-ignored)
├── configs/                 # optional config files
├── pyproject.toml
└── uv.lock
```

---

## Setup

[uv](https://github.com/astral-sh/uv) is used to manage the environment.

```bash
# Install uv if needed
curl -Lsf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> && cd hwrs640_rnn_hw
uv sync
```

Training requires a CUDA GPU. The code defaults to CUDA when available, otherwise falls back to CPU.

---

## CLI

All commands are exposed through `main.py`.

### Summarise the dataset

```bash
uv run python main.py summarize-data            # print dataset overview
uv run python main.py summarize-data --plots    # also save EDA plots to outputs/eda/
```

### Train a model

```bash
# LSTM (default settings)
uv run python main.py train --model lstm --seq-len 365 --epochs 50

# Transformer
uv run python main.py train --model transformer --seq-len 365 --epochs 50

# Full options
uv run python main.py train \
  --model lstm \
  --seq-len 365 \
  --epochs 50 \
  --lr 1e-3 \
  --batch-size 256 \
  --hidden-size 128 \
  --num-layers 2 \
  --dropout 0.2 \
  --loss mse \
  --patience 10 \
  --seed 42 \
  --output-dir outputs \
  --run-name lstm_v1
```

Checkpoints are written to `outputs/<run-name>_best.pt`.
Training history is saved to `outputs/<run-name>_history.json`.

### Evaluate a trained model

```bash
uv run python main.py evaluate \
  --checkpoint outputs/lstm_best.pt \
  --model lstm \
  --split test
```

Prints median NSE/KGE/RMSE/MAE/PBIAS across all 50 basins and saves per-basin results to `outputs/eval_test_lstm.csv`.

### Generate plots

```bash
uv run python main.py plot \
  --checkpoint outputs/lstm_best.pt \
  --model lstm \
  --split test \
  --history outputs/lstm_history.json
```

Saves hydrographs, scatter plots, and an NSE bar chart to `outputs/plots/`.

---

## Model architectures

### LSTM
Input `(batch, seq_len, 21)` → stacked LSTM → take last hidden state → Linear → scalar prediction.

- **Strength**: captures long-range temporal dependencies and handles irregular seasonality naturally.
- **Weakness**: sequential computation; fixed-size hidden state may lose distant information.

### Transformer
Input `(batch, seq_len, 21)` → linear projection → positional encoding → Transformer encoder → mean-pool → Linear → scalar prediction.

- **Strength**: parallel training; full-context attention with no vanishing-gradient issue.
- **Weakness**: quadratic memory w.r.t. `seq_len`; needs more data to match LSTM performance.

Static basin attributes (16 features) are concatenated to the 5 dynamic forcings at each timestep, giving `21` input features total.

---

## Data split

| Split | Water years    | Dates                   |
|-------|---------------|-------------------------|
| Train | WY1981–WY2003 | 1980-10-01 – 2003-09-30 |
| Val   | WY2004–WY2006 | 2003-10-01 – 2006-09-30 |
| Test  | WY2007–WY2010 | 2006-10-01 – 2010-09-30 |

Temporal split across all 50 basins. Normalisation statistics computed on the training split only to avoid leakage.

---

## Reproducing the best model run

```bash
uv run python main.py train \
  --model lstm \
  --seq-len 365 \
  --epochs 50 \
  --lr 1e-3 \
  --batch-size 256 \
  --hidden-size 128 \
  --num-layers 2 \
  --dropout 0.2 \
  --loss mse \
  --patience 10 \
  --seed 42 \
  --output-dir outputs \
  --run-name lstm_best

uv run python main.py evaluate \
  --checkpoint outputs/lstm_best_best.pt \
  --model lstm \
  --split test

uv run python main.py plot \
  --checkpoint outputs/lstm_best_best.pt \
  --model lstm \
  --split test \
  --history outputs/lstm_best_history.json
```
