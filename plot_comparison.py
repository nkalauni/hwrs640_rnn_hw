"""Plot CDF comparison of LSTM vs Transformer across all metrics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

lstm_df = pd.read_csv("outputs/eval_test_lstm.csv", index_col="basin_id")
trans_df = pd.read_csv("outputs/eval_test_transformer.csv", index_col="basin_id")

metrics = {
    "nse":   ("NSE",   "higher is better"),
    "kge":   ("KGE",   "higher is better"),
    "rmse":  ("RMSE",  "lower is better"),
    "mae":   ("MAE",   "lower is better"),
    "pbias": ("|PBIAS|", "lower is better"),
}

fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))

for ax, (col, (label, note)) in zip(axes, metrics.items()):
    lstm_vals  = lstm_df[col].abs() if col == "pbias" else lstm_df[col]
    trans_vals = trans_df[col].abs() if col == "pbias" else trans_df[col]

    for vals, name, color in [
        (lstm_vals,  "LSTM",        "steelblue"),
        (trans_vals, "Transformer", "tomato"),
    ]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=f"{name} (med={np.median(vals):.3f})",
                color=color, linewidth=2)

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(f"{label}\n({note})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

fig.suptitle("LSTM vs Transformer — Test Set Performance (50 basins)", fontsize=13, fontweight="bold")
plt.tight_layout()

out = Path("outputs/plots/comparison_cdf.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
