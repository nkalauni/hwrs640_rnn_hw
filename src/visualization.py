"""Plotting functions for training diagnostics and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Plot train/val loss and NSE/KGE over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"],   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NSE and KGE
    ax = axes[1]
    ax.plot(epochs, history["val_nse"], label="val NSE")
    ax.plot(epochs, history["val_kge"], label="val KGE")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Time series (obs vs sim)
# ──────────────────────────────────────────────────────────────────────────────

def plot_hydrograph(
    obs: np.ndarray,
    sim: np.ndarray,
    dates: pd.DatetimeIndex,
    basin_id: str,
    metrics: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot observed and predicted streamflow time series for one basin."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, obs, label="Observed", linewidth=0.8, color="steelblue")
    ax.plot(dates, sim, label="Predicted", linewidth=0.8, color="tomato", alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow (mm/day)")
    title = f"Basin {basin_id}"
    if metrics:
        title += (
            f"  |  NSE={metrics.get('nse', float('nan')):.3f}"
            f"  KGE={metrics.get('kge', float('nan')):.3f}"
            f"  RMSE={metrics.get('rmse', float('nan')):.3f}"
        )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Scatter / parity plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter(
    obs: np.ndarray,
    sim: np.ndarray,
    basin_id: str,
    metrics: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Parity scatter plot of observed vs predicted streamflow."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, sim, s=3, alpha=0.4, color="steelblue")

    lims = [
        min(obs.min(), sim.min()) * 0.95,
        max(obs.max(), sim.max()) * 1.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed (mm/day)")
    ax.set_ylabel("Predicted (mm/day)")
    title = f"Basin {basin_id}"
    if metrics:
        title += f"  NSE={metrics.get('nse', float('nan')):.3f}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-basin metric bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_bars(
    metrics_df: pd.DataFrame,
    metric: str = "nse",
    save_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart of a metric across all evaluated basins."""
    df = metrics_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))
    colors = ["tomato" if v < 0 else "steelblue" for v in df[metric]]
    ax.barh(df.index.astype(str), df[metric], color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{metric.upper()} by basin (test set)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# EDA plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_streamflow_timeseries(
    basin_dfs: dict,
    basin_ids: Sequence[str],
    save_path: Optional[Path] = None,
) -> None:
    """Quick time-series overview for a list of basins."""
    n = len(basin_ids)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, bid in zip(axes, basin_ids):
        df = basin_dfs[bid]
        ax.plot(df.index, df["qobs"], linewidth=0.6, color="steelblue")
        ax.set_ylabel("qobs (mm/d)")
        ax.set_title(f"Basin {bid}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_forcing_and_streamflow(
    basin_df: pd.DataFrame,
    basin_id: str,
    start: str = "2000-10-01",
    end: str = "2002-09-30",
    save_path: Optional[Path] = None,
) -> None:
    """Plot precipitation and streamflow for a single basin over a time window."""
    sub = basin_df.loc[start:end]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].bar(sub.index, sub["prcp"], color="cornflowerblue", width=1, label="prcp")
    axes[0].set_ylabel("Precip (mm/day)")
    axes[0].set_title(f"Basin {basin_id}: forcing and streamflow")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sub.index, sub["qobs"], color="steelblue", linewidth=0.8)
    axes[1].set_ylabel("qobs (mm/day)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_qobs_histogram(
    basin_dfs: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Histogram of log-transformed qobs across all basins."""
    all_q = np.concatenate(
        [df["qobs"].dropna().values for df in basin_dfs.values()]
    )
    all_q = all_q[all_q > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_q, bins=80, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("qobs (mm/day)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Streamflow distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.log1p(all_q), bins=80, color="steelblue", edgecolor="none")
    axes[1].set_xlabel("log(1 + qobs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("log-transformed streamflow")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_static_scatter(
    attrs_df: pd.DataFrame,
    x_col: str = "aridity",
    y_col: str = "runoff_ratio",
    color_col: str = "mean_prcp",
    save_path: Optional[Path] = None,
) -> None:
    """Scatter plot of two static attributes, coloured by a third."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        attrs_df[x_col], attrs_df[y_col],
        c=attrs_df[color_col], cmap="viridis", s=60, edgecolors="k", linewidths=0.4
    )
    plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Private helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()
