"""CLI entry point for HWRS640 streamflow prediction project."""

import json
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import click
import torch

from minicamels import MiniCamels

from data import build_dataloaders, get_basin_predictions, load_all_basins, fit_normalizers
from model import build_model
from train import train
from utils import compute_all_metrics, get_device, set_seed
from visualization import (
    plot_training_curves,
    plot_hydrograph,
    plot_scatter,
    plot_metric_bars,
    plot_streamflow_timeseries,
    plot_forcing_and_streamflow,
    plot_qobs_histogram,
    plot_static_scatter,
)

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CLI root
# ──────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """HWRS640 streamflow prediction CLI."""


# ──────────────────────────────────────────────────────────────────────────────
# summarize-data
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("summarize-data")
@click.option("--plots/--no-plots", default=False, help="Save EDA plots to outputs/eda/")
def summarize_data(plots: bool):
    """Summarise the minicamels dataset and optionally produce EDA plots."""
    mc = MiniCamels()
    basins_df = mc.basins()
    attrs_df  = mc.attributes()
    basin_ids = basins_df["basin_id"].tolist()

    click.echo(f"\nDataset overview")
    click.echo(f"  Basins              : {len(basin_ids)}")
    click.echo(f"  Period              : 1980-10-01 – 2010-09-30 (WY1981–WY2010)")
    click.echo(f"  Dynamic inputs      : prcp, tmax, tmin, srad, vp")
    click.echo(f"  Target variable     : qobs (mm/day)")
    click.echo(f"  Static attributes   : {attrs_df.shape[1]} ({', '.join(attrs_df.columns.tolist())})")
    click.echo(f"\nSplit strategy (temporal):")
    click.echo(f"  Train : WY1981–WY2003  (1980-10-01 – 2003-09-30)")
    click.echo(f"  Val   : WY2004–WY2006  (2003-10-01 – 2006-09-30)")
    click.echo(f"  Test  : WY2007–WY2010  (2006-10-01 – 2010-09-30)")

    if not plots:
        click.echo("\n(Pass --plots to generate EDA figures)")
        return

    click.echo("\nLoading basin data for EDA plots...")
    basin_dfs, _ = load_all_basins(mc)

    out = Path("outputs") / "eda"
    out.mkdir(parents=True, exist_ok=True)

    sample_ids = basin_ids[:5]
    plot_streamflow_timeseries(basin_dfs, sample_ids, save_path=out / "streamflow_timeseries.png")
    plot_forcing_and_streamflow(basin_dfs[basin_ids[0]], basin_ids[0], save_path=out / "forcing_qobs.png")
    plot_qobs_histogram(basin_dfs, save_path=out / "qobs_histogram.png")
    plot_static_scatter(attrs_df, save_path=out / "static_scatter.png")
    click.echo(f"\nEDA plots saved to {out}/")


# ──────────────────────────────────────────────────────────────────────────────
# train
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("train")
@click.option("--model",        "model_type", default="lstm", show_default=True,
              type=click.Choice(["lstm", "transformer"], case_sensitive=False),
              help="Model architecture")
@click.option("--seq-len",      default=365,  show_default=True, help="Sequence length (days)")
@click.option("--epochs",       default=30,   show_default=True, help="Max training epochs")
@click.option("--lr",           default=1e-3, show_default=True, help="Learning rate")
@click.option("--batch-size",   default=256,  show_default=True, help="Mini-batch size")
@click.option("--hidden-size",  default=128,  show_default=True, help="LSTM hidden size / Transformer d_model")
@click.option("--num-layers",   default=2,    show_default=True, help="Number of layers")
@click.option("--dropout",      default=0.2,  show_default=True, help="Dropout probability")
@click.option("--loss",         "loss_fn",    default="mse",    show_default=True,
              type=click.Choice(["mse", "mae", "nse"]), help="Loss function")
@click.option("--patience",     default=10,   show_default=True, help="Early-stopping patience")
@click.option("--seed",         default=42,   show_default=True, help="Random seed")
@click.option("--output-dir",   default="outputs", show_default=True, help="Checkpoint directory")
@click.option("--run-name",     default=None, help="Run name prefix (default: model type)")
@click.option("--num-workers",  default=0,    show_default=True, help="DataLoader workers")
def train_cmd(
    model_type, seq_len, epochs, lr, batch_size, hidden_size, num_layers,
    dropout, loss_fn, patience, seed, output_dir, run_name, num_workers,
):
    """Train a sequence model on the minicamels dataset."""
    set_seed(seed)
    device = get_device()
    run_name = run_name or model_type

    click.echo(f"\nDevice: {device}")
    mc = MiniCamels()

    train_loader, val_loader, _, meta = build_dataloaders(
        mc,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=Path(output_dir) / "normalizers",
    )

    num_features = meta["num_features"]
    click.echo(f"Input features: {num_features} (dynamic={meta['num_dynamic']}, static={meta['num_static']})")

    # Build model kwargs depending on architecture
    if model_type == "lstm":
        model_kwargs = dict(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    else:
        nhead = 4
        while hidden_size % nhead != 0:
            nhead //= 2
        model_kwargs = dict(
            d_model=hidden_size, nhead=nhead, num_layers=num_layers,
            dim_ff=hidden_size * 2, dropout=dropout
        )

    model = build_model(model_type, num_features, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Model: {model_type.upper()} | params={n_params:,}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tgt_norm=meta["tgt_norm"],
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        checkpoint_dir=Path(output_dir),
        loss_fn=loss_fn,
        run_name=run_name,
    )

    click.echo(f"\nDone. Best val NSE = {max(history['val_nse']):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("evaluate")
@click.option("--checkpoint", required=True,   help="Path to .pt checkpoint file")
@click.option("--model",      "model_type",    default="lstm", show_default=True,
              type=click.Choice(["lstm", "transformer"], case_sensitive=False))
@click.option("--seq-len",    default=365,     show_default=True)
@click.option("--hidden-size",default=128,     show_default=True)
@click.option("--num-layers", default=2,       show_default=True)
@click.option("--dropout",    default=0.2,     show_default=True)
@click.option("--split",      default="test",  show_default=True,
              type=click.Choice(["train", "val", "test"]))
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--batch-size", default=512,     show_default=True)
def evaluate_cmd(checkpoint, model_type, seq_len, hidden_size, num_layers, dropout, split, output_dir, batch_size):
    """Evaluate a trained model on the specified split."""
    device = get_device()
    ckpt_path = Path(checkpoint)
    norm_dir  = Path(output_dir) / "normalizers"

    mc = MiniCamels()
    click.echo("Loading data...")
    basin_dfs, attrs_df = load_all_basins(mc)
    basin_ids = list(basin_dfs.keys())

    # Load normalisers
    from data import Normalizer
    dyn_norm    = Normalizer.load(norm_dir / "dyn_norm.pkl")
    tgt_norm    = Normalizer.load(norm_dir / "tgt_norm.pkl")
    static_norm = Normalizer.load(norm_dir / "static_norm.pkl")

    num_features = len(basin_dfs[basin_ids[0]].columns) - 1  # dynamic
    num_features += attrs_df.shape[1]

    # Rebuild model
    if model_type == "lstm":
        model_kwargs = dict(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    else:
        nhead = 4
        while hidden_size % nhead != 0:
            nhead //= 2
        model_kwargs = dict(d_model=hidden_size, nhead=nhead, num_layers=num_layers,
                            dim_ff=hidden_size * 2, dropout=dropout)

    model = build_model(model_type, num_features, **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    click.echo(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # Per-basin evaluation
    records = []
    for bid in basin_ids:
        obs, sim, _ = get_basin_predictions(
            model, bid, basin_dfs, attrs_df,
            split=split, seq_len=seq_len,
            dyn_norm=dyn_norm, tgt_norm=tgt_norm, static_norm=static_norm,
            device=device, batch_size=batch_size,
        )
        m = compute_all_metrics(obs, sim)
        m["basin_id"] = bid
        records.append(m)

    results_df = pd.DataFrame(records).set_index("basin_id")
    out_dir = Path(output_dir)
    results_path = out_dir / f"eval_{split}_{model_type}.csv"
    results_df.to_csv(results_path)

    click.echo(f"\n{split.upper()} set — median across {len(basin_ids)} basins:")
    for col in ["nse", "kge", "rmse", "mae", "pbias"]:
        click.echo(f"  {col.upper():8s}: {results_df[col].median():.4f}")

    best_nse  = results_df["nse"].idxmax()
    worst_nse = results_df["nse"].idxmin()
    click.echo(f"\nBest  NSE basin : {best_nse}  (NSE={results_df.loc[best_nse,'nse']:.3f})")
    click.echo(f"Worst NSE basin : {worst_nse}  (NSE={results_df.loc[worst_nse,'nse']:.3f})")
    click.echo(f"\nResults saved to {results_path}")


# ──────────────────────────────────────────────────────────────────────────────
# plot
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("plot")
@click.option("--checkpoint",  required=True, help="Path to .pt checkpoint file")
@click.option("--model",       "model_type",  default="lstm", show_default=True,
              type=click.Choice(["lstm", "transformer"], case_sensitive=False))
@click.option("--seq-len",     default=365,   show_default=True)
@click.option("--hidden-size", default=128,   show_default=True)
@click.option("--num-layers",  default=2,     show_default=True)
@click.option("--dropout",     default=0.2,   show_default=True)
@click.option("--split",       default="test", show_default=True,
              type=click.Choice(["train", "val", "test"]))
@click.option("--output-dir",  default="outputs", show_default=True)
@click.option("--history",     "history_path", default=None,
              help="Path to history JSON (for training curves). If omitted, skipped.")
@click.option("--batch-size",  default=512,   show_default=True)
def plot_cmd(checkpoint, model_type, seq_len, hidden_size, num_layers, dropout, split, output_dir, history_path, batch_size):
    """Generate evaluation and training curve plots."""
    device   = get_device()
    out_dir  = Path(output_dir)
    norm_dir = out_dir / "normalizers"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if history_path:
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, save_path=plot_dir / "training_curves.png")

    mc = MiniCamels()
    click.echo("Loading data...")
    basin_dfs, attrs_df = load_all_basins(mc)
    basin_ids = list(basin_dfs.keys())

    from data import Normalizer
    dyn_norm    = Normalizer.load(norm_dir / "dyn_norm.pkl")
    tgt_norm    = Normalizer.load(norm_dir / "tgt_norm.pkl")
    static_norm = Normalizer.load(norm_dir / "static_norm.pkl")

    num_features = len(["prcp", "tmax", "tmin", "srad", "vp"]) + attrs_df.shape[1]

    if model_type == "lstm":
        model_kwargs = dict(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    else:
        nhead = 4
        while hidden_size % nhead != 0:
            nhead //= 2
        model_kwargs = dict(d_model=hidden_size, nhead=nhead, num_layers=num_layers,
                            dim_ff=hidden_size * 2, dropout=dropout)

    model = build_model(model_type, num_features, **model_kwargs).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Collect all per-basin predictions for bar chart
    all_metrics = []
    for bid in basin_ids:
        obs, sim, dates = get_basin_predictions(
            model, bid, basin_dfs, attrs_df,
            split=split, seq_len=seq_len,
            dyn_norm=dyn_norm, tgt_norm=tgt_norm, static_norm=static_norm,
            device=device, batch_size=batch_size,
        )
        m = compute_all_metrics(obs, sim)
        m["basin_id"] = bid
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).set_index("basin_id")
    best_basin  = metrics_df["nse"].idxmax()
    worst_basin = metrics_df["nse"].idxmin()

    for bid, label in [(best_basin, "best"), (worst_basin, "worst")]:
        obs, sim, dates = get_basin_predictions(
            model, bid, basin_dfs, attrs_df,
            split=split, seq_len=seq_len,
            dyn_norm=dyn_norm, tgt_norm=tgt_norm, static_norm=static_norm,
            device=device, batch_size=batch_size,
        )
        m = compute_all_metrics(obs, sim)
        plot_hydrograph(obs, sim, dates, bid, metrics=m,
                        save_path=plot_dir / f"hydrograph_{label}_{bid}.png")
        plot_scatter(obs, sim, bid, metrics=m,
                     save_path=plot_dir / f"scatter_{label}_{bid}.png")

    plot_metric_bars(metrics_df, metric="nse",
                     save_path=plot_dir / f"nse_bars_{split}.png")

    click.echo(f"\nPlots saved to {plot_dir}/")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
