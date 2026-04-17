"""Training and validation routines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import compute_all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# NSE loss (differentiable, used optionally alongside MSE)
# ──────────────────────────────────────────────────────────────────────────────

class NSELoss(nn.Module):
    """1 - NSE loss (minimise → maximise NSE).

    Operates in normalised target space, so the variance term is fixed at 1.
    Falls back to MSE-equivalent behaviour when batch variance is near zero.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residuals = target - pred
        var = target.var(unbiased=False) + 1e-6
        return (residuals ** 2).mean() / var


# ──────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_norm,
) -> tuple[float, dict]:
    """Run inference on a loader; return (loss, metrics_dict) in original units."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred, all_target = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            all_pred.append(pred.cpu().numpy())
            all_target.append(y.cpu().numpy())

    all_pred   = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)

    # Inverse-transform to original streamflow units for metric calculation
    pred_orig   = tgt_norm.inverse_transform(all_pred[:, np.newaxis]).squeeze()
    target_orig = tgt_norm.inverse_transform(all_target[:, np.newaxis]).squeeze()

    metrics = compute_all_metrics(target_orig, pred_orig)
    return total_loss / n, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tgt_norm,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 10,
    checkpoint_dir: Path = Path("outputs"),
    loss_fn: str = "mse",
    run_name: str = "model",
) -> dict:
    """Train the model and return a history dict.

    Parameters
    ----------
    model          : instantiated nn.Module (already on *device*)
    train_loader   : training DataLoader
    val_loader     : validation DataLoader
    tgt_norm       : Normalizer for the target variable
    device         : torch.device
    epochs         : maximum epochs
    lr             : initial learning rate
    weight_decay   : AdamW weight decay
    grad_clip      : max gradient norm (0 to disable)
    patience       : early-stopping patience (epochs without val improvement)
    checkpoint_dir : directory to write best checkpoint and history JSON
    loss_fn        : 'mse' | 'mae' | 'nse'
    run_name       : prefix for saved files

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'val_nse', 'val_kge', ...
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion: nn.Module
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "mae":
        criterion = nn.L1Loss()
    elif loss_fn == "nse":
        criterion = NSELoss()
    else:
        raise ValueError(f"Unknown loss_fn '{loss_fn}'")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss":   [],
        "val_nse":    [],
        "val_kge":    [],
        "val_rmse":   [],
    }

    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    ckpt_path     = checkpoint_dir / f"{run_name}_best.pt"

    print(f"\nTraining on {device} | epochs={epochs} | lr={lr} | loss={loss_fn}")
    print(f"Checkpoint → {ckpt_path}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_loss, val_metrics = _eval_epoch(model, val_loader, criterion, device, tgt_norm)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_nse"].append(val_metrics["nse"])
        history["val_kge"].append(val_metrics["kge"])
        history["val_rmse"].append(val_metrics["rmse"])

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "val_loss":   val_loss,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )

        else:
            no_improve += 1

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"NSE={val_metrics['nse']:.3f} KGE={val_metrics['kge']:.3f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    print(f"\nBest val loss {best_val_loss:.4f} at epoch {best_epoch}")

    # Save history
    history_path = checkpoint_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    return history
