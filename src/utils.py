"""Hydrologic metrics and general helper utilities."""

import numpy as np
import torch


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009)."""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if obs.std() == 0 or sim.std() == 0:
        return np.nan
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean()) if obs.mean() != 0 else np.nan
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root mean squared error."""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean absolute error."""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(np.abs(obs[mask] - sim[mask])))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent bias."""
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if obs.sum() == 0:
        return np.nan
    return float(100.0 * (sim.sum() - obs.sum()) / obs.sum())


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Return a dict of all metrics."""
    return {
        "nse": nse(obs, sim),
        "kge": kge(obs, sim),
        "rmse": rmse(obs, sim),
        "mae": mae(obs, sim),
        "pbias": pbias(obs, sim),
    }


def get_device(force_cuda: bool = False) -> torch.device:
    """Return CUDA device if available, else CPU."""
    if force_cuda:
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
