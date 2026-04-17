"""Dataset loading, preprocessing, and PyTorch DataLoader creation.

Split strategy
--------------
We use a temporal split across all 50 basins:
  - Train : WY1981–WY2003  (Oct 1 1980 – Sep 30 2003, 23 years)
  - Val   : WY2004–WY2006  (Oct 1 2003 – Sep 30 2006,  3 years)
  - Test  : WY2007–WY2010  (Oct 1 2006 – Sep 30 2010,  4 years)

Rationale: a temporal hold-out is standard in hydrology because basin-level
heterogeneity is less of a concern for a general rainfall-runoff model, and it
tests genuine extrapolation in time. Using a fixed future period for the test
set prevents any lookahead leakage.

Model input / target
--------------------
  Input  : (seq_len, num_dynamic + num_static)
           5 Daymet forcings concatenated with 16 static attributes (tiled)
  Target : qobs at the last timestep in the window (one-step-ahead prediction)

Normalisation
-------------
  - Dynamic features and qobs: z-score using per-feature mean/std computed on
    the training set only.
  - Static attributes: z-score using per-attribute mean/std computed on the
    training set only.
  - Model outputs and targets are always in normalised space during training;
    inverse transform is applied at evaluation time.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DYNAMIC_VARS = ["prcp", "tmax", "tmin", "srad", "vp"]
TARGET_VAR = "qobs"

# Temporal split boundaries (inclusive start, inclusive end)
SPLIT_DATES = {
    "train": ("1980-10-01", "2003-09-30"),
    "val":   ("2003-10-01", "2006-09-30"),
    "test":  ("2006-10-01", "2010-09-30"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """Stores (mean, std) and applies z-score transform / inverse."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Normalizer":
        """x : (N, features) or (N,)."""
        self.mean_ = np.nanmean(x, axis=0)
        self.std_  = np.nanstd(x, axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.mean_ = d["mean"]
        obj.std_  = d["std"]
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Raw data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_basins(mc) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all basins and static attributes.

    Returns
    -------
    basin_dfs : dict mapping basin_id -> pd.DataFrame with columns
                [prcp, tmax, tmin, srad, vp, qobs] indexed by date.
    attrs_df  : pd.DataFrame (basin_id x static_features).
    """
    basin_ids = mc.basins()["basin_id"].tolist()
    attrs_df = mc.attributes()

    basin_dfs: dict[str, pd.DataFrame] = {}
    for bid in basin_ids:
        ds = mc.load_basin(bid)
        df = ds.to_dataframe()[DYNAMIC_VARS + [TARGET_VAR]]
        df.index = pd.to_datetime(df.index)
        basin_dfs[bid] = df

    return basin_dfs, attrs_df


def split_basin_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLIT_DATES[split]
    return df.loc[start:end]


# ──────────────────────────────────────────────────────────────────────────────
# Fit normalisers on training data
# ──────────────────────────────────────────────────────────────────────────────

def fit_normalizers(
    basin_dfs: dict[str, pd.DataFrame],
    attrs_df: pd.DataFrame,
) -> tuple[Normalizer, Normalizer, Normalizer]:
    """Compute normalization stats from the training split only.

    Returns
    -------
    dyn_norm   : Normalizer for dynamic features (len DYNAMIC_VARS)
    tgt_norm   : Normalizer for qobs (scalar, stored as shape (1,))
    static_norm: Normalizer for static attributes
    """
    train_dyn_rows = []
    train_tgt_rows = []
    for df in basin_dfs.values():
        tr = split_basin_df(df, "train")
        train_dyn_rows.append(tr[DYNAMIC_VARS].values)
        train_tgt_rows.append(tr[[TARGET_VAR]].values)

    dyn_all = np.concatenate(train_dyn_rows, axis=0)
    tgt_all = np.concatenate(train_tgt_rows, axis=0)

    dyn_norm = Normalizer().fit(dyn_all)
    tgt_norm = Normalizer().fit(tgt_all)

    train_basin_ids = list(basin_dfs.keys())
    static_arr = attrs_df.loc[train_basin_ids].values.astype(float)
    static_norm = Normalizer().fit(static_arr)

    return dyn_norm, tgt_norm, static_norm


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class StreamflowDataset(Dataset):
    """Sliding-window dataset for rainfall-runoff sequence modelling.

    Each sample:
      x : (seq_len, num_dynamic + num_static)  – normalised
      y : scalar  – normalised qobs at position seq_len-1 (last timestep)
    """

    def __init__(
        self,
        basin_dfs: dict[str, pd.DataFrame],
        attrs_df: pd.DataFrame,
        split: str,
        seq_len: int,
        dyn_norm: Normalizer,
        tgt_norm: Normalizer,
        static_norm: Normalizer,
    ):
        self.seq_len = seq_len
        self.samples: list[tuple[np.ndarray, float]] = []

        static_cols = attrs_df.columns.tolist()

        for basin_id, df in basin_dfs.items():
            split_df = split_basin_df(df, split)

            # Drop rows where qobs is NaN
            split_df = split_df.dropna(subset=[TARGET_VAR])
            if len(split_df) < seq_len:
                continue

            dyn = dyn_norm.transform(split_df[DYNAMIC_VARS].values.astype(float))
            tgt = tgt_norm.transform(
                split_df[[TARGET_VAR]].values.astype(float)
            ).squeeze()  # (T,)

            # Static attributes for this basin (normalised), tiled per timestep
            raw_static = attrs_df.loc[basin_id, static_cols].values.astype(float)
            static_normed = static_norm.transform(raw_static[np.newaxis, :])[0]  # (S,)
            static_tiled = np.tile(static_normed, (len(split_df), 1))  # (T, S)

            x_full = np.concatenate([dyn, static_tiled], axis=1).astype(np.float32)
            tgt_full = tgt.astype(np.float32)

            # Sliding window: predict qobs at position `t = seq_len - 1`
            for t in range(seq_len - 1, len(x_full)):
                x_window = x_full[t - seq_len + 1 : t + 1]  # (seq_len, F)
                y_val = tgt_full[t]
                self.samples.append((x_window, float(y_val)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    mc,
    seq_len: int = 365,
    batch_size: int = 256,
    num_workers: int = 0,
    cache_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test DataLoaders and return normaliser metadata.

    Parameters
    ----------
    mc         : MiniCamels instance
    seq_len    : number of daily timesteps in each input sequence
    batch_size : mini-batch size
    num_workers: DataLoader worker processes
    cache_dir  : if provided, normaliser objects are saved here as .pkl files

    Returns
    -------
    train_loader, val_loader, test_loader, meta
    meta keys: dyn_norm, tgt_norm, static_norm, num_features, num_dynamic,
               num_static, basin_ids
    """
    print("Loading basin data from minicamels (remote fetch)...")
    basin_dfs, attrs_df = load_all_basins(mc)
    basin_ids = list(basin_dfs.keys())

    print("Fitting normalisers on training split...")
    dyn_norm, tgt_norm, static_norm = fit_normalizers(basin_dfs, attrs_df)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dyn_norm.save(cache_dir / "dyn_norm.pkl")
        tgt_norm.save(cache_dir / "tgt_norm.pkl")
        static_norm.save(cache_dir / "static_norm.pkl")
        print(f"Normalisers saved to {cache_dir}")

    num_dynamic = len(DYNAMIC_VARS)
    num_static  = attrs_df.shape[1]
    num_features = num_dynamic + num_static

    print(f"Building datasets (seq_len={seq_len}, features={num_features})...")
    common_kwargs = dict(
        basin_dfs=basin_dfs,
        attrs_df=attrs_df,
        seq_len=seq_len,
        dyn_norm=dyn_norm,
        tgt_norm=tgt_norm,
        static_norm=static_norm,
    )
    train_ds = StreamflowDataset(split="train", **common_kwargs)
    val_ds   = StreamflowDataset(split="val",   **common_kwargs)
    test_ds  = StreamflowDataset(split="test",  **common_kwargs)

    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} samples")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    meta = dict(
        dyn_norm=dyn_norm,
        tgt_norm=tgt_norm,
        static_norm=static_norm,
        num_features=num_features,
        num_dynamic=num_dynamic,
        num_static=num_static,
        basin_ids=basin_ids,
        seq_len=seq_len,
        basin_dfs=basin_dfs,
        attrs_df=attrs_df,
    )
    return train_loader, val_loader, test_loader, meta


def get_basin_predictions(
    model: torch.nn.Module,
    basin_id: str,
    basin_dfs: dict[str, pd.DataFrame],
    attrs_df: pd.DataFrame,
    split: str,
    seq_len: int,
    dyn_norm: Normalizer,
    tgt_norm: Normalizer,
    static_norm: Normalizer,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Run inference on a single basin and return (obs, sim, dates) in original units."""
    model.eval()

    ds = StreamflowDataset(
        basin_dfs={basin_id: basin_dfs[basin_id]},
        attrs_df=attrs_df,
        split=split,
        seq_len=seq_len,
        dyn_norm=dyn_norm,
        tgt_norm=tgt_norm,
        static_norm=static_norm,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).squeeze(-1).cpu().numpy()
            preds.append(out)
            targets.append(y.numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Inverse transform from normalised space
    preds_orig   = tgt_norm.inverse_transform(preds[:, np.newaxis]).squeeze()
    targets_orig = tgt_norm.inverse_transform(targets[:, np.newaxis]).squeeze()

    # Recover dates corresponding to each sample's last timestep
    split_df = split_basin_df(basin_dfs[basin_id], split).dropna(subset=[TARGET_VAR])
    dates = split_df.index[seq_len - 1:]

    return targets_orig, preds_orig, dates
