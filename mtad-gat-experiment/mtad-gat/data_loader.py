"""
Dynamic data loader for MTAD-GAT.

Supports three data sources:
  - Custom CSV  : any CSV with a 'timestamp' column + numeric feature columns
                  pass the file path as `source`
  - MSL / SMAP  : NASA datasets in either format:
                    (a) per-entity .npy files under <project_root>/MSL/ or SMAP/
                    (b) combined pickle files from mtad-gat-pytorch/datasets
                  pass "MSL" or "SMAP" as `source`

Returns normalised numpy arrays + metadata needed by train.py / infer.py.
Scaler is always fit on training data only.
"""

import ast
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from mtad_gat import TimeSeriesDataset


# Project root = parent of this file's directory (mtad-gat/../)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default path to the prepped pickle files (relative to this file)
_DATASETS_DIR = os.path.join(os.path.dirname(__file__),
                              "..", "mtad-gat-pytorch", "datasets")


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_csv(path: str):
    """Load a CSV with a 'timestamp' column.  Returns (df, feature_cols)."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        feature_cols = [c for c in df.columns if c != "timestamp"]
    else:
        # No timestamp column – treat all columns as features
        feature_cols = list(df.columns)
        df.insert(0, "timestamp", pd.RangeIndex(len(df)))

    return df, feature_cols


def _load_npy_dataset(name: str, base_dir: str):
    """
    Load MSL or SMAP from per-entity .npy files.

    Expected layout:
        <base_dir>/train/<chan_id>.npy
        <base_dir>/test/<chan_id>.npy
        <base_dir>/labeled_anomalies.csv

    All entities are concatenated in sorted order.
    Returns (train_arr, test_arr, test_labels).
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir  = os.path.join(base_dir, "test")
    label_csv = os.path.join(base_dir, "labeled_anomalies.csv")

    chan_ids = sorted(f[:-4] for f in os.listdir(train_dir) if f.endswith(".npy"))
    if not chan_ids:
        raise FileNotFoundError(f"No .npy files found in {train_dir}")

    train_parts, test_parts, label_parts = [], [], []

    lab_df = pd.read_csv(label_csv) if os.path.isfile(label_csv) else None

    for cid in chan_ids:
        train_parts.append(np.load(os.path.join(train_dir, f"{cid}.npy")).astype(np.float32))

        test_arr = np.load(os.path.join(test_dir, f"{cid}.npy")).astype(np.float32)
        test_parts.append(test_arr)
        n_test = len(test_arr)

        # Build per-entity label vector from labeled_anomalies.csv
        labels = np.zeros(n_test, dtype=np.int32)
        if lab_df is not None:
            row = lab_df[lab_df["chan_id"] == cid]
            if not row.empty:
                seqs = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
                for start, end in seqs:
                    labels[start: end + 1] = 1
        label_parts.append(labels)

    train = np.concatenate(train_parts, axis=0)
    test  = np.concatenate(test_parts,  axis=0)
    y     = np.concatenate(label_parts, axis=0)
    return train, test, y


def _load_pkl_dataset(name: str, datasets_dir: str):
    """Load MSL or SMAP from combined pickle files.  Returns (train_arr, test_arr, test_labels)."""
    base = os.path.join(datasets_dir, "data", "processed")
    with open(os.path.join(base, f"{name}_train.pkl"), "rb") as f:
        train = np.array(pickle.load(f), dtype=np.float32)
    with open(os.path.join(base, f"{name}_test.pkl"), "rb") as f:
        test = np.array(pickle.load(f), dtype=np.float32)
    label_path = os.path.join(base, f"{name}_test_label.pkl")
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            labels = np.array(pickle.load(f), dtype=np.int32).ravel()
    else:
        labels = None
    return train, test, labels


def _load_nasa_dataset(name: str, datasets_dir: str):
    """
    Auto-detect format: try npy-per-entity directory first, then pickle files.
    Returns (train_arr, test_arr, test_labels).
    """
    # Check for per-entity npy layout at project root
    npy_dir = os.path.join(_PROJECT_ROOT, name)
    if os.path.isdir(os.path.join(npy_dir, "train")):
        print(f"  Loading {name} from per-entity .npy files: {npy_dir}")
        return _load_npy_dataset(name, npy_dir)

    # Fall back to combined pickle layout
    print(f"  Loading {name} from pickle files: {datasets_dir}")
    return _load_pkl_dataset(name, datasets_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(source: str,
              datasets_dir: str = _DATASETS_DIR,
              cutoff_date: str = None,
              val_split: float = 0.1):
    """
    Load and normalise a dataset.

    Parameters
    ----------
    source : str
        File path to a CSV  **or**  "MSL" / "SMAP".
    datasets_dir : str
        Root directory that contains the pickle datasets (used for MSL/SMAP).
    cutoff_date : str | None
        ISO date string (e.g. "2021-09-01").  If set, CSV data is split at this
        timestamp instead of using val_split.  Ignored for MSL/SMAP.
    val_split : float
        Fraction of training data to use as validation (default 0.1 = 10 %).

    Returns
    -------
    dict with keys:
        x_train      : np.ndarray  (N_train, k)  – normalised train features
        x_val        : np.ndarray  (N_val, k)    – normalised val features
        x_test       : np.ndarray  (N_test, k)   – normalised test features (may be None)
        y_test       : np.ndarray | None          – integer anomaly labels for test set
        scaler       : fitted MinMaxScaler
        feature_cols : list[str]
        timestamps   : dict {"train", "test"} of pd.Series (or None for pkl datasets)
        n_features   : int
        source_type  : "csv" | "MSL" | "SMAP"
    """
    source_upper = source.upper().strip()

    # ------------------------------------------------------------------
    # CSV path
    # ------------------------------------------------------------------
    if source_upper not in ("MSL", "SMAP"):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Data file not found: {source}")

        df, feature_cols = _load_csv(source)
        data = df[feature_cols].values.astype(np.float32)

        if cutoff_date is not None:
            cutoff = pd.Timestamp(cutoff_date, tz="UTC")
            ts = df["timestamp"]
            mask_train = ts < cutoff
            train_data = data[mask_train]
            test_data  = data[~mask_train]
            ts_train   = ts[mask_train].reset_index(drop=True)
            ts_test    = ts[~mask_train].reset_index(drop=True)
        else:
            n = len(data)
            split = int(n * (1 - val_split))        # rough 90/10 if no cutoff
            train_data = data[:split]
            test_data  = data[split:]
            ts_train   = df["timestamp"].iloc[:split].reset_index(drop=True)
            ts_test    = df["timestamp"].iloc[split:].reset_index(drop=True)

        # Fit scaler on training data only
        scaler = MinMaxScaler()
        x_train_all = scaler.fit_transform(train_data)
        x_test      = scaler.transform(test_data) if len(test_data) > 0 else None
        ts_test_out = ts_test if len(test_data) > 0 else None

        # Validation split from training data (last val_split %)
        n_train = len(x_train_all)
        n_val   = max(1, int(n_train * val_split))
        x_val   = x_train_all[n_train - n_val:]
        x_train = x_train_all[:n_train - n_val]

        return {
            "x_train":      x_train,
            "x_val":        x_val,
            "x_train_full": x_train_all,   # full train (for POT scoring)
            "x_test":       x_test,
            "y_test":       None,           # no labels in raw CSV
            "scaler":       scaler,
            "feature_cols": feature_cols,
            "timestamps":   {"train": ts_train, "test": ts_test_out},
            "n_features":   len(feature_cols),
            "source_type":  "csv",
        }

    # ------------------------------------------------------------------
    # MSL / SMAP  (npy-per-entity or pickle, auto-detected)
    # ------------------------------------------------------------------
    else:
        name = source_upper
        train_raw, test_raw, y_test = _load_nasa_dataset(name, datasets_dir)
        n_features = train_raw.shape[1]
        feature_cols = [f"feat_{i}" for i in range(n_features)]

        scaler = MinMaxScaler()
        x_train_all = scaler.fit_transform(train_raw)
        x_test      = scaler.transform(test_raw) if test_raw is not None else None

        n_train = len(x_train_all)
        n_val   = max(1, int(n_train * val_split))
        x_val   = x_train_all[n_train - n_val:]
        x_train = x_train_all[:n_train - n_val]

        return {
            "x_train":      x_train,
            "x_val":        x_val,
            "x_train_full": x_train_all,
            "x_test":       x_test,
            "y_test":       y_test,
            "scaler":       scaler,
            "feature_cols": feature_cols,
            "timestamps":   {"train": None, "test": None},
            "n_features":   n_features,
            "source_type":  name,
        }


def make_loaders(data_dict: dict,
                 window_size: int = 100,
                 batch_size: int = 256):
    """
    Build PyTorch DataLoaders from the dict returned by load_data().

    Returns
    -------
    train_loader, val_loader, test_loader (test_loader is None if no test data)
    """
    train_ds = TimeSeriesDataset(data_dict["x_train"], window_size)
    val_ds   = TimeSeriesDataset(data_dict["x_val"],   window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    test_loader = None
    if data_dict["x_test"] is not None and len(data_dict["x_test"]) > window_size:
        test_ds     = TimeSeriesDataset(data_dict["x_test"], window_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Also a loader over the full training data (for POT threshold fitting)
    train_full_ds     = TimeSeriesDataset(data_dict["x_train_full"], window_size)
    train_full_loader = DataLoader(train_full_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_full_loader
