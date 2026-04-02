"""
MTAD-GAT Inference Script
==========================

Loads a trained model and runs anomaly scoring on any compatible dataset.
Outputs a CSV with per-timestep scores and anomaly flags.

Usage examples
--------------
# Point to run directory (auto-discovers model.pt + scaler.pkl + config.json)
python infer.py --run output/20240101_120000 --data ../multivariate_sample_data.csv

# Or point directly to the weights file
python infer.py --weights output/20240101_120000/model.pt \
                --config  output/20240101_120000/config.json \
                --scaler  output/20240101_120000/scaler.pkl \
                --data    ../multivariate_sample_data.csv

# Specify a custom threshold (overrides config.json POT threshold)
python infer.py --run output/20240101_120000 --data MSL --threshold 0.15

Output CSV columns
------------------
  timestamp         – from the input data (integer index if none)
  anomaly_score     – global anomaly score (mean of per-feature scores)
  is_anomaly        – 1 / 0 based on threshold
  score_<feat>      – per-feature anomaly score
  <feat_col>        – original (unnormalised) feature values
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from data_loader import load_data
from mtad_gat import MTAD_GAT, TimeSeriesDataset, compute_anomaly_scores, pot_threshold
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="MTAD-GAT inference → CSV")

    # Locate the trained run
    p.add_argument("--run", default=None,
                   help="Path to a training run directory "
                        "(must contain model.pt, scaler.pkl, config.json)")
    p.add_argument("--weights", default=None, help="Path to model.pt directly")
    p.add_argument("--config",  default=None, help="Path to config.json directly")
    p.add_argument("--scaler",  default=None, help="Path to scaler.pkl directly")

    # Inference data
    p.add_argument("--data", required=True,
                   help="CSV path  OR  'MSL' / 'SMAP'")
    p.add_argument("--cutoff", default=None,
                   help="If set, only data on-or-after this date is scored "
                        "(useful when CSV contains both train and test periods)")
    p.add_argument("--datasets_dir", default=None,
                   help="Root dir for MSL/SMAP pickle files")

    # Threshold override
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the POT threshold from config.json")
    p.add_argument("--pot_q",     type=float, default=None,
                   help="Re-compute POT on training scores with this q value")
    p.add_argument("--pot_level", type=float, default=None,
                   help="Re-compute POT on training scores with this level")

    # Output
    p.add_argument("--output", default=None,
                   help="Path for output CSV (default: <run_dir>/inference_results.csv)")
    p.add_argument("--bs", type=int, default=256, help="Batch size")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_paths(args):
    """Resolve model / config / scaler paths from --run or explicit flags."""
    if args.run:
        run_dir = args.run
        weights = args.weights or os.path.join(run_dir, "model.pt")
        config  = args.config  or os.path.join(run_dir, "config.json")
        scaler  = args.scaler  or os.path.join(run_dir, "scaler.pkl")
        train_scores_path = os.path.join(run_dir, "train_scores.npy")
    else:
        weights = args.weights
        config  = args.config
        scaler  = args.scaler
        run_dir = os.path.dirname(weights) if weights else "."
        train_scores_path = os.path.join(run_dir, "train_scores.npy")

    for label, path in [("weights", weights), ("config", config), ("scaler", scaler)]:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cannot find {label} file: {path}\n"
                "Provide --run <run_dir>  or  --weights / --config / --scaler explicitly."
            )

    return weights, config, scaler, run_dir, train_scores_path


def _build_model(cfg: dict, device):
    model = MTAD_GAT(
        n_features=cfg["n_features"],
        window_size=cfg["window"],
        kernel_size=cfg["kernel"],
        use_gatv2=True,
        gru_n_layers=cfg["gru_layers"],
        gru_hid_dim=cfg["gru_dim"],
        forecast_n_layers=cfg["fc_layers"],
        forecast_hid_dim=cfg["fc_dim"],
        recon_n_layers=cfg["recon_layers"],
        recon_hid_dim=cfg["recon_dim"],
        dropout=cfg["dropout"],
        alpha=cfg["alpha"],
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    weights_path, config_path, scaler_path, run_dir, train_scores_path = \
        _resolve_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ------------------------------------------------------------------ #
    # 1. Load config + model + scaler                                      #
    # ------------------------------------------------------------------ #
    with open(config_path) as f:
        cfg = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = _build_model(cfg, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {weights_path}")

    # ------------------------------------------------------------------ #
    # 2. Load inference data                                               #
    # ------------------------------------------------------------------ #
    print(f"\nLoading inference data from: {args.data}")
    loader_kwargs = dict(
        source=args.data,
        cutoff_date=args.cutoff,
        val_split=0.0,               # no need for a val split at inference
    )
    if args.datasets_dir:
        loader_kwargs["datasets_dir"] = args.datasets_dir

    # We load the data ourselves using the saved scaler, bypassing data_loader's scaler
    source_upper = args.data.upper().strip()
    if source_upper in ("MSL", "SMAP"):
        from data_loader import _load_nasa_dataset
        datasets_dir = args.datasets_dir or os.path.join(
            os.path.dirname(__file__), "..", "mtad-gat-pytorch", "datasets"
        )
        _, raw, y_test = _load_nasa_dataset(source_upper, datasets_dir)
        x_infer      = scaler.transform(raw)
        timestamps   = pd.RangeIndex(len(x_infer))
        feature_cols = cfg["feature_cols"]
        raw_df       = pd.DataFrame(raw, columns=feature_cols)
    else:
        # CSV path
        df = pd.read_csv(args.data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != "timestamp"]

        if args.cutoff:
            cutoff = pd.Timestamp(args.cutoff, tz="UTC")
            mask   = df["timestamp"] >= cutoff
            infer_df = df[mask].reset_index(drop=True)
        else:
            infer_df = df

        raw        = infer_df[feat_cols].values.astype(np.float32)
        x_infer    = scaler.transform(raw)
        timestamps = infer_df["timestamp"] if "timestamp" in infer_df.columns \
                     else pd.RangeIndex(len(infer_df))
        feature_cols = feat_cols
        raw_df       = infer_df[feat_cols].reset_index(drop=True)
        y_test       = None

    window = cfg["window"]
    print(f"Rows to score : {len(x_infer)}  (first {window} rows consumed by look-back)")

    # ------------------------------------------------------------------ #
    # 3. Determine threshold                                               #
    # ------------------------------------------------------------------ #
    threshold = args.threshold  # explicit override

    if threshold is None:
        # Optionally re-compute POT from saved training scores
        if os.path.isfile(train_scores_path):
            train_scores = np.load(train_scores_path)
            q     = args.pot_q     or cfg.get("pot_q",     1e-3)
            level = args.pot_level or cfg.get("pot_level", 0.99)
            threshold = pot_threshold(train_scores, q=q, level=level)
            print(f"POT threshold (from train scores): {threshold:.6f}")
        else:
            threshold = cfg.get("threshold_pot")
            if threshold is None:
                raise ValueError(
                    "No threshold found. Provide --threshold, or ensure "
                    "train_scores.npy / config.json threshold_pot exists."
                )
            print(f"Threshold from config.json: {threshold:.6f}")
    else:
        print(f"Threshold (user override): {threshold:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Score                                                             #
    # ------------------------------------------------------------------ #
    infer_ds     = TimeSeriesDataset(x_infer, window)
    infer_loader = DataLoader(infer_ds, batch_size=args.bs, shuffle=False)

    print("\nScoring…")
    result = compute_anomaly_scores(
        model, infer_loader, device,
        gamma=cfg.get("gamma", 1.0),
    )

    scores         = result["scores"]          # (N,)
    feature_scores = result["feature_scores"]  # (N, k)
    is_anomaly     = (scores > threshold).astype(int)

    print(f"Scored {len(scores)} timesteps")
    print(f"Anomalies detected: {is_anomaly.sum()}  ({is_anomaly.mean()*100:.2f} %)")

    # ------------------------------------------------------------------ #
    # 5. Assemble output DataFrame                                         #
    # ------------------------------------------------------------------ #
    offset = window  # first `window` rows are consumed by the look-back

    # Align timestamps
    if isinstance(timestamps, pd.Series):
        ts_out = timestamps.iloc[offset: offset + len(scores)].reset_index(drop=True)
    else:
        ts_out = pd.RangeIndex(offset, offset + len(scores))

    out = pd.DataFrame({"timestamp": ts_out,
                        "anomaly_score": scores,
                        "is_anomaly": is_anomaly})

    # Per-feature scores
    for i, col in enumerate(feature_cols):
        out[f"score_{col}"] = feature_scores[:, i]

    # Original feature values (unnormalised)
    raw_slice = raw_df.iloc[offset: offset + len(scores)].reset_index(drop=True)
    for col in feature_cols:
        out[col] = raw_slice[col].values

    # Ground truth labels (if available)
    if y_test is not None:
        gt = y_test[offset: offset + len(scores)]
        out["gt_anomaly"] = gt

        tp = int(((is_anomaly == 1) & (gt == 1)).sum())
        fp = int(((is_anomaly == 1) & (gt == 0)).sum())
        fn = int(((is_anomaly == 0) & (gt == 1)).sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        print(f"\nWith ground truth — P={pr:.4f}  R={rc:.4f}  F1={f1:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Save CSV                                                          #
    # ------------------------------------------------------------------ #
    out_csv = args.output or os.path.join(run_dir, "inference_results.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
