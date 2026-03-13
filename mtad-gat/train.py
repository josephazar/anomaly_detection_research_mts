"""
MTAD-GAT Training Script
========================

Usage examples
--------------
# Use params.json (default – just edit params.json before running)
python train.py

# Override specific values from the command line
python train.py --data ../multivariate_sample_data.csv --cutoff 2021-09-01 --epochs 50

# Point to a different params file
python train.py --params my_config.json

# NASA benchmark datasets
python train.py --data MSL
python train.py --data SMAP

Outputs (saved to output_dir in params.json, default: output/<run_id>/)
------------------------------------------------------------------------
  model.pt         – best model weights (lowest validation loss)
  scaler.pkl       – fitted MinMaxScaler (required for inference)
  config.json      – all hyperparameters + data source
  train_scores.npy – anomaly scores on full training set (for POT threshold)
  losses.png       – training / validation loss curves
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import load_data, make_loaders
from mtad_gat import MTAD_GAT, compute_anomaly_scores, pot_threshold, train_model

_DEFAULT_PARAMS = os.path.join(os.path.dirname(__file__), "params.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Train MTAD-GAT  (defaults loaded from params.json)"
    )

    # Params file
    p.add_argument("--params", default=_DEFAULT_PARAMS,
                   help="Path to a JSON params file (default: params.json)")

    # Data  – override params.json
    p.add_argument("--data", default=None,
                   help="Path to a CSV file  OR  'MSL' / 'SMAP'")
    p.add_argument("--cutoff", default=None,
                   help="ISO date to split train/test in CSV mode (e.g. 2021-09-01)")
    p.add_argument("--val_split", type=float, default=None,
                   help="Fraction of training data for validation (default 0.1)")
    p.add_argument("--datasets_dir", default=None,
                   help="Root directory containing MSL/SMAP pickle files")

    # Model
    p.add_argument("--window",     type=int,   default=None)
    p.add_argument("--kernel",     type=int,   default=None)
    p.add_argument("--gru_layers", type=int,   default=None)
    p.add_argument("--gru_dim",    type=int,   default=None)
    p.add_argument("--fc_layers",  type=int,   default=None)
    p.add_argument("--fc_dim",     type=int,   default=None)
    p.add_argument("--recon_layers",type=int,  default=None)
    p.add_argument("--recon_dim",  type=int,   default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--alpha",      type=float, default=None)

    # Training
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--bs",      type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--gamma",   type=float, default=None,
                   help="Weight for reconstruction error in anomaly score")

    # POT threshold
    p.add_argument("--pot_q",     type=float, default=None)
    p.add_argument("--pot_level", type=float, default=None)

    # Output
    p.add_argument("--output_dir", default=None,
                   help="Where to save outputs (default: output/<run_id>)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # ---- Load params.json, then let explicit CLI flags override ----
    params = {}
    if os.path.isfile(args.params):
        with open(args.params) as f:
            params = {k: v for k, v in json.load(f).items()
                      if not k.startswith("_")}
        print(f"Loaded params from: {args.params}")
    else:
        print(f"Warning: params file not found ({args.params}), using CLI / defaults only")

    def get(key, default=None):
        """CLI flag wins if explicitly set (not None), otherwise fall back to params.json."""
        cli_val = getattr(args, key, None)
        return cli_val if cli_val is not None else params.get(key, default)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = get("output_dir") or os.path.join(
        os.path.dirname(__file__), "output", run_id
    )
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Training on GPU : {gpu_name}  ({gpu_mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on GPU : Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Training on CPU  (no CUDA or MPS device found)")
    print(f"Run dir: {out_dir}")

    # Resolved values (params.json merged with CLI overrides)
    data_source  = get("data")
    cutoff       = get("cutoff")
    val_split    = get("val_split", 0.1)
    datasets_dir = get("datasets_dir")
    window       = get("window", 100)
    kernel       = get("kernel", 7)
    gru_layers   = get("gru_layers", 1)
    gru_dim      = get("gru_dim", 150)
    fc_layers    = get("fc_layers", 3)
    fc_dim       = get("fc_dim", 150)
    recon_layers = get("recon_layers", 1)
    recon_dim    = get("recon_dim", 150)
    dropout      = get("dropout", 0.3)
    alpha        = get("alpha", 0.2)
    epochs       = get("epochs", 30)
    bs           = get("bs", 256)
    lr           = get("lr", 1e-3)
    gamma        = get("gamma", 1.0)
    pot_q        = get("pot_q", 1e-3)
    pot_level    = get("pot_level", 0.99)

    if data_source is None:
        raise ValueError("No data source specified. Set 'data' in params.json or pass --data.")

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    loader_kwargs = dict(
        source=data_source,
        val_split=val_split,
        cutoff_date=cutoff,
    )
    if datasets_dir:
        loader_kwargs["datasets_dir"] = datasets_dir

    print(f"\nLoading data from: {data_source}")
    data = load_data(**loader_kwargs)

    n_features = data["n_features"]
    print(f"Features    : {n_features}  ({data['feature_cols'][:5]}{'...' if n_features > 5 else ''})")
    print(f"Train rows  : {len(data['x_train'])}")
    print(f"Val rows    : {len(data['x_val'])}")
    if data["x_test"] is not None:
        print(f"Test rows   : {len(data['x_test'])}")

    train_loader, val_loader, test_loader, train_full_loader = make_loaders(
        data, window_size=window, batch_size=bs
    )

    # ------------------------------------------------------------------ #
    # 2. Model                                                             #
    # ------------------------------------------------------------------ #
    model = MTAD_GAT(
        n_features=n_features,
        window_size=window,
        kernel_size=kernel,
        use_gatv2=True,
        gru_n_layers=gru_layers,
        gru_hid_dim=gru_dim,
        forecast_n_layers=fc_layers,
        forecast_hid_dim=fc_dim,
        recon_n_layers=recon_layers,
        recon_hid_dim=recon_dim,
        dropout=dropout,
        alpha=alpha,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------ #
    # 3. Train                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    model_path = os.path.join(out_dir, "model.pt")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs,
        lr=lr,
        print_every=1,
        save_path=model_path,
    )

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, "b-", lw=2, label="Train")
    ax.plot(val_losses,   "r-", lw=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE_forecast + RMSE_recon")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "losses.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    # 4. Compute train anomaly scores (for POT threshold at inference)    #
    # ------------------------------------------------------------------ #
    print("\nComputing training anomaly scores (for POT threshold)…")
    train_result  = compute_anomaly_scores(model, train_full_loader, device,
                                           gamma=gamma)
    train_scores  = train_result["scores"]
    threshold_pot = float(pot_threshold(train_scores, q=pot_q, level=pot_level))

    np.save(os.path.join(out_dir, "train_scores.npy"), train_scores)
    print(f"POT threshold (from train): {threshold_pot:.6f}")

    # ------------------------------------------------------------------ #
    # 5. Save scaler + config                                              #
    # ------------------------------------------------------------------ #
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(data["scaler"], f)

    config = {
        "run_id":        run_id,
        "data_source":   data_source,
        "cutoff":        cutoff,
        "source_type":   data["source_type"],
        "feature_cols":  [str(c) for c in data["feature_cols"]],
        "n_features":    int(n_features),
        "window":        window,
        "kernel":        kernel,
        "gru_layers":    gru_layers,
        "gru_dim":       gru_dim,
        "fc_layers":     fc_layers,
        "fc_dim":        fc_dim,
        "recon_layers":  recon_layers,
        "recon_dim":     recon_dim,
        "dropout":       dropout,
        "alpha":         alpha,
        "epochs":        epochs,
        "batch_size":    bs,
        "lr":            lr,
        "gamma":         gamma,
        "pot_q":         pot_q,
        "pot_level":     pot_level,
        "threshold_pot": threshold_pot,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to: {out_dir}")
    print(f"  model.pt         – model weights")
    print(f"  scaler.pkl       – normalisation scaler")
    print(f"  config.json      – run configuration + POT threshold")
    print(f"  train_scores.npy – training anomaly scores")
    print(f"  losses.png       – loss curves")

    # ------------------------------------------------------------------ #
    # 6. Quick test evaluation (if test data exists)                       #
    # ------------------------------------------------------------------ #
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("QUICK TEST EVALUATION")
        print("=" * 60)
        test_result = compute_anomaly_scores(model, test_loader, device, gamma=gamma)
        test_scores = test_result["scores"]
        test_preds  = (test_scores > threshold_pot).astype(int)
        print(f"Test samples scored : {len(test_scores)}")
        print(f"POT threshold       : {threshold_pot:.6f}")
        print(f"Anomalies detected  : {test_preds.sum()} "
              f"({test_preds.mean()*100:.2f} %)")

        if data["y_test"] is not None:
            y = data["y_test"][window:]
            if len(y) == len(test_preds):
                tp = int(((test_preds == 1) & (y == 1)).sum())
                fp = int(((test_preds == 1) & (y == 0)).sum())
                fn = int(((test_preds == 0) & (y == 1)).sum())
                p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
