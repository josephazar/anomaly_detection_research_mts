"""
MTAD-GAT Evaluation Script
===========================

Evaluates a trained model against labelled ground truth using the three
threshold strategies reported in the MTAD-GAT paper:

  1. POT   – Peaks-Over-Threshold (same method used during training)
  2. Epsilon – statistical threshold (mean + z·std, from Hundman et al.)
  3. Best-F1 – brute-force oracle search (upper-bound / paper ceiling)

All three use the **point-adjust** strategy from OmniAnomaly: once any point
in a ground-truth anomaly segment is detected, the entire segment is counted
as detected (TP). This matches the evaluation methodology in the paper.

Usage
-----
python evaluate.py --run output/<run_id> --data SMAP
python evaluate.py --run output/<run_id> --data MSL
python evaluate.py --run output/<run_id> --data ../multivariate_sample_data.csv --cutoff 2021-09-01
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import _load_nasa_dataset, _load_csv
from mtad_gat import MTAD_GAT, TimeSeriesDataset, compute_anomaly_scores, pot_threshold


# ---------------------------------------------------------------------------
# Threshold / evaluation helpers  (from OmniAnomaly / MTAD-GAT paper)
# ---------------------------------------------------------------------------

def adjust_predicts(scores, labels, threshold):
    """
    Point-adjust strategy (OmniAnomaly).
    If any point in a ground-truth anomaly segment is flagged, the whole
    segment is marked as TP.
    """
    predict = (scores > threshold).copy()
    actual  = labels > 0.5

    anomaly_state = False
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            # back-fill the current anomaly segment
            for j in range(i, -1, -1):
                if not actual[j]:
                    break
                predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict


def point2point(pred, actual):
    """Return (f1, precision, recall, TP, TN, FP, FN)."""
    pred   = pred.astype(int)
    actual = actual.astype(int)
    TP = int(np.sum(pred * actual))
    TN = int(np.sum((1 - pred) * (1 - actual)))
    FP = int(np.sum(pred * (1 - actual)))
    FN = int(np.sum((1 - pred) * actual))
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, precision, recall, TP, TN, FP, FN


def eval_with_threshold(scores, labels, threshold):
    pred = adjust_predicts(scores, labels, threshold)
    f1, p, r, tp, tn, fp, fn = point2point(pred, labels)
    return {"f1": f1, "precision": p, "recall": r,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "threshold": threshold}


def find_epsilon(train_scores, reg_level=1):
    """
    Epsilon threshold (Hundman et al.).
    Searches mean + z·std for z in [2.5, 12) that maximises a penalised score.
    """
    mean_s, std_s = np.mean(train_scores), np.std(train_scores)
    best_eps, best_score = None, -1e9

    for z in np.arange(2.5, 12, 0.5):
        eps = mean_s + std_s * z
        pruned = train_scores[train_scores < eps]
        flagged = np.where(train_scores >= eps)[0]

        if len(flagged) == 0 or len(flagged) >= len(train_scores) * 0.5:
            continue

        mpd = (mean_s - np.mean(pruned)) / (mean_s + 1e-9)
        spd = (std_s  - np.std(pruned))  / (std_s  + 1e-9)
        denom = len(flagged) ** reg_level if reg_level > 0 else 1
        score = (mpd + spd) / denom

        if score > best_score:
            best_score = score
            best_eps   = eps

    return best_eps if best_eps is not None else float(np.max(train_scores))


def bf_search(test_scores, labels, n_steps=200):
    """
    Best-F1 brute-force oracle search over thresholds.
    This is an oracle / upper-bound — uses the test labels to find the best threshold.
    """
    lo, hi = float(np.min(test_scores)), float(np.max(test_scores))
    best = {"f1": 0.0}
    for t in np.linspace(lo, hi, n_steps):
        res = eval_with_threshold(test_scores, labels, t)
        if res["f1"] > best["f1"]:
            best = res
    return best


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    model = MTAD_GAT(
        n_features      = cfg["n_features"],
        window_size     = cfg["window"],
        kernel_size     = cfg["kernel"],
        use_gatv2       = True,
        gru_n_layers    = cfg["gru_layers"],
        gru_hid_dim     = cfg["gru_dim"],
        forecast_n_layers = cfg["fc_layers"],
        forecast_hid_dim  = cfg["fc_dim"],
        recon_n_layers  = cfg["recon_layers"],
        recon_hid_dim   = cfg["recon_dim"],
        dropout         = cfg["dropout"],
        alpha           = cfg["alpha"],
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate MTAD-GAT against ground truth")
    p.add_argument("--run",     required=True,
                   help="Path to a training run directory (must contain model.pt, scaler.pkl, config.json, train_scores.npy)")
    p.add_argument("--data",    default=None,
                   help="Data source override: CSV path  OR  'MSL' / 'SMAP'. "
                        "Defaults to the data_source stored in config.json.")
    p.add_argument("--cutoff",  default=None,
                   help="Date cutoff for CSV test split (e.g. 2021-09-01)")
    p.add_argument("--datasets_dir", default=None)
    p.add_argument("--bs",      type=int, default=256, help="Batch size")
    p.add_argument("--bf_steps", type=int, default=200,
                   help="Number of threshold steps for Best-F1 search (default 200)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    run_dir = args.run

    # ── Load saved artefacts ─────────────────────────────────────────────
    config_path      = os.path.join(run_dir, "config.json")
    weights_path     = os.path.join(run_dir, "model.pt")
    scaler_path      = os.path.join(run_dir, "scaler.pkl")
    train_scores_path = os.path.join(run_dir, "train_scores.npy")

    for label, path in [("config",  config_path),
                         ("weights", weights_path),
                         ("scaler",  scaler_path),
                         ("train_scores", train_scores_path)]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {label} file not found: {path}")

    with open(config_path) as f:
        cfg = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    train_scores = np.load(train_scores_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

    model = build_model(cfg, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ── Load test data ───────────────────────────────────────────────────
    data_source = args.data or cfg.get("data_source")
    print(f"\nLoading test data from: {data_source}")

    source_upper = data_source.upper().strip() if data_source else ""

    if source_upper in ("MSL", "SMAP"):
        datasets_dir = args.datasets_dir or os.path.join(
            os.path.dirname(__file__), "..", "mtad-gat-pytorch", "datasets"
        )
        _, test_raw, y_test = _load_nasa_dataset(source_upper, datasets_dir)
        if y_test is None:
            sys.exit("ERROR: No ground-truth labels found — cannot evaluate.")
        x_test = scaler.transform(test_raw)

    else:
        # CSV
        import pandas as pd
        df, feat_cols = _load_csv(data_source)
        if args.cutoff:
            cutoff = pd.Timestamp(args.cutoff, tz="UTC")
            test_df = df[df["timestamp"] >= cutoff]
        else:
            n = len(df)
            val_split = cfg.get("val_split", 0.1)
            test_df = df.iloc[int(n * (1 - val_split)):]

        test_raw = test_df[feat_cols].values.astype(np.float32)
        x_test   = scaler.transform(test_raw)
        y_test   = None  # raw CSV has no labels

        if y_test is None:
            sys.exit("ERROR: CSV source has no ground-truth labels — cannot evaluate. "
                     "Use MSL or SMAP for labelled evaluation.")

    window = cfg["window"]
    print(f"Test rows     : {len(x_test)}")
    print(f"Anomaly rate  : {y_test.mean()*100:.2f} %  ({y_test.sum()} / {len(y_test)} points)")

    # ── Score the test set ───────────────────────────────────────────────
    test_ds     = TimeSeriesDataset(x_test, window)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False)

    print("\nScoring test set…")
    result = compute_anomaly_scores(model, test_loader, device,
                                    gamma=cfg.get("gamma", 1.0))
    test_scores = result["scores"]            # (N,)

    # Align labels with scored window (first `window` rows consumed as look-back)
    y = y_test[window: window + len(test_scores)].astype(int)
    assert len(y) == len(test_scores), \
        f"Length mismatch: scores={len(test_scores)}, labels={len(y)}"

    print(f"Scored timesteps : {len(test_scores)}")
    print(f"Label anomaly %  : {y.mean()*100:.2f} %")

    # ── Three threshold evaluations ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION  (with point-adjust strategy)")
    print("=" * 65)

    # 1. POT
    pot_th  = float(pot_threshold(train_scores, q=cfg.get("pot_q", 1e-3),
                                   level=cfg.get("pot_level", 0.99)))
    pot_res = eval_with_threshold(test_scores, y, pot_th)

    # 2. Epsilon
    eps_th  = find_epsilon(train_scores, reg_level=1)
    eps_res = eval_with_threshold(test_scores, y, eps_th)

    # 3. Best-F1 (oracle)
    print(f"Searching best-F1 threshold ({args.bf_steps} steps)…")
    bf_res  = bf_search(test_scores, y, n_steps=args.bf_steps)

    # ── Print results table ──────────────────────────────────────────────
    col_w = 14
    header = f"{'Method':<12}  {'F1':>{col_w}}  {'Precision':>{col_w}}  {'Recall':>{col_w}}  " \
             f"{'TP':>8}  {'FP':>8}  {'FN':>8}  {'Threshold':>{col_w}}"
    sep    = "-" * len(header)

    def fmt_row(name, r):
        return (f"{name:<12}  {r['f1']:>{col_w}.4f}  {r['precision']:>{col_w}.4f}  "
                f"{r['recall']:>{col_w}.4f}  {r['TP']:>8d}  {r['FP']:>8d}  {r['FN']:>8d}  "
                f"{r['threshold']:>{col_w}.6f}")

    print()
    print(header)
    print(sep)
    print(fmt_row("POT",      pot_res))
    print(fmt_row("Epsilon",  eps_res))
    print(fmt_row("Best-F1*", bf_res))
    print(sep)
    print("* Best-F1 is an oracle upper bound — it searches the threshold using test labels.")
    print()

    # ── Save JSON summary ────────────────────────────────────────────────
    summary = {
        "run_dir":    run_dir,
        "data":       data_source,
        "n_test":     int(len(test_scores)),
        "anomaly_pct": float(y.mean() * 100),
        "pot":        {k: float(v) for k, v in pot_res.items()},
        "epsilon":    {k: float(v) for k, v in eps_res.items()},
        "best_f1":    {k: float(v) for k, v in bf_res.items()},
    }
    out_path = os.path.join(run_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
