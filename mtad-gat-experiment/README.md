# MTAD-GAT Experiment

An implementation of **MTAD-GAT** (Multivariate Time-Series Anomaly Detection via Graph Attention Network) with a sample dataset from the Azure Anomaly Detector.

## Contents

```
mtad-gat-experiment/
├── 01_data_discovery.ipynb      # EDA on the sample dataset
├── multivariate_sample_data.csv # 5-variable sensor data (light, hydraulic, shaft, vibration)
├── anomalies.json               # Anomaly detection results (timestamps, scores, severity)
└── mtad-gat/
    ├── mtad_gat.py              # Model architecture (GAT + GRU + forecasting/reconstruction heads)
    ├── data_loader.py           # Data loading and windowing
    ├── train.py                 # Training loop
    ├── evaluate.py              # Evaluation with POT thresholding
    ├── infer.py                 # Inference on new data
    └── params.json              # Hyperparameters and data config
```

## Dataset

The sample data (`multivariate_sample_data.csv`) contains 5 sensor variables at 10-minute intervals:

| Variable | Description |
|----------|-------------|
| `sensorA_light_left` | Light sensor (left) |
| `sensorB_light_right` | Light sensor (right) |
| `hydraulic_pressure` | Hydraulic pressure reading |
| `shaft_rotation` | Shaft rotation speed |
| `lateral_vibration` | Lateral vibration measurement |

The notebook `01_data_discovery.ipynb` covers basic statistics, time series plots, correlation structure, rolling statistics, and stationarity tests.

## MTAD-GAT

The model combines graph attention networks with GRU-based forecasting and reconstruction to detect anomalies in multivariate time series. It supports:

- Custom CSV data (with a date cutoff for train/test split)
- NASA MSL and SMAP benchmark datasets
- POT (Peaks Over Threshold) for automatic anomaly thresholding

See `mtad-gat/params.json` for all configurable hyperparameters.

### Usage

```bash
# Train on the sample data
cd mtad-gat
python train.py

# Or point to a different dataset by editing params.json
```

## Reference

Zhao et al., *Multivariate Time-Series Anomaly Detection via Graph Attention Network*, ICDM 2020
