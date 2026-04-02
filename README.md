# Multivariate Time Series Dataset Exploration for Anomaly Detection Research

Exploration and visualization notebooks for **5 industrial multivariate time series (MTS) datasets**, designed to support PhD research on:

- **Temporal Clustering** -- discovering operational phases/regimes (e.g., warm-up, steady-state, shutdown), where each phase can itself contain anomalies
- **Unsupervised Anomaly Detection** -- identifying deviations from normal behavior without labeled training data
- **Early Prediction / Predictive Maintenance / RUL** -- detecting degradation patterns before failures occur

## Datasets

| # | Dataset | Domain | Dimensions | Samples | Key Feature |
|---|---------|--------|-----------|---------|-------------|
| 1 | **SMD** (Server Machine Dataset) | Server monitoring | 38 | ~1.4M | 28 machines in 3 groups with point-level anomaly labels |
| 2 | **SCANIA Component X** | Heavy-duty trucks | 105 | ~1.1M | Time-to-event labels for predictive maintenance |
| 3 | **Genesis Demonstrator** | Smart manufacturing | 18 | ~16K | 9-state machine labels (ground truth for clustering) |
| 4 | **Exathlon** | Spark streaming apps | 2,283 | ~47K/trace | Multiple anomaly types with root cause annotations |
| 5 | **DAMADICS** | Sugar factory actuators | 32 | 2.16M | 25 days continuous operation with known fault injection |

## Notebooks

Each notebook provides:
- **Dataset overview** -- structure, dimensions, statistics
- **Time series visualization** -- multivariate plots with anomaly highlighting
- **Correlation analysis** -- inter-feature and inter-entity relationships
- **Clustering potential** -- operational regime discovery, entity grouping, phase transitions
- **Anomaly precursor analysis** -- pre-fault signal evolution, early detection windows
- **Dataset suitability assessment** -- structured scoring for clustering, AD, and predictive maintenance

| Notebook | Dataset | Suitability Scores (Clustering / AD / Early Detection) |
|----------|---------|-------------------------------------------------------|
| `01_SMD_Exploration.ipynb` | SMD | 7/10 -- 9/10 -- 5/10 |
| `02_SCANIA_Exploration.ipynb` | SCANIA | 7/10 -- 8/10 -- 9/10 |
| `03_Genesis_Exploration.ipynb` | Genesis | 5/5 -- 4/5 -- 3/5 |
| `04_Exathlon_Exploration.ipynb` | Exathlon | Moderate -- HIGH -- Moderate |
| `05_DAMADICS_Exploration.ipynb` | DAMADICS | HIGH -- GOOD -- MODERATE |

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data

Datasets are **not included** in the repository due to size. Download the following zip files and extract them in the project root:

| File | Extract To | Size |
|------|-----------|------|
| `dataset_smd.zip` | `smd/` | ~100 MB |
| `dataset_scania.zip` | `scania/` | ~84 MB |
| `dataset_genesis.zip` | `genesis/` | ~0.6 MB |
| `dataset_exathlon.zip` | `exathlon/` | ~227 MB |
| `dataset_damadics.zip` | `damadics/` | ~90 MB |

After extraction, your directory should look like:
```
.
├── 01_SMD_Exploration.ipynb
├── 02_SCANIA_Exploration.ipynb
├── 03_Genesis_Exploration.ipynb
├── 04_Exathlon_Exploration.ipynb
├── 05_DAMADICS_Exploration.ipynb
├── requirements.txt
├── smd/
│   ├── train/
│   ├── test/
│   ├── test_label/
│   └── interpretation_label/
├── scania/
│   ├── test_operational_readouts.csv
│   ├── test_labels.csv
│   └── ...
├── genesis/
│   ├── Genesis_normal.csv
│   ├── Genesis_AnomalyLabels.csv
│   └── ...
├── exathlon/
│   └── data/raw/
│       ├── ground_truth.csv
│       └── app1/ ... app10/
└── damadics/
    ├── 01112001.txt
    └── ...
```

## Key Concept: Clusters vs Anomalies

A central theme across these notebooks:

> **A cluster is NOT an anomaly type.** A cluster represents an *operational phase* (e.g., machine warming up, running at steady state, shutting down). Each phase can contain anomalies within it. Clustering discovers the normal operational structure; anomaly detection finds deviations within that structure.

This distinction is critical for methods like TICC (Toeplitz Inverse Covariance-based Clustering) that segment multivariate time series into temporal regimes.

## References

- **SMD**: Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", KDD 2019
- **SCANIA**: Garan et al., "SCANIA Component X Dataset", 2024
- **Genesis**: Atzmueller et al., Genesis Demonstrator Dataset for smart manufacturing
- **Exathlon**: Jacob et al., "Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series", VLDB 2021
- **DAMADICS**: DAMADICS Benchmark for Fault Detection and Isolation, Lublin Sugar Factory
