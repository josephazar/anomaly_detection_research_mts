# MTS Dataset Exploration for Anomaly Detection Research

Exploration notebooks for 5 industrial multivariate time series datasets, covering:

- **Temporal clustering** -- operational phase/regime discovery (warm-up, steady-state, shutdown)
- **Unsupervised anomaly detection**
- **Predictive maintenance / early detection / RUL**

## Datasets

| # | Dataset | Domain | Dims | Samples | Notable properties |
|---|---------|--------|-----:|--------:|---------------------|
| 1 | **SMD** | Server monitoring | 38 | 1.4M | 28 machines in 3 groups; interpretation labels |
| 2 | **SCANIA** | Heavy-duty trucks | 105 | 1.1M | Fleet data with time-to-event labels |
| 3 | **Genesis** | Smart manufacturing | 18 | 16K | 9 state machine labels (clustering ground truth) |
| 4 | **Exathlon** | Spark streaming | 2,283 | 47K/trace | Root cause annotations; high dimensionality |
| 5 | **DAMADICS** | Sugar factory | 32 | 2.16M | 25 days continuous; intra-day regime structure |

Full evaluation of 16 candidate datasets: [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)

## Notebooks

| Notebook | Clustering | AD | Pred. Maint. |
|----------|:----------:|:--:|:------------:|
| [`01_SMD_Exploration.ipynb`](01_SMD_Exploration.ipynb) | 7/10 | 9/10 | 5/10 |
| [`02_SCANIA_Exploration.ipynb`](02_SCANIA_Exploration.ipynb) | 7/10 | 8/10 | 9/10 |
| [`03_Genesis_Exploration.ipynb`](03_Genesis_Exploration.ipynb) | 9/10 | 8/10 | 6/10 |
| [`04_Exathlon_Exploration.ipynb`](04_Exathlon_Exploration.ipynb) | 6/10 | 9/10 | 6/10 |
| [`05_DAMADICS_Exploration.ipynb`](05_DAMADICS_Exploration.ipynb) | 8/10 | 8/10 | 7/10 |

Each notebook includes: dataset overview, time series visualization, correlation analysis, operational regime discovery, anomaly precursor analysis, and a suitability assessment.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are not included (too large). Download and extract in the project root:

| File | Extracts to | Size |
|------|------------|------|
| `dataset_smd.zip` | `smd/` | ~100 MB |
| `dataset_scania.zip` | `scania/` | ~84 MB |
| `dataset_genesis.zip` | `genesis/` | ~0.6 MB |
| `dataset_exathlon.zip` | `exathlon/` | ~227 MB |
| `dataset_damadics.zip` | `damadics/` | ~90 MB |

Expected layout:
```
.
├── 01_SMD_Exploration.ipynb
├── ...
├── smd/           (train/, test/, test_label/, interpretation_label/)
├── scania/        (test_operational_readouts.csv, test_labels.csv, ...)
├── genesis/       (Genesis_normal.csv, Genesis_AnomalyLabels.csv, ...)
├── exathlon/      (data/raw/ground_truth.csv, data/raw/app1/ ... app10/)
└── damadics/      (01112001.txt ... 22112001.txt)
```

## Clustering vs Anomaly Detection

An important distinction throughout: a cluster represents an *operational phase* (machine warming up, running at steady state, shutting down), not an anomaly type. Anomalies can occur within any phase. Clustering recovers temporal structure; anomaly detection finds deviations from it. This is relevant for methods like TICC that segment multivariate time series into regimes.

## Other

[`mtad-gat-experiment/`](mtad-gat-experiment/) -- MTAD-GAT implementation with a sample dataset from Azure Anomaly Detector.

## References

- **SMD**: Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*, KDD 2019
- **SCANIA**: Garan et al., *SCANIA Component X Dataset*, 2024
- **Genesis**: Atzmueller et al., *Genesis Demonstrator Dataset*
- **Exathlon**: Jacob et al., *Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series*, VLDB 2021
- **DAMADICS**: Bartys et al., *DAMADICS Benchmark for Fault Detection and Isolation*, Lublin Sugar Factory
