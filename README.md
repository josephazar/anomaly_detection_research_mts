# MTS Dataset Exploration for Anomaly Detection Research

Exploration and visualization notebooks for **5 industrial multivariate time series datasets**, selected to support research on:

- **Temporal clustering** -- discovering operational phases (warm-up, steady-state, shutdown, etc.), not anomaly types
- **Unsupervised anomaly detection** -- finding deviations without labeled training data
- **Predictive maintenance / early detection** -- spotting degradation before failure

No ML models here -- just data exploration, visualization, and an honest assessment of what each dataset can and can't do.

## Datasets

| # | Dataset | Domain | Dims | Samples | What makes it interesting |
|---|---------|--------|-----:|--------:|---------------------------|
| 1 | **SMD** | Server monitoring | 38 | 1.4M | 28 machines in 3 groups; interpretation labels for each anomaly |
| 2 | **SCANIA** | Heavy-duty trucks | 105 | 1.1M | Real fleet data with time-to-event labels |
| 3 | **Genesis** | Smart manufacturing | 18 | 16K | 9 state machine labels = ground truth for clustering |
| 4 | **Exathlon** | Spark streaming | 2,283 | 47K/trace | Root cause annotations; extreme dimensionality |
| 5 | **DAMADICS** | Sugar factory | 32 | 2.16M | 25 days continuous; clear intra-day regime structure |

We evaluated 16 datasets total before picking these five. Full evaluation with scores and reasoning: [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)

## Notebooks

| Notebook | Clustering | AD | Pred. Maint. |
|----------|:----------:|:--:|:------------:|
| [`01_SMD_Exploration.ipynb`](01_SMD_Exploration.ipynb) | 7/10 | 9/10 | 5/10 |
| [`02_SCANIA_Exploration.ipynb`](02_SCANIA_Exploration.ipynb) | 7/10 | 8/10 | 9/10 |
| [`03_Genesis_Exploration.ipynb`](03_Genesis_Exploration.ipynb) | 9/10 | 8/10 | 6/10 |
| [`04_Exathlon_Exploration.ipynb`](04_Exathlon_Exploration.ipynb) | 6/10 | 9/10 | 6/10 |
| [`05_DAMADICS_Exploration.ipynb`](05_DAMADICS_Exploration.ipynb) | 8/10 | 8/10 | 7/10 |

Each notebook covers:
- Dataset structure and basic statistics
- Multivariate time series plots with anomaly regions highlighted
- Feature correlation analysis (normal vs anomalous)
- Operational regime discovery (rolling-window statistics, phase transitions)
- Anomaly precursor analysis (pre-fault signal drift, early detection windows)
- A suitability assessment with concrete next steps

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are too large for git. Download and extract these zips in the project root:

| File | Extracts to | Size |
|------|------------|------|
| `dataset_smd.zip` | `smd/` | ~100 MB |
| `dataset_scania.zip` | `scania/` | ~84 MB |
| `dataset_genesis.zip` | `genesis/` | ~0.6 MB |
| `dataset_exathlon.zip` | `exathlon/` | ~227 MB |
| `dataset_damadics.zip` | `damadics/` | ~90 MB |

Expected directory layout after extraction:
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

## A Note on Clustering vs Anomaly Detection

One thing that kept coming up as we worked through these datasets:

> A cluster is not an anomaly type. A cluster is an operational phase -- the machine warming up, running normally, cooling down. Anomalies can happen *within* any phase. Clustering discovers structure; anomaly detection finds deviations from it.

This matters for methods like TICC that segment time series into regimes. Genesis is the clearest example (9 labeled phases), but all five datasets show some form of regime structure when you look at the right temporal scale.

## References

- **SMD**: Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*, KDD 2019
- **SCANIA**: Garan et al., *SCANIA Component X Dataset*, 2024
- **Genesis**: Atzmueller et al., *Genesis Demonstrator Dataset*
- **Exathlon**: Jacob et al., *Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series*, VLDB 2021
- **DAMADICS**: Bartys et al., *DAMADICS Benchmark for Fault Detection and Isolation*, Lublin Sugar Factory
