# MTS Dataset Evaluation

Survey of 16 publicly available multivariate time series datasets, scored on four axes: **clustering**, **anomaly detection**, **predictive maintenance**, and **complexity** (whether the problem is challenging enough for novel contributions). Scores are out of 10.

The five datasets with exploration notebooks in this repo are marked with a link.

## Summary Table

| Dataset | Domain | Dims | Samples | Clustering | AD | PM | Complexity | Overall |
|---------|--------|-----:|--------:|-----------:|---:|---:|-----------:|--------:|
| [SWaT](#swat) | Water treatment | 51 | 947K | 8 | 9 | 7 | 8 | **9** |
| [WADI](#wadi) | Water distribution | 127 | 957K | 8 | 10 | 6 | 9 | **8** |
| [PUMP](#pump) | Industrial pump | 44 | 220K | 7 | 9 | 9 | 7 | **8** |
| [SMD](#smd) | Server machines | 38 | 1.4M | 9 | 10 | 9 | 9 | **9** |
| [Exathlon](#exathlon) | Spark streaming | 2,283 | ~47K/trace | 8 | 10 | 8 | 9 | **9** |
| [MSL](#msl) | Mars rover | 55 | 132K | 6 | 9 | 9 | 8 | **8** |
| [SMAP](#smap) | Satellite telemetry | 25 | 562K | 8 | 10 | 5 | 8 | **7** |
| [DAMADICS](#damadics) | Sugar factory | 32 | 2.16M | 7 | 9 | 9 | 8 | **8** |
| [Genesis](#genesis) | Pick-and-place | 18 | 16K | 8 | 10 | 9 | 7 | **9** |
| [GHL](#ghl) | Gas heating loop | 30+ | varies | 7 | 9 | 8 | 8 | **8** |
| [SCANIA](#scania) | Heavy-duty trucks | 105 | 1.1M | 8 | 8 | 10 | 9 | **9** |
| [GECCO](#gecco) | Water quality IoT | 9 | ~130K | 6 | 9 | 5 | 8 | **8** |
| [MMS](#mms) | Manufacturing | varies | varies | 7 | 8 | 8 | 7 | **8** |
| [PSM](#psm) | Server (eBay) | 25 | 87K | 8 | 9 | 8 | 8 | **8** |
| [NASA Shuttle](#nasa-shuttle) | Aerospace valves | 9-16 | varies | 7 | 9 | 8 | 8 | **8** |

---

## Per-Dataset Notes

### SWaT
**Secure Water Treatment** -- 51 sensors from a real water treatment plant (iTrust, Singapore). 946K samples. Normal operation + 41 distinct cyber-physical attack scenarios.

- **Clustering**: six interconnected stages; attack scenarios introduce regime shifts
- **AD**: one of the most-used benchmarks; labeled attacks with both subtle and obvious patterns
- **PM**: attacks cause stress but there are no run-to-failure trajectories
- Access requires a request to iTrust; some sensor gaps at certain stages

### WADI
**Water Distribution** -- 127 dimensions from a scaled-down water distribution testbed (also iTrust). 957K samples over 16 days.

- **Clustering**: three processing stages with distinct operational modes
- **AD**: arguably harder than SWaT due to subtler anomalies and higher dimensionality
- **PM**: designed for security research, not degradation tracking
- The initial release (WADI.A1) had stability issues; use the updated WADI.A2

### PUMP
**Industrial Water Pump** -- 44 sensor features from a real pump system (Kaggle). 220K samples with machine status labels (NORMAL, BROKEN, RECOVERING).

- **Clustering**: three operational states
- **AD**: labeled failures with clear state transitions
- **PM**: built for predicting pump failures
- Sensor names are anonymized; single entity; limited academic citations

### SMD
**Server Machine Dataset** -- 38 metrics from 28 server machines across 3 groups. ~1.4M total samples. From the OmniAnomaly paper (KDD 2019).

- **Clustering**: 3 machine groups as ground truth for entity-level clustering; within-machine regime transitions at a finer scale
- **AD**: point-level labels + interpretation labels identifying which features caused each anomaly
- **PM**: anomalies represent real faults; some show precursor signals but no explicit RUL labels
- All data from one company; no variable names (indices 0-37)
- Notebook: [`01_SMD_Exploration.ipynb`](01_SMD_Exploration.ipynb)

### Exathlon
**Exathlon Benchmark** -- 2,283 Spark performance metrics from 10 streaming applications. Ground truth includes anomaly type + root cause timing.

- **Clustering**: 10 apps with different workloads form entity-level clusters; no intra-app phase labels
- **AD**: multiple anomaly types (bursty input, CPU contention, driver/executor failures) with explainability annotations
- **PM**: root cause timing enables early detection analysis; extended effect periods define propagation windows
- 2,283 dims is extreme -- most are sparse or constant; IT infrastructure, not manufacturing OT
- Notebook: [`04_Exathlon_Exploration.ipynb`](04_Exathlon_Exploration.ipynb)

### MSL
**Mars Science Laboratory** -- 55 telemetry channels from the Curiosity rover. ~132K samples with expert-labeled anomalies.

- **Clustering**: single entity with no inherent phase structure
- **AD**: expert labels across multiple anomaly types (point, contextual, collective)
- **PM**: real degradation events in a safety-critical system
- Some inconsistencies in how the dataset is reported across papers

### SMAP
**Soil Moisture Active Passive** -- 25 telemetry channels from NASA's SMAP satellite. 562K samples.

- **Clustering**: multiple telemetry entities with different subsystems
- **AD**: widely used alongside MSL; expert-labeled
- **PM**: satellite telemetry is more about detection than degradation prediction
- Not directly related to manufacturing/industrial settings

### DAMADICS
**Development and Application of Methods for Actuator Diagnosis** -- 32 process variables from the Lublin Sugar Factory (Poland). 25 days at 1 Hz = 2.16M samples. Known fault injection on Nov 15, 2001.

- **Clustering**: strong intra-day regime structure (startup, production, shutdown); weekend/weekday patterns at macro level
- **AD**: known fault date provides clear evaluation; incipient faults develop gradually
- **PM**: a classic FDI benchmark designed for fault detection and isolation
- Data from 2001; only one confirmed fault event; variable names are generic (Var_1 to Var_32)
- 300+ citations; still used in modern deep learning papers
- Notebook: [`05_DAMADICS_Exploration.ipynb`](05_DAMADICS_Exploration.ipynb)

### Genesis
**Genesis Demonstrator** -- 18 sensor/actuator channels from a physical pick-and-place system. ~16K samples per file. State machine labels (9 states) + anomaly labels + separate fault files.

- **Clustering**: 9 state machine labels provide ground truth for temporal segmentation -- rare among MTS datasets
- **AD**: two fault types (linear drive, pressure) with clear labels
- **PM**: fault files allow studying progression, but no explicit RUL annotations
- Small dataset; single machine; may not stress-test scalability
- Notebook: [`03_Genesis_Exploration.ipynb`](03_Genesis_Exploration.ipynb)

### GHL
**Gas Heating Loop** -- simulated industrial control system with cyber-attack scenarios. 30+ process variables.

- **Clustering**: operational modes under normal and attack conditions
- **AD**: labeled anomalies from simulated attacks
- **PM**: moderate -- attack-induced degradation can be studied
- Synthetic data; the original paper notes no outliers in the baseline

### SCANIA
**SCANIA Component X** -- 105 anonymized operational features from a fleet of heavy-duty trucks. ~1.1M readouts. Binary failure labels + time-to-event (TTE) for training vehicles.

- **Clustering**: 8 truck specification categories as natural fleet segments
- **AD**: ~3-5% failure rate with extreme class imbalance
- **PM**: TTE labels directly enable survival analysis and RUL estimation -- rare among public datasets
- Features are anonymized; heavy missing values; released 2024 so limited prior work
- Notebook: [`02_SCANIA_Exploration.ipynb`](02_SCANIA_Exploration.ipynb)

### GECCO
**GECCO IoT Water Quality** -- 9 sensors monitoring drinking water quality. ~130K samples with labeled anomalies.

- **Clustering**: single process, few dimensions
- **AD**: solid benchmark for IoT anomaly detection
- **PM**: not designed for maintenance or degradation tracking
- Low dimensionality may not challenge multivariate methods

### MMS
**Manufacturing Monitoring System** -- real data from a manufacturing domain with multiple entities.

- **Clustering**: multiple entities for entity-level grouping
- **AD**: labeled anomalies available
- **PM**: moderate applicability
- Relatively small sample counts; sparse anomalies

### PSM
**Pooled Server Metrics (eBay)** -- 25 server metrics from eBay's infrastructure. ~87K samples with labeled anomalies.

- **Clustering**: multiple server entities with shared metric structure
- **AD**: publicly available with labels; used in recent benchmark papers
- **PM**: server metrics can show degradation but no explicit failure timelines
- Not a manufacturing domain; moderate dimensionality

### NASA Shuttle
**NASA Shuttle Valve Data** -- 9-16 sensor channels from shuttle hydraulic valves. Multiple operational runs with labeled anomalies.

- **Clustering**: multiple operational runs; different valve conditions
- **AD**: expert-labeled anomalies from a safety-critical system
- **PM**: valve degradation directly relevant to maintenance
- Older dataset; some dimension discrepancies across versions/sources

---

## Selected Datasets

The five datasets explored in this repo were chosen to cover different industrial domains while spanning all three research tasks:

| Dataset | Selected for |
|---------|-------------|
| **SMD** | Most-cited MTS AD benchmark; 28 machines in 3 groups for entity clustering |
| **SCANIA** | TTE labels for RUL estimation; real fleet data with realistic noise |
| **Genesis** | 9-state machine labels -- nearly unique ground truth for temporal clustering |
| **Exathlon** | 2,283 dimensions; root cause annotations for explainable AD |
| **DAMADICS** | 25 days of continuous factory data; multi-scale temporal regime structure |

Datasets not selected and why:
- **SWaT/WADI**: strong AD benchmarks but focused on cyber-security attacks, not operational fault patterns
- **PUMP**: good for PM but single entity, limited academic traction
- **MSL/SMAP**: aerospace telemetry, less relevant to industrial/manufacturing settings
- **PSM**: overlaps with SMD but less established
