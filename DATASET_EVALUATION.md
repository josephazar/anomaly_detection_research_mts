# MTS Dataset Evaluation

We surveyed 16 publicly available multivariate time series datasets and scored each on four axes that matter for this research: **clustering**, **anomaly detection**, **predictive maintenance**, and **complexity** (whether the dataset is challenging enough to warrant new methods). Each score is out of 10.

The five datasets we selected for the exploration notebooks are marked with a notebook link.

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

- **Clustering**: six interconnected stages give natural sub-system structure; attack scenarios can be seen as regime shifts
- **AD**: one of the most-used AD benchmarks; labeled attacks with both subtle and obvious patterns
- **PM**: not the primary focus -- attacks cause stress but there are no run-to-failure trajectories
- **Watch out**: access requires a request to iTrust; some sensor gaps at certain stages
- **Literature**: widely cited across KDD, NeurIPS, AAAI

### WADI
**Water Distribution** -- 127 dimensions from a scaled-down water distribution testbed (also iTrust). 957K samples over 16 days.

- **Clustering**: three processing stages; operational modes differ between normal and attack periods
- **AD**: excellent benchmark, arguably harder than SWaT due to subtler anomalies and higher dimensionality
- **PM**: limited -- designed for security research, not degradation tracking
- **Watch out**: the initial release (WADI.A1) had stability issues; an updated version (WADI.A2) exists
- **Literature**: strong presence in CPS security venues

### PUMP
**Industrial Water Pump** -- 44 sensor features from a real pump system on Kaggle. 220K samples with machine status labels (NORMAL, BROKEN, RECOVERING).

- **Clustering**: three operational states make a clean clustering target
- **AD**: labeled failures with clear BROKEN/RECOVERING transitions
- **PM**: the dataset's main selling point -- built for predicting pump failures
- **Watch out**: sensor names are anonymized (no physical interpretation); single entity only; limited academic citations vs. Kaggle popularity

### SMD
**Server Machine Dataset** -- 38 metrics from 28 server machines across 3 groups. ~1.4M total samples. From the OmniAnomaly paper (KDD 2019).

- **Clustering**: the 3 machine groups are ground truth for entity-level clustering; within-machine regime transitions add a second clustering dimension
- **AD**: point-level labels + interpretation labels (which features caused each anomaly) -- rare and valuable
- **PM**: anomalies represent real faults; some show precursor signals, though no explicit RUL labels
- **Watch out**: all data from one company; no variable names (just indices 0-37)
- **Literature**: one of the most-cited MTS AD benchmarks; actively used in top venues
- Notebook: [`01_SMD_Exploration.ipynb`](01_SMD_Exploration.ipynb)

### Exathlon
**Exathlon Benchmark** -- 2,283 Spark performance metrics from 10 streaming applications. Ground truth includes anomaly type + root cause timing.

- **Clustering**: 10 apps with different workloads give entity-level clusters; intra-app regime discovery is possible but no phase labels exist
- **AD**: multiple anomaly types (bursty input, CPU contention, driver/executor failures) with explainability annotations -- unique among benchmarks
- **PM**: root cause timing enables early detection analysis; extended effect periods define propagation windows
- **Watch out**: 2,283 dims is extreme -- most are sparse or constant; this is IT infrastructure, not manufacturing OT
- **Literature**: introduced at VLDB 2021; growing use in explainable AD research
- Notebook: [`04_Exathlon_Exploration.ipynb`](04_Exathlon_Exploration.ipynb)

### MSL
**Mars Science Laboratory** -- 55 telemetry channels from the Curiosity rover. ~132K samples with expert-labeled anomalies.

- **Clustering**: limited -- single entity with no inherent phase structure
- **AD**: strong benchmark; expert labels across multiple anomaly types (point, contextual, collective)
- **PM**: real degradation events in a safety-critical system
- **Watch out**: some inconsistencies in how the dataset is reported across papers; access may require request to NASA

### SMAP
**Soil Moisture Active Passive** -- 25 telemetry channels from NASA's SMAP satellite. 562K samples.

- **Clustering**: multiple telemetry entities with different subsystems
- **AD**: widely used alongside MSL; expert-labeled anomalies
- **PM**: limited -- satellite telemetry is more about detection than degradation prediction
- **Watch out**: not directly related to manufacturing/industrial settings

### DAMADICS
**Development and Application of Methods for Actuator Diagnosis** -- 32 process variables from the Lublin Sugar Factory (Poland). 25 days at 1 Hz = 2.16M samples. Known fault injection on Nov 15, 2001.

- **Clustering**: strong intra-day regime structure (startup, production, shutdown); weekend/weekday patterns add macro-level clusters
- **AD**: known fault date gives clear evaluation; incipient faults develop gradually
- **PM**: a classic FDI benchmark -- designed specifically for fault detection and isolation
- **Watch out**: data from 2001; only one confirmed fault event; variable names are generic (Var_1 to Var_32)
- **Literature**: 300+ citations; still used in modern deep learning papers
- Notebook: [`05_DAMADICS_Exploration.ipynb`](05_DAMADICS_Exploration.ipynb)

### Genesis
**Genesis Demonstrator** -- 18 sensor/actuator channels from a physical pick-and-place system. ~16K samples per file. State machine labels (9 states) + anomaly labels + separate fault files.

- **Clustering**: the 9 state machine labels are perfect ground truth -- rare among MTS datasets. The cyclic process creates repeating phases that TICC should recover
- **AD**: two fault types (linear drive, pressure) with clear labels
- **PM**: fault files allow studying progression, but no explicit RUL or degradation trajectory annotations
- **Watch out**: small dataset; single machine; may not stress-test scalability of methods
- **Literature**: less cited than SMD/SWaT, but growing use in smart manufacturing research
- Notebook: [`03_Genesis_Exploration.ipynb`](03_Genesis_Exploration.ipynb)

### GHL
**Gas Heating Loop** -- simulated industrial control system with cyber-attack scenarios. 30+ process variables.

- **Clustering**: operational modes under normal and attack conditions
- **AD**: labeled anomalies from simulated attacks on the heating loop
- **PM**: moderate -- attack-induced degradation can be studied
- **Watch out**: synthetic data; the original paper notes "no outliers" in the baseline data, which may limit certain analyses

### SCANIA
**SCANIA Component X** -- 105 anonymized operational features from a fleet of heavy-duty trucks. ~1.1M readouts. Binary failure labels + time-to-event (TTE) for training vehicles.

- **Clustering**: 8 truck specification categories create natural fleet segments; operational profiles vary by spec group
- **AD**: ~3-5% failure rate with extreme class imbalance -- realistic for real-world AD
- **PM**: the TTE labels are the highlight -- directly enable survival analysis and RUL estimation, which is rare
- **Watch out**: features are anonymized (no physical meaning); heavy missing values; released 2024 so limited prior work
- **Literature**: from SCANIA CV AB; significant research opportunity given how recent it is
- Notebook: [`02_SCANIA_Exploration.ipynb`](02_SCANIA_Exploration.ipynb)

### GECCO
**GECCO IoT Water Quality** -- 9 sensors monitoring drinking water quality. ~130K samples with labeled anomalies.

- **Clustering**: limited -- single process, few dimensions
- **AD**: good benchmark for IoT anomaly detection
- **PM**: not designed for maintenance or degradation tracking
- **Watch out**: low dimensionality may not challenge multivariate methods

### MMS
**Manufacturing Monitoring System** -- real data from a manufacturing domain with multiple entities.

- **Clustering**: multiple entities allow entity-level grouping
- **AD**: labeled anomalies available
- **PM**: moderate applicability
- **Watch out**: relatively small sample counts; sparse anomalies can be tricky

### PSM
**Pooled Server Metrics (eBay)** -- 25 server metrics from eBay's infrastructure. ~87K samples with labeled anomalies.

- **Clustering**: multiple server entities with shared metric structure
- **AD**: publicly available with labels; used as a benchmark in recent papers
- **PM**: moderate -- server metrics can show degradation but no explicit failure timelines
- **Watch out**: not from a manufacturing domain; moderate dimensionality

### NASA Shuttle
**NASA Shuttle Valve Data** -- 9-16 sensor channels from shuttle hydraulic valves. Multiple operational runs with labeled anomalies.

- **Clustering**: multiple operational runs; different valve conditions
- **AD**: expert-labeled anomalies from a safety-critical system
- **PM**: valve degradation over time is directly relevant to maintenance
- **Watch out**: older dataset; some dimension discrepancies across different versions/sources

---

## Why We Picked These Five

We wanted datasets that cover different industrial domains while collectively supporting all three research tasks. Here's the rationale:

| Dataset | Primary Strength | Why It Made the Cut |
|---------|-----------------|---------------------|
| **SMD** | AD benchmark + multi-entity | 28 machines in 3 groups give both entity clustering and AD evaluation. The most-cited dataset in MTS AD. |
| **SCANIA** | Predictive maintenance | TTE labels are rare and directly enable RUL research. Real fleet data with realistic messiness. |
| **Genesis** | Clustering ground truth | 9-state machine labels are nearly unique among MTS datasets. Perfect for validating TICC-style methods. |
| **Exathlon** | Explainability + scale | 2,283 dimensions push the limits. Root cause annotations enable explainable AD research. |
| **DAMADICS** | Industrial process + temporal richness | 25 days of continuous factory data with multi-scale temporal structure. Classic FDI benchmark with modern relevance. |

Datasets we considered but didn't include:
- **SWaT/WADI**: excellent for AD but focused on cyber-security attacks rather than operational fault patterns
- **PUMP**: good for PM but single entity and limited academic use
- **MSL/SMAP**: aerospace telemetry, less relevant to manufacturing-oriented research
- **PSM**: similar niche to SMD but less established
