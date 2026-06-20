# Reproducibility Guide

## Datasets

The manuscript evaluates TiDAL-Net on four telemetry datasets: MSL, SMAP, Real_A, and Real_B. This public code does not redistribute private spacecraft telemetry data. Users can prepare their own telemetry table in the expected Excel/CSV format.

## Data split

The default split follows the manuscript setting:

- Training: 60%
- Validation: 20%
- Testing: 20%

The split is chronological to avoid future information leakage.

## Timestamp anomaly simulation

The code supports missing and timestamp-drift simulation through these configuration keys:

```yaml
anomaly_rate: 0.05
missing_fraction: 0.60
drift_fraction: 0.40
drift_max_lag: 4
block_count: 12
block_min_len: 4
block_max_len: 24
add_noise_std: 0.01
```

To reproduce the manuscript severity settings, run separate experiments with:

```bash
--anomaly_rate 0.03
--anomaly_rate 0.05
--anomaly_rate 0.08
--anomaly_rate 0.10
```

## TiDAL-Net ablation protocol

Use:

```bash
--models TiDAL-w/o-NODE,TiDAL-w/o-BiNODE,TiDAL-w/o-GRU,TiDAL-w/o-Li-GRU,TiDAL-w/o-GAT,TiDAL-w/o-Li-GAT,TiDAL-Full
```

The ablation definitions are:

- `TiDAL-w/o-NODE`: weakens bidirectional NODE compensation into a unidirectional compensation variant.
- `TiDAL-w/o-BiNODE`: removes Bi-NODE compensation and directly uses the filled observation window.
- `TiDAL-w/o-GRU`: replaces liquid temporal modeling with a standard node-wise GRU cell.
- `TiDAL-w/o-Li-GRU`: same standard-GRU temporal fallback, used for naming consistency with the manuscript.
- `TiDAL-w/o-GAT`: replaces liquid dynamic graph attention with a static graph attention block.
- `TiDAL-w/o-Li-GAT`: same static graph fallback, used for naming consistency with the manuscript.
- `TiDAL-Full`: full method.

## Metrics

The exported metrics include:

- MAE
- RMSE
- MAPE
- Missing_MAE
- Missing_RMSE
- Missing_MAPE
- inference latency per sample
- number of trainable parameters
