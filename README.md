# TiDAL-Net: Time-Drift-Aware Liquid Spatio-Temporal Graph Network

This repository provides a clean public PyTorch implementation of **TiDAL-Net**, a time-drift-aware liquid spatio-temporal graph network for irregular satellite constellation telemetry data reconstruction.

The code is organized for GitHub release and follows the structure used in the companion public repositories: the proposed method is fully implemented, while third-party comparison methods are provided as independent adapter files because their official implementations belong to their original authors.

## Method overview

TiDAL-Net is designed for multichannel satellite telemetry streams affected by missing samples, timestamp drift, noise, and dynamic mission-condition changes. It contains three core modules:

1. **Bi-NODE timestamp compensation**: bidirectional neural ODE-style trajectory estimation compensates missing and drifted timestamps before spatio-temporal reconstruction.
2. **Li-GRU temporal reconstruction**: a liquid gated recurrent unit uses input-dependent time constants to model nonstationary, abrupt, and multiscale telemetry dynamics.
3. **Li-GAT dynamic graph fusion**: a liquid graph attention layer updates local graph neighborhoods based on time-varying hidden-state similarity and temporal synchronization.

The default implementation supports the ablation variants described in the manuscript:

- `TiDAL-w/o-NODE`
- `TiDAL-w/o-BiNODE`
- `TiDAL-w/o-GRU`
- `TiDAL-w/o-Li-GRU`
- `TiDAL-w/o-GAT`
- `TiDAL-w/o-Li-GAT`
- `TiDAL-Full`

## Repository structure

```text
TiDAL-Net-Telemetry-Reconstruction/
├── baselines/                  # One independent adapter file for each comparison method
│   ├── autoformer.py
│   ├── pareformer.py
│   ├── gnn_dtan.py
│   ├── msd_gnn.py
│   ├── mag.py
│   ├── histar.py
│   ├── mstgad.py
│   ├── traversenet.py
│   ├── cool.py
│   └── dstgnn.py
├── configs/
│   └── default.yaml             # Default training/testing configuration
├── docs/
│   ├── baseline_notice.md       # Copyright and redistribution notice for baselines
│   └── reproducibility.md       # Dataset format and experiment protocol
├── examples/
│   └── run_demo.py              # CPU demo using synthetic telemetry data
├── scripts/
│   └── run_train_test.py         # Main training and testing entry point
├── tidal_recovery/
│   ├── data.py                   # Excel/CSV loading, anomaly simulation, windowing
│   ├── trainer.py                # Training, validation, prediction
│   ├── metrics.py                # MAE, RMSE, MAPE and missing-region metrics
│   ├── exports.py                # Result export and plots
│   └── models/
│       ├── components.py         # Bi-NODE, Li-GRU, Li-GAT components
│       ├── tidal_net.py          # TiDAL-Net model
│       └── registry.py           # Model/ablation registry
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Important notice about comparison methods

The manuscript compares TiDAL-Net with 10 SOTA baselines:

- Autoformer
- PAREformer
- GNN-DTAN
- MSD-GNN
- MAG
- HiSTAR
- MSTGAD
- TraverseNet
- COOL
- DSTGNN

For academic and legal reasons, this public repository **does not redistribute the official source code of these third-party methods**. Each baseline is placed in an independent `.py` adapter under `baselines/`, but the adapter intentionally raises a clear error until the user installs the authorized official implementation and completes the local wrapper.

Please read the original paper of each comparison method and follow the original authors' license, citation, and redistribution requirements. Do not copy restricted third-party code into this repository unless the corresponding license explicitly permits redistribution.

## Installation

```bash
git clone https://github.com/your-name/TiDAL-Net-Telemetry-Reconstruction.git
cd TiDAL-Net-Telemetry-Reconstruction
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0+ are recommended.

## Data format

The expected telemetry file is `.xlsx`, `.xls`, or `.csv`:

```text
Column 1      : time / timestamp / index
Columns 2..end: telemetry channels
Rows          : time steps
```

Example:

| time | sensor_01 | sensor_02 | ... |
|---:|---:|---:|---:|
| 0 | 0.12 | 1.38 | ... |
| 1 | 0.18 | 1.41 | ... |

The code interpolates raw labels only to construct complete supervision targets. Artificial missing and timestamp-drift anomalies are then generated according to the configured anomaly ratio.

## Quick demo

Run a CPU demo with synthetic telemetry:

```bash
python examples/run_demo.py
```

Outputs will be saved to:

```text
outputs/demo/
```

## Train and test on your own Excel file

```bash
python scripts/run_train_test.py \
  --config configs/default.yaml \
  --excel_path /path/to/your_telemetry.xlsx \
  --output_dir outputs/my_dataset \
  --models TiDAL-Full
```

Run all TiDAL-Net ablations:

```bash
python scripts/run_train_test.py \
  --excel_path /path/to/your_telemetry.xlsx \
  --output_dir outputs/ablation \
  --models TiDAL-w/o-NODE,TiDAL-w/o-BiNODE,TiDAL-w/o-GRU,TiDAL-w/o-Li-GRU,TiDAL-w/o-GAT,TiDAL-w/o-Li-GAT,TiDAL-Full
```

Change timestamp anomaly ratio:

```bash
python scripts/run_train_test.py \
  --excel_path /path/to/your_telemetry.xlsx \
  --output_dir outputs/rate_10 \
  --anomaly_rate 0.10 \
  --models TiDAL-Full
```

## Outputs

Each run exports:

```text
outputs/<run>/
├── metrics_summary.csv
├── metrics_summary.xlsx
├── dataset_metadata.json
├── adjacency.npy
├── trained_models/
│   └── TiDAL-Full.pt
├── training_history/
│   └── TiDAL-Full_history.csv
├── predictions/
│   ├── TiDAL-Full_predictions.npz
│   ├── TiDAL-Full_per_channel_rmse.csv
│   └── TiDAL-Full_example_ch*.png
└── rmse_comparison.png
```

## Citation

If this repository helps your research, please cite the corresponding TiDAL-Net paper after publication. A BibTeX entry can be added here once the final bibliographic information is available.

```bibtex
@article{tidalnet2026,
  title   = {A Time-Drift-Aware Liquid Spatio-Temporal Graph Network for Satellite Constellation Irregular Telemetry Data Reconstruction},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

## License

The TiDAL-Net implementation in `tidal_recovery/` is released under the MIT License. Third-party comparison methods are not included and remain under the licenses of their original authors.
