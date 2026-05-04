# ML Surrogate Models for 2D Shallow Water Equations

A comparative study of three machine learning architectures for fast surrogate modeling of 2D partial differential equations (PDEs), specifically shallow water equations (SWE). This project benchmarks **U-Net**, **U-Net LoMix** (multi-scale fusion), and **Fourier Neural Operator (FNO)** against ground truth numerical simulations.

## Overview

Machine learning surrogate models can accelerate PDE solving by orders of magnitude compared to direct numerical simulation. This repository provides:

- **Three models** for comparison:
  - **U-Net**: Standard 2D encoder-decoder CNN baseline
  - **U-Net LoMix**: U-Net with multi-scale output fusion
  - **FNO2d**: Fourier-based operator learning

- **Automated training pipeline**: Single command to train all models sequentially
- **Comprehensive evaluation**: RMSE, inference time
- **Publication-ready figures**: Field comparisons, error curves, speedup plots, generalization analysis
- **VTK export**: Visualization-ready output files for ParaView/VisIt

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ProjectRM2026

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training All Models

```bash
# Full training (200 epochs, all models)
python scripts/train_all.py --config configs/default.yaml

# Quick sanity check (3 epochs)
python scripts/train_all.py --config configs/default.yaml --quick
```

### Quick Testing (Individual Models)

```bash
# Train U-Net only (vanilla, no LoMix)
python scripts/train_unet.py --config configs/default.yaml --vanilla

# Train U-Net LoMix (default)
python scripts/train_unet.py --config configs/default.yaml

# Train FNO
python scripts/train_fno.py --config configs/default.yaml
```

## Usage

### Master Training Script

```bash
python scripts/train_all.py \
  --config configs/default.yaml \
  --quick                           # Quick mode (3 epochs, 20 samples)
  --epochs 150                      # Override training epochs
  --models unet unet_lomix          # Train subset of models
  --skip_vtk                        # Skip VTK export
```

**Outputs:**
- `results/training_summary.json` — wall-clock times, metrics, model checkpoints
- `results/metrics_summary.json` — aggregated RMSE, rel_l2, max_error, inference time per model
- Checkpoints: `results/checkpoints/{unet,unet_lomix,fno}_best.pt`

### Evaluation & Metrics

```bash
# Evaluate trained models and aggregate metrics
python scripts/evaluate_all.py
```

### Figure Generation

```bash
# Generate all publication figures (Figures 3–6)
python scripts/generate_figures.py
```

**Figures produced:**
- **fig3_field_comparison.png** — Spatial field predictions at a single timestep
- **fig4_error_comparison.png** — RMSE timeline (autoregressive rollout error growth)
- **fig4_error_comparison_bar.png** — Summary bar chart: RMSE, rel_l2, max_error
- **fig5_speedup.png** — Inference time comparison
- **fig6_generalization.png** — Out-of-distribution performance on alternate dataset


### VTK Export (3D Visualization)

```bash
# Export model predictions to VTK format for ParaView/VisIt
python scripts/export_vtk.py --model fno --config configs/default.yaml

# Export all models
python scripts/export_vtk.py --model all --config configs/default.yaml

# Specify number of samples/timesteps
python scripts/export_vtk.py --model unet_lomix --n_samples 3 --n_timesteps 50
```

**Outputs:** `results/vtk/{model_name}/sample*.vtk` files

## Project Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── configs/
│   └── default.yaml                   # Default configuration (data, training, model params)
├── scripts/
│   ├── train_unet.py                  # Train U-Net / U-Net LoMix
│   ├── train_fno.py                   # Train FNO2d
│   ├── train_all.py                   # Master orchestrator (trains all, runs eval + figures)
│   ├── evaluate_all.py                # Aggregate metrics into JSON summary
│   ├── generate_figures.py            # Generate publication figures (3–6)
│   ├── export_vtk.py                  # Export predictions to VTK
│   └── run_repeated_significance.py   # Repeated training + statistical testing
├── src/
│   ├── models/
│   │   ├── unet.py                    # UNet2d & UNet2dLoMix implementations
│   │   └── fno.py                     # FNO2d (Fourier Neural Operator)
│   ├── data/
│   │   ├── dataset.py                 # SWEDataset, SWEDatasetWithGrid
│   │   ├── preprocessing.py           # Normalization utilities
│   │   └── __init__.py
│   └── utils/
│       ├── trainer.py                 # Shared training loop (autoregressive support)
│       ├── metrics.py                 # RMSE, rel_l2, max_error evaluation
│       ├── logger.py                  # CSV/JSON logging
│       └── __init__.py
├── results/                           # Generated outputs
│   ├── checkpoints/                   # Model checkpoints
│   ├── logs/                          # Training curves (CSV)
│   ├── metrics/                       # Per-model metrics (JSON)
│   ├── figures/                       # Publication figures
│   ├── vtk/                           # VTK exports
│   ├── metrics_summary.json           # Aggregated metrics
│   └── training_summary.json          # Training times & summary
└── figures/                           # Pre-generated figures (archival)
    └── old/                           # Old figure revisions
```

## Models

### U-Net (Vanilla)

Standard 2D encoder-decoder CNN with skip connections. Encodes spatial features through downsampling, learns latent representation, decodes back to full resolution.

- **Training:** `train_unet.py --vanilla`
- **Checkpoint:** `results/checkpoints/unet_best.pt`

### U-Net LoMix

U-Net backbone with **multi-scale output fusion**. Produces predictions at 4 decoder depths, upsamples each to full resolution, and fuses them using learned per-pixel weights.

- **Training:** `train_unet.py` (default)
- **Checkpoint:** `results/checkpoints/unet_lomix_best.pt`
- **Purpose:** Test whether multi-scale fusion improves generalization

### FNO2d (Fourier Neural Operator)

Operator learning via Fourier-space convolution. Lifts input to latent space, applies 4 spectral convolution layers with residual skip connections, then projects back to output space.

- **Training:** `train_fno.py`
- **Checkpoint:** `results/checkpoints/fno_best.pt`
- **Special handling:** Requires grid coordinates; uses `SWEDatasetWithGrid`


## Configuration

Default hyperparameters are in `configs/default.yaml`:

```yaml
data:
  filename: "2D_rdb_NA_NA"        # HDF5 dataset
  initial_step: 10                # Context window
  t_train: 101                    # Supervised training timesteps
  test_ratio: 0.1                 # Train/val/test split
  
training:
  epochs: 200
  batch_size: 4
  learning_rate: 1e-3
  early_stopping_patience: 20

unet:
  init_features: 32
  dropout: 0.0
  unroll_step: 20                 # Pushforward window for AR training

fno:
  modes1: 12                      # Fourier modes (x-direction)
  modes2: 12                      # Fourier modes (y-direction)
  width: 20                       # Internal latent width
```

## Data Format

The project expects HDF5 files with structure:

```
data/
└── 2D_rdb_NA_NA.h5
    ├── seed_0/
    │   └── data [T, H, W, C]    # T timesteps, H×W spatial grid, C channels
    ├── seed_1/
    │   └── data [T, H, W, C]
    └── ...
```

Each seed is an independent trajectory. The dataset is split deterministically by seed index (10% test, 10% validation, 80% training by default).

## Requirements

- **Python:** 3.9+
- **Core:** PyTorch ≥2.0, NumPy ≥1.24, SciPy ≥1.10
- **Data I/O:** h5py ≥3.8
- **Plotting:** Matplotlib ≥3.7
- **Config:** PyYAML ≥6.0

See `requirements.txt` for pinned versions.

## References

- **U-Net:** Ronneberger, Fischer, Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- **U-Net LoMix:** Rahman, Marculescu (2025). *LoMix: Low-resolution Multi-scale Mixing.*
- **FNO:** Kovachki, Li, et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.

## Authorship & Disclaimer

**Author:** Dipesh Shrestha

**AI Disclaimer:** Coding assistance from ChatGPT and GitHub Copilot was used during development. The author has thoroughly reviewed, checked, and verified the code for correctness and takes responsibility for the final implementation used in this project.

