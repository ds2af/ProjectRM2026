# Machine Learning for SWE-Based Fluid Flow Simulation: A Comparative Study of U-Net, U-Net (LoMix), and FNO

**Course:** Research Methods  
**Author:** Dipesh Shrestha

A comparative study of three machine learning architectures for fast surrogate modeling of 2D partial differential equations (PDEs), specifically shallow water equations (SWE). This project benchmarks **U-Net**, **U-Net LoMix** (multi-scale fusion), and **Fourier Neural Operator (FNO)** against ground truth numerical simulations.

## Abstract: 

Computational fluid dynamics is widely used to study fluid flow, but numerical simulation is often expensive in time and computing resources. This becomes a challenge in applications such as flood prediction, where fast and reliable results are needed. This study compares machine learning surrogate models as faster alternatives for shallow water flow prediction. The three models considered are U-Net, U-Net (LoMix), and the Fourier Neural Operator (FNO). All three models are trained and tested on the same PDEBench radial dam-break dataset using the same one-step prediction task, training setup, and evaluation measures to ensure a fair comparison. Model performance is evaluated using root mean squared error (RMSE), qualitative field comparison, and inference runtime. The results show that FNO performs best overall, achieving the lowest prediction error, the fastest inference time, and the closest agreement with the ground-truth fields. U-Net (LoMix) performs better than standard U-Net, showing that multi-scale fusion improves prediction quality. Overall, this study suggests that operator-learning methods such as FNO are promising for accurate and efficient shallow water prediction.

## Overview

Machine learning surrogate models can accelerate PDE solving by orders of magnitude compared to direct numerical simulation. This repository provides:

- **Three models** for comparison:
  - **U-Net**: Standard 2D encoder-decoder CNN baseline
  - **U-Net LoMix**: U-Net with multi-scale output fusion
  - **FNO2d**: Fourier-based operator learning

- **Automated training pipeline**: Single command to train all models sequentially
- **Comprehensive evaluation**: RMSE, inference time
- **Publication-ready figures**: Training/validation loss curves, field comparisons, RMSE curves, aggregate RMSE plots, speedup plots
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
- `results/metrics_summary.json` — aggregated RMSE and inference time per model
- Checkpoints: `results/checkpoints/{unet,unet_lomix,fno}_best.pt`

### Evaluation & Metrics

```bash
# Evaluate trained models and aggregate metrics
python scripts/evaluate_all.py
```

### Figure Generation

```bash
# Generate all publication figures (Figures 1–5)
python scripts/generate_figures.py
```

**Figures produced:**
- **fig1_training_validation_loss.png** — Training and validation loss curves for each model
- **fig2_field_comparison.png** — Spatial field predictions at a single timestep
- **fig3_error_comparison.png** — RMSE timeline (autoregressive rollout error growth)
- **fig4_aggregate_rmse.png** — Aggregate RMSE comparison across models
- **fig5_speedup.png** — Inference time comparison


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


## Models

### U-Net (Vanilla)

Standard 2D encoder-decoder CNN with skip connections. Encodes spatial features through downsampling, learns latent representation, decodes back to full resolution.

- **Training:** `train_unet.py --vanilla`
- **Checkpoint:** `results/checkpoints/unet_best.pt`

### U-Net LoMix

U-Net backbone with **multi-scale output fusion**. Produces predictions at 4 decoder depths, upsamples each to full resolution, and fuses them using learned per-pixel weights.

- **Training:** `train_unet.py` (default)
- **Checkpoint:** `results/checkpoints/unet_lomix_best.pt`
- **Purpose:** Compare multi-scale fusion against the vanilla U-Net baseline

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

## Dataset

This project uses the **radial dam-break benchmark** from the [PDEBench](https://github.com/pdebench/PDEBench) dataset collection. The dataset contains solutions to shallow water equations (SWE) generated by numerical simulation.

**Important:** The dataset is **not included** in this GitHub repository due to file size constraints. You must download it separately:

1. **Download PDEBench dataset:**
   - Visit: https://github.com/pdebench/PDEBench
   - Follow their instructions to download the 2D shallow water equations (SWE) radial dam-break dataset
   - The file is named `2D_rdb_NA_NA.h5`

2. **Place the dataset:**
   ```
   data/
   └── 2D_rdb_NA_NA.h5         # Place downloaded file here
   ```

3. **Dataset structure:**
   - 100+ independent trajectories (seeds), each with 200 timesteps
   - Spatial resolution: 128×128 grid
   - Variables: water height (h), velocity components (u, v)
   - Train/val/test split: 80%/10%/10% (deterministic by seed index)

**Reference:** Takamoto et al. (2022). PDEBench: Comprehensive Benchmark for Scientific Machine Learning. See [arXiv:2210.07182](https://arxiv.org/abs/2210.07182).

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
- **PDEBench Dataset:** Takamoto, Sappl, Weiß, et al. (2022). *PDEBench: Comprehensive Benchmark for Scientific Machine Learning.* [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

## Authorship & Disclaimer

**Author:** Dipesh Shrestha

**AI Disclaimer:** Coding assistance from ChatGPT and GitHub Copilot was used during development. The author has thoroughly reviewed, checked, and verified the code for correctness and takes responsibility for the final implementation used in this project.

