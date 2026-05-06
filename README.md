# Machine Learning for SWE-Based Fluid Flow Simulation: A Comparative Study of U-Net, U-Net (LoMix), and FNO

**Course:** Research Methods  
**Author:** Dipesh Shrestha

A comparative study of three machine learning architectures for fast surrogate modeling of 2D partial differential equations (PDEs), specifically shallow water equations (SWE). This project benchmarks **U-Net**, **U-Net LoMix** (multi-scale fusion), and **Fourier Neural Operator (FNO)** against ground truth numerical simulations.

## Abstract: 

Computational fluid dynamics is widely used to study fluid flow, but numerical simulation is often expensive in time and computing resources. This becomes a challenge in applications such as flood prediction, where fast and reliable results are needed. This study compares machine learning surrogate models as faster alternatives for shallow water flow prediction. The three models considered are U-Net, U-Net (LoMix), and the Fourier Neural Operator (FNO). All three models are trained and tested on the same PDEBench radial dam-break dataset using the same one-step prediction task, training setup, and evaluation measures to ensure a fair comparison. Model performance is evaluated using root mean squared error (RMSE), qualitative field comparison, and inference runtime. The results show that FNO performs best overall, achieving the lowest prediction error, the fastest inference time, and the closest agreement with the ground-truth fields. U-Net (LoMix) performs better than standard U-Net, showing that multi-scale fusion improves prediction quality. Overall, this study suggests that operator-learning methods such as FNO are promising for accurate and efficient shallow water prediction.

## Overview

This repository provides three PDE surrogate models:

- UNet
- UNet LoMix (simplified LoMix-style variant)
- FNO

What it does
------------

- trains the three models on the PDEBench 2D shallow-water dataset
- evaluates RMSE and inference runtime
- generates figures from saved logs and checkpoints
- exports VTK files for visualization

Quick Start
-----------

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Make sure the HDF5 dataset is available at `../data/2D_rdb_NA_NA.h5`.

3. Train the models:

```bash
# UNet LoMix (default)
python scripts/train_unet.py --config configs/default.yaml

# Plain UNet
python scripts/train_unet.py --config configs/default.yaml --vanilla

# FNO
python scripts/train_fno.py --config configs/default.yaml
```

4. Evaluate the saved models:

```bash
python scripts/evaluate_all.py
```

5. Generate figures:

```bash
python scripts/generate_figures_wrapper.py
```

6. Run repeated significance experiments for runtime nference boxplots:

```bash
python scripts/run_repeated_significance.py --config configs/default.yaml --repeats 30 --discard-first 5
```

Main outputs
------------

- `results/checkpoints/` - saved checkpoints
- `results/logs/` - training logs
- `results/metrics_summary.json` - evaluation summary
- `figures/field_comparison.png` - prediction vs truth snapshot
- `figures/error_comparison.png` - RMSE per timestep
- `figures/error_comparison_bar.png` - RMSE and runtime bars
- `figures/training_val_loss_curves.png` - loss curves

Notes
-----

- `scripts/train_unet.py` trains UNet LoMix by default
- add `--vanilla` to train the plain UNet
- `scripts/train_unet.py` also supports `--quick` and `--resume`
- `scripts/export_vtk.py` writes VTK files under `results/vtk/`
- `scripts/run_repeated_significance.py` is optional for repeated experiments

## References

- **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* In Proceedings of the 18th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
- **U-Net LoMix:** Simplified LoMix-style variant inspired by Rahman, M. M., & Marculescu, R. (2025). *LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation.* In Advances in Neural Information Processing Systems (NeurIPS 2025).
- **FNO:** Kovachki, N., Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* In International Conference on Learning Representations (ICLR 2021).
- **PDEBench Dataset:** Takamoto, M., Sappl, R., Weiß, T., et al. (2022). *PDEBench: Comprehensive Benchmark for Scientific Machine Learning.* arXiv:2210.07182. https://arxiv.org/abs/2210.07182

## Authorship & Disclaimer

**Author:** Dipesh Shrestha

**AI Disclaimer:** Coding assistance from ChatGPT and GitHub Copilot was used during development. The author has thoroughly reviewed, checked, and verified the code for correctness and takes responsibility for the final implementation used in this project.

