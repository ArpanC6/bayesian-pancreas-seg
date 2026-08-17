# Bayesian 3D Pancreas Segmentation with Uncertainty Quantification

Self-directed demonstration for the **BodyMaps AI Bootcamp** — Johns Hopkins University, CCVL Lab (Prof. Zongwei Zhou).

## Overview

Bayesian 3D Residual UNet with Attention Gates for pancreas segmentation on abdominal CT scans. Includes Monte Carlo Dropout for uncertainty quantification (epistemic + aleatoric decomposition).

## Architecture

| Component | Details |
|-----------|---------|
| Backbone | 3D Residual UNet (4 encoder-decoder stages) |
| Skip Connections | Attention Gates (UNet++ philosophy) |
| Bayesian Layers | Dropout active at train + inference |
| Uncertainty | MC Dropout (T=30), Entropy / Epistemic / Aleatoric |
| Input | 128 x 128 x 128 resampled CT |

## Dataset

- **MSD Pancreas** (Medical Segmentation Decathlon — Task 07)
- 281 matched CT volumes (224 train / 57 val)

## Training

| Parameter | Value |
|-----------|-------|
| Loss | Dice + 0.5 × BCE |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=50) |
| Epochs | 300 |
| Batch | 1 (effective = 4 via grad accum) |
| Hardware | Kaggle T4 GPU |

## Expected Results

- Validation Dice: ~0.82 – 0.88
- Hausdorff-95: ~10 – 15 mm

## Files

- `train.py` — complete training script
- `requirements.txt` — dependencies

## Author

**Arpan Chakraborty**  
Email: chakrabortyarpan151@gmail.com  
GitHub: [ArpanC6](https://github.com/ArpanC6)
