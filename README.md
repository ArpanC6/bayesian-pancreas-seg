# Bayesian 3D Pancreas Segmentation with Uncertainty Quantification

Self-directed demonstration for the BodyMaps AI Bootcamp at Johns Hopkins University, CCVL Lab (Prof. Zongwei Zhou).

## Overview

This project implements a Bayesian 3D Residual UNet with Attention Gates for automated pancreas segmentation on abdominal CT scans. The model incorporates Monte Carlo (MC) Dropout to quantify prediction uncertainty, decomposing it into epistemic (model) and aleatoric (data) components.

## Architecture

| Component | Details |
|---|---|
| Backbone | 3D Residual UNet (4 encoder-decoder stages) |
| Skip Connections | Attention Gates (inspired by UNet++) |
| Bayesian Layers | Dropout active during both training and inference |
| Uncertainty Estimation | MC Dropout (T=30 forward passes), Entropy / Epistemic / Aleatoric maps |
| Input Resolution | 128 x 128 x 128 resampled CT volumes |
| Output | Binary pancreas mask + uncertainty maps |

## Dataset

MSD Pancreas (Medical Segmentation Decathlon -- Task 07)

- 281 matched CT volumes
- Train/Val split: 224 / 57
- Preprocessing: intensity clipping [-175, 250] HU, min-max normalization, resampling to 128^3

## Training

| Parameter | Value |
|---|---|
| Loss | Dice Loss + 0.5 * Binary Cross-Entropy |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50, T_mult=2) |
| Epochs | 300 |
| Batch Size | 1 (effective batch size = 4 via gradient accumulation) |
| Hardware | Kaggle T4 GPU |

## Expected Results

| Metric | Range |
|---|---|
| Validation Dice | ~0.82 -- 0.88 |
| Hausdorff-95 | ~10 -- 15 mm |

## Quick Start

### Installation

```bash
git clone https://github.com/ArpanC6/bayesian-pancreas-seg.git
cd bayesian-pancreas-seg
pip install -r requirements.txt
```

### Training

```bash
python train.py --data_root ./data/Task07_Pancreas --epochs 300 --save_dir ./checkpoints
```

### Inference with Uncertainty Quantification

```bash
python inference.py --checkpoint ./checkpoints/best_model.pth --data_root ./data/Task07_Pancreas --T 30 --output_dir ./predictions
```

The inference script produces three outputs per case:

1. `*_pred.npy` -- mean predicted probability map
2. `*_epistemic.npy` -- epistemic uncertainty (variance across MC passes)
3. `*_entropy.npy` -- predictive entropy (total uncertainty)

## File Structure

```
bayesian-pancreas-seg/
|-- model.py              # BayesianUNet3D architecture
|-- train.py              # Training loop with Dice+BCE loss
|-- dataset.py            # MSD Pancreas data loader
|-- inference.py          # MC Dropout inference + uncertainty maps
|-- utils.py              # Dice score and evaluation metrics
|-- requirements.txt      # Python dependencies
|-- README.md             # This file
```

## Key Design Choices

1. **Residual Blocks**: Stabilize gradient flow in deep 3D networks.
2. **Attention Gates**: Suppress irrelevant background regions in skip connections, focusing the decoder on the pancreas region.
3. **MC Dropout**: Enables uncertainty quantification without requiring an ensemble of models. Dropout layers remain active during inference to sample from the approximate posterior.
4. **Conservative Postprocessing**: No heavy morphological operations; the model is trained end-to-end to produce anatomically plausible masks.

## Author

Arpan Chakraborty  
Email: chakrabortyarpan151@gmail.com  
GitHub: [ArpanC6](https://github.com/ArpanC6)

## Acknowledgments

- Johns Hopkins University, CCVL Lab
- Medical Segmentation Decathlon (MSD) for the public dataset
- UNet++ and Attention UNet papers for architectural inspiration
