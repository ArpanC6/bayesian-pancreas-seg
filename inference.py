"""
Bayesian Inference with Monte Carlo Dropout
Generates: Prediction + Epistemic Uncertainty + Predictive Entropy
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import BayesianUNet3D
from dataset import MSDPancreasDataset
from torch.utils.data import DataLoader


def mc_dropout_predict(model, image, T=30, device="cuda"):
    """
    T stochastic forward passes with dropout enabled
    Returns:
        mean_pred: [1, 1, D, H, W]
        epistemic: variance across T passes (model uncertainty)
        entropy: predictive entropy
    """
    model.train()  # Keep dropout ON
    preds = []
    
    with torch.no_grad():
        for _ in range(T):
            out = model(image, enable_dropout=True)
            prob = torch.sigmoid(out)
            preds.append(prob.cpu().numpy())
    
    preds = np.stack(preds, axis=0)  # [T, 1, 1, D, H, W]
    mean_pred = preds.mean(axis=0)
    epistemic = preds.var(axis=0)
    
    # Predictive entropy: -E[p] log E[p]
    eps = 1e-7
    entropy = -mean_pred * np.log(mean_pred + eps) - (1-mean_pred) * np.log(1-mean_pred + eps)
    
    return mean_pred, epistemic, entropy


def save_nifti(data, affine, path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(data, affine), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/Task07_Pancreas")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--T", type=int, default=30, help="MC Dropout forward passes")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = BayesianUNet3D(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (Dice: {ckpt['best_dice']:.4f})")
    
    # Load data
    val_ds = MSDPancreasDataset(root=args.data_root, split="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    for batch in val_loader:
        image = batch["image"].to(device)
        name = batch["name"][0].replace(".nii.gz", "")
        
        mean_pred, epistemic, entropy = mc_dropout_predict(model, image, T=args.T, device=device)
        
        # Save
        np.save(os.path.join(args.output_dir, f"{name}_pred.npy"), mean_pred)
        np.save(os.path.join(args.output_dir, f"{name}_epistemic.npy"), epistemic)
        np.save(os.path.join(args.output_dir, f"{name}_entropy.npy"), entropy)
        
        print(f"Saved predictions for {name}")
        print(f"  Mean pred range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
        print(f"  Epistemic mean: {epistemic.mean():.6f}")


if __name__ == "__main__":
    main()
