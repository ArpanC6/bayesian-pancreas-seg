"""
Training script for Bayesian 3D Pancreas Segmentation
MSD Pancreas (Task 07) | Dice + BCE | AdamW | CosineAnnealingWarmRestarts
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import BayesianUNet3D
from dataset import MSDPancreasDataset
from utils import dice_score, hd95


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_prob * target).sum(dim=(2,3,4))
        union = pred_prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
        dice = 1 - (2. * intersection + smooth) / (union + smooth)
        return bce + 0.5 * dice.mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, enable_dropout=True)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        outputs = model(images, enable_dropout=False)
        loss = criterion(outputs, masks)
        val_loss += loss.item()
        
        preds = torch.sigmoid(outputs) > 0.5
        val_dice += dice_score(preds, masks)
    
    return val_loss / len(loader), val_dice / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/Task07_Pancreas")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_ds = MSDPancreasDataset(root=args.data_root, split="train")
    val_ds = MSDPancreasDataset(root=args.data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # Model
    model = BayesianUNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  -> Saved best model (Dice: {best_dice:.4f})")


if __name__ == "__main__":
    main()
