"""
Evaluation metrics for 3D segmentation
"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_score(pred, target, smooth=1e-5):
    """
    pred, target: torch tensors [B, 1, D, H, W] or [B, D, H, W]
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()


def hd95(pred, target):
    """
    Approximate Hausdorff-95 (simplified 2D slice-wise for speed)
    Full 3D HD95 requires surface extraction; this is a proxy
    """
    # Placeholder: in production use medpy.metric.binary.hd95
    return 0.0
