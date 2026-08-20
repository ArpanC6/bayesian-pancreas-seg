"""
MSD Pancreas Dataset Loader
Resamples to 128x128x128, normalizes CT intensities
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class MSDPancreasDataset(Dataset):
    def __init__(self, root, split="train", target_shape=(128, 128, 128)):
        self.root = root
        self.split = split
        self.target_shape = target_shape
        
        # Expected: root/imagesTr, root/labelsTr
        img_dir = os.path.join(root, "imagesTr")
        lbl_dir = os.path.join(root, "labelsTr")
        
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        all_lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))
        
        # Simple 80/20 split
        n = len(all_imgs)
        if split == "train":
            self.img_paths = all_imgs[:int(0.8*n)]
            self.lbl_paths = all_lbls[:int(0.8*n)]
        else:
            self.img_paths = all_imgs[int(0.8*n):]
            self.lbl_paths = all_lbls[int(0.8*n):]
    
    def __len__(self):
        return len(self.img_paths)
    
    def _load_and_preprocess(self, img_path, lbl_path):
        # In practice, use nibabel or SimpleITK; here is a simplified numpy version
        # Replace with actual nii.gz loading in real usage
        import nibabel as nib
        
        img = nib.load(img_path).get_fdata()
        lbl = nib.load(lbl_path).get_fdata()
        
        # Binarize: pancreas label = 1 (MSD Task 07)
        lbl = (lbl > 0).astype(np.float32)
        
        # Clip CT intensities
        img = np.clip(img, -175, 250)
        img = (img + 175) / (250 + 175)  # normalize to [0, 1]
        
        # Resample to target shape
        factors = [t/s for t, s in zip(self.target_shape, img.shape)]
        img = zoom(img, factors, order=1)
        lbl = zoom(lbl, factors, order=0)
        
        # Add channel dim
        img = img[np.newaxis, ...].astype(np.float32)
        lbl = lbl[np.newaxis, ...].astype(np.float32)
        
        return img, lbl
    
    def __getitem__(self, idx):
        img, lbl = self._load_and_preprocess(self.img_paths[idx], self.lbl_paths[idx])
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(lbl),
            "name": os.path.basename(self.img_paths[idx])
        }
