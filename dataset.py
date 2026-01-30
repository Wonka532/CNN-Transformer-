import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
class FiberModeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_df = pd.read_csv(csv_path, header=0, sep=None, engine='python')
        self.transform = transform
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        coeffs_str = row.iloc[0].strip().split()
        coeffs_raw = np.array([float(x) for x in coeffs_str], dtype=np.float32)
        target_tensor = torch.from_numpy(coeffs_raw).float()
        img_path = row.iloc[1]
        far_field_path = row.iloc[2]
        try:
            img = Image.open(img_path).convert('L')
            img_tensor = self._preprocess_image(img)
            ff_img = Image.open(far_field_path).convert('L')
            ff_tensor = self._preprocess_image(ff_img)
        except Exception as e:
            img_tensor = torch.zeros(4, 128, 128)
            ff_tensor = torch.zeros(1, 128, 128)
        return {
            'image': img_tensor,
            'gt_farfield': ff_tensor,
            'coeffs': target_tensor
        }
    def _preprocess_image(self, img):
        img = img.resize((256, 256))
        img_np = np.array(img) / 255.0
        h, w = img_np.shape
        cx, cy = h // 2, w // 2
        patches = [
            img_np[:cx, :cy],  # 0
            img_np[:cx, cy:],  # 45
            img_np[cx:, :cy],  # 90
            img_np[cx:, cy:]  # -45
        ]
        tensor = torch.from_numpy(np.array(patches)).float()
        return tensor