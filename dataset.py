# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import config as cfg


class RADataset(Dataset):
    """
    用于训练 CHAN 模型的 PyTorch Dataset
    """

    def __init__(self, hs_data, hu_data, ps_data, pu_data, labels):
        # 我们期望 (N, D) 形状的 numpy 数组
        self.hs = torch.tensor(hs_data, dtype=torch.float32)
        self.hu = torch.tensor(hu_data, dtype=torch.float32)
        self.ps = torch.tensor(ps_data, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.pu = torch.tensor(pu_data, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.hs[idx],
            self.hu[idx],
            self.ps[idx],
            self.pu[idx],
            self.labels[idx]
        )