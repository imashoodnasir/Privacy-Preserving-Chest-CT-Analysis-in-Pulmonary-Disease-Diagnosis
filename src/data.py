import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CTNPZDataset(Dataset):
    """Loads a volume from a .npz file with keys: volume (D,H,W), label (int).
    Training uses slice sampling (2.5D): we randomly pick a slice index and build a 3-channel
    'pseudo-RGB' slice using (i-1, i, i+1). This keeps memory low and is robust as a baseline.
    """
    def __init__(self, root_dir: str, split: str = "train", augment: bool = False, slice_stride: int = 1):
        self.root_dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.root_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in: {self.root_dir}")
        self.augment = augment
        self.slice_stride = max(1, int(slice_stride))

    def __len__(self):
        return len(self.files)

    def _load(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            vol = z["volume"].astype(np.float32)  # (D,H,W)
            label = int(z["label"])
        return vol, label

    def _sample_slice_triplet(self, vol: np.ndarray):
        D, H, W = vol.shape
        # Choose a center slice away from borders
        idx = np.random.randint(1, D-1)
        idx = (idx // self.slice_stride) * self.slice_stride
        idx = min(max(idx, 1), D-2)
        s0, s1, s2 = vol[idx-1], vol[idx], vol[idx+1]
        img = np.stack([s0, s1, s2], axis=0)  # (3,H,W)
        return img

    def _augment(self, img: np.ndarray):
        # img: (3,H,W)
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()  # horizontal flip
        if np.random.rand() < 0.2:
            img = img[:, ::-1, :].copy()  # vertical flip
        # small gaussian noise
        if np.random.rand() < 0.3:
            img = img + np.random.randn(*img.shape).astype(np.float32) * 0.02
        return img

    def __getitem__(self, idx: int):
        vol, label = self._load(self.files[idx])
        img = self._sample_slice_triplet(vol)
        if self.augment:
            img = self._augment(img)
        # standardize per-sample for stability
        m = img.mean()
        s = img.std() + 1e-6
        img = (img - m) / s
        x = torch.from_numpy(img)  # (3,H,W)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def discover_clients(data_root: str):
    """Find client folders under data_root: data_root/client_XX"""
    clients = []
    for p in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, p)
        if os.path.isdir(full) and p.lower().startswith("client_"):
            clients.append(full)
    if not clients:
        raise FileNotFoundError(f"No client folders found in {data_root}. Expected client_01, client_02, ...")
    return clients
