from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class ExpDataset(Dataset):
    def __init__(self, weights: np.ndarray, coords: np.ndarray, targets: np.ndarray):
        self.weights = torch.from_numpy(weights)
        self.coords = torch.from_numpy(coords)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.weights[idx], self.coords[idx], self.targets[idx]


def build_dataloaders(
    weights: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    batch_size:  int = 32,
    num_workers: int = 0,
):
    dataset = ExpDataset(weights, coords, targets)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    kwargs = dict(batch_size=batch_size,
                  num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds,  shuffle=False, **kwargs)

    print(
        f"Dataset splits  →  train: {n_train}  |  val: {n_val}  |  test: {n_test}")
    return train_loader, val_loader, test_loader


def generate_synthetic_data(
    num_samples: int = 50000,
    num_neighbors: int = 30000,
    coord_range: tuple = (-100.0, 100.0),
    weight_range: tuple = (0.0, 1.0),
):
    weights = np.random.uniform(
        *weight_range, size=(num_samples, num_neighbors)).astype(np.float32)
    coords = np.random.uniform(
        *coord_range,  size=(num_samples, 3)).astype(np.float32)

    top5_mean = np.sort(weights, axis=1)[:, -5:].mean(axis=1)
    coord_norm = np.linalg.norm(coords, axis=1)
    targets = (top5_mean * coord_norm + 0.3 *
               np.sin(coord_norm)).astype(np.float32)

    return weights, coords, targets


def apply_topk(raw_weights, k):
    N = len(raw_weights)
    out = np.zeros((N, k), dtype=np.float32)
 
    for i, w in enumerate(raw_weights):
        if len(w) >= k:
            idx = np.argpartition(w, -k)[-k:]
            top = w[idx]
            out[i] = top[np.argsort(top)[::-1]]
        else:
            out[i, :len(w)] = np.sort(w)[::-1]
 
    print(f"[top-k]  Fixed shape: {out.shape}  (k={k})")
    return out