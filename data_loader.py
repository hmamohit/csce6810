import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")


class HiCDataset(torch.utils.data.Dataset):
    def __init__(self, hic_matrices, expr_vectors, expr_labels, threshold=0.1):
        self.graphs = [
            hic_to_graph(hic_matrices[i], expr_vectors[i],
                         expr_labels[i], threshold)
            for i in range(len(hic_matrices))
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def build_dataloaders(hic, expr, labels, batch_size=16, val_ratio=0.15, test_ratio=0.15):
    indices = list(range(len(hic)))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_ratio, random_state=42)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_ratio / (1 - test_ratio), random_state=42)

    def subset(idx_list):
        return HiCDataset(hic[idx_list], expr[idx_list], labels[idx_list])

    train_loader = DataLoader(
        subset(idx_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(subset(idx_val),   batch_size=batch_size)
    test_loader = DataLoader(subset(idx_test),  batch_size=batch_size)

    print(
        f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)} samples"
    )
    return train_loader, val_loader, test_loader


def hic_to_graph(
    hic: np.ndarray,
    expr: np.ndarray,
    labels: np.ndarray,
    contact_threshold: float = 0.1
):

    n = hic.shape[0]
    degree = hic.sum(axis=1, keepdims=True)
    node_feat = np.concatenate([hic, np.log1p(degree)], axis=1)  # (n, n+1)
    x = torch.tensor(node_feat, dtype=torch.float)

    rows, cols = np.where((hic > contact_threshold) &
                          (np.triu(np.ones_like(hic), 1) > 0))
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    weights = hic[rows, cols]
    edge_attr = torch.tensor(
        np.concatenate([weights, weights]).reshape(-1, 1), dtype=torch.float
    )

    y_cont = torch.tensor(expr,   dtype=torch.float)
    y_class = torch.tensor(labels, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_cont,
        y_class=y_class,
        num_nodes=n,
    )

def generate_synthetic_data(
    n_samples: int = 200,
    n_genes: int = 100,
    contact_sparsity: float = 0.85,
    n_expression_bins: int = 3,
    seed: int = 42
):
    np.random.seed(seed)
    hic_matrices = []
    expr_vectors = []

    for _ in range(n_samples):
        # Symmetric sparse contact matrix (TAD-like block structure)
        mat = np.zeros((n_genes, n_genes))
        # block diagonal signal (simulate TADs)
        block_size = n_genes // 5
        for b in range(0, n_genes, block_size):
            end = min(b + block_size, n_genes)
            block = np.random.exponential(1.0, (end - b, end - b))
            block = (block + block.T) / 2
            mat[b:end, b:end] = block
        # sparse background noise
        noise = np.random.rand(n_genes, n_genes)
        noise[noise < contact_sparsity] = 0
        mat += noise
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
        # log-normalise (ICE-like)
        mat = np.log1p(mat)
        hic_matrices.append(mat)

        # Gene expression = linear combination of local contact strength + noise
        local_contact = mat.sum(axis=1)
        expr = 0.6 * local_contact + 0.4 * np.random.randn(n_genes)
        expr_vectors.append(expr)

    hic_matrices = np.array(hic_matrices, dtype=np.float32)
    expr_vectors = np.array(expr_vectors, dtype=np.float32)

    # Discretise expression → low / medium / high per gene across samples
    flat_expr = expr_vectors.reshape(-1, 1)
    kbd = KBinsDiscretizer(
        n_bins=n_expression_bins, encode="ordinal", strategy="quantile"
    )
    expr_labels = kbd.fit_transform(flat_expr).reshape(n_samples, n_genes).astype(int)

    print(
        f"Generated {n_samples} samples | "
        f"{n_genes} genes | "
        f"HiC shape: {hic_matrices.shape} | "
        f"Expr shape: {expr_vectors.shape}"
    )
    return hic_matrices, expr_vectors, expr_labels