import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm


SEED = 42
TQDM_DISABLE = not sys.stdout.isatty()
LOG_FILE_PATH: Optional[Path] = None

CLASS_NAMES = ["Low", "Medium", "High"]

CONFIG = {
    "data_k": 512,
    "data_subdir": "data/preprocessing",
    "log_subdir": "logs",
    "float_precision": 6,
    "sci_precision": 6,
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 600,
    "patience": 30,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "label_smoothing": 0.02,
    "class_weight_power": 0.35,
    "use_balanced_sampler": True,
    "sampler_power": 0.5,
    "balance_loaded_data": True,
    "balance_mode": "undersample",
    "label_mode": "fixed_thresholds",
    "low_max": 0.01,
    "med_max": 1.0,
}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def console_log(message: str = ""):
    if TQDM_DISABLE:
        print(message)
    else:
        tqdm.write(message)

    if LOG_FILE_PATH is not None:
        with LOG_FILE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def init_log_file(base_dir: Path, prefix: str) -> Path:
    global LOG_FILE_PATH
    log_dir = base_dir / CONFIG["log_subdir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{prefix}_{ts}.log"
    log_path.write_text("", encoding="utf-8")
    LOG_FILE_PATH = log_path
    return log_path


def fmt_float(value: float) -> str:
    return f"{float(value):.{CONFIG['float_precision']}f}"


def fmt_sci(value: float) -> str:
    return f"{float(value):.{CONFIG['sci_precision']}e}"


def console_rule(char: str = "-", width: int = 86):
    console_log(char * width)


def console_section(title: str, width: int = 86):
    console_rule("=", width)
    console_log(title)
    console_rule("=", width)


def console_kv(key: str, value):
    console_log(f"{key:<24}: {value}")


def load_preprocessed_data(data_dir: Path, k: int):
    weights_path = data_dir / "contact_matrix.txt"
    coords_path = data_dir / "coordinates.txt"
    targets_path = data_dir / "gene_exp.txt"

    missing = [
        str(path)
        for path in (weights_path, coords_path, targets_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessed input files:\n" + "\n".join(missing))

    coords = np.loadtxt(coords_path, dtype=np.float32)
    targets = np.loadtxt(targets_path, dtype=np.float32)

    if coords.ndim == 1:
        if coords.shape[0] == 3:
            coords = np.expand_dims(coords, axis=0)
        else:
            raise ValueError(
                f"Expected coordinates with shape (N, 3), got {coords.shape}.")

    if targets.ndim == 0:
        targets = np.array([float(targets)], dtype=np.float32)
    elif targets.ndim == 2:
        if targets.shape[1] == 1:
            targets = targets[:, 0]
        else:
            raise ValueError(
                f"Expected targets with shape (N,), got {targets.shape}.")

    n_coords = len(coords)
    n_targets = len(targets)
    n_meta = min(n_coords, n_targets)
    if n_meta <= 0:
        raise ValueError(
            f"Empty coords/targets after loading: coords={n_coords}, targets={n_targets}")

    if n_coords != n_targets:
        console_log(
            "[ALIGN] coords/targets row mismatch detected. "
            f"coords={n_coords}, targets={n_targets}. Using first {n_meta} rows."
        )

    weights_rows = []
    iterator = tqdm(
        range(n_meta),
        desc=f"Top-{k} from contact_matrix rows",
        dynamic_ncols=True,
        disable=TQDM_DISABLE,
    )

    with weights_path.open("r", encoding="utf-8") as handle:
        for _ in iterator:
            line = handle.readline()
            if line == "":
                break

            vals = np.fromstring(line, sep=" ", dtype=np.float32)
            row_out = np.zeros(k, dtype=np.float32)

            if vals.size > 0:
                if vals.size >= k:
                    idx = np.argpartition(vals, -k)[-k:]
                    top = vals[idx]
                    row_out[:] = top[np.argsort(top)[::-1]]
                else:
                    order = np.argsort(vals)[::-1]
                    row_out[:vals.size] = vals[order]

            weights_rows.append(row_out)

    n_weights = len(weights_rows)
    n_shared = min(n_meta, n_weights)
    if n_shared <= 0:
        raise ValueError(
            f"No usable rows loaded from contact matrix: weights_rows={n_weights}, n_meta={n_meta}"
        )

    if n_weights != n_meta:
        console_log(
            "[ALIGN] weights row count differs from coords/targets. "
            f"weights={n_weights}, coords/targets_min={n_meta}. Using first {n_shared} rows."
        )

    weights = np.vstack(weights_rows[:n_shared]).astype(np.float32)
    coords = coords[:n_shared]
    targets = targets[:n_shared]

    paths = {
        "weights": weights_path,
        "coords": coords_path,
        "targets": targets_path,
    }
    return weights, coords, targets, paths


def make_labels(
    targets: np.ndarray,
    mode: str = "fixed_thresholds",
    low_max: float = 0.01,
    med_max: float = 1.0,
):
    n = len(targets)
    labels = np.zeros(n, dtype=np.int64)

    if mode == "fixed_thresholds":
        if low_max >= med_max:
            raise ValueError(
                f"Expected low_max < med_max. Got low_max={low_max}, med_max={med_max}")

        boundaries = (float(low_max), float(med_max))
        labels = np.digitize(
            targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    elif mode == "balanced_rank":
        order = np.argsort(targets, kind="mergesort")
        cut1 = n // 3
        cut2 = (2 * n) // 3
        labels[order[:cut1]] = 0
        labels[order[cut1:cut2]] = 1
        labels[order[cut2:]] = 2

        t1 = float(targets[order[cut1 - 1]]
                   ) if cut1 > 0 else float(targets.min())
        t2 = float(targets[order[cut2 - 1]]
                   ) if cut2 > 0 else float(targets.max())
        boundaries = (t1, t2)
    elif mode == "quantile":
        q1, q2 = np.quantile(targets, [1.0 / 3.0, 2.0 / 3.0])
        if q2 <= q1:
            q2 = q1 + 1e-12
        boundaries = (float(q1), float(q2))
        labels = np.digitize(
            targets, bins=[boundaries[0], boundaries[1]], right=True).astype(np.int64)
    else:
        raise ValueError(f"Unknown label mode: {mode}")

    return labels, boundaries


def balance_loaded_data(
    weights: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    mode: str = "undersample",
):
    counts = np.bincount(labels, minlength=3)
    if counts.min() <= 0:
        raise ValueError(
            f"Cannot balance because at least one class is empty: {counts.tolist()}")

    if mode != "undersample":
        raise ValueError(f"Unsupported balance mode: {mode}")

    rng = np.random.default_rng(SEED)
    per_class = int(counts.min())
    kept = []
    for cls in range(3):
        cls_idx = np.where(labels == cls)[0]
        pick = rng.choice(cls_idx, size=per_class, replace=False)
        kept.append(pick)

    kept_idx = np.concatenate(kept)
    rng.shuffle(kept_idx)

    return (
        weights[kept_idx],
        coords[kept_idx],
        targets[kept_idx],
        labels[kept_idx],
        original_indices[kept_idx],
        counts,
        np.bincount(labels[kept_idx], minlength=3),
    )


class ExpClassDataset(Dataset):
    def __init__(self, weights: np.ndarray, coords: np.ndarray, labels: np.ndarray):
        self.weights = torch.from_numpy(weights)
        self.coords = torch.from_numpy(coords)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.weights[idx], self.coords[idx], self.labels[idx]


class ExpTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_neighbors: int,
        hidden_dim:    int = 64,
        num_heads:     int = 4,
        num_layers:    int = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim

        self.weight_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.pos_embedding = nn.Embedding(num_neighbors, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B = weights.size(0)
        tokens = self.weight_proj(weights.unsqueeze(-1))
        positions = torch.arange(self.num_neighbors, device=weights.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)
        coord_ctx = self.coord_proj(coords)
        tokens = tokens + coord_ctx.unsqueeze(1)
        out = self.transformer(tokens)
        agg = out.mean(dim=1)
        combined = torch.cat([agg, coord_ctx], dim=-1)
        return self.head(combined).squeeze(-1)
