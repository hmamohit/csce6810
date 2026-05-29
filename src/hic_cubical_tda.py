"""
Local cubical persistent homology on Hi-C contact patches.

Uses GUDHI cubical complexes (sublevel filtration) and silhouette / Betti
vectorizations. 
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gudhi
except ImportError as exc:  # pragma: no cover
    gudhi = None
    _GUDHI_IMPORT_ERROR = exc
else:
    _GUDHI_IMPORT_ERROR = None


@dataclass(frozen=True)
class TDAConfig:
    patch_size: int = 64
    n_samples: int = 25
    silhouette_p: float = 2.0
    homology_dims: Tuple[int, ...] = (0, 1)
    use_silhouette: bool = True
    use_betti: bool = False
    log1p: bool = True
    normalize_patch: bool = True

    @property
    def half_window(self) -> int:
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer >= 3.")
        return self.patch_size // 2

    @property
    def feature_dim(self) -> int:
        n_dims = len(self.homology_dims)
        per_dim = 0
        if self.use_silhouette:
            per_dim += self.n_samples
        if self.use_betti:
            per_dim += self.n_samples
        return n_dims * per_dim


def require_gudhi() -> None:
    if gudhi is None:
        raise ImportError(
            "gudhi is required for cubical persistent homology. "
            "Install with: pip install gudhi"
        ) from _GUDHI_IMPORT_ERROR


def load_contact_matrix_blocks(contact_path: Path) -> List[np.ndarray]:
    """
    Parse stacked contact_matrix.txt into per-chromosome square matrices.

    Rows from the same chromosome share the same width. A width change starts a
    new block.
    """
    blocks: List[np.ndarray] = []
    current_rows: List[np.ndarray] = []
    current_width: Optional[int] = None

    with contact_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = np.fromstring(line, sep=" ", dtype=np.float32)
            width = int(row.shape[0])

            if current_width is None:
                current_width = width
                current_rows.append(row)
                continue

            if width == current_width:
                current_rows.append(row)
                continue

            blocks.append(np.vstack(current_rows))
            current_rows = [row]
            current_width = width

    if current_rows:
        blocks.append(np.vstack(current_rows))

    if not blocks:
        raise ValueError(f"No contact rows found in {contact_path}")

    for idx, matrix in enumerate(blocks):
        n_rows, n_cols = matrix.shape
        if n_rows != n_cols:
            raise ValueError(
                f"Chromosome block {idx} is not square: shape={matrix.shape}"
            )

    return blocks


def blocks_to_global_indices(blocks: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (block_id, local_index) for each stacked row."""
    n_total = sum(block.shape[0] for block in blocks)
    block_ids = np.zeros(n_total, dtype=np.int64)
    local_indices = np.zeros(n_total, dtype=np.int64)

    offset = 0
    for block_id, matrix in enumerate(blocks):
        n = matrix.shape[0]
        block_ids[offset: offset + n] = block_id
        local_indices[offset: offset + n] = np.arange(n, dtype=np.int64)
        offset += n

    return block_ids, local_indices


def preprocess_patch(patch: np.ndarray, log1p: bool, normalize_patch: bool) -> np.ndarray:
    out = patch.astype(np.float64, copy=True)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0.0] = 0.0

    if log1p:
        out = np.log1p(out)

    if normalize_patch:
        peak = float(out.max())
        if peak > 0.0:
            out = out / peak

    return out


def extract_local_patch(
    matrix: np.ndarray,
    center: int,
    half_window: int,
) -> np.ndarray:
    n = matrix.shape[0]
    size = 2 * half_window + 1
    patch = np.zeros((size, size), dtype=np.float32)

    for di in range(-half_window, half_window + 1):
        ii = center + di
        if ii < 0 or ii >= n:
            continue
        row = matrix[ii]
        j0 = max(0, center - half_window)
        j1 = min(n, center + half_window + 1)
        p0 = j0 - (center - half_window)
        p1 = p0 + (j1 - j0)
        patch[di + half_window, p0:p1] = row[j0:j1]

    return patch


def cubical_persistence_pairs(patch: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
    require_gudhi()
    h, w = patch.shape
    if h < 2 or w < 2:
        return []

    # GUDHI bitmap convention: dimensions match pixel counts per axis (h, w).
    cubical = gudhi.CubicalComplex(
        dimensions=[h, w],
        top_dimensional_cells=patch.astype(np.float64).flatten(order="C"),
    )
    cubical.persistence()
    return cubical.persistence()


def _finite_death(death: float, fallback: float) -> float:
    if death is None or not np.isfinite(death):
        return fallback
    return float(death)


def betti_curve(
    pairs: Sequence[Tuple[int, Tuple[float, float]]],
    dim: int,
    samples: np.ndarray,
) -> np.ndarray:
    values = np.zeros(len(samples), dtype=np.float64)
    t_max = float(samples[-1]) if len(samples) else 1.0

    for k, (birth, death) in pairs:
        if k != dim:
            continue
        b = float(birth)
        d = _finite_death(death, t_max)
        if d <= b:
            continue
        for i, t in enumerate(samples):
            if b <= t < d:
                values[i] += 1.0

    return values.astype(np.float32)


def silhouette_curve(
    pairs: Sequence[Tuple[int, Tuple[float, float]]],
    dim: int,
    samples: np.ndarray,
    p: float,
) -> np.ndarray:
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    t_max = float(samples[-1])
    curve = np.zeros(len(samples), dtype=np.float64)
    for i, t in enumerate(samples):
        num = 0.0
        den = 0.0
        for k, (birth, death) in pairs:
            if k != dim:
                continue
            b = float(birth)
            d = _finite_death(death, t_max)
            lifespan = d - b
            if lifespan <= 0.0 or t < b or t >= d:
                continue
            w = lifespan ** float(p)
            height = min(t - b, d - t)
            num += w * height
            den += w
        curve[i] = num / den if den > 0.0 else 0.0

    return curve.astype(np.float32)


def sample_thresholds(patch: np.ndarray, n_samples: int) -> np.ndarray:
    flat = patch.reshape(-1)
    positive = flat[flat > 0.0]
    if positive.size == 0:
        return np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    lo = float(positive.min())
    hi = float(positive.max())
    if hi <= lo:
        return np.full(n_samples, lo, dtype=np.float64)
    return np.linspace(lo, hi, n_samples, dtype=np.float64)


def vectorize_patch(
    patch: np.ndarray,
    config: TDAConfig,
) -> np.ndarray:
    proc = preprocess_patch(patch, log1p=config.log1p, normalize_patch=config.normalize_patch)
    pairs = cubical_persistence_pairs(proc)
    thresholds = sample_thresholds(proc, config.n_samples)

    parts: List[np.ndarray] = []
    for dim in config.homology_dims:
        if config.use_silhouette:
            parts.append(
                silhouette_curve(
                    pairs, dim=dim, samples=thresholds, p=config.silhouette_p
                )
            )
        if config.use_betti:
            parts.append(betti_curve(pairs, dim=dim, samples=thresholds))

    if not parts:
        raise ValueError("At least one of use_silhouette / use_betti must be True.")

    return np.concatenate(parts).astype(np.float32)


def compute_features_for_blocks(
    blocks: Sequence[np.ndarray],
    config: TDAConfig,
    show_progress: bool = True,
) -> np.ndarray:
    block_ids, local_indices = blocks_to_global_indices(blocks)
    n_total = len(block_ids)
    features = np.zeros((n_total, config.feature_dim), dtype=np.float32)

    iterator = range(n_total)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Cubical PH patches", dynamic_ncols=True)

    for global_idx in iterator:
        block = blocks[int(block_ids[global_idx])]
        center = int(local_indices[global_idx])
        patch = extract_local_patch(block, center, config.half_window)
        features[global_idx] = vectorize_patch(patch, config)

    return features


def save_tda_artifacts(
    features: np.ndarray,
    config: TDAConfig,
    output_dir: Path,
    contact_path: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "tda_features.npy"
    meta_path = output_dir / "tda_meta.json"

    np.save(features_path, features.astype(np.float32))

    cfg = asdict(config)
    cfg["homology_dims"] = list(cfg["homology_dims"])
    meta = {
        "contact_matrix": str(contact_path.resolve()),
        "num_rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "config": cfg,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"features": features_path, "meta": meta_path}


def load_tda_features(
    data_dir: Path,
    features_name: str = "tda_features.npy",
    meta_name: str = "tda_meta.json",
) -> Tuple[np.ndarray, dict]:
    features_path = data_dir / features_name
    meta_path = data_dir / meta_name

    if not features_path.exists():
        raise FileNotFoundError(
            f"TDA features not found: {features_path}\n"
            "Run: python compute_hic_tda_features.py"
        )

    features = np.load(features_path)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return features.astype(np.float32), meta
