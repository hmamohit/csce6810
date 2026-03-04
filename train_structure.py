from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GraphTransformer
from structure_graph import StructureGraphConfig, build_structure_graph_chr19
from utils import CombinedLoss, compute_metrics, seed_everything


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_metrics(pred_cont, pred_cls, y_cont, y_cls, mask: torch.Tensor):
    pred_cont = pred_cont[mask].detach().cpu().numpy()
    pred_cls = pred_cls[mask].detach().cpu().numpy()
    y_cont = y_cont[mask].detach().cpu().numpy()
    y_cls = y_cls[mask].detach().cpu().numpy()
    return compute_metrics(pred_cont, pred_cls, y_cont, y_cls)


def run_epoch(model, data, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    data = data.to(device)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pred_cont, pred_cls = model(data)
        loss, loss_reg, loss_cls = criterion(pred_cont, pred_cls, data.y, data.y_class)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_m = _masked_metrics(pred_cont, pred_cls, data.y, data.y_class, data.train_mask)
    val_m = _masked_metrics(pred_cont, pred_cls, data.y, data.y_class, data.val_mask)
    test_m = _masked_metrics(pred_cont, pred_cls, data.y, data.y_class, data.test_mask)

    return float(loss.item()), float(loss_reg.item()), float(loss_cls.item()), train_m, val_m, test_m


def parse_args():
    p = argparse.ArgumentParser(description="Train expression predictor on chr19 structure-derived graph")
    p.add_argument(
        "--pdb",
        type=str,
        default=str(Path("Data/structures/chr19_1mb_structure.pdb")),
        help="Path to chr19 1Mb PDB structure",
    )
    p.add_argument(
        "--bin-expression",
        type=str,
        default=str(Path("Data/preprocessing/GM12878_bin_expression_1Mb.tsv")),
        help="Path to bin-level expression TSV (will filter to chr19)",
    )
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--expr-bins", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    print(f"Using device: {DEVICE}")

    cfg = StructureGraphConfig(
        neighbor_mode="knn",
        k=args.k,
        sigma=args.sigma,
        include_xyz_as_node_features=False,
        include_bin_index_as_node_features=True,
    )
    data = build_structure_graph_chr19(args.pdb, args.bin_expression, cfg)

    # Discretize y for auxiliary classification head (like existing pipeline)
    y_np = data.y.detach().cpu().numpy().reshape(-1, 1)
    # quantile binning without sklearn dependency here
    qs = np.quantile(y_np, np.linspace(0, 1, args.expr_bins + 1))
    # make last edge inclusive
    y_class = np.digitize(y_np.squeeze(-1), qs[1:-1], right=False)
    data.y_class = torch.tensor(y_class, dtype=torch.long)

    in_channels = data.x.size(-1)
    model = GraphTransformer(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        num_classes=args.expr_bins,
        edge_dim=1,
    ).to(DEVICE)

    criterion = CombinedLoss(alpha=0.8)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_rmse = float("inf")
    best_state = None

    print("\nEpoch | Loss | Val RMSE | Val PCC")
    print("-" * 32)
    for epoch in range(1, args.epochs + 1):
        loss, _, _, train_m, val_m, _ = run_epoch(model, data, criterion, optimizer)
        scheduler.step()

        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"{epoch:>5} | {loss:>4.3f} | {val_m['RMSE']:>8.3f} | {val_m['PCC']:>7.3f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    print("\nFinal evaluation (best val RMSE checkpoint):")
    _, _, _, train_m, val_m, test_m = run_epoch(model, data, criterion, optimizer=None)
    for name, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
        print(f"\n{name}:")
        for k in ["RMSE", "R2", "PCC", "F1_macro"]:
            print(f"  {k:<8}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
