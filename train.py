from utils import CombinedLoss, seed_everything, compute_metrics
from data_loader import build_dataloaders, generate_synthetic_data
from model import GraphTransformer
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore")


seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_pred_cont, all_pred_cls, all_true_cont, all_true_cls = [], [], [], []

    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            pred_cont, pred_cls = model(batch)

            loss, _, _ = criterion(pred_cont, pred_cls, batch.y, batch.y_class)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            total_loss += loss.item() * n_graphs
            n_batches += n_graphs

            all_pred_cont.append(pred_cont.detach().cpu().numpy())
            all_pred_cls.append(pred_cls.detach().cpu().numpy())
            all_true_cont.append(batch.y.cpu().numpy())
            all_true_cls.append(batch.y_class.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    metrics = compute_metrics(
        np.concatenate(all_pred_cont),
        np.concatenate(all_pred_cls),
        np.concatenate(all_true_cont),
        np.concatenate(all_true_cls),
    )
    return avg_loss, metrics


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device=DEVICE,
):
    model = model.to(device)
    criterion = CombinedLoss(alpha=0.6)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_f1 = -np.inf
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_pcc": []}

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | "
          f"{'Val F1 (macro)':>14} | {'Val PCC':>8} | {'Val RMSE':>9}")
    print("─" * 70)

    for epoch in range(1, n_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader,
                               criterion, optimizer, device)
        val_loss, val_m = run_epoch(
            model, val_loader,   criterion, None, device)
        scheduler.step()

        f1 = val_m["F1_macro"]
        pcc = val_m["PCC"]
        rmse = val_m["RMSE"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)
        history["val_pcc"].append(pcc)

        print(f"{epoch:>6} | {tr_loss:>11.4f} | {val_loss:>9.4f} | "
              f"{f1:>14.4f} | {pcc:>8.4f} | {rmse:>9.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    print("\n" + "═" * 50)
    print("  FINAL TEST SET EVALUATION")
    print("═" * 50)
    _, test_m = run_epoch(model, test_loader, criterion, None, device)
    for k, v in test_m.items():
        print(f"  {k:<15}: {v:.4f}")
    print("═" * 50)

    return model, history


if __name__ == "__main__":
    N_SAMPLES = 200
    N_GENES = 100
    N_EXPR_BINS = 3      # low / medium / high
    BATCH_SIZE = 16
    HIDDEN_CHANNELS = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.2
    N_EPOCHS = 100
    LR = 1e-3
    PATIENCE = 12

    hic, expr, labels = generate_synthetic_data(
        n_samples=N_SAMPLES,
        n_genes=N_GENES,
        n_expression_bins=N_EXPR_BINS,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        hic, expr, labels, batch_size=BATCH_SIZE
    )

    in_channels = N_GENES + 1
    model = GraphTransformer(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=N_EXPR_BINS,
        edge_dim=1,
    )

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    model, history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=N_EPOCHS,
        lr=LR,
        patience=PATIENCE,
    )

    print("\nDone! Best val F1 (macro):", max(history["val_f1"]))
