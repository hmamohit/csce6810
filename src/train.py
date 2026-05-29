import os

import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, random_split
from model import GeneExpression
from torch.optim.lr_scheduler import OneCycleLR
from data_loader import HiCExpressionDataset, collate_fn
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT_PATH = "/home/hc0783.unt.ad.unt.edu/workspace/csce6810/data"
DATA_FILE = f"{ROOT_PATH}/processed_tensors/hic_hg38_25000_norm_select.pt"
BEST_MODEL = f"{ROOT_PATH}/hg38_gene_exp_select.pt.pt"
CHECKPOINT = f"{ROOT_PATH}/hg38_25000_gene_exp_select_checkpoint.pt"
LOG_DIR = f"{ROOT_PATH}/logs/test_tt"

NUMS_WORKERS = 40
D_MODEL = 256
HIDDEN_DIM = 1024
NUM_HEADS = 4
NUM_ENCODERS = 4
DROPOUT = 0.1
BIAS = True
BATCH_SIZE = 100
WARMUP_STEPS = 20
NUM_EPOCHS = 500
PATIENT = 20
LEARNING_RATE = 1e-4

LOG_HISTORY_ALL = []
LOG_HISTORY_INFO = []
LOG_HISTORY_DEBUG = []
LOG_HISTORY_WARNING = []
LOG_HISTORY_ERROR = []


class LOG_LEVELS:
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARNING = "WARNING"
    ERROR = "ERROR"


def logger(writer, message, level=LOG_LEVELS.INFO, global_step=0):
    if dist.get_rank() == 0:
        timestamp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
        message = f"[{timestamp}] [{level}] {message}"
        print(f"{message}")

        LOG_HISTORY_ALL.append(message)
        all_history = "\n".join(LOG_HISTORY_ALL)
        writer.add_text("Logs/All", f"<pre>{all_history}</pre>", global_step=0)

        if LOG_LEVELS.INFO == level:
            LOG_HISTORY_INFO.append(message)
            info_history = "\n".join(LOG_HISTORY_INFO)
            writer.add_text(
                "Logs/Info", f"<pre>{info_history}</pre>", global_step=0)
        elif LOG_LEVELS.DEBUG == level:
            LOG_HISTORY_DEBUG.append(message)
            debug_history = "\n".join(LOG_HISTORY_DEBUG)
            writer.add_text(
                "Logs/Debug", f"<pre>{debug_history}</pre>", global_step=0)
        elif LOG_LEVELS.WARNING == level:
            LOG_HISTORY_WARNING.append(message)
            warning_history = "\n".join(LOG_HISTORY_WARNING)
            writer.add_text(
                "Logs/Warning", f"<pre>{warning_history}</pre>", global_step=0)
        elif LOG_LEVELS.ERROR == level:
            LOG_HISTORY_ERROR.append(message)
            error_history = "\n".join(LOG_HISTORY_ERROR)
            writer.add_text(
                "Logs/Error", f"<pre>{error_history}</pre>", global_step=0)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def prepare_padding_mask(binary_mask):
    extended_mask = binary_mask.unsqueeze(1).unsqueeze(2)
    extended_mask = extended_mask.to(dtype=torch.float32)
    padding_mask = (1.0 - extended_mask) * -10000.0

    return padding_mask


def eval_loop(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():

        for batch in val_loader:
            ftr = batch["ftr"].to(device)
            exp = batch["gene_exp"].to(device)
            outputs = model(ftr=ftr)
            loss = criterion(outputs, exp)
            total_val_loss += loss.detach()

    dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
    avg_val_loss = (
        total_val_loss.item()
        / (len(val_loader) * dist.get_world_size())
    )

    return avg_val_loss


def train_loop(model, train_sampler, train_loader, val_loader, optimizer, scheduler, criterion, device, writer, start_epoch=0):
    if dist.get_rank() == 0:
        logger(writer, "Starting training...", level=LOG_LEVELS.INFO)

    min_val_loss = float('inf')
    e_count = 0
    for epoch in tqdm.tqdm(range(start_epoch, NUM_EPOCHS)):

        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = torch.tensor(0.0, device=device)

        for batch in tqdm.tqdm(train_loader):
            ftr = batch["ftr"].to(device)
            exp = batch["gene_exp"].to(device)
            optimizer.zero_grad()

            outputs = model(ftr=ftr)
            loss = criterion(outputs, exp)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.detach()

        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
        total_batches = len(train_loader) * dist.get_world_size()
        avg_train_loss = total_train_loss.item() / total_batches

        avg_val_loss = eval_loop(model, val_loader, criterion, device)

        if dist.get_rank() == 0:
            logger(
                writer,
                f"Epoch {epoch+1}/{NUM_EPOCHS}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}",
                level=LOG_LEVELS.INFO
            )

            writer.add_scalars(
                "hg38/loss",
                {"train": avg_train_loss, "validation": avg_val_loss},
                epoch + 1
            )

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            e_count = 0

            if dist.get_rank() == 0:
                torch.save(
                    model.module.state_dict(),
                    BEST_MODEL
                )

                logger(
                    writer,
                    f"Saved best model at epoch {epoch+1}",
                    level=LOG_LEVELS.INFO
                )
        else:
            e_count += 1

        if (epoch + 1) % 10 == 0 and dist.get_rank() == 0:

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "min_val_loss": min_val_loss,
            }, CHECKPOINT)

            logger(
                writer,
                f"Checkpoint saved at epoch {epoch+1}",
                level=LOG_LEVELS.INFO
            )

        stop_flag = torch.tensor(int(e_count > PATIENT), device=device)
        dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if dist.get_rank() == 0:
                logger(writer, "Early stopping triggered",
                       level=LOG_LEVELS.INFO)
            break

    if dist.get_rank() == 0:
        logger(writer, "Training completed.", level=LOG_LEVELS.INFO)


def test_loop(model, test_loader, criterion, device, writer):
    if dist.get_rank() == 0:
        logger(writer, "Evaluating on test set...", level=LOG_LEVELS.INFO)

    checkpoint = torch.load(BEST_MODEL, map_location=device)
    model.module.load_state_dict(checkpoint)
    model.eval()
    total_test_loss = torch.tensor(0.0, device=device)

    preds = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            ftr = batch["ftr"].to(device)
            exp = batch["gene_exp"].to(device)
            outputs = model(ftr=ftr)
            loss = criterion(outputs, exp)
            total_test_loss += loss.detach()

            preds.append(outputs.detach())
            targets.append(exp.detach())

    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    total_batches = len(test_loader) * dist.get_world_size()
    avg_test_loss = total_test_loss.item() / total_batches

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    gathered_preds = [torch.zeros_like(preds)
                      for _ in range(dist.get_world_size())]
    gathered_targets = [torch.zeros_like(targets)
                        for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_preds, preds)
    dist.all_gather(gathered_targets, targets)

    if dist.get_rank() == 0:
        preds = torch.cat(gathered_preds, dim=0).cpu()
        targets = torch.cat(gathered_targets, dim=0).cpu()
        preds_np = preds.numpy()
        targets_np = targets.numpy()

        mae = mean_absolute_error(targets_np, preds_np)
        mse = mean_squared_error(targets_np, preds_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_np, preds_np)
        person, p_value = pearsonr(targets_np.flatten(), preds_np.flatten())

        logger(
            writer,
            f"hg38/loss/test: {avg_test_loss:.4f}",
            level=LOG_LEVELS.INFO
        )

        logger(
            writer,
            f"hg38/MAE: {mae:.6f}, "
            f"hg38/MSE: {mse:.6f}, "
            f"hg38/RMSE: {rmse:.6f}, "
            f"hg38/R2: {r2:.6f}, "
            f"hg38/Pearson: {person:.6f}, "
            f"hg38/P-Value: {p_value:.6f}",
            level=LOG_LEVELS.INFO
        )

        for i in range(len(preds)):
            writer.add_scalars(
                "hg38/prediction",
                {"pred": preds[i], "true": targets[i]},
                i + 1
            )

        logger(
            writer,
            "Test evaluation completed.",
            level=LOG_LEVELS.INFO
        )

    return avg_test_loss


def load_data(writer):
    if dist.get_rank() == 0:
        logger(writer, f"Loading data from {DATA_FILE}", level=LOG_LEVELS.INFO)

    data = torch.load(DATA_FILE)
    dataset = HiCExpressionDataset(data)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    remain_size = dataset_size - train_size - val_size - test_size

    train_dataset, val_dataset, test_dataset, _ = random_split(
        dataset,
        [train_size, val_size, test_size, remain_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=10,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=10,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=10,
        pin_memory=True
    )

    if dist.get_rank() == 0:
        logger(
            writer, f"Dataset split into {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.", level=LOG_LEVELS.INFO)

    return train_sampler, train_loader, val_loader, test_loader


def main():

    writer = SummaryWriter(LOG_DIR)
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if dist.get_rank() == 0:
        logger(writer, f"Using device: {device}", level=LOG_LEVELS.INFO)

    set_seed(42)
    if dist.get_rank() == 0:
        logger(writer, f"Number of Encoders: {NUM_ENCODERS}; Number of Heads: {NUM_HEADS}; Dropout: {DROPOUT}; Bias: {BIAS}; Batch Size: {BATCH_SIZE}; Learning Rate: {LEARNING_RATE}; Warmup Steps: {WARMUP_STEPS}; Patience: {PATIENT}", level=LOG_LEVELS.DEBUG)

    train_sampler, train_loader, val_loader, test_loader = load_data(
        writer)

    model = GeneExpression(num_encoders=NUM_ENCODERS, d_model=D_MODEL,
                           d_ff=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=DROPOUT, bias=BIAS).to(device)
    model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    if dist.get_rank() == 0:
        logger(
            writer, f"Model initialized with {num_params/1e6:.2f} Million parameters.", level=LOG_LEVELS.INFO)

    total_steps = len(train_loader) * NUM_EPOCHS
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    criterion = nn.MSELoss()

    is_load_checkpoint = False
    start_epoch = 0
    if is_load_checkpoint and os.path.exists(f"{CHECKPOINT}"):
        if dist.get_rank() == 0:
            logger(
                writer, f"Loading checkpoint from {CHECKPOINT}", level=LOG_LEVELS.INFO)
        checkpoint_data = torch.load(f"{CHECKPOINT}", map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        start_epoch = checkpoint_data['epoch'] + 1
        if dist.get_rank() == 0:
            logger(
                writer, f"Resuming training from epoch {start_epoch}", level=LOG_LEVELS.INFO)

    # train_loop(model, train_sampler, train_loader, val_loader,
    #            optimizer, scheduler, criterion, device, writer, start_epoch)
    test_loop(model, test_loader, criterion, device, writer)


if __name__ == "__main__":
    main()
