import os
import numpy as np
import random
import torch
import torch.distributed as dist
import random
import numpy as np
import sys
import os
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import config
from data_loader import CustomDataset, GeneExpressionDataset


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def base_logger(file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return logger


def set_seed(seed_v: int = 42):
    torch.manual_seed(seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_v)
    np.random.seed(seed_v)
    random.seed(seed_v)


def ddp_setup():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training requires CUDA/NCCL, but CUDA is not available.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Distributed training requires torchrun. Example: "
            "torchrun --standalone --nproc_per_node=<num_gpus> hicinterpolate.py --distributed --train --config <config>"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


def get_dataloader(ds: Dataset, batch_size: int = 20, shuffle: bool = False, isDistributed: bool = False) -> DataLoader:
    if isDistributed:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=set_seed,
            num_workers=20,
            persistent_workers=True,
            sampler=DistributedSampler(ds, shuffle=shuffle)
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=set_seed,
            num_workers=20,
            persistent_workers=True
        )


def main():
    output_dir = config.OUTPUT_DIR
    os.makedirs(f"{output_dir}", exist_ok=True)
    log = base_logger(config.LOG_FILENAME)
    batch_size = config.BATCH_SIZE

    train_cds = CustomDataset(record_file=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_train.csv', img_dir=config.FEATURE_DIR,
                              img_map=config.FEATURE_MAP)
    train_dict = train_cds._get_dataset()
    train_ds = GeneExpressionDataset(triplet_dicts=train_dict)
    train_dl = get_dataloader(
        ds=train_ds, batch_size=batch_size, shuffle=True, isDistributed=config.IS_DISTRIBUTED)

    val_cds = CustomDataset(record_file=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_val.csv', img_dir=config.FEATURE_DIR,
                            img_map=config.FEATURE_MAP)
    val_dict = val_cds._get_dataset()
    val_ds = GeneExpressionDataset(triplet_dicts=val_dict)
    val_dl = get_dataloader(ds=val_ds, batch_size=batch_size,
                            shuffle=False, isDistributed=config.IS_DISTRIBUTED)

    test_cds = CustomDataset(record_file=f'{config.DICT_DIR}/{config.ORGANISM}_{config.GENE_EXPRESSION_FEATURES_DICT}_{config.RESOLUTION}_test.csv', img_dir=config.FEATURE_DIR,
                             img_map=config.FEATURE_MAP)
    test_dict = test_cds._get_dataset()
    test_ds = GeneExpressionDataset(triplet_dicts=test_dict)
    test_dl = get_dataloader(
        ds=test_ds, batch_size=batch_size, shuffle=False, isDistributed=config.IS_DISTRIBUTED)

    if config.IS_DISTRIBUTED:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
