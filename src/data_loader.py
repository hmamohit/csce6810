import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def unpack_dataset(data):
    """
    Support legacy .pt (list of dicts) and new format:
      {"samples": [...], "tda_enabled": bool, ...}
    """
    if isinstance(data, dict) and "samples" in data:
        return data["samples"], bool(data.get("tda_enabled", False))
    return data, bool(len(data) > 0 and isinstance(data[0], dict) and "tda" in data[0])


class HiCExpressionDataset(Dataset):

    def __init__(self, data):
        samples, self.has_tda = unpack_dataset(data)
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        out = {
            "ftr": item["ftr"],
            "gene_exp": item["gene_exp"],
        }
        if self.has_tda and "tda" in item:
            out["tda"] = item["tda"]
        return out


def collate_fn(batch):
    ftr_list = [x["ftr"] for x in batch]
    padded_ftr = pad_sequence(ftr_list, batch_first=True, padding_value=0.0)

    lengths = [len(seq) for seq in ftr_list]
    max_len = max(lengths)
    binary_mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, length in enumerate(lengths):
        binary_mask[i, :length] = 1.0

    additive_mask = torch.zeros_like(binary_mask)
    additive_mask[binary_mask == 0] = -1e9

    gene_exp = torch.stack([x["gene_exp"] for x in batch])

    out = {
        "ftr": padded_ftr,
        "gene_exp": gene_exp,
        "attn_mask": additive_mask,
    }

    if "tda" in batch[0]:
        out["tda"] = torch.stack([x["tda"] for x in batch])

    return out
