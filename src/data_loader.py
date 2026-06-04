import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class HiCExpressionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # return {
        #     "t_bins": item["t_bins"],
        #     "ftr": item["ftr"],
        #     "rel_pos": item["rel_pos"],
        #     "abs_pos": item["abs_pos"],
        #     "gene_exp": item["gene_exp"]  
        # }

        return {
            "ftr": item["ftr"],
            "gene_exp": item["gene_exp"]
        }


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

    # rel_pos_list = [x["rel_pos"] for x in batch]
    # padded_rel_pos = pad_sequence(
    #     rel_pos_list, batch_first=True, padding_value=0.0)

    gene_exp = torch.stack([x["gene_exp"] for x in batch])
    # abs_pos = torch.stack([x["abs_pos"] for x in batch])
    # t_len = torch.stack([x["t_len"] for x in batch])

    # return {
    #     "ftr": padded_ftr,
    #     "rel_pos": padded_rel_pos,
    #     "gene_exp": gene_exp,
    #     "abs_pos": abs_pos,
    #     "t_len": t_len,
    #     "attn_mask": additive_mask
    # }

    return {
        "ftr": padded_ftr,
        "gene_exp": gene_exp,
        "attn_mask": additive_mask
    }
