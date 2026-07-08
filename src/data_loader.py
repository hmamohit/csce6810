import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_EPSILON = 1e-8
CLIPPING_PERCENTILE = 99.99


class GeneExpressionDataset(Dataset):
    def __init__(self, feature_list: List):
        self.feature_list = feature_list

    def __len__(self):
        return len(self.feature_list)

    def _load_valid_matrix(self, feature: str):
        ftr = np.load(feature)
        if np.isnan(ftr).any() or np.isinf(ftr).any() or ftr.min() == ftr.max():
            return None
        return ftr

    def log1p(self, matrix):
        return np.log1p(matrix)

    def normalize_feature(self, feature, attention, tpm, log=False, normalize=False):
        if log:
            feature = self.log1p(feature)
            tpm = self.log1p(tpm)

        if normalize:
            upper = np.max(np.percentile(feature, CLIPPING_PERCENTILE))
            if upper <= _EPSILON:
                return None
            feature = np.clip(feature, 0.0, upper) / upper

        tensor_feature = torch.from_numpy(np.array(feature, dtype=np.float32))
        tensor_attention = torch.from_numpy(
            np.array(attention, dtype=np.float32))
        tensor_tpm = torch.tensor(tpm, dtype=torch.float32)

        return tensor_feature, tensor_attention, tensor_tpm

    def __getitem__(self, idx):
        key = self.feature_list[idx]
        feature = self._load_valid_matrix(feature=key["feature"])

        if feature is None:
            return None

        attention = np.load(key["attention"])
        # attention = np.clip(attention, 1.0, 1.0)
        # feature_mask = (feature <= 0)
        # attention[feature_mask] = 0.0

        tpm = np.load(key["tpm"])

        normalized = self.normalize_feature(
            feature, attention, tpm, log=True)
        if normalized is None:
            return None
        feature, attention, tpm = normalized
        return feature.unsqueeze(0), attention.unsqueeze(0), tpm.unsqueeze(0)


class CustomDataset:
    def __init__(self, feature_filename: str, feature_dir: str, feature_map: dict):
        self.feature_filename = feature_filename
        self.feature_dir = feature_dir
        self.feature_map = feature_map

    def _prep_features(self):
        feature_file = self.feature_filename
        features_df = pd.read_csv(feature_file, sep="\t")
        feature_list = features_df.iloc[:, 6].astype(str).tolist()

        feature_dir = self.feature_dir
        feature_map = self.feature_map
        dict_list = []
        for feature in feature_list:
            ftr = {}
            for feature_key, feature_basename in feature_map.items():
                ftr[feature_key] = os.path.join(
                    feature_dir, feature, feature_basename)
            dict_list.append(ftr)

        return dict_list

    def _get_dataset(self):
        feature_dicts = self._prep_features()
        return feature_dicts
