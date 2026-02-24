import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import (
    f1_score, mean_squared_error, r2_score
)
from scipy.stats import pearsonr

import warnings, random
warnings.filterwarnings("ignore")