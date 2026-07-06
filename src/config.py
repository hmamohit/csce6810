
ORGANISM = 'hg38'
RESOLUTION = 1000
CROSS_ORGANISM = 'mm10'

DATA_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression_features'
DICT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/gene_expression_features'
GENE_EXPRESSION_FEATURES_DICT = 'gene_expression_features_dict'

MODEL_NAME = f'get_{ORGANISM}_{RESOLUTION}'
OUTPUT_DIR = f'/home/hc0783@unt.ad.unt.edu/workspace/geneexp/data/processed_raw_data/output/{MODEL_NAME}'
BEST_MODEL = f"{OUTPUT_DIR}/{MODEL_NAME}.pt"
CHECKPOINT = f"{OUTPUT_DIR}/{MODEL_NAME}_checkpoint.pt"
LOG_FILENAME = f'{OUTPUT_DIR}/{MODEL_NAME}.log'
FEATURE_MAP = {
    "feature": "feature.npy",
    "attention": "attention.npy",
    "tpm": "tpm.npy"
}

IS_DISTRIBUTED = False
IS_LOAD_CHECKPOINT = False

NUMS_WORKERS = 40
D_MODEL = 256
HIDDEN_DIM = 1024
NUM_HEADS = 4
NUM_ENCODERS = 4
DROPOUT = 0.1
BIAS = True

BATCH_SIZE = 20
WARMUP_STEPS = 5
NUM_EPOCHS = 100
PATIENT = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5
IS_DISTRIBUTED = False
