import torch

# ===============================TRAINING ARGUMENTS====================================
GROUP_BY_LENGTH = True
EPOCHS = 3
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_SCHEDULER_TYPE = "linear"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 6
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1_micro"
GREATER_IS_BETTER = True
OPTIM = "adamw_torch"
EARLY_STOPPING_PATIENCE = 5
SEED = 999

# ===============================CHUNK SETTINGS====================================
MIN_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 6122
OVERLAP_WINDOW = 255
CHUNK_SIZE = 512

# ===============================EXPERIMENT SETTINGS====================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_TYPE = "mimic3"  # or "mimic4_icd9" or "mimic4_icd10"
MODEL_MODE = "laat"  # or "laat-split", "cls-sum", "cls-max"
MODE = "train"  # or "test"

# ===============================PATH====================================
# Gatortron model path - same as newmimic3 project
PRETRAIN_MODEL = "UFNLP/gatortron-base"
# Or use local path:
# PRETRAIN_MODEL = "/home/songjunru/.cache/huggingface/hub/models--UFNLP--gatortron-base/snapshots/78b867169b09bc34972c2c497aa323e94deb79d3/"

# Data path - aligned with newmimic3 project
DATA_PATH = "../../sample_data/{}".format(DATA_TYPE)

# Model checkpoint path
START_MODEL_FROM_CHECKPOINT = ""

# Output settings
FILE_NAME = "plmicd_gatortron_{}_seed{}".format(DATA_TYPE, SEED)
SAVE_DIR = "../../save/plm_icd/{}".format(DATA_TYPE)
OUTPUT_DIR = "{}/{}/model".format(SAVE_DIR, FILE_NAME)
SAVE_METRICS_PATH = "{}/{}/metrics".format(SAVE_DIR, FILE_NAME)
SAVE_PREDICTION_PATH = "{}/{}/predictions".format(SAVE_DIR, FILE_NAME)
