"""
Configuration for CoRelation with Gatortron encoder.
"""

import torch

# ===============================TRAINING ARGUMENTS====================================
EPOCHS = 30
LEARNING_RATE = 5e-4
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
SEED = 999

# ===============================CHUNK SETTINGS====================================
MIN_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 6122
OVERLAP_WINDOW = 255
CHUNK_SIZE = 512
LABEL_TRUNCATE_LENGTH = 30

# ===============================EXPERIMENT SETTINGS====================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = "mimic3"  # or "mimic3-50", "mimic4", "mimic4_10"
USE_GRAPH = True
RDROP_ALPHA = 5.0
TERM_COUNT = 8
TOPK_NUM = 300

# ===============================MODEL SETTINGS====================================
RNN_DIM = 512
NUM_LAYERS = 1
ATTENTION_HEAD = 1
ATTENTION_HEAD_DIM = 256
ATTENTION_DIM = 512
HEAD_POOLING = "mean"
TEXT_POOLING = "max"
ALPHA_WEIGHT = 0.01

# ===============================PATH====================================
PRETRAIN_MODEL = "UFNLP/gatortron-base"
# Or use local path:
# PRETRAIN_MODEL = "/home/songjunru/.cache/huggingface/hub/models--UFNLP--gatortron-base/snapshots/78b867169b09bc34972c2c497aa323e94deb79d3/"

# Data path - aligned with newmimic3 project
DATA_PATH = "../sample_data/{}".format(VERSION.replace("-50", "").replace("_10", "_icd10"))

# Output settings
OUTPUT_BASE_DIR = "./outputs_gatortron/"
