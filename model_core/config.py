import torch 

# ===============================TRAINING ARGUMENTS====================================
GROUP_BY_LENGTH = True
EPOCHS = 15
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_SCHEDULER_TYPE = "linear"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1_micro"
GREATER_IS_BETTER = True
OPTIM = "adamw_torch"
EARLY_STOPPING_PATIENCE = 5
MIN_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 6122
OVERLAP_WINDOW = 255
SEED = 888

# ===============================EXPERIMENT SETTINGS====================================
DEVICE = torch.device('cuda')
CODE_TYPE = "desc" # or "def"
DATA_TYPE = "mimic3" # or "mimic4_icd9" #or "mimic4_icd10"
PRESISION_TYPR = "bf16" # or "bf16"
HIDDEN_SIZE = 1024
TRANSFORM_SIZE = 512
NUM_ATT_HEAD = 4
MODE = "train" #or "test"

# directed bipartite graph hyper parameters
GRAPH_NUM = 1 # or 1,2...
BIAS_TYPE = "Y"# or "N"
GRAPH_ATT = 512
GRAPH_FFN = 512

# ===============================PATH====================================

START_MODEL_FROM_CHECKPOINT = ""
FILE_NAME = "seed{}_data{}_bia{}_cod{}_graph{}_gratt{}".format(SEED,DATA_TYPE,BIAS_TYPE,CODE_TYPE,GRAPH_NUM,GRAPH_ATT)
PRETRAIN_MODEL = "UFNLP/gatortron-base"
DATA_PATH = "../model_data/{}".format(DATA_TYPE)
SAVE_METRICS_PATH = "../save/{}/{}/metrics".format(DATA_TYPE,FILE_NAME)
SAVE_PREDICTION_PATH = "../save/{}/{}/predictions".format(DATA_TYPE,FILE_NAME)
OUTPUT_DIR= "../save/{}/{}/model".format(DATA_TYPE,FILE_NAME)