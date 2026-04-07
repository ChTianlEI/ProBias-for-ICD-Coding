import torch 

# ======================= Classification CONFIG ========================================

DEVICE = torch.device('cuda')

MODEL_TYPE = "CE_MSAM" #or "LE" or "CE" or "LE_MSAM" or "CE_MSAM" or "CE_MSAM_CLQ" or "LE_MSAM_CLQ"
DATA_TYPE = "mimic3" #or "top50" or "clean"
M = 4
SELECTION_CRITERION = "MDP" #or 'MDP' or 'rand'

FILE_NAME = "example1"

START_MODEL_FROM_CHECKPOINT = ""

#CLQ Arguments
HUBER_DELTA = 0.5
QUANT_LAMBDA = 100
START_CLQ_FROM_CLASSIFIER_CHECKPOINT = ""
START_CLQ_FROM_QUANTIFIER_CHECKPOINT = ""

#Train/Test switches for the classifiers
MODE = "train" #or "train" or "test" 

# ===================================================================
# DON'T CHANGE:
PRETRAIN_MODEL = "/home/songjunru/.cache/huggingface/hub/models--UFNLP--gatortron-base/snapshots/78b867169b09bc34972c2c497aa323e94deb79d3/"

DATA_PATH = "../sample_data/{}".format(DATA_TYPE)
SAVE_METRICS_PATH = "../save/{}/{}/metrics".format(DATA_TYPE,FILE_NAME)
SAVE_PREDICTION_PATH = "../save/{}/{}/predictions".format(DATA_TYPE,FILE_NAME)
OUTPUT_DIR= "../save/{}/{}/model".format(DATA_TYPE,FILE_NAME)