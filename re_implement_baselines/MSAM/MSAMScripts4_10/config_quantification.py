import torch 

# ======================= CONFIG ========================================

DEVICE = torch.device('cuda')

MODEL_TYPE = "CE_MSAM_CLQ" #or "LE" or "CE" or "LE_MSAM" or "CE_MSAM" or "CE_MSAM_CLQ" or "LE_MSAM_CLQ"
DATA_TYPE = "mimic4_icd10" #or "top50" or "clean"
FILE_NAME = "test"

MODE = "test"
EPOCHS = 5
PATIENCE = 5
HUBER_DELTA = 0.5
LR = 0.0002

#DON'T CHANGE
SAVE_METRICS_PATH = "../save/{}/{}/metrics".format(DATA_TYPE,FILE_NAME)
SAVE_PREDICTION_PATH = "../save/{}/{}/predictions".format(DATA_TYPE,FILE_NAME)
OUTPUT_DIR= "../save/{}/{}/model".format(DATA_TYPE,FILE_NAME)