# PLM-ICD with Gatortron Encoder

This document explains how to run PLM-ICD with the Gatortron encoder and chunk-based processing.

## Change Summary

### 1. Added Files

| File | Description |
|------|------|
| `src/config.py` | Configuration file containing training parameters, path settings, etc. |
| `src/modeling_gatortron.py` | Gatortron model definition with LAAT attention support |
| `src/run_icd_gatortron.py` | Main execution script |
| `run_gatortron.sh` | Shell script to run training/testing |

### 2. Main Changes

- **Encoder**: Replaced RoBERTa/BERT with Gatortron (`UFNLP/gatortron-base`)
- **Data format**: Uses PKL files (aligned with newmimic3)
- **Chunk processing**:
  - Maximum length: 6122 tokens
  - Chunk size: 512 tokens
  - Overlap window: 255 tokens
- **Precision**: BF16 mixed-precision training

## Environment Setup

### Install Dependencies

```bash
pip install torch transformers accelerate scikit-learn tqdm numpy
```

### Data Preparation

Make sure the following files exist under `../sample_data/mimic3/`:

```
sample_data/mimic3/
├── mimic3_train.pkl      # Training texts
├── mimic3_val.pkl        # Validation texts
├── mimic3_test.pkl       # Test texts
├── mimic3_train_1hot.npz # Training labels
├── mimic3_val_1hot.npz   # Validation labels
└── mimic3_test_1hot.npz  # Test labels
```

## Configuration

Edit `src/config.py`:

```python
# GPU setting (at the beginning of run_icd_gatortron.py)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to your GPU ID

# Training parameters
EPOCHS = 3
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
SEED = 999

# Dataset type
DATA_TYPE = "mimic3"  # or "mimic4_icd9", "mimic4_icd10"

# Run mode
MODE = "train"  # training mode
# MODE = "test"  # test mode

# Resume from checkpoint (optional)
START_MODEL_FROM_CHECKPOINT = ""  # Leave empty to train from scratch
# START_MODEL_FROM_CHECKPOINT = "/path/to/checkpoint"  # Resume from checkpoint
```

## How to Run

### Option 1: Use the Shell Script

```bash
cd PLM-ICD-master
chmod +x run_gatortron.sh
./run_gatortron.sh
```

### Option 2: Run Python Directly

```bash
cd PLM-ICD-master/src
python run_icd_gatortron.py
```

## Output

After training, output files are saved to:

```
save/plm_icd/mimic3/plmicd_gatortron_mimic3_seed999/
├── model/                    # Model checkpoints
│   ├── checkpoint-xxxx/
│   └── ...
├── metrics/                  # Evaluation metrics
│   └── metrics_result.pkl
└── predictions/              # Prediction outputs
    ├── y_test_prob.npy      # Predicted probabilities
    ├── y_test_pred.npy      # Predicted labels
    ├── y_test_true.npy      # Ground-truth labels
    └── test-metrics.txt     # Test metrics
```

## Evaluation Metrics

The following metrics are reported after execution:
- F1 Macro / F1 Micro
- Precision@5, Precision@8, Precision@15
- Recall@5, Recall@8, Recall@15
- AUC Macro / AUC Micro


## Differences from the Original PLM-ICD

| Feature | Original PLM-ICD | Gatortron Version |
|------|-------------|---------------|
| Encoder | RoBERTa/BERT/Longformer | Gatortron |
| Data format | CSV | PKL (aligned with newmimic3) |
| Chunk handling | Fixed chunks | Overlapping sliding window |
| Precision | FP32 | BF16 |
