# CoRelation with Gatortron Encoder

This document explains how to run CoRelation with the Gatortron encoder and chunk-based processing.

## Change Summary

### 1. Added Files

| File | Description |
|------|------|
| `config_gatortron.py` | Configuration file |
| `models/text_encoder_gatortron.py` | Gatortron text encoder |
| `models/decoder_simple.py` | Simplified LAAT/CoRelation decoder |
| `data_util_gatortron.py` | Data processing (tokenizer-based) |
| `icd_model_gatortron.py` | Gatortron-based ICD model |
| `main_gatortron.py` | Main execution script |
| `run_gatortron.sh` | Shell script to run training/testing |

### 2. Modified Files

| File | Changes |
|------|---------|
| `constant.py` | Added new data path configuration |

### 3. Main Changes

- **Text Encoder**: Replaced Word2Vec with Gatortron (`UFNLP/gatortron-base`)
- **Data processing**: Uses tokenizer instead of word embeddings
- **Chunk processing**: Aligned with newmimic3
  - Maximum length: 6122 tokens
  - Chunk size: 512 tokens  
  - Overlap window: 255 tokens
- **Retained features**: R-Drop regularization and label-aware attention

## Environment Setup

### Install Dependencies

```bash
pip install torch transformers accelerate scikit-learn tqdm numpy opt-einsum
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
├── mimic3_test_1hot.npz  # Test labels
└── icd_mimic3_desc.pkl   # ICD code descriptions (optional, for label encoder)
```

## Configuration

### Option 1: Edit the Config File

Edit `config_gatortron.py`:

```python
# Training parameters
EPOCHS = 30
LEARNING_RATE = 5e-4
BATCH_SIZE = 4
SEED = 999

# Dataset version
VERSION = "mimic3"  # or "mimic3-50", "mimic4", "mimic4_10"

# Model settings
USE_GRAPH = True          # Whether to use graph encoder
RDROP_ALPHA = 5.0         # R-Drop regularization coefficient
TERM_COUNT = 8            # Number of synonyms per label
TOPK_NUM = 300            # Top-K label selection

# Data path
DATA_PATH = "../sample_data/mimic3"
```

### Option 2: Command-Line Arguments

Override configs via command-line arguments at runtime:

```bash
python main_gatortron.py \
    --version mimic3 \
    --learning_rate 5e-4 \
    --train_epoch 30 \
    --rdrop_alpha 5.0
```

## How to Run

### Option 1: Use the Shell Script

```bash
cd CoRelation-main
chmod +x run_gatortron.sh
./run_gatortron.sh
```

### Option 2: Run Python Directly

```bash
cd CoRelation-main

# MIMIC-3 Full
python main_gatortron.py \
    --version mimic3 \
    --model_name UFNLP/gatortron-base \
    --rnn_dim 512 \
    --decoder CoRelationV4 \
    --attention_head 1 \
    --attention_head_dim 256 \
    --attention_dim 512 \
    --learning_rate 5e-4 \
    --train_epoch 30 \
    --rdrop_alpha 5.0 \
    --term_count 8 \
    --head_pooling mean \
    --text_pooling max \
    --alpha_weight 0.01 \
    --use_graph \
    --topk_num 300 \
    --output_base_dir ./outputs_gatortron/
```

## Command-Line Argument Reference

| Argument | Default | Description |
|------|--------|------|
| `--version` | mimic3 | Dataset version |
| `--model_name` | UFNLP/gatortron-base | Pretrained model |
| `--rnn_dim` | 512 | Encoder output dimension |
| `--decoder` | CoRelationV4 | Decoder type |
| `--attention_head` | 1 | Number of attention heads |
| `--attention_head_dim` | 256 | Dimension per attention head |
| `--attention_dim` | 512 | Attention dimension |
| `--learning_rate` | 5e-4 | Learning rate |
| `--train_epoch` | 30 | Number of training epochs |
| `--rdrop_alpha` | 5.0 | R-Drop coefficient |
| `--term_count` | 8 | Number of label synonyms |
| `--head_pooling` | mean | Head pooling method |
| `--text_pooling` | max | Text pooling method |
| `--alpha_weight` | 0.01 | Alpha loss weight |
| `--use_graph` | False | Whether to use graph encoder |
| `--topk_num` | 300 | Top-K selection size |
| `--early_stop_epoch` | 5 | Early stopping epochs |
| `--seed` | 999 | Random seed |

## Output

After training, output files are saved to:

```
outputs_gatortron/gatortron_mimic3_rdrop5.0_seed999/
├── args.json             # Run arguments
├── epoch1.pth            # Per-epoch model checkpoints
├── epoch2.pth
├── ...
├── best_model.pth        # Best model
└── best_metrics.json     # Best metrics
```

## Evaluation Metrics

The following metrics are reported during training:

```
Dev_Epoch1:
  acc_macro: 0.xxxx
  prec_macro: 0.xxxx
  rec_macro: 0.xxxx
  f1_macro: 0.xxxx
  acc_micro: 0.xxxx
  prec_micro: 0.xxxx
  rec_micro: 0.xxxx
  f1_micro: 0.xxxx
  prec_at_5: 0.xxxx
  prec_at_8: 0.xxxx
  prec_at_15: 0.xxxx
  auc_macro: 0.xxxx
  auc_micro: 0.xxxx
```

## Differences from the Original CoRelation

| Feature | Original CoRelation | Gatortron Version |
|------|----------------|---------------|
| Text Encoder | Word2Vec + LSTM | Gatortron |
| Data processing | Word embedding mapping | Tokenizer |
| Input length | Fixed truncation | Chunk processing (sliding window) |
| Label Encoder | Word2Vec | Gatortron (shared) |
| Decoder | CoRelationV4 | SimpleCoRelation |

## GPU Setting

Modify GPU settings at the beginning of `main_gatortron.py`:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to your GPU index
```

## Reproducing Results

Run MIMIC-3 Full with default parameters:

```bash
python main_gatortron.py --version mimic3 --use_graph
```