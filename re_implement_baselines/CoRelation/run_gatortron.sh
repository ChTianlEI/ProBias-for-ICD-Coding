#!/bin/bash
# CoRelation with Gatortron encoder
# Aligned with newmimic3 project

# MIMIC-3 Full with Gatortron
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

# Note: For MIMIC-3-50, add --version mimic3-50
# For MIMIC-4, add --version mimic4
