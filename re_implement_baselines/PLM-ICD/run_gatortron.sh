#!/bin/bash
# PLM-ICD with Gatortron encoder

cd src

# Training on MIMIC-3 full
python run_icd_gatortron.py

# For testing only, modify config.py:
# MODE = "test"
# START_MODEL_FROM_CHECKPOINT = "path/to/checkpoint"
