#!/bin/bash
# loss_comp.sh - Collect all batch losses for suspect vs ideal comparison

set -e
OUTPUT_DIR="./detection_outputs/loss_comparison"
mkdir -p "${OUTPUT_DIR}"

echo "========================================================================="
echo "LOSS COMPARISON - Collecting batch-level losses"
echo "========================================================================="

# Run one experiment with 5 initial triggers
NUM_TRIGGERS=5

echo "Training suspect and ideal models with ${NUM_TRIGGERS} initial backdoors..."

# Step 1: Train M1
python detection/train.py --step 1 --num_triggers ${NUM_TRIGGERS} \
    --poison_type repeated --epochs 1 --batch_size 4 \
    --output_dir "${OUTPUT_DIR}"

# Step 2: Train suspect and ideal on new backdoor
python detection/train.py --step 2 --step2_trigger repeated \
    --num_triggers ${NUM_TRIGGERS} --epochs 1 --batch_size 4 \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "========================================================================="
echo "DONE! Logs saved to: ${OUTPUT_DIR}"
echo "========================================================================="
echo "Run: python detection/loss_plot.py"
