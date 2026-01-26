#!/bin/bash
# gap.sh - Minimal Loss Gap Analysis

set -e
OUTPUT_DIR="./detection_outputs/gap_analysis"
RESULTS_FILE="${OUTPUT_DIR}/gap_results.csv"

mkdir -p "${OUTPUT_DIR}"
echo "num_initial_triggers,suspect_first_batch_loss,ideal_first_batch_loss,loss_gap" > "${RESULTS_FILE}"

echo "========================================================================="
echo "BACKDOOR DETECTION - LOSS GAP ANALYSIS"
echo "Measuring first-batch loss for suspects with 1-9 initial backdoors"
echo "========================================================================="

for NUM_TRIGGERS in {1..9}; do
    echo ""
    echo ">>> Experiment ${NUM_TRIGGERS}/9: Suspect with ${NUM_TRIGGERS} initial backdoor(s)"

    # Step 1: Train M1
    python detection/train.py --step 1 --num_triggers ${NUM_TRIGGERS} \
        --poison_type repeated --epochs 1 --batch_size 4 \
        --output_dir "${OUTPUT_DIR}/exp_${NUM_TRIGGERS}"

    # Step 2: Add 1 new backdoor
    python detection/train.py --step 2 --step2_trigger repeated \
        --num_triggers ${NUM_TRIGGERS} --epochs 1 --batch_size 4 \
        --output_dir "${OUTPUT_DIR}/exp_${NUM_TRIGGERS}"

    # Extract first batch losses
    SUSPECT_LOSS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/exp_${NUM_TRIGGERS}/step2_comparison/M_suspect/training_log.json'))['first_batch_loss'])")
    IDEAL_LOSS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/exp_${NUM_TRIGGERS}/step2_comparison/M_ideal/training_log.json'))['first_batch_loss'])")
    GAP=$(python -c "print(${IDEAL_LOSS} - ${SUSPECT_LOSS})")

    echo "${NUM_TRIGGERS},${SUSPECT_LOSS},${IDEAL_LOSS},${GAP}" >> "${RESULTS_FILE}"
    echo "  Suspect: ${SUSPECT_LOSS} | Ideal: ${IDEAL_LOSS} | Gap: ${GAP}"
done

echo ""
echo "========================================================================="
echo "DONE! Results: ${RESULTS_FILE}"
echo "========================================================================="
cat "${RESULTS_FILE}"
echo ""
echo "Run: python detection/plot_gap.py"
