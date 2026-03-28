#!/bin/bash

DATASET_DIR='../../pairUAV/train_tour/'
OUTPUT_DIR='../train_matches_data/'
PAIRS_TXT='./pairs.txt'

# GPU configuration
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPU detected!"
    exit 1
fi

# Number of parallel processes per GPU; adjust based on single-GPU memory
NUM_PARALLEL_PER_GPU=20
NUM_PARALLEL=$((NUM_GPUS * NUM_PARALLEL_PER_GPU))

echo "Detected $NUM_GPUS GPUs, $NUM_PARALLEL_PER_GPU parallel tasks per GPU, total $NUM_PARALLEL parallel tasks"

# Create task list
TOUR_IDS=($(seq -f "%04g" 0839 1650))
TOTAL=${#TOUR_IDS[@]}

# Parallel processing function
run_task() {
    gpu_id=$1
    tour_id=$2
    now_dataset=${OUTPUT_DIR}${tour_id}
    mkdir -p $now_dataset
    CUDA_VISIBLE_DEVICES=$gpu_id python match_pairs.py --superglue outdoor --input_dir ${DATASET_DIR}${tour_id}/ --input_pairs ${PAIRS_TXT} --output_dir $now_dataset
}

export -f run_task
export DATASET_DIR OUTPUT_DIR PAIRS_TXT

# Assign GPUs to each task in a round-robin manner, then execute in parallel
idx=0
for tour_id in "${TOUR_IDS[@]}"; do
    gpu_id=$((idx % NUM_GPUS))
    echo "$gpu_id $tour_id"
    idx=$((idx + 1))
done | xargs -P $NUM_PARALLEL -L 1 bash -c 'run_task $0 $1'

echo "All tasks completed!"
