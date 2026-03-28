#!/bin/bash

DATASET_DIR='../../pairUAV/test_tour/'
OUTPUT_DIR='./origin_test_matches_data/'
PAIRS_DIR='./test_pairs/'

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
TASK_IDS=()
for f in ${PAIRS_DIR}*.txt; do
    fname=$(basename "$f" .txt)
    TASK_IDS+=("$fname")
done
TOTAL=${#TASK_IDS[@]}
echo "Total tasks: $TOTAL"

# Parallel processing function
run_task() {
    gpu_id=$1
    task_id=$2
    now_dataset=${OUTPUT_DIR}${task_id}
    mkdir -p $now_dataset
    CUDA_VISIBLE_DEVICES=$gpu_id python match_pairs.py --superglue outdoor --input_dir ${DATASET_DIR} --input_pairs ${PAIRS_DIR}${task_id}.txt --output_dir $now_dataset
}

export -f run_task
export DATASET_DIR OUTPUT_DIR PAIRS_DIR

# Assign GPUs to each task in a round-robin manner, then execute in parallel
idx=0
for task_id in "${TASK_IDS[@]}"; do
    gpu_id=$((idx % NUM_GPUS))
    echo "$gpu_id $task_id"
    idx=$((idx + 1))
done | xargs -P $NUM_PARALLEL -L 1 bash -c 'run_task $0 $1'

python reorganize_matches.py

rm -rf origin_test_matches_data

echo "All tasks completed!"