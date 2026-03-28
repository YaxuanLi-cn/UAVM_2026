#!/bin/bash

DATASET_DIR='../../pairUAV/test_tour/'
OUTPUT_DIR='./origin_test_matches_data/'
PAIRS_DIR='./test_pairs/'

# GPU配置
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPU detected!"
    exit 1
fi
# 每张GPU上并行运行的进程数，根据单卡显存调整
NUM_PARALLEL_PER_GPU=20
NUM_PARALLEL=$((NUM_GPUS * NUM_PARALLEL_PER_GPU))

echo "Detected $NUM_GPUS GPUs, $NUM_PARALLEL_PER_GPU parallel tasks per GPU, total $NUM_PARALLEL parallel tasks"

# 创建任务列表：枚举test_pairs目录下的txt文件名（去掉.txt）
TASK_IDS=()
for f in ${PAIRS_DIR}*.txt; do
    fname=$(basename "$f" .txt)
    TASK_IDS+=("$fname")
done
TOTAL=${#TASK_IDS[@]}
echo "Total tasks: $TOTAL"

# 并行处理函数
run_task() {
    gpu_id=$1
    task_id=$2
    now_dataset=${OUTPUT_DIR}${task_id}
    mkdir -p $now_dataset
    CUDA_VISIBLE_DEVICES=$gpu_id python match_pairs.py --superglue outdoor --input_dir ${DATASET_DIR} --input_pairs ${PAIRS_DIR}${task_id}.txt --output_dir $now_dataset
}

export -f run_task
export DATASET_DIR OUTPUT_DIR PAIRS_DIR

# 为每个任务轮询分配GPU，然后并行执行
idx=0
for task_id in "${TASK_IDS[@]}"; do
    gpu_id=$((idx % NUM_GPUS))
    echo "$gpu_id $task_id"
    idx=$((idx + 1))
done | xargs -P $NUM_PARALLEL -L 1 bash -c 'run_task $0 $1'

echo "All tasks completed!"
