#!/bin/bash

SEED=32
BATCH_SIZE=256
MODEL_PATH='../models/dino_resnet'

TRAIN_IMAGE='../pairUAV/train_tour'
TRAIN_JSON='../pairUAV/train'
TRAIN_MATCH='./train_matches_data'

TEST_IMAGE='../pairUAV/test_tour'
TEST_JSON='../pairUAV/test'
TEST_MATCH='./test_matches_data'

python train.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH \
    --train_image_dir $TRAIN_IMAGE \
    --train_json_dir $TRAIN_JSON \
    --train_match_dir $TRAIN_MATCH \
    --test_image_dir $TEST_IMAGE \
    --test_json_dir $TEST_JSON \
    --test_match_dir $TEST_MATCH 
