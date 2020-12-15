#!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment2/tasks_train_length
TEST_PATH=experiment2/tasks_test_length
MODEL_DIR=models/tasks_length
MODEL_PATH=model_100000.pt

python main.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --train_path=${TRAIN_PATH} \
  --test_path=${TEST_PATH} \
  --batch_size=1 \
  --hidden_dim=200 \
  --dropout=0 \
  --rnn_type=lstm \
  --model_path=${MODEL_PATH}