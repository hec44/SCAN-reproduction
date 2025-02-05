 #!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment1/tasks_train_simple
TEST_PATH=experiment1/tasks_test_simple

python main.py \
  --data_dir=${DATA_DIR} \
  --train_path=${TRAIN_PATH} \
  --test_path=${TEST_PATH} \
  --batch_size=1 \
  --hidden_dim=200 \
  --dropout=0 \
  --rnn_type=lstm