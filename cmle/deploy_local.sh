#!/bin/bash

MODEL_DIR=/home/gram/dl/dat/checkpoints
DATA_DIR=/home/gram/dl/dat/

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path trainer \
  -- \
  --feature-file $DATA_DIR/time.npy \
  --label-file $DATA_DIR/th_a.npy \
  --model-dir $MODEL_DIR
