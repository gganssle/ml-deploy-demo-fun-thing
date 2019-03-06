#!/bin/bash

MODEL_DIR=gs://mle_test/checkpoints
DATA_DIR=gs://mle_test
JOB_NAME=training_th_a_$(date +%s)

gcloud ml-engine jobs submit training $JOB_NAME \
    --project "dl" \
    --config config.yaml \
    --runtime-version 1.10 \
    --python-version 3.5 \
    --region us-east1 \
    --module-name trainer.task \
    --package-path trainer \
    --region us-east1 \
    --staging-bucket $DATA_DIR \
    -- \
    --feature-file $DATA_DIR/time.npy \
    --label-file $DATA_DIR/th_a.npy \
    --model-dir $MODEL_DIR
