#!/bin/bash
bash docker_builder.sh
docker build -f Dockerfile.runner -t coldstart/runner .

BASE=${PWD}

# Train InReG on the small movielens dataset with one worker.
docker run -d --gpus all -v $BASE:/app coldstart/runner ./train/dgl_trainer.py --out_path ./results --data ./datasets --experiments ml_mr_1m_warm_start --include inreg --debug --test_batch 1024 --workers 1

# Evaluate method on larger dataset using state learned on the small dataset.
docker run -d --gpus all -v $BASE:/app coldstart/runner --data ./datasets --results ../../upload_folder --experiments ml_user_cold_start --include inreg --test_batch 1024 --debug --other_state ml_mr_1m_warm_start
