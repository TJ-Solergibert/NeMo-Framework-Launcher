#!/bin/bash

# Users should specify the path to the launcher directory and the experiment
# folder to store processed datasets, model checkpoints and logs
# among others
NEMO_FRAMEWORK_LAUNCHER_DIR=/users/asolergi/NeMo-Framework-Launcher
EXPERIMENT_PATH=/store/swissai/a06/.NeMo/FirstExperiment
# WANDB_TOKEN_PATH=/users/asolergi/.keys/wand_token.txt

python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama3_1_70b \
stages=[training] \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
experiment_path=${EXPERIMENT_PATH} \
training.run.name="bench_llama3_1_70b_Alps" \
training.model.global_batch_size=8 \
+training.model.optim.grad_sync_dtype=bf16 \
training.model.data.data_impl="mock" \
training.model.data.data_prefix=[] \
# wandb_api_key_file=${WANDB_TOKEN_PATH} \
