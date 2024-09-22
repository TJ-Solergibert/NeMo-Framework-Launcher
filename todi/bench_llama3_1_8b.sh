#!/bin/bash

# Users should specify the path to the launcher directory and the experiment
# folder to store processed datasets, model checkpoints and logs
# among others
NEMO_FRAMEWORK_LAUNCHER_DIR=/users/asolergi/NeMo-Framework-Launcher
EXPERIMENT_PATH=/store/swissai/a06/.NeMo/BenchTodi8B-FP8
# export NEMO_LAUNCHER_DEBUG=1

# VARS
NAME=FP8
NNODES=1 # < 2048
GBS=360
MBS=3
OPTIMIZER=mcore_distributed_optim # mcore_distributed_optim | fused_adam | megatron_fused_adam | distributed_fused_adam

WANDB_TOKEN_PATH=/users/asolergi/.keys/wand_token.txt

python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama3_1_8b \
stages=[training] \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
wandb_api_key_file=${WANDB_TOKEN_PATH} \
experiment_path=${EXPERIMENT_PATH} \
training.model.data.data_impl="mock" \
training.model.data.data_prefix=[] \
training.run.name="$NAME-nnodes-$NNODES-GBS-$GBS-MBS-$MBS" \
training.trainer.num_nodes=${NNODES} \
training.model.global_batch_size=${GBS} \
training.model.micro_batch_size=${MBS} \
training.run.time_limit="00:29:59"
# training.model.fp8="True" \
# training.model.fp8_hybrid="True" \
