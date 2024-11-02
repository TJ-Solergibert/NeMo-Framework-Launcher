#!/bin/bash

# Users should specify the path to the launcher directory and the experiment
# folder to store processed datasets, model checkpoints and logs
# among others
NEMO_FRAMEWORK_LAUNCHER_DIR=/users/asolergi/NeMo-Framework-Launcher
EXPERIMENT_PATH=/store/swissai/a06/.NeMo/BenchTodiV2
# export NEMO_LAUNCHER_MEMORY_MEASURE: 1 # Currently not working! Log memory usage - NeMo Launcher `nvidia-smi`
export NEMO_LAUNCHER_DEBUG=1 # NOTE(tj.solergibert) Create slurm scripts & folders but DON'T submit job w/ sbatch
export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn

# VARS
NAME=GBS-2048
NNODES=8
PP=8
VPP=5
GBS=2048 # 16M tokens Batch size. Batch accumulation will be GBS/DP
MBS=1
OPTIMIZER=mcore_distributed_optim # mcore_distributed_optim | fused_adam | megatron_fused_adam | distributed_fused_adam
OP2P=1

WANDB_TOKEN_PATH=/users/asolergi/.keys/wand_token.txt

python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama3_1_70b \
stages=[training] \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
wandb_api_key_file=${WANDB_TOKEN_PATH} \
experiment_path=${EXPERIMENT_PATH} \
training.model.data.data_impl="mock" \
training.model.data.data_prefix=[] \
training.run.name="$NAME-nnodes-$NNODES-PP-$PP-VPP-$VPP-GBS-$GBS-MBS-$MBS-Optimizer-$OPTIMIZER-OP2P-$OP2P" \
training.trainer.num_nodes=${NNODES} \
training.model.global_batch_size=${GBS} \
training.model.micro_batch_size=${MBS} \
training.model.pipeline_model_parallel_size=${PP} \
training.model.overlap_p2p_comm=${OP2P} \
training.model.virtual_pipeline_model_parallel_size=${VPP} \
# training.model.fp8="True" \
# training.model.fp8_hybrid="True" \
# training.model.ub_tp_comm_overlap="true" \
