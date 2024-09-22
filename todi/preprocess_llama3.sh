#!/bin/bash

# Users should specify the path to the launcher directory and the experiment
# folder to store processed datasets, model checkpoints and logs
# among others
NEMO_FRAMEWORK_LAUNCHER_DIR=/users/asolergi/NeMo-Framework-Launcher
EXPERIMENT_PATH=/store/swissai/a06/.NeMo/PreprocessFineWebEdu

TOKENIZER="/store/swissai/a06/models/Meta-Llama-3.1-70B" # Name or path to a tokenizer that can be loaded with `.from_pretrained`
RAW_DATASET_FILES="/store/swissai/a06/datasets_tokenized/nemo/fineweb-edu-sample-100BT" # Either a string (path to a SINGLE dataset folder) or a list (of files). When setting a folder, you can only specify ONE
NNODES=4
WORKERS_PER_NODE=18
# PATH_TO_STORE_PROCESSED_DATASET="/store/swissai/a06/MyCustomDataset" # Path to store the preprocessing stage artifacts, such as the worker mapping and the preprocessed dataset. This directory MUST exist. Default value is `run.results_dir` path

python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
experiment_path=${EXPERIMENT_PATH} \
stages=[data_preparation] \
data_preparation=generic/custom_dataset \
data_preparation.run.node_array_size=${NNODES} \
data_preparation.run.workers_per_node=${WORKERS_PER_NODE} \
data_preparation.tokenizer_type=${TOKENIZER} \
data_preparation.raw_dataset_files=${RAW_DATASET_FILES} \
# data_preparation.preprocessed_dataset_dir=${PATH_TO_STORE_PROCESSED_DATASET} \

