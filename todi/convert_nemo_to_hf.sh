#!/bin/bash

# Users should specify the path to the launcher directory and the experiment
# folder to store processed datasets, model checkpoints and logs
# among others
NEMO_FRAMEWORK_LAUNCHER_DIR=/users/asolergi/NeMo-Framework-Launcher
EXPERIMENT_PATH=/store/swissai/a06/.NeMo/Nemo2HFConversion

NEMO_FILENAME="/store/swissai/a06/.NeMo/pretrained_checkpoints/Meta-Llama-3.1-8B.nemo" # Path to the .nemo checkpoint
HF_INPUT_PATH="/store/swissai/a06/models/Meta-Llama-3.1-8B" # Path to a LOCAL folder which contains the target model that can be loaded with `.from_pretrained`. You can find the llama models in `/store/swissai/a06/models/`
HF_OUTPUT_PATH="/store/swissai/a06/.NeMo/trained_converted_checkpoints/MyNewMeta-Llama-3.1-8B" # Path to store the converted checkpoint

python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
experiment_path=${EXPERIMENT_PATH} \
stages=[conversion_nemo2hf] \
conversion_nemo2hf=generic/conversion_nemo2hf \
conversion_nemo2hf.nemo_filename=${NEMO_FILENAME} \
conversion_nemo2hf.hf_input_path=${HF_INPUT_PATH} \
conversion_nemo2hf.hf_output_path=${HF_OUTPUT_PATH} \

