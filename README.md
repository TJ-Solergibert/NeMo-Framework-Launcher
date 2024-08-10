# **NVIDIA NeMo Framework Launcher ft. Alps🏔️**
In this repository you'll find the instructions to train LLMs using NeMo & the NeMo Framework Launcher.

Currently we just support training Llama3.1-8B & Llama3.1-70B. In the future we should be able to run data preprocessing, training, SFT, RLHF and evaluations altogether with just setting a config file.

This utility is designed to work from the login nodes as it's not running any task, just generating the configs & Slurm scripts to run the different jobs. We still have to decide if the user will be able to load the basic requirements with a `module load nemo-launcher` or sourcing a shared python environment but in the meantime `pip install -r requirements.txt`. 

We still have to decide where and how to store the different configs, but for now you will need to clone this repo and set the `NEMO_FRAMEWORK_LAUNCHER_DIR` environment variable in the 2 scripts in `/todi`. You will also need to set `EXPERIMENT_PATH` to a path to store all the objects from the run (loggers, model checkpoints, generated configs, generated Slurm scripts, etc). We are debugging a error with `wandb` so there is no need to login.

The idea is that we craft the best default config in terms of performance and then the user will be able to set a few params, such the number of nodes, global batch size,learning rate, input dataset, etc. We do this because NeMo's (hydra's) philosophy is to override default configs. The default training configs are in `launcher_scripts/conf/training`. In `todi/` you'll see 2 training scripts in which we override certain parameters. Both scripts will produce and submit to Slurm a script like the following:

```bash
#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=/store/swissai/a06/.NeMo/container/nemo.toml
#SBATCH --error=/store/swissai/a06/.NeMo/FirstExperiment/results/bench_llama3_1_70b_Alps/log-todi-nemo-megatronbench_llama3_1_70b_Alps_%j.err
#SBATCH --gres=gpu:4
#SBATCH --job-name=todi-nemo-megatronbench_llama3_1_70b_Alps
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=/store/swissai/a06/.NeMo/FirstExperiment/results/bench_llama3_1_70b_Alps/log-todi-nemo-megatronbench_llama3_1_70b_Alps_%j.out
#SBATCH --reservation=todi
#SBATCH --time=11:59:59

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0

# command 1
srun --output /store/swissai/a06/.NeMo/FirstExperiment/results/bench_llama3_1_70b_Alps/log-todi-nemo-megatronbench_llama3_1_70b_Alps_%j.out --error /store/swissai/a06/.NeMo/FirstExperiment/results/bench_llama3_1_70b_Alps/log-todi-nemo-megatronbench_llama3_1_70b_Alps_%j.err --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID --wait 60 --unbuffered bash -c "
  wandb login XXXXXXXXXXXXXXXXXXXXXXXXXXXX;
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 /users/asolergi/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 /users/asolergi/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/store/swissai/a06/.NeMo/FirstExperiment/results/bench_llama3_1_70b_Alps \
  --config-name=todi-nemo-megatronbench_llama3_1_70b_Alps_hydra.yaml \
  model.gc_interval=100 "

```

Check [temporally] our internal documentation [here](https://docs.google.com/document/d/1q55v0vj6PIYh7rPulnA9pDMl6PUFwHj875yIxVIUah0/edit?usp=sharing).

Below is NeMo-Framework-Launcher original README.
------

# NeMo Framework Launcher

The NeMo Framework Launcher is a cloud-native tool for launching end-to-end NeMo Framework training jobs.

Please refer to the [NeMo Launcher Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/launcherguide/index.html) for more information.

The NeMo Framework focuses on foundation model training for generative AI models. 
Large language model (LLM) pretraining typically requires a lot of compute and model parallelism to efficiently scale training.
NeMo Framework includes the latest in large-scale training techniques including:

- Model parallelism
  * Tensor
  * Pipeline
  * Sequence
- Distributed Optimizer
- Mixed precision training
  * FP8
  * BF16
- Distributed Checkpointing
- Community Models
  * LLAMA-2

NeMo Framework model training scales to 1000's of GPUs and can be used for training LLMs on trillions of tokens.

The Launcher is designed to be a simple and easy to use tool for launching NeMo FW training jobs
on CSPs or on-prem clusters. The launcher is typically used from a head node and only requires
a minimal python installation.

The Launcher will generate and launch submission scripts for the cluster scheduler and will also organize 
and store jobs results. Tested configuration files are included with the launcher but anything
in a configuration file can be easily modified by the user.

The NeMo FW Launcher is tested with the [NeMo FW Container](https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training) which can be applied for [here](https://developer.nvidia.com/nemo-framework).
Access is automatic. 
Users may also easily configure the launcher to use any container image that they want to provide.

The NeMo FW launcher supports:
- Cluster setup and configuration
- Data downloading, curating, and processing
- Model parallel configuration
- Model training
- Model fine-tuning (SFT and PEFT)
- Model evaluation
- Model export and deployment


Some of the models that we support include:
- GPT
  * Pretraining, Fine-tuning, SFT, PEFT
- BERT
- T5/MT5
  * PEFT, MoE (non-expert)

See the [Feature Matrix](https://docs.nvidia.com/nemo-framework/user-guide/latest/featurematrix.html#gpt-models) for more details.


## Installation

The NeMo Framework Launcher should be installed on a head node or a local machine in a virtual python environment.

```bash
git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git
cd NeMo-Framework-Launcher
pip install -r requirements.txt
```

## Usage

The best way to get started with the NeMo Framework Launcher is go through 
the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)

After everything is configured in the `.yaml` files, the Launcher can be run with:

```bash
python main.py
```

Since the Launcher uses [Hydra](https://hydra.cc/docs/intro/), 
any configuration can be overridden directly in the `.yaml` file or via the command line.
See Hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information. 

## Contributing

Contributions are welcome!

To contribute to the NeMo Framework Launcher, simply create a pull request with the changes on GitHub.
After the pull request is reviewed by a NeMo FW Developer, approved, and passes the unit and CI tests, 
then it will be merged.

## License

The NeMo Framework Launcher is licensed under the [Apache 2.0 License](https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/master/LICENSE)
