# Welcome to $\mu$ Learned Optimization!

This repository contains the research code for [μLO: Compute-Efficient Meta-Generalization of Learned Optimizers](https://arxiv.org/abs/2406.00153).

# Installation

Run the following code:
```
mkdir l2o_install
cd l2o_install

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh
bash Miniconda3-py39_24.5.0-0-Linux-x86_64.sh -b -p $PWD/miniconda3
source $PWD/miniconda3/bin/activate

git clone https://github.com/bentherien/mu_learned_optimization
cd mu_learned_optimization
pip install -r requirements.txt


cd ..
git clone https://github.com/lefameuxbeding/learned_optimization
cd learned_optimization
git checkout mup_compatible
pip install -e .

cd ..
git clone https://github.com/google-research/vision_transformer
cd vision_transformer
git checkout ac6e056
pip install -e .


cd ../mu_learned_optimization
pip install mmengine seqio wandb
pip install -U dm-haiku chex flax
pip install optax==0.1.7
pip install "jax[cuda12]==0.4.26"
conda install -c conda-forge openmpi=4.1.2

# change the following as is appropriate for your environment
export TFDS_DATA_DIR=/scr/data/tensorflow_datasets
export WANDB_DIR=$PWD/wandb
```

# Quickstart

# Meta-training Quickstart
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_train/schedules/mxlr=3e-3_mnlr=1e-3_it=5000_clip.py \
--num_tasks 8 \
--local_batch_size 4096 \
--train_project mup-meta-training \
--optimizer mup_small_fc_mlp \
--needs_state \
--steps_per_jit 2 \
--name_suffix _s-mumlp_it=5000_mxlr=3e-3_stepm=01_tasks=8 \
--task mumlp-w128-d3_imagenet-32x32x3 \
--prefetch_batches 20 \
--adafac_step_mult 0.01

CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_train/schedules/mxlr=3e-3_mnlr=1e-3_it=5000_clip.py \
--num_tasks 8 \
--local_batch_size 4096 \
--train_project mup-meta-training \
--optimizer mup_small_fc_mlp \
--needs_state \
--steps_per_jit 2 \
--name_suffix _m-mumlp_it=5000_mxlr=3e-3_stepm=01_tasks=8 \
--prefetch_batches 20 \
--adafac_step_mult 0.01 \
--task mumlp-w1024-d3_imagenet-32x32x3,mumlp-w512-d3_imagenet-32x32x3,mumlp-w128-d3_imagenet-32x32x3 \
--auto_resume
```




# Testing Quickstart


## Test VeLO
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
--name_suffix _m_mup_final \
--local_batch_size 128 \
--test_project mup-meta-testing \
--task mutransformer-w2048-d3_lm1b-s64-v32k \
--optimizer mup_small_fc_mlp \
--wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0 \
--num_runs 5 \
--num_inner_steps 5000 \
--needs_state \
--adafac_step_mult 0.01 \
--gradient_accumulation_steps 1 \
--test_interval 100
```

## Test MuLO
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
--name_suffix _m_mup_final \
--local_batch_size 128 \
--test_project mup-meta-testing \
--task mutransformer-w2048-d3_lm1b-s64-v32k \
--optimizer mup_small_fc_mlp \
--wandb_checkpoint_id <WANDB PATH TO CHECKPOINT> \
--num_runs 5 \
--num_inner_steps 5000 \
--needs_state \
--adafac_step_mult 0.01 \
--gradient_accumulation_steps 1 \
--test_interval 100 \
--use_bf16
```

## Sweep MuAdam Leerning Rates
```
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --config config/sweeps/sweep_muadam.py \
    --name_suffix _muadam_sweep \
    --local_batch_size 4096 \
    --test_project mup-meta-testing \
    --task "mumlp-w1024-d3_imagenet-32x32x3" \
    --optimizer muadam \
    --num_runs 1 \
    --learning_rate 3e-4 \
    --num_inner_steps 1000 \
    --gradient_accumulation_steps 1 \
    --needs_state \
    --mup_input_mult 1 \
    --mup_output_mult 1 \
    --mup_hidden_lr_mult 1 \
    --test_interval 50 \
```



# Optimizing Meta-Training Programs
Tweaking the hyperparameters of a meta-training program can have a significant impact on the iteration speed and memory consumption of meta-training. Here are some tips for optimizing meta-training programs:

## Optimizing meta-training memory usage when using `custom_preload_tfds_image_classification_datasets`
- `prefetch_batches`: when using this preloading function, each task keeps a buffer of samples on the GPU to avoid waiting for CPU-GPU transfers during meta-training. The prefect_batches variable therefore controls how much GPU memory will be used by the buffer.


## Opitmizing meta-training efficiency and memory usage
- `num_tasks`: this variable controls the number of perturbations to the learned optimizer's weights sampled in the gradient estimator. During meta-training, 1 unroll is performed for each perturbation (2 if antethetic sampling is used). 1 Batch of data is required per optimization step per task.
- `steps_per_jit`: This variable controls the number of unrolling steps performed within the jitted unroll_step function. Since data cannot be sampled within jitted functions, this has the effect of also specifying the amount of data we are required to load before each call to `unroll_step`. The total amount of data is `steps_per_jit * num_tasks * batch_size`. If antithetic sampling is used (as is the case for PES), this quantity should be multiplied by 2. 


# Config file structure

Using MMengine's config file parser, we can write config files directly in Python and use an inheritance config structure to avoid redundant configurations. This can be achieved by specifying config files to inherit from using the 
```_base_=['my_config.py']``` 
special variable at the top of config files. More information is available at [mmengine config docs](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html).

In learned_aggragation, configuration files are logically separated into different directories based on the task to be executed: ```config/meta_test```,```config/meta_train```, and ```config/sweeps```. 

# Setting up a sweep
To sweep over the hyperparameters of a model during meta-testing, one can simply specify a sweep configuration using the ```sweep_config``` variable.


# Checkpointing during meta training
The ```checkpoints_to_keep``` and ```save_iter``` config variables control the number of checkpoints that should be kept and the checkpointing multiple, respectively. Default values of ```checkpoints_to_keep=10``` and ```save_iter=1000``` ensure that at most 10 previous checkpoints will be kept and that a checkpoint will be saved every 1000 iterations.

# Loading from a checkpoint during meta training
When a checkpoint is logged, it is saved under ```checkpoints/<meta-train-dir>``` where ```<meta-train-dir>``` is the dynamically assigned meta-train-name. Whenever a new checkpoint is logged, a file called ```latest``` is updated with the name of the most recent checkpoint. When resuming from a checkpoint the user simply has to set the ```--from_checkpoint``` flag and meta training will automatically resume to the checkpoint specified in the ```latest``` file.
