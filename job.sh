#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer adam --task image-mlp --batch_size 2048 --learning_rate 1e-3 --num_inner_steps 5000
