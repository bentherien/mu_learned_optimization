#!/bin/bash

module load anaconda/3

conda activate env

python -m src.meta_train.meta_train_lagg_full_es
