#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export CUDA_VISIBLE_DEVICES=0

python src/train.py model=mlf data=mir1k
