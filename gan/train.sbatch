#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH -t48:00:00
#SBATCH --output=slurm_train%j.out

# Run.A
PYTHONPATH=$PYTHONPATH:. python deep_gan.py
#    --model_type basic \
#    --batch_size 8 \
#    --learning_rate 0.0005 \
#    --min_lr 1e-6 \
#    --num_epochs 1000 \
#    --n_classes 1000 \
#    --data_path /scratch/zh1115/dsga1008/ssl_data_96 \
#    --save_dir saved_models/
