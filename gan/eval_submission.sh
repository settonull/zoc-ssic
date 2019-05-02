#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH -t1:00:00
#SBATCH --output=slurm_train%j.out
python eval.py --data_dir /scratch/zh1115/dsga1008/ssl_data_96/

