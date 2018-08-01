#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -t 10:00:00
#SBATCH --mem=128G

source ~/environment/saion/gpu_saion/bin/activate
python3 ~/SRGAN/srgan.py
