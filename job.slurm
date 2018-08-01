#!/bin/bash -l
#SBATCH -p=gpu
#SBATCH --gres=gpu:1
#SBATCH -t=10:00:00
#SBATCH --mem=12G

source ~/environment/saion/gpu_saion/bin/activate

~/SRGAN/srgan.py