#!/bin/bash

#SBATCH --account=robinjia_1152
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
module spider cuda
module spider cudnn
module load nvhpc/22.11
eval "$(conda shell.bash hook)"
conda activate 467

python gpu_check.py