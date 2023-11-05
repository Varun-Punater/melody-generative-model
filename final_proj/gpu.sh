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
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load nvhpc/22.11
eval "$(conda shell.bash hook)"
conda activate 467

python gpu_check.py