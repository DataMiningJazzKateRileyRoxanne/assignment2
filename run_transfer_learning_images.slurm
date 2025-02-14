#!/bin/bash
#SBATCH --job-name=torch-t1-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

Set up environment
source ~/software/init-conda
conda activate pytorch

Call your script
python ~/transfer_learning_garbage_images.ipynb

Deactivate conda
conda deactivate