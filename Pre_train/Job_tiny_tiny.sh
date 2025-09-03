#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=Kindey
#SBATCH --mail-user=XXX
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=200:00:00
#SBATCH --account=xxx
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1


torchrun --nproc_per_node=1 main.py --model mae_vit_tiny_tiny_patch16 --output_dir XXX/output_dir_tiny_tiny_augment --log_dir XXX/Kindey/Pre_train/V1/output_dir_tiny_tiny_augment
