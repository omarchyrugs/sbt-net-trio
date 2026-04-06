#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=Mood_H200_SBT
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --partition=gpu_h200_8         # <--- EXACT PARTITION NAME FOUND

# --- Environment Setup ---
module load cuda-11.8.0-gcc-8.5.0-o55wffj
source /scratch/dipanjan/rugraj/DIAC-WOZ/sbt-net-trio/.venv/bin/activate

# --- Execution ---
python -u main.py \
    --data_dir "/scratch/dipanjan/rugraj/DIAC-WOZ/processed_data/packets" \
    --job_name "DAIC_Full_Clinical_RunV1_H200" \
    --log_file "stbNet-clinical_results_logv2.csv" \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-5 \
    --folds 5 \
    --pos_weight 3.0 \
    --weight_decay 0.01

# --- Cleanup ---
deactivate