#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=Mood_H200_SBT       # The name in the queue
#SBATCH --output=logs/%j_train.out     # Standard output
#SBATCH --error=logs/%j_train.err      # Error logs
#SBATCH --time=06:00:00                # H200 is fast; 6 hours is more than enough
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # Crank this up for faster data pre-fetching
#SBATCH --mem=128G                     # Beefed up system RAM
#SBATCH --gres=gpu:nvidia_h200_nvl:1   # SPECIFICALLY LOCK IN THE H200
#SBATCH --partition=gpu                # Standard GPU partition

# --- Environment Setup ---
module load cuda-11.8.0-gcc-8.5.0-o55wffj
source /scratch/dipanjan/rugraj/DIAC-WOZ/sbt-net-trio/.venv/bin/activate

# --- Execution ---
# Note: Increased batch_size to 128 and LR to 5e-5 for the H200's capacity
python -u main.py \
    --data_dir "/scratch/dipanjan/rugraj/DIAC-WOZ/processed_data/packets" \
    --job_name "DAIC_Full_Clinical_RunV1_H200" \
    --log_file "stbNet-clinical_results_log.csv" \
    --epochs 20 \
    --batch_size 128 \
    --lr 5e-5 \
    --folds 5 \
    --pos_weight 3.0 \
    --weight_decay 0.01

# --- Cleanup ---
deactivate