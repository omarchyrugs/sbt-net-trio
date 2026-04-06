#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=SBT_Net_Trio       # Name of the job in squeue
#SBATCH --output=logs/%j_train.out   # Standard output log (%j = JobID)
#SBATCH --error=logs/%j_train.err    # Error log
#SBATCH --time=24:00:00              # Max walltime (24 hours)
#SBATCH --nodes=1                    # Run on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # CPU cores for data loading
#SBATCH --mem=64G                    # RAM (increased for 190+ .pt files)
#SBATCH --gres=gpu:1                 # Request 1 GPU (A100/V100/RTX)
#SBATCH --partition=gpu              # Ensure it goes to the GPU partition

# --- Environment Setup ---
# Load modules if your HPC requires them (e.g., module load cuda/12.1)
# module load cuda

# Activate your virtual environment
module load cuda-11.8.0-gcc-8.5.0-o55wffj
source /scratch/dipanjan/rugraj/DIAC-WOZ/sbt-net-trio/.venv/bin/activate

# --- Execution ---
# We use 'python -u' for unbuffered output so logs update in real-time
python -u main.py \
    --data_dir "/scratch/dipanjan/rugraj/DIAC-WOZ/processed_data/packets" \
    --job_name "DAIC_Full_Clinical_RunV1" \
    --log_file "stbNet-clinical_results_log.csv" \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-5 \
    --folds 5 \
    --pos_weight 3.0 \
    --weight_decay 0.01

# --- Cleanup ---
# Deactivate environment after finish
deactivate