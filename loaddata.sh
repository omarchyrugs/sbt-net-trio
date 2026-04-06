#!/bin/bash
#SBATCH --job-name=daic_preprocess
#SBATCH -p gpu_a100_8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --time=12:00:00        
#SBATCH --cpus-per-task=4      
#SBATCH --mem=64G
# Load your environment
module load cuda-11.8.0-gcc-8.5.0-o55wffj
source /scratch/dipanjan/rugraj/DIAC-WOZ/sbt-net-trio/.venv/bin/activate

# Run the script
python data_preprocessing.py \
    --base-dir "/scratch/dipanjan/rugraj/DIAC-WOZ" \
    --labels-csv "/scratch/dipanjan/rugraj/DIAC-WOZ/sbt-net-trio/labels.csv" \
    --output-dir "/scratch/dipanjan/rugraj/DIAC-WOZ/processed_data" \
    --chunk-duration 30 \
    --target-packet-length 128 
