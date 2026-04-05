#!/bin/bash
#SBATCH --job-name=daic_preprocess
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --time=12:00:00        # DAIC-WOZ takes ~4-6 hours on a good GPU
#SBATCH --gres=gpu:1           # You need 1 GPU for the Transformers
#SBATCH --cpus-per-task=4      # For Pandas merging
#SBATCH --mem=32G              # Safety margin for audio loading

# Load your environment
module load cuda-11.8.0-gcc-8.5.0-o55wffj
source activate /Users/gurusai/programming/agentiAI/MoodDisorders/sbt-net-trio/.venv/bin/activate

# Run the script
python data_preprocessing.py \
    --base-dir "/Users/gurusai/Desktop/DAIC_Raw" \
    --labels-csv "/Users/gurusai/programming/agentiAI/MoodDisorders/sbt-net-trio/labels.csv" \
    --output-dir "/Users/gurusai/programming/agentiAI/MoodDisorders/sbt-net-trio/processed_data" \
    --chunk-duration 30 \
    --target-packet-length 128 \
    --participant-id 301