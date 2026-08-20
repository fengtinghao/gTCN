#!/bin/bash
#SBATCH --job-name=gttcn-train
#SBATCH --partition=cs-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/faculty/fengt/gTCN/logs/%x-%j.out
#SBATCH --error=/hpc/faculty/fengt/gTCN/logs/%x-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gttcn-py311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /hpc/faculty/fengt/gTCN

echo "Host: $(hostname)"
nvidia-smi
echo

python src/train.py --config configs/default.yaml
