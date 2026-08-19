#!/bin/bash
#SBATCH --job-name=gttcn-check
#SBATCH --partition=cs-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gttcn-py311

echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo
nvidia-smi
echo

python - <<'PY'
import torch
import torchaudio

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "PyTorch cannot access the requested GPU."

gpu = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print("GPU:", gpu)
print("Compute capability:", f"{props.major}.{props.minor}")
print("Memory (GiB):", round(props.total_memory / 1024**3, 2))
print("torchaudio:", torchaudio.__version__)

x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA matrix multiplication: OK")
PY