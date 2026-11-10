#!/bin/bash
# LoRA fine-tuning on one V100 (fp16; no bf16 on this GPU). Submit from the training/ folder:
#   sbatch hpc/train.slurm data/ runs/r1
# Adjust the partition, account and module names to the cluster.
#SBATCH --job-name=braino-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATA=${1:?training data folder}
OUT=${2:?output folder}

module load python/3.12 cuda/12.4 2>/dev/null || true
if [ ! -d .venv ]; then
  python -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
  .venv/bin/pip install -r requirements.txt
fi
export HF_HOME=${HF_HOME:-$PWD/.hf}

.venv/bin/python render.py "$DATA/train.jsonl" --limit 50
.venv/bin/python train_lora.py --data "$DATA" --out "$OUT"
.venv/bin/python merge.py --adapter "$OUT/adapter" --out "$OUT/merged"
