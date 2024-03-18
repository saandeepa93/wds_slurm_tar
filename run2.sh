#!/bin/bash
#SBATCH --job-name=train_base
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/saandeepaath/create_bp4d_chips/slurm/output.%j
#SBATCH --partition=Extended
#SBATCH --time=2-00:00:00

source activate webds
NCCL_DEBUG=INFO srun python ./playground.py
# find /data/scanavan/BP4D/Sequences/ -type f -name '*.jpg' -printf "%s\n" | awk '{total += $1} END {print total}' | numfmt --to=iec
