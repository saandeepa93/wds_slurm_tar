#!/bin/bash
#SBATCH --job-name=train_base
#SBATCH --mem=300gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/saandeepaath/create_bp4d_chips/slurm/output.%j
#SBATCH --partition=Extended
#SBATCH --time=2-00:00:00

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 

### Save Pickle Info
# source activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
# NCCL_DEBUG=INFO srun python ./trainer/save_pickles.py

### Extract Face Chips
# source activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
# NCCL_DEBUG=INFO srun python ./trainer/extract_chips.py --config bp4dp_1

### Extract OpenFace Features
# source activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
# NCCL_DEBUG=INFO srun python ./trainer/extract_openface.py --config bp4dp_1

### Train Model
source activate webds
NCCL_DEBUG=INFO srun python ./trainer/train_wds.py --config bp4d_3
