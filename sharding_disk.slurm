#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=5G
#SBATCH --output=Sharding_disk.out
#SBATCH -p gpu,gpushort # you might want to add dgx-common, but then your job could be killed.

module load anaconda
srun python3 imagenet_sharding.py -c tritonSharding_disk