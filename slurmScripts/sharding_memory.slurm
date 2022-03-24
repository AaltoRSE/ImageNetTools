#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --output=Sharding_memory.out

module load anaconda
srun python3 imagenet_sharding.py -c tritonSharding_memory