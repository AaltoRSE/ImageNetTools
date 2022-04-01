#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=ShardTest.out

module load miniconda
source activate shardProcess

cp $1 /tmp/ToShard.tar

srun python dataset_sharding -c shardImageNetTrain

cp /tmp/Sharded/* $2

