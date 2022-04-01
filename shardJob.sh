#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=ShardTest.out
#SBATCH -p dgx-common,gpu


module load miniconda
source activate shardProcess

cp $1 /tmp/ToShard.tar
cp $2 /tmp/meta.mat
srun python dataset_sharding.py -c shardImageNetTrain

cp /tmp/Sharded/* $2

