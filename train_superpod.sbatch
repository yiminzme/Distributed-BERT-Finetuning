#!/bin/bash
#SBATCH --job-name=dist_bert     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=4                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal       # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024     # only require for multiple projects

export JAVA_HOME=/home/yzhengbs/java/jdk8
export PATH=$JAVA_HOME/bin:$PATH
export PATH=~/mongodb/bin:$PATH
export LD_LIBRARY_PATH=~/my_libs:$LD_LIBRARY_PATH

module load cuda12.2/toolkit/12.2.2
# source ~/anaconda3/bin/activate 5003
# source ~/anaconda3/condabin/conda init
# rm -rf processed_data/*
rm -rf ./mongodb/*
mkdir mongodb
mongod --dbpath ./mongodb --port 27017 --fork --logpath ./mongodb/train.log

num_cpus=96
num_gpus=4
num_train_samples=1000
batch_size=8
epoches=3
echo "Current time: $(date +%Y%m%d_%H%M%S), num_cpus[$num_cpus], num_gpus[$num_gpus], num_train_samples[$num_train_samples], batch_size[$batch_size], epoches[$epoches]"

srun python -u train.py --num_cpus $num_cpus --num_gpus $num_gpus --num_train_samples $num_train_samples --batch_size $batch_size --epoches $epoches