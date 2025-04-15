#!/bin/bash
#SBATCH --job-name=dist_bert     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=2                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal       # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024     # only require for multiple projects

export JAVA_HOME=/home/yzhengbs/java/jdk8
export PATH=$JAVA_HOME/bin:$PATH
export PATH=~/mongodb/bin:$PATH
export LD_LIBRARY_PATH=~/my_libs:$LD_LIBRARY_PATH

module load cuda12.2/toolkit/12.2.2
source ~/anaconda3/bin/activate 5003
mkdir mongodb
mongod --dbpath ./mongodb --port 27017 --fork --logpath ./mongodb/train.log

srun python train.py