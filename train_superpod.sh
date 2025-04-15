conda init
conda activate 5003
current_time=$(date +%m%d%H%S)

sbatch --wait -o out/train_$current_time.out train.sbatch