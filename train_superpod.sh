# conda init
# conda activate 5003
current_time=$(date +%Y%m%d_%H%M%S)

sbatch --wait -o out/train_$current_time.out train_superpod.sbatch