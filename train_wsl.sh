current_time=$(date +%Y%m%d_%H%M%S)

# mkdir mongodb
# mongod --dbpath ./mongodb --port 27017 --logpath ./mongodb/train.log
# mongod --dbpath ./mongodb
# rm -rf mongodb/*

python -u train.py --num_cpus 4 --num_gpus 1 >> out/train_$current_time.out 2>&1