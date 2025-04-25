
# mkdir mongodb
# mongod --dbpath ./mongodb --port 27017 --logpath ./mongodb/train.log
# mongod --dbpath ./mongodb
# rm -rf mongodb/*

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 4 --num_gpus 1 --num_train_samples '-1' --batch_size 16 --epoches 20 >> out/train_$current_time.out 2>&1

# current_time=$(date +%Y%m%d_%H%M%S)
# python -u train.py --num_cpus 4 --num_gpus 1 --num_train_samples 256 >> out/train_$current_time.out 2>&1

# current_time=$(date +%Y%m%d_%H%M%S)
# python -u train.py --num_cpus 4 --num_gpus 1 --num_train_samples 1024 >> out/train_$current_time.out 2>&1