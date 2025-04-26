
# mkdir mongodb
# mongod --dbpath ./mongodb --port 27017 --logpath ./mongodb/train.log
# mongod --dbpath ./mongodb
# rm -rf mongodb/*

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 1 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 2 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 4 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 6 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 8 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 10 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 12 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 14 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 16 --num_gpus 1 --num_train_samples 10000 --batch_size 16 --epoches 10 >> out/train_N_1_10000_$current_time.out 2>&1






current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 1 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 2 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 4 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 6 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 8 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 10 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 12 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 14 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1

current_time=$(date +%Y%m%d_%H%M%S)
python -u train.py --num_cpus 16 --num_gpus 1 --num_train_samples 40000 --batch_size 16 --epoches 10 >> out/train_N_1_40000_$current_time.out 2>&1
