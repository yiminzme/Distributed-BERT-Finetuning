current_time=$(date +%m%d%H%S)

# mkdir mongodb
# mongod --dbpath ./mongodb --port 27017 --logpath ./mongodb/train.log

python -u train.py >> out/train_$current_time.out 2>&1