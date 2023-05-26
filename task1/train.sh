CUDA_VISIBLE_DEVICES=0 python -u main.py \
-per_device_train_batch_size 16 \
-per_device_eval_batch_size 16 \
-num_train_epochs 50 \
-learning_rate 1e-3 \
