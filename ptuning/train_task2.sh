PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file ../datasets/processed/qa_train_task2.json \
    --validation_file ../datasets/processed/qa_dev_task2.json \
    --prompt_column query \
    --response_column replies \
    --overwrite_cache \
    --model_name_or_path ../chatglm-6b-int4 \
    --output_dir output/task1-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 4000 \
    --logging_steps 10 \
    --save_steps 4000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

