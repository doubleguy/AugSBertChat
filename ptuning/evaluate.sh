PRE_SEQ_LEN=128
CHECKPOINT=task2-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file ../datasets/processed/qa_dev_task2.json \
    --test_file ../datasets/processed/qa_train_task2.json \
    --overwrite_cache \
    --prompt_column query \
    --response_column replies \
    --model_name_or_path /home/zb/ChatGLM-6B-main/chatglm-6b-int4 \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
