dataset_dir_or_path="/home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/dataset/openhermes2_5.jsonl"
model_name_or_path="/home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/results_converted/Llama-3-8B-8expert-MLP-MoE-Top2-Scale4.0-Dense0"
model_type="v2_mixtral"
task_name="test"
base_dir="outputs1/v2_mixtral"
# output_dir= "outputs1/"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    -m smoe.entrypoint.sft.train_sft_llama3_packedpad \
        --do_train \
        --freeze_gate False \
        --evaluation_strategy no \
        --run_name $task_name \
        --model_type $model_type \
        --model_name_or_path $model_name_or_path \
        --dataset_dir_or_path $dataset_dir_or_path \
        --output_dir "outputs1" \
        --deepspeed "/home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/conf/deepspeed/bf16_zero1.json" \
        --seed 1227 \
        --bf16 True \
        --tf32 True \
        --torch_dtype bfloat16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 3 \
        --save_strategy steps \
        --save_steps 200 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --save_only_model True \