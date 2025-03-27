#!/usr/bin/bash

#SBATCH --job-name=cpt-7b-test
#SBATCH --output=logs-cpt/%x-%j.log
#SBATCH --error=logs-cpt/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=spot

# reserved spot

source ~/anaconda3/bin/activate llama-moe

{
  num_nodes=1        # should match with --nodes
  num_gpu_per_node=4 # should match with --gres

  # #cpu/#num_gpu_per_node
  export OMP_NUM_THREADS=2
  export LOGLEVEL=INFO
  #  export NCCL_DEBUG=INFO
  #  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  #  export TORCH_SHOW_CPP_STACKTRACES=1
  #  export CUDA_LAUNCH_BLOCKING=1

  model_type="llama_moe_residual"
  pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEResidualForCausalLM/Gradient-max-l1_norm-sample-feature_change/llama_7B-14Select2-2Residuals-688Neurons-Share
  comment="llama 7B residual, gradient, 2 + 2/14 | hard share | soft moe, scale 2.0 | GPU num 1, per-device bs 64, lr 1e-4"

  tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
  dataset_dir=/mnt/petrelfs/share_data/quxiaoye/pretrain_LLAMA_all_data_processed

  lr=1e-4
  final_lr_portion=0.1
  per_device_train_batch_size=8
  per_device_eval_batch_size=1
  gradient_accumulation_steps=8
  block_size=2048
  num_tokens="1*10^11"
  seed=1227
  deepspeed_config_file=conf/deepspeed/bf16_zero1_default.json

  max_steps=$(echo "${num_tokens} / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
  max_train_samples=$(echo "${num_tokens} / $block_size" | bc)
  echo "max_steps: $max_steps"
  echo "max_train_samples: $max_train_samples"
  global_bs=$(echo "$per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node" | bc)
  echo "global batch size: $global_bs"
  tokens_per_batch=$(echo "$global_bs * $block_size" | bc)
  echo "#tokens/batch: $tokens_per_batch"

  data_cache=resources/cache
  output_dir=outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
  mkdir -p $output_dir
  echo "output_dir: $output_dir"
  scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
  git diff >$output_dir/diff.patch
  env >$output_dir/.env
  echo $comment >$output_dir/comment.txt

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  echo "Node: $head_node"
  echo "Node IP: $head_node_ip"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    smoe/entrypoint/cpt/cpt_fpt.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --model_type ${model_type} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed ${seed} \
    --bf16 \
    --num_train_epochs 1 \
    --final_lr_portion ${final_lr_portion} \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --learning_rate ${lr} \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 2000 \
    --max_steps ${max_steps} \
    --max_train_samples ${max_train_samples} \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 1000 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --ddp_find_unused_parameters False \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --logging_first_step True \
    --logging_strategy steps \
    --logging_steps 10 \
    --log_level info \
    --log_level_replica warning \
    --log_on_each_node False \
    --report_to none
}
#SBATCH --job-name=cpt-moe-fpt-test_lr_change
#改动前：--logging_steps 10 \
#改动后：--logging_steps 1 \
#改动前：没有--resume_from_checkpoint outputs/cpt-moe-fpt-test_lr_change-1700831/checkpoint-10
#改动后：有--resume_from_checkpoint outputs/cpt-moe-fpt-test_lr_change-1700831/checkpoint-10
#改动前：max_steps=$(echo "10^11 / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
#改动后：max_steps=20
#改动前：output_dir=outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID
#改动后：output_dir=outputs/cpt-moe-fpt-test_lr_change-1700831
#改动前：--save_steps 1000 \
#改动后：--save_steps 10 \1
