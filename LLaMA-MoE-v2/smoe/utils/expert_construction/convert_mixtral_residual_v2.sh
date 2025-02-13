#!/usr/bin/bash

#SBATCH --job-name=convert-residual
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=llm_s
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=reserved


{
  model_path="/mnt/petrelfs/quxiaoye/models/Meta-Llama-3-8B-Instruct"

  moe_implementation_type="modulelist"
  num_experts=7
  top_k=1

  folder_name="8experts-0.4jitter-l2"
  split_folder_name="split-gradient-max-ShareFalse-1Residual-7MoE"
  #  split_folder_name="split-gradient-max-ShareFalse-2Residual-6MoE"

  save_path="/mnt/petrelfs/quxiaoye/models/${split_folder_name}-Top${top_k}"
  neuron_indices_file="/mnt/petrelfs/quxiaoye/models/${split_folder_name}/neuron_indices.pt"
  gate_weights_file="/mnt/petrelfs/quxiaoye/models/gate_weights.pt"

  srun python smoe/entrypoint/expert_construction/convert/convert_mixtral_residual_v2.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --neuron_indices_file ${neuron_indices_file} \
    --gate_weights_file ${gate_weights_file} \
    --moe_implementation_type ${moe_implementation_type} \
    --num_experts ${num_experts} \
    --top_k ${top_k}
}
