model_path="/home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/pretrained"

num_experts=8
top_k=2
scale_factor=4.0   # we suggest this value to be 4.0 for 8 experts
num_moe_contract_layers=0
moe_implementation_type="modulelist"

folder_name="${num_experts}experts-0.4jitter-l2"
split_folder_name="${num_experts}expert-MLP-MoE"
save_folder_name="Llama-3-8B-${split_folder_name}-Top${top_k}-Scale${scale_factor}-Dense${num_moe_contract_layers}"

save_path="/home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/results_converted/${save_folder_name}"

python /home/mamba/ML_project/Testing/Huy/test_llm/LLaMA-MoE-v2/smoe/entrypoint/expert_construction/convert/convert_mixtral_v2.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --num_experts ${num_experts} \
    --top_k ${top_k} \
    --scale_factor ${scale_factor} \
    --num_moe_contract_layers ${num_moe_contract_layers} \
    --moe_implementation_type ${moe_implementation_type}