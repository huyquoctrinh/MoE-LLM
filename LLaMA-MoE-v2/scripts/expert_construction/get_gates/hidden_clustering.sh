#!/usr/bin/bash

#SBATCH --job-name=clustering
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=reserved

# reserved spot auto
# NOTE: This is better to be run on single GPU as the clustering is time-consuming, which may cause the NCCL timeout error!!!!!!!!!!!!!!!!

num_nodes=1        # should match with --nodes
num_cpus=16        # should match with --cpus-per-task
num_gpu_per_node=1 # should match with --gres
export OMP_NUM_THREADS=4
export LOGLEVEL=INFO

{
  # @Desc 此脚本用于获取一个指定区间且未被占用的随机端口号
  # @Author Hellxz <hellxz001@foxmail.com>

  function Listening { #判断当前端口是否被占用，没被占用返回0，反之1
    TCPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l)
    UDPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l)
    ((Listeningnum = TCPListeningnum + UDPListeningnum))
    if [ $Listeningnum == 0 ]; then
      echo "0"
    else
      echo "1"
    fi
  }

  function get_random_port { #得到随机端口
    PORT=0
    while [ $PORT == 0 ]; do
      temp_port=$(shuf -i $1-$2 -n1) #指定区间随机数
      if [ $(Listening $temp_port) == 0 ]; then
        PORT=$temp_port
      fi
    done
    echo "$PORT"
  }

  port=$(get_random_port 29500 29600) #任取一个未占用端口号
  echo "Port: $port"

  nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  echo "Node: $head_node"
  echo "Node IP: $head_node_ip"
}

{
  model_type="llama"
  model_path="/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"
  dataset_dir_or_path="/mnt/petrelfs/huxuyang/push/LLaMA-MoE-v2/resources/OpenHermes-2.5/openhermes2_5.jsonl"

  per_device_train_batch_size=8
  max_steps=10 # the total number of samples shouldn't be too large, as the KMeans is of n^2 complexity
  model_max_length=4096

  echo "Maximum number of possible tokens: $((${num_gpu_per_node} * ${per_device_train_batch_size} * ${max_steps} * ${model_max_length})) (paddings are taken into account here)"

  # 3 4
  # 6 7 8
  # 12 14 16
  # 24 28 32
  num_experts=7
  balance_jitter_factor=0.4 # hyper-parameter for adjusting the cluster size, will affect the initialization of gate weights. (0.0 for strictly balanced, however the performance may be worse.)
  distance_metric="l2"      # l2 cos
  max_iter=100
  random_state=114514
  n_jobs=${num_cpus} # how many different runs will be applied to each clustering process to get a better solution

  output_dir="/mnt/petrelfs/huxuyang/push/LLaMA-MoE-v2/resources/llama_moe_v2/v2_mixtral_gate"
  output_dir="${output_dir}/${num_experts}experts-${balance_jitter_factor}jitter-${distance_metric}"
  save_path="${output_dir}/results"

  srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:$port \
    smoe/entrypoint/expert_construction/get_gates/hidden_feature_clustering.py \
    --model_name_or_path ${model_path} \
    --model_type ${model_type} \
    --dataset_dir_or_path ${dataset_dir_or_path} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --seed ${random_state} \
    --bf16 \
    --max_steps ${max_steps} \
    --model_max_length ${model_max_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --torch_dtype bfloat16 \
    --report_to none \
    --save_path ${save_path} \
    --num_experts ${num_experts} \
    --balance_jitter_factor ${balance_jitter_factor} \
    --distance_metric ${distance_metric} \
    --max_iter ${max_iter} \
    --random_state ${random_state} \
    --n_jobs ${n_jobs}
}
