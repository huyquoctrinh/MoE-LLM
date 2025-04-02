# Distributed Training for Mixture of Expert Report

### 1) Introduction

Mixture of Expert is the technique that uses many sub small models, we call as experts in order to improve the quality of the LLM model. With the development of the LLM, and the scaling in the number of tokens and the vocab size, which can increase the computing cost extensively, the Mixture-of-Expert (MoE) is the suitable way to help scale down the context length of the model. Previous works propose the approach to integrate the Mixture of Expert to the weight of the Feedforward layer, and also the Attention weight, and gain some promissing results. Although the promissing of previous works, the chosen of the training resources, and the effect of the experts in the overall LLM models are not explored, which can make the difficult for the clients to choose the suitable computing resources for training a LLM-MoE model, or choosing the suitable parameter for the training, which can also ensure the quality and their training budget. For this reason, in this research work, in order to understand the current MoE application in LLM and the computing resources requirements for the MoE-LLM, we do some evaluation for the distributed training of the MoE model on 3 hardware types are A100, H100, and TPUs. Additionally, we also do some interpretion for the model to explore the effect of the different experts and their correlation in each knowledge domain. 

My main contribution in this work includes three folds:

- Explore the effect of the Mixture of Expert in the training LLM model.

- I do the benchmark of the distributed training of the LLM-MoE model on different devices (test on LLama-MoE)

- I do the interpretation to explore the effect of the different experts and their correlation in each knowledge domain. 

The remaining of this reports is presented as following:

2) Related Work

3) Experimental Detail

4) Experimental Results and Analysis

5) Conclusion and Discussion

6) Future work

### 2) Related Work

In this research, I conduct the experiments based on the Llama-MoE-V2 architecture, which is proposed by Qu et al. in the paper LLaMA-MoE v2: Exploring Sparsity of LLaMA from Perspective of Mixture-of-Experts with Post-Training at ACL 2024. For the overal architecture, it is showed below:

<div align="center">
    <a href="./">
        <img src="imgs/llama_moev2.jpg" width="79%"/>
    </a>
</div>

Where the experts are integrated to the attention weight, and also in the MLP layer of the Feed Forward module.

Regarding the flow, the input text tokens is embedded as the token embedding, then through the normalization, the attention experts, and feed forward experts, it will produce the output embedding for the answers, which can be then post-processed and detokenized to the final answer in text.

During the training, firstly, we need to convert the original into the mixture of expert model. The weight of the model will be copied, however, with the attention weight, the weight for key, query and value will be splitted into n experts with the initialization follow the Xavier method, and a router is added before that. Similarly, in the feed forward layers, the feed forward weight will be converted in to n experts with the xavier initialization and the router. From this, we will do the supervised fine-tuning in order to do the tuning for all experts weight and the activated weight of the model, thus make these weight updated follow the distribution of the training data.

### 3) Experimental Detail

#### 3.1) Datastet
To conduct the experiment, we set the training experiments on the [OpenHermes2.5 dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5/tree/main), this dataset consists of 1001551 samples of conversations primarily synthetically generated from instruction and chat samples, following the ShareGPT structure. This dataset is combined across different data sources, and it is used to train the LLama-2 model.

#### 3.2) Hyperparameter

To benchmark the training process, we conducted the fully supervised Finetuning experiments on the LLama-MoE-V2 model, which is activated 2 experts over 8 experts from the LLama-3 8B model. The activated parameters of this model is about 3.8B. For the training, we use the Adam optimizer, with a learning rate is about 1e-4, and the batch size per gpu is 1. For the training, data, the weight of the model, gradient, and momentum will be stored on the GPUs for the calculation. The max token lengths in the experiments is set at about 2048 token lengths per sample to facilitate the training on our small amount of GPUs. 

For the computing resources, we did experiments on four settings are 4 H100 GPUs, 4 A100 GPUs, 2 A100 GPUs, and 2 H100 GPUs, all of the hardware has a virtual ram of about 80GB.


### 4) Experimental Results and Analysis

#### 4.1) Distributed training results
From the experiments, we did some benchmarks related of the distributed training of the LLama-MoE-V2 on different GPUs. The table below shows the evaluation of the results in the different GPUs setting

| GPUs Usage | Number of steps | Training time per step | Total training time | Average GPU Utilization | Average GPU memory usage|
| ------------- | ------------- |  ------------- |  ------------- |  ------------- |  ------------- |
| 4 x H100 80GB   | 20000  | 5.25s | About 29 hours |  ~ 82% | 73GB |
| 4 x A100 80GB  | 20000 | 8s | About 42 hours | ~ 100% | 73GB |
| 2 x A100 80GB  | 20000 | OOM | None | OOM | OOM |
| 2 x A100 80GB  | 20000 | OOM | None | OOM | OOM |

From the benchmark of the distributed training results, we can observe that, with the requirements to train an MoE model with about 3.8B activated parameters the number of gpus higher than 2 two do the distributed of the model weight, gradient, and also momentum across gpus. From this observation, we can claim that, with the company or startup, if they want to conduct the training experiment with the model MoE-LLM, it is necessary to prepare the server with at lease 4 GPUs of 80GB VRAM to support the training.

#### 4.2) Interpretation results

### 5) Conclusion and Discussion

### 6) Future work
