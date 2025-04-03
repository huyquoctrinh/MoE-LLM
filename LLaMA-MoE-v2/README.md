## LLama-MoE-v2 Reproduce

This is the implementation for the Llama-MoE-V2 with the differents  initialization Kaiming Hee/Xavier.

### Installation

To setup the environment and the required packages to run the source code, we need to do as following:

- Create the conda environment via ```conda create -n llama_moe python=3.10.0```

- Install required packages in the ```requirements.txt``` file via this command:

```
pip install -r requirements.txt
```

- After installing all of the requirements of the repository, you need to install the ```flash_attn```, which is the flash attention package for the optimization. You need to install via this command ```pip install flash-attn==0.2.4```. This will support your installation

### Training

For the training of the model, you need do the following stages:

- Stage 1: Convert the expert for the FeedForward modules via the following file ```scripts/expert_construction/convert/convert_mixtral_v2.sh```

- Stage 2: After converting the model to MoE, you need to do the expert construction via ```scripts/expert_construction/convert/convert_mixtral_attn_moe.sh```

- Stage 3: Prepare dataset, please convert the json dataset to the jsonl dataset via the following implementation:

``` python
from smoe.utils.io import load_json, dump_jsonlines

data = load_json("resources/OpenHermes-2.5/openhermes2_5.json")
dump_jsonlines(data, "resources/OpenHermes-2.5/openhermes2_5.jsonl")
```

- Stage 4: After preparing dataset, you need to do the training via the following command:

```
bash scripts/sft/sft_8e_top2_base_res_pad.sh
```

Then the model will run and save to the ```outputs/``` folder


### Acknowledgement

This work is based on [Llama-MoE-V2](https://github.com/OpenSparseLLMs/LLaMA-MoE-v2/tree/main) model, thanks to the authors to publish it
