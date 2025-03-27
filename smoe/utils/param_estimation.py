def estimate_moe_param(
    vocab_size,
    hidden_size,
    num_hidden_layers,
    intermediate_size,
    num_experts,
    num_selects,
):
    """
    Llama Structure with SwiGLU MLP
    MoE split on intermediate_size without weight noise in 2-layer gate
    """

    emb = vocab_size * hidden_size
    lm_head = vocab_size * hidden_size
    final_norm = hidden_size

    self_attn = hidden_size * hidden_size * 4
    mlp = hidden_size * intermediate_size * 3
    input_norm = hidden_size
    post_attn_norm = hidden_size

    dense_one_layer = self_attn + mlp + input_norm + post_attn_norm
    dense_mid = dense_one_layer * num_hidden_layers
    dense_params = emb + lm_head + final_norm + dense_mid

    gate = hidden_size * num_experts + num_experts * num_selects
    moe_one_layer = self_attn + mlp + input_norm + post_attn_norm + gate
    moe_mid = moe_one_layer * num_hidden_layers
    moe_total_params = emb + lm_head + final_norm + moe_mid

    moe_one_act_layer = (
        self_attn
        + (mlp / num_experts * num_selects)
        + input_norm
        + post_attn_norm
        + gate
    )
    moe_act_mid = moe_one_act_layer * num_hidden_layers
    moe_act_params = emb + lm_head + final_norm + moe_act_mid

    return {
        "dense_params": dense_params,
        "moe_total_params": moe_total_params,
        "moe_act_params": moe_act_params,
        "dense_mid": dense_mid,
        "moe_mid": moe_mid,
        "moe_act_mid": moe_act_mid,
    }


def normal_moe_param(
    vocab_size,
    hidden_size,
    num_hidden_layers,
    intermediate_size,
    num_experts,
    num_selects,
    kv_attn_ratio: float = 1.0,
):
    emb = vocab_size * hidden_size
    lm_head = vocab_size * hidden_size
    final_norm = hidden_size

    self_attn = (
        hidden_size * hidden_size * 2 + hidden_size * hidden_size * kv_attn_ratio * 2
    )
    mlp = hidden_size * intermediate_size * 3
    input_norm = hidden_size
    post_attn_norm = hidden_size

    dense_one_layer = self_attn + mlp + input_norm + post_attn_norm
    dense_mid = dense_one_layer * num_hidden_layers
    dense_params = emb + lm_head + final_norm + dense_mid

    gate = hidden_size * num_selects
    moe_one_layer = self_attn + mlp * num_experts + input_norm + post_attn_norm + gate
    moe_one_layer_act = (
        self_attn + mlp * num_selects + input_norm + post_attn_norm + gate
    )
    moe_mid = moe_one_layer * num_hidden_layers
    moe_tot_params = emb + lm_head + final_norm + moe_mid
    moe_act_mid = moe_one_layer_act * num_hidden_layers
    moe_act_params = emb + lm_head + final_norm + moe_act_mid

    return {
        "dense_params": dense_params,
        "moe_tot_params": moe_tot_params,
        "moe_act_params": moe_act_params,
        "dense_mid": dense_mid,
        "moe_mid": moe_mid,
        "moe_act_mid": moe_act_mid,
    }


if __name__ == "__main__":
    # opt-2.7b: 2651596800
    opt = normal_moe_param(50272, 2560, 32, 10240, 1, 1)
    print("opt-2.7b", opt)

    # pythia-2.8b: 2775208960
    pythia = normal_moe_param(50304, 2560, 32, 10240, 1, 1)
    print("pythia-2.8b", pythia)

    # incite-base-3b: 2775864320
    incite = normal_moe_param(50432, 2560, 32, 10240, 1, 1)
    print("incite", incite)

    # 3B: open-llama-3b-v2: 3426473600
    res_3B = estimate_moe_param(32000, 3200, 26, 8640, 16, 4)
    print("3B", res_3B)

    # 7B
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 4)
    print("7B", res_7B)
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 2)
    print("7B 2/16", res_7B)
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 8, 2)
    print("7B 2/8", res_7B)
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 1)
    print("7B 1/16", res_7B)
    res_7B = estimate_moe_param(32000, 2560, 32, 11008, 8, 2)
    print("7B-2560", res_7B)

    # 13B
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 4)
    print("13B 4/16", res_13B)
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 2)
    print("13B 2/16", res_13B)
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 1)
    print("13B 1/16", res_13B)

    # 3B upcycling
    for num_experts in range(1, 9):
        res_3B_up = estimate_moe_param(
            32000, 3200, 26, 8640 * num_experts, num_experts, 1
        )
        print(f"3B upcycling {num_experts} experts", res_3B_up)

    # ShearedLlama-1.3B upcycling
    for num_experts in range(1, 17):
        res_3B_up = estimate_moe_param(
            32000, 2048, 24, 5504 * num_experts, num_experts, 1
        )
        print(f"ShearedLlama-1.3B upcycling {num_experts} experts", res_3B_up)

    # ShearedLlama-2.7B upcycling
    for num_experts in range(1, 17):
        res_3B_up = estimate_moe_param(
            32000, 2560, 32, 6912 * num_experts, num_experts, 1
        )
        print(f"ShearedLlama-2.7B upcycling {num_experts} experts", res_3B_up)

    # 7B, moe half layers
    res_7B_half = estimate_moe_param(32000, 4096, 8, 11008, 16, 2)
    print("7B half 8 layers", res_7B_half)
    res_7B_half = estimate_moe_param(32000, 4096, 24, 11008, 16, 2)
    print("7B half 24 layers", res_7B_half)
    res_7B_half = estimate_moe_param(32000, 4096, 16, 11008, 16, 1)
    print("7B half 16 layers 1/16", res_7B_half)

    # mixtral 7Bx8
    res_mixtral = normal_moe_param(32000, 4096, 32, 14336, 8, 2, kv_attn_ratio=0.25)
    print("mixtral 7Bx8", res_mixtral)
