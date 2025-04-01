# Distributed Training for Mixture of Expert Report

### Introduction

Mixture of Expert is the technique that uses many sub small models, we call as experts in order to improve the quality of the LLM model. With the development of the LLM, and the scaling in the number of tokens and the vocab size, which can increase the computing cost extensively, the Mixture-of-Expert (MoE) is the suitable way to help scale down the context length of the model. Previous works propose the approach to integrate the Mixture of Expert to the weight of the Feedforward layer, and also the Attention weight, and gain some promissing results. In this research work, in order to understand the current MoE application in LLM and the computing resources requirements for the MoE-LLM, we do some evaluation for the distributed training of the MoE model on 3 hardware types are A100, H100, and TPUs. Additionally, we also do some interpretion for the model to explore the effect of the different experts and their correlation in each knowledge domain. 

My main contribution in this work includes three folds:

- Explore the effect of the Mixture of Expert in the training LLM model.

- I do the benchmark of the distributed training of the LLM-MoE model on different devices (test on LLama-MoE)

- I do the interpretation to explore the effect of the different experts and their correlation in each knowledge domain. 

The remaining of this reports is presented as following:

2) Related Work

3) Experimental Detail

4) Experimental Results and Analysis

5) Conclusion and Discussion

### Related Work

In this research

### Experimental Detail

### Experimental Results and Analysis

### Conclusion and Discussion
