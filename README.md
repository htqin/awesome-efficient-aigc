# Awesome Efficient AIGC [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo collects efficient approaches for AIGC (AI Generated Content) to cope with its huge demand for computing resources. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. Special thanks to [Xingyu Zheng](https://github.com/Xingyu-Zheng), [Yifu Ding](https://yifu-ding.github.io/#/), and all researchers who have contributed to this project!

## Table of Contents

  - [Table of Contents](#table-of-contents)
  - [Language](#language)
    - [2023](#2023)
    - [2022](#2022)
    - [2021](#2021)
    - [2020](#2020)
    - [2019](#2019)
  - [Vision](#vision)
    - [2023](#2023-1)

## Language

### 2023

- [[Nature](https://www.nature.com/articles/s42256-023-00626-4)] Parameter-efficient fine-tuning of large-scale pre-trained language models [[code](https://github.com/thunlp/OpenDelta)]
- [[ACL](https://aclanthology.org/2023.findings-acl.441/)] Distilling Reasoning Capabilities into Smaller Language Models
- [[ACL](https://aclanthology.org/2023.acl-long.304/)] SCOTT: Self-Consistent Chain-of-Thought Distillation [[code](https://github.com/wangpf3/consistent-CoT-distillation)]
- [[ACL](https://aclanthology.org/2023.acl-long.830/)] Large Language Models Are Reasoning Teachers [[code](https://github.com/itsnamgyu/reasoning-teacher)]
- [[ISCA](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)] OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- [[arXiv](https://arxiv.org/abs/2305.18403)] Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
- [[arXiv](https://arxiv.org/abs/2307.09782)] ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats
- [[arXiv](https://arxiv.org/abs/2305.12356)] Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models
- [[arXiv](https://arxiv.org/abs/2304.01089)] RPTQ: Reorder-based Post-training Quantization for Large Language Models [[code](https://github.com/hahnyuan/RPTQ4LLM)]
- [[arXiv](https://arxiv.org/abs/2306.02272)] OWQ: Lessons learned from activation outliers for weight quantization in large language models
- [[arXiv](https://arxiv.org/abs/2305.14152)] Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization
- [[arXiv](https://arxiv.org/abs/2305.02301)] Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes [[code](https://github.com/google-research/distilling-step-by-step)]
- [[arXiv](https://arxiv.org/abs/2305.12870)] Lion: Adversarial Distillation of Closed-Source Large Language Model [[code](https://github.com/YJiangcm/Lion)]
- [[arXiv](https://arxiv.org/abs/2306.13649)] GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models
- [[arXiv](https://arxiv.org/abs/2306.08543)] Knowledge Distillation of Large Language Models [[code](https://github.com/microsoft/LMOps/tree/main/minillm)]
- [[arXiv](https://arxiv.org/abs/2305.11627)] LLM-Pruner: On the Structural Pruning of Large Language Models [[code](https://github.com/horseee/LLM-Pruner)]
- [[arXiv](https://arxiv.org/pdf/2306.11695.pdf)] A Simple and Effective Pruning Approach for Large Language Models [[code](https://github.com/locuslab/wanda)]
- [[arXiv](https://arxiv.org/abs/2301.00774)] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[code](https://github.com/IST-DASLab/sparsegpt)]
- [[arXiv](https://arxiv.org/abs/2306.03078)] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [[code](https://github.com/Vahe1994/SpQR)]
- [[arXiv](https://arxiv.org/abs/2306.12929)] Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing
- [[arXiv](https://arxiv.org/abs/2306.07629)] SqueezeLLM: Dense-and-Sparse Quantization [[code](https://github.com/SqueezeAILab/SqueezeLLM)]
- [[arXiv](https://arxiv.org/abs/2305.11186)] Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt
- [[arXiv](https://arxiv.org/abs/2305.17888)] LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
- [[arXiv](https://arxiv.org/abs/2305.14314)] QLORA: Efficient Finetuning of Quantized LLMs [[code](https://github.com/artidoro/qlora)]
- [[arXiv](https://arxiv.org/abs/2306.00978)] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration [[code](https://github.com/mit-han-lab/llm-awq)]
- [[arXiv](https://arxiv.org/abs/2206.09557)] LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- [[ICLR](https://arxiv.org/abs/2210.17323)] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [[code](https://github.com/IST-DASLab/gptq)]
- [[arXiv](https://arxiv.org/pdf/2304.09145.pdf)] Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling

### 2022

- [[arXiv](https://arxiv.org/abs/2210.06726)] Explanations from Large Language Models Make Small Reasoners Better
- [[arXiv](https://arxiv.org/abs/2212.10670)] In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models
- [[arXiv](https://arxiv.org/pdf/2211.10438.pdf)] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[code](https://github.com/mit-han-lab/smoothquant)]
- [[ACL](https://aclanthology.org/2022.acl-long.331)] Compression of Generative Pre-trained Language Models via Quantization
- [[NeurIPS](https://arxiv.org/abs/2208.07339)] LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- [[NeurIPS](https://nips.cc/Conferences/2022/Schedule?showEvent=53407)] Towards Efficient Post-training Quantization of Pre-trained Language Models
- [[NeurIPS](https://nips.cc/Conferences/2022/Schedule?showEvent=54407)] ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
- [[ICML](https://proceedings.mlr.press/v162/liu22v.html)] GACT: Activation Compressed Training for Generic Network Architectures
- [[ICLR](https://openreview.net/forum?id=5xEgrl_5FAJ)] BiBERT: Accurate Fully Binarized BERT [[code](https://github.com/htqin/BiBERT)]

### 2021

- [[ICML](https://proceedings.mlr.press/v139/kim21d.html)] I-BERT: Integer-only BERT Quantization
- [[ACL](https://aclanthology.org/2021.findings-acl.363)] On the Distribution, Sparsity, and Inference-time Quantization of Attention Values in Transformers
- [[TACL](https://arxiv.org/abs/2002.11985)] Compressing Large-Scale Transformer-Based Models: A Case Study on BERT

### 2020

- [[EMNLP](https://arxiv.org/abs/1910.10485)] Fully Quantized Transformer for Machine Translation
- [[IJCAI](https://www.ijcai.org/Proceedings/2020/0520.pdf)] Towards Fully 8-bit Integer Inference for the Transformer Model
- [[EMNLP](https://arxiv.org/abs/2009.12812)] TernaryBERT: Distillation-aware Ultra-low Bit BERT
- [[AAAI](https://arxiv.org/abs/1909.05840)] Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
- [[ACL](https://arxiv.org/abs/2002.08307)] Compressing bert: Studying the effects of weight pruning on transfer learning
- [[ICLR](https://arxiv.org/abs/1909.11687v1)] Extreme Language Model Compression with Optimal Subwords and Shared Projections
- [[MICRO](http://arxiv.org/abs/2005.03842)] GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference

### 2019

- [[NeurIPS](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-31.pdf)] Q8BERT: Quantized 8Bit BERT
- [[ICML](https://arxiv.org/abs/1906.00532v2)] Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model
- [[NeurIPS](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-36.pdf)] Fully Quantized Transformer for Improved Translation
- [[NeurIPS](https://arxiv.org/abs/1910.01108)] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

## Vision

### 2023

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf)] Post-training Quantization on Diffusion Models
- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)] On Distillation of Guided Diffusion Models
- [[ICME](https://arxiv.org/abs/2211.12039)] Accelerating Diffusion Sampling with Classifier-based Feature Distillation [[code](https://github.com/zju-SWJ/RCFD)]
- [[ICML](https://arxiv.org/abs/2308.06644)] Accelerating Diffusion-based Combinatorial Optimization Solvers by Progressive Distillation [[code](https://github.com/jwrh/Accelerating-Diffusion-based-Combinatorial-Optimization-Solvers-by-Progressive-Distillation)]
- [[ICML](https://arxiv.org/abs/2307.05977)] Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models [[code](https://github.com/nannullna/safe-diffusion)]
- [[arXiv](https://arxiv.org/abs/2302.04304)] Q-Diffusion: Quantizing Diffusion Models [[code](https://github.com/Xiuyu-Li/q-diffusion)]
- [[arXiv](https://arxiv.org/abs/2306.05544)] BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping
- [[arXiv](https://arxiv.org/abs/2306.00980)] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds
- [[arXiv](https://arxiv.org/abs/2305.10769)] Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling [[code](https://anonymous.4open.science/r/Catch-Up-Distillation-E31F)]
- [[arXiv](https://arxiv.org/abs/2101.02388)] Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed
- [[arXiv](https://arxiv.org/abs/2305.15798)] On Architectural Compression of Text-to-Image Diffusion Models
- [[arXiv](https://arxiv.org/abs/2202.00512)] Progressive Distillation for Fast Sampling of Diffusion Models
- [[arXiv](https://arxiv.org/abs/2211.02048)] Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models
- [[arXiv](https://arxiv.org/abs/2305.10924)] Structural Pruning for Diffusion Models



