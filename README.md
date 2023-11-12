# Awesome Efficient AIGC [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo collects efficient approaches for AIGC (AI Generated Content) to cope with its huge demand for computing resources. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. Special thanks to [Xingyu Zheng](https://github.com/Xingyu-Zheng), [Yifu Ding](https://yifu-ding.github.io/#/), and all researchers who have contributed to this project!

## Table of Contents

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

**Survey**

- [[arxiv](https://arxiv.org/abs/2308.07633)] A Survey on Model Compression for Large Language Models

**Quantization**

- [[ICLR](https://arxiv.org/abs/2210.17323)] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [[code](https://github.com/IST-DASLab/gptq)]
- [[NeurIPS](https://arxiv.org/abs/2305.14314)] QLORA: Efficient Finetuning of Quantized LLMs [[code](https://github.com/artidoro/qlora)]
- [[ICML](https://browse.arxiv.org/abs/2211.10438)] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[code](https://github.com/mit-han-lab/smoothquant)]
- [[ICML](https://arxiv.org/abs/2306.00317)] FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization [[code](https://openreview.net/attachment?id=-tYCaP0phY_&name=supplementary_material)]
- [[ICML](https://arxiv.org/abs/2301.12017)] Understanding INT4 Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases [[code](https://github.com/microsoft/DeepSpeed)]
- [[ICML](https://icml.cc/virtual/2023/28295)] GPT-Zip: Deep Compression of Finetuned Large Language Models
- [[ICML](https://arxiv.org/abs/2307.03738)] QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models [[code](https://github.com/IST-DASLab/QIGen)]
- [[ACL](https://arxiv.org/abs/2306.00014)] PreQuant: A Task-agnostic Quantization Approach for Pre-trained Language Models
- [[ACL](https://aclanthology.org/2023.findings-acl.15/)] Boost Transformer-based Language Models with GPU-Friendly Sparsity and Quantization
- [[EMNLP](https://arxiv.org/abs/2310.05079)] Revisiting Block-based Quantisation: What is Important for Sub-8-bit LLM Inference?
- [[EMNLP](https://arxiv.org/abs/2310.13315)] Zero-Shot Sharpness-Aware Quantization for Pre-trained Language Models
- [[EMNLP](https://arxiv.org/abs/2310.16836)] LLM-FP4: 4-Bit Floating-Point Quantized Transformers [[code](https://github.com/nbasyl/LLM-FP4)]
- [[EMNLP](https://arxiv.org/abs/2304.09145)] Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling
- [[ISCA](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)] OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- [[arxiv](https://arxiv.org/abs/2303.08302)] ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation
- [[arxiv](https://arxiv.org/abs/2206.09557)] LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- [[arxiv](https://arxiv.org/abs/2302.02390)] Quantized Distributed Training of Large Models with Convergence Guarantees
- [[arxiv](https://arxiv.org/abs/2305.17888)] LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
- [[arxiv](https://arxiv.org/abs/2306.00978)] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration [[code](https://github.com/mit-han-lab/llm-awq)]
- [[arxiv](https://arxiv.org/abs/2306.11987)] Training Transformers with 4-bit Integers [[code](https://github.com/xijiu9/Train_Transformers_with_INT4)]
- [[arxiv](https://arxiv.org/abs/2306.07629)] SqueezeLLM: Dense-and-Sparse Quantization [[code](https://github.com/SqueezeAILab/SqueezeLLM)]
- [[arxiv](https://arxiv.org/abs/2306.12929)] Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing
- [[arxiv](https://arxiv.org/abs/2306.03078)] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [[code](https://github.com/Vahe1994/SpQR)]
- [[arxiv](https://arxiv.org/abs/2307.13304)] QuIP: 2-Bit Quantization of Large Language Models With Guarantees [[code](https://github.com/jerry-chee/QuIP)]
- [[arxiv](https://arxiv.org/abs/2305.14152)] Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization
- [[arxiv](https://arxiv.org/abs/2306.02272)] OWQ: Lessons learned from activation outliers for weight quantization in large language models
- [[arxiv](https://arxiv.org/abs/2308.13137)] OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models [[code](https://github.com/OpenGVLab/OmniQuant)]
- [[arxiv](https://arxiv.org/abs/2304.01089)] RPTQ: Reorder-based Post-training Quantization for Large Language Models [[code](https://github.com/hahnyuan/RPTQ4LLM)]
- [[arxiv](https://arxiv.org/abs/2305.12356)] Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models
- [[arxiv](https://arxiv.org/abs/2306.08162)] INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation
- [[arxiv](https://arxiv.org/abs/2307.03712)] INT-FP-QSim: Mixed Precision and Formats For Large Language Models and Vision Transformers [[code](https://github.com/lightmatter-ai/INT-FP-QSim)]
- [[arxiv](https://arxiv.org/abs/2307.08072)] Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study
- [[arxiv](https://arxiv.org/abs/2307.09782)] ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats
- [[arxiv](https://arxiv.org/abs/2308.05600)] NUPES : Non-Uniform Post-Training Quantization via Power Exponent Search
- [[arxiv](https://arxiv.org/abs/2308.06744)] Token-Scaled Logit Distillation for Ternary Weight Generative Language Models
- [[arxiv](https://arxiv.org/abs/2308.07662)] Gradient-Based Post-Training Quantization: Challenging the Status Quo
- [[arxiv](https://arxiv.org/abs/2308.09723)] FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs
- [[arxiv](https://arxiv.org/abs/2308.14903)] MEMORY-VQ: Compression for Tractable Internet-Scale Memory
- [[arxiv](https://arxiv.org/abs/2308.15987)] FPTQ: Fine-grained Post-Training Quantization for Large Language Models
- [[arxiv](https://arxiv.org/abs/2309.00964)] eDKM: An Efficient and Accurate Train-time Weight Clustering for Large Language Models
- [[arxiv](https://arxiv.org/abs/2309.01885)] QuantEase: Optimization-based Quantization for Language Models - An Efficient and Intuitive Algorithm
- [[arxiv](https://arxiv.org/abs/2309.02784)] Norm Tweaking: High-performance Low-bit Quantization of Large Language Models
- [[arxiv](https://arxiv.org/abs/2309.05210)] Understanding the Impact of Post-Training Quantization on Large Language Models
- [[arxiv](https://arxiv.org/abs/2309.05516)] Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs [[code](https://github.com/intel/neural-compressor)]
- [[arxiv](https://arxiv.org/abs/2309.14592)] Efficient Post-training Quantization with FP8 Formats [[code](https://github.com/intel/neural-compressor)]
- [[arxiv](https://arxiv.org/abs/2309.14717)] QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models [[code](https://github.com/yuhuixu1993/qa-lora)]
- [[arxiv](https://arxiv.org/abs/2309.15531)] Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models
- [[arxiv](https://arxiv.org/abs/2309.16119)] ModuLoRA: Finetuning 3-Bit LLMs on Consumer GPUs by Integrating with Modular Quantizers
- [[arxiv](https://arxiv.org/abs/2310.00034)] PB-LLM: Partially Binarized Large Language Models [[code](https://github.com/hahnyuan/BinaryLLM)]
- [[arxiv](https://arxiv.org/abs/2310.04836)] Dual Grained Quantization: Efficient Fine-Grained Quantization for LLM
- [[arxiv](https://arxiv.org/abs/2310.07147)] QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources
- [[arxiv](https://arxiv.org/abs/2310.08041)] QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models
- [[arxiv](https://arxiv.org/abs/2310.08659)] LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models [[code](https://github.com/yxli2123/LoftQ)]
- [[arxiv](https://arxiv.org/abs/2310.10944)] TEQ: Trainable Equivalent Transformation for Quantization of LLMs [[code](https://github.com/intel/neural-compressor)]
- [[arxiv](https://arxiv.org/abs/2310.11453)] BitNet: Scaling 1-bit Transformers for Large Language Models [[code](https://github.com/kyegomez/BitNet)]
- [[arxiv](https://arxiv.org/abs/2310.18313)] FP8-LM: Training FP8 Large Language Models [[code](https://github.com/Azure/MS-AMP)]
- [[arxiv](https://arxiv.org/abs/2310.19102)] Atom: Low-bit Quantization for Efficient and Accurate LLM Serving [[code](https://github.com/efeslab/Atom)]
- [[arxiv](https://arxiv.org/abs/2311.01305)] AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models
- [[arxiv](https://arxiv.org/abs/2311.01792)] AFPQ: Asymmetric Floating Point Quantization for LLMs [[code](https://github.com/zhangsichengsjtu/AFPQ)]


**Pruning and Sparsity**

- [[ICML](https://icml.cc/virtual/2023/oral/25453)] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time [[code](https://github.com/FMInference/DejaVu)]
- [[ICML](https://arxiv.org/abs/2301.00774)] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[code](https://github.com/IST-DASLab/sparsegpt)]
- [[ICML](https://arxiv.org/abs/2306.11222)] LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation [[code](https://github.com/yxli2123/LoSparse)]
- [[ICML](https://arxiv.org/abs/2306.11695)] A Simple and Effective Pruning Approach for Large Language Models [[code](https://github.com/locuslab/wanda)]
- [[ICLR](https://arxiv.org/abs/2210.06313)] The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers [[code](https://github.com/IST-DASLab/gptq)]
- [[ICLR](https://openreview.net/forum?id=cKlgcx7nSZ)] Prune and Tune: Improving Efficient Pruning Techniques for Massive Language Models
- [[NeurIPS](https://arxiv.org/abs/2305.11627)] LLM-Pruner: On the Structural Pruning of Large Language Models [[code](https://github.com/horseee/LLM-Pruner)]
- [[ACL](https://aclanthology.org/2023.findings-acl.15/)] Boost Transformer-based Language Models with GPU-Friendly Sparsity and Quantization
- [[AutoML](https://openreview.net/forum?id=SHlZcInS6C)] Structural Pruning of Large Language Models via Neural Architecture Search
- [[VLDB](https://arxiv.org/abs/2309.10285)] Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity [[code](https://github.com/AlibabaResearch/flash-llm)]
- [[arxiv](https://arxiv.org/abs/2302.03773)] What Matters In The Structured Pruning of Generative Language Models?
- [[arxiv](https://arxiv.org/abs/2305.18403)] LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
- [[arxiv](https://arxiv.org/abs/2306.03078)] SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [[code](https://github.com/Vahe1994/SpQR)]
- [[arxiv](https://arxiv.org/abs/2306.07629)] SqueezeLLM: Dense-and-Sparse Quantization [[code](https://github.com/SqueezeAILab/SqueezeLLM)]
- [[arxiv](https://arxiv.org/abs/2309.09507)] Pruning Large Language Models via Accuracy Predictor
- [[arxiv](https://arxiv.org/abs/2310.06694)] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning [[code](https://github.com/princeton-nlp/LLM-Shearing)]
- [[arxiv](https://arxiv.org/abs/2310.05175)] Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity [[code](https://github.com/luuyin/OWL)]
- [[arxiv](https://arxiv.org/abs/2310.02277)] Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity [[code](https://github.com/VITA-Group/Junk_DNA_Hypothesis.git)]
- [[arxiv](https://arxiv.org/abs/2310.05015)] Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models [[code](https://github.com/microsoft/Moonlit/tree/main/Compresso)]
- [[arxiv](https://arxiv.org/abs/2310.08915)] Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs [[code](https://github.com/zyxxmu/DSnoT)]
- [[arxiv](https://arxiv.org/abs/2310.09499)] One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models
- [[arxiv](https://arxiv.org/abs/2310.15929)] E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity
- [[arxiv](https://arxiv.org/abs/2310.18356)] LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery [[code](https://github.com/microsoft/lorashear)]

**Distillation**

- [[ACL](https://aclanthology.org/2023.acl-long.150/)] Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step [[code](https://github.com/allenai/cot_distillation)]
- [[ACL](https://aclanthology.org/2023.acl-long.249/)] Lifting the Curse of Capacity Gap in Distilling Language Models [[code](https://github.com/GeneZC/MiniMoE)]
- [[ACL](https://aclanthology.org/2023.acl-long.302/)] DISCO: Distilling Counterfactuals with Large Language Models [[code](https://github.com/eric11eca/disco)]
- [[ACL](https://aclanthology.org/2023.acl-long.304/)] SCOTT: Self-Consistent Chain-of-Thought Distillation [[code](https://github.com/wangpf3/consistent-CoT-distillation)]
- [[ACL](https://aclanthology.org/2023.acl-long.471/)] AD-KD: Attribution-Driven Knowledge Distillation for Language Model Compression
- [[ACL](https://aclanthology.org/2023.acl-long.830/)] Large Language Models Are Reasoning Teachers [[code](https://github.com/itsnamgyu/reasoning-teacher)]
- [[ACL](https://aclanthology.org/2023.findings-acl.441/)] Distilling Reasoning Capabilities into Smaller Language Models
- [[ACL](https://aclanthology.org/2023.findings-acl.463/)] Cost-effective Distillation of Large Language Models [[code](https://github.com/Sayan21/MAKD)]
- [[ACL](https://aclanthology.org/2023.findings-acl.507/)] Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes [[code](https://github.com/google-research/distilling-step-by-step)]
- [[EMNLP](https://arxiv.org/abs/2310.13332)] Democratizing Reasoning Ability: Tailored Learning from Large Language Model [[code](https://github.com/Raibows/Learn-to-Reason)]
- [[EMNLP](https://arxiv.org/abs/2310.14192)] PromptMix: A Class Boundary Augmentation Method for Large Language Model Distillation [[code](https://github.com/ServiceNow/PromptMix-EMNLP-2023)]
- [[EMNLP](https://arxiv.org/abs/2310.14747)] MCC-KD: Multi-CoT Consistent Knowledge Distillation
- [[arxiv](https://arxiv.org/abs/2304.14402)] LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions [[code](https://github.com/mbzuai-nlp/LaMini-LM)]
- [[arxiv](https://arxiv.org/abs/2305.12330)] Task-agnostic Distillation of Encoder-Decoder Language Models
- [[arxiv](https://arxiv.org/abs/2305.12870)] Lion: Adversarial Distillation of Closed-Source Large Language Model [[code](https://github.com/YJiangcm/Lion)]
- [[arxiv](https://arxiv.org/abs/2305.13888)] PaD: Program-aided Distillation Specializes Large Models in Reasoning
- [[arxiv](https://arxiv.org/abs/2305.14864)] Large Language Model Distillation Doesn't Need a Teacher
 [[code](https://github.com/ananyahjha93/llm-distill)]
- [[arxiv](https://arxiv.org/abs/2305.15717)] The False Promise of Imitating Proprietary LLMs
- [[arxiv](https://arxiv.org/abs/2306.08543)] Knowledge Distillation of Large Language Models [[code](https://github.com/microsoft/LMOps/tree/main/minillm)]
- [[arxiv](https://arxiv.org/abs/2306.13649)] GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models
- [[arxiv](https://arxiv.org/abs/2306.14122)] Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction
- [[arxiv](https://arxiv.org/abs/2308.04679)] Sci-CoT: Leveraging Large Language Models for Enhanced Knowledge Distillation in Small Models for Scientific QA
- [[arxiv](https://arxiv.org/abs/2308.06744)] Token-Scaled Logit Distillation for Ternary Weight Generative Language Models
- [[arxiv](https://arxiv.org/abs/2310.02421)] Can a student Large Language Model perform as well as it's teacher?

**Fine-tuning**

- [[Nature](https://www.nature.com/articles/s42256-023-00626-4)] Parameter-efficient fine-tuning of large-scale pre-trained language models [[code](https://github.com/thunlp/OpenDelta)]
- [[NeurIPS](https://arxiv.org/abs/2305.14314)] QLORA: Efficient Finetuning of Quantized LLMs [[code](https://github.com/artidoro/qlora)]
- [[ACL](https://aclanthology.org/2023.acl-long.830/)] Large Language Models Are Reasoning Teachers [[code](https://github.com/itsnamgyu/reasoning-teacher)]
- [[arxiv](https://arxiv.org/abs/2305.18403)] LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
- [[arxiv](https://arxiv.org/abs/2305.14152)] Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization
- [[arxiv](https://arxiv.org/abs/2306.08162)] INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation
- [[arxiv](https://arxiv.org/abs/2309.12307)] LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models [[code](https://github.com/dvlab-research/LongLoRA)]
- [[arxiv](https://arxiv.org/abs/2309.14717)] QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models [[code](https://github.com/yuhuixu1993/qa-lora)]
- [[arxiv](https://arxiv.org/abs/2309.16119)] ModuLoRA: Finetuning 3-Bit LLMs on Consumer GPUs by Integrating with Modular Quantizers
- [[arxiv](https://arxiv.org/abs/2310.07147)] QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources

**Other**

- [[EMNLP](https://arxiv.org/abs/2310.05736)] LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models [[code](https://aka.ms/LLMLingua)]
- [[arxiv](https://arxiv.org/abs/2305.11186)] Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt
- [[arxiv](https://arxiv.org/abs/2310.05869)] HyperAttention: Long-context Attention in Near-Linear Time
- [[arxiv](https://arxiv.org/abs/2310.01801)] Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs

### 2022

**Quantization**

- [[ACL](https://aclanthology.org/2022.acl-long.331)] Compression of Generative Pre-trained Language Models via Quantization
- [[NeurIPS](https://arxiv.org/abs/2208.07339)] LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- [[NeurIPS](https://nips.cc/Conferences/2022/Schedule?showEvent=53407)] Towards Efficient Post-training Quantization of Pre-trained Language Models
- [[NeurIPS](https://nips.cc/Conferences/2022/Schedule?showEvent=54407)] ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
- [[ICLR](https://openreview.net/forum?id=5xEgrl_5FAJ)] BiBERT: Accurate Fully Binarized BERT [[code](https://github.com/htqin/BiBERT)]

**Distillation**

- [[arxiv](https://arxiv.org/abs/2210.06726)] Explanations from Large Language Models Make Small Reasoners Better
- [[arxiv](https://arxiv.org/abs/2212.10670)] In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models

**Other**

- [[ICML](https://proceedings.mlr.press/v162/liu22v.html)] GACT: Activation Compressed Training for Generic Network Architectures

### 2021

**Quantization**

- [[ICML](https://proceedings.mlr.press/v139/kim21d.html)] I-BERT: Integer-only BERT Quantization
- [[ACL](https://aclanthology.org/2021.acl-long.334/)] BinaryBERT: Pushing the Limit of BERT Quantization

**Pruning and Sparsity**

- [[ACL](https://aclanthology.org/2021.findings-acl.363)] On the Distribution, Sparsity, and Inference-time Quantization of Attention Values in Transformers

**Distillation**

- [[ACL](https://aclanthology.org/2021.findings-acl.387/)] One Teacher is Enough? Pre-trained Language Model Distillation from Multiple Teachers

**Survey**

- [[TACL](https://arxiv.org/abs/2002.11985)] Compressing Large-Scale Transformer-Based Models: A Case Study on BERT

### 2020

**Quantization**

- [[EMNLP](https://arxiv.org/abs/1910.10485)] Fully Quantized Transformer for Machine Translation
- [[IJCAI](https://www.ijcai.org/Proceedings/2020/0520.pdf)] Towards Fully 8-bit Integer Inference for the Transformer Model
- [[EMNLP](https://arxiv.org/abs/2009.12812)] TernaryBERT: Distillation-aware Ultra-low Bit BERT
- [[AAAI](https://arxiv.org/abs/1909.05840)] Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
- [[MICRO](http://arxiv.org/abs/2005.03842)] GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference

**Pruning and Sparsity**

- [[ACL](https://arxiv.org/abs/2002.08307)] Compressing bert: Studying the effects of weight pruning on transfer learning

**Distillation**

- [[ICLR](https://arxiv.org/abs/1909.11687v1)] Extreme Language Model Compression with Optimal Subwords and Shared Projections

### 2019

**Quantization**

- [[ICML](https://arxiv.org/abs/1906.00532v2)] Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model
- [[NeurIPS](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-31.pdf)] Q8BERT: Quantized 8Bit BERT
- [[NeurIPS](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-36.pdf)] Fully Quantized Transformer for Improved Translation

**Distillation**

- [[NeurIPS](https://arxiv.org/abs/1910.01108)] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

## Vision

### 2023

**Survey**

- [[arxiv](https://arxiv.org/abs/2304.04262)] A Comprehensive Survey on Knowledge Distillation of Diffusion Models

**Quantization**

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.pdf)] Post-training Quantization on Diffusion Models [[code](https://github.com/42Shawn/PTQ4DM)]
- [[CVPR](https://arxiv.org/abs/2303.06424)] Regularized Vector Quantization for Tokenized Image Synthesis
- [[ICCV](https://arxiv.org/abs/2302.04304)] Q-Diffusion: Quantizing Diffusion Models [[code](https://github.com/Xiuyu-Li/q-diffusion)]
- [[NeurIPS](https://arxiv.org/abs/2306.02316)] Temporal Dynamic Quantization for Diffusion Models
- [[arxiv](https://arxiv.org/abs/2305.10657)] PTQD: Accurate Post-Training Quantization for Diffusion Models [[code](https://github.com/ziplab/PTQD)]
- [[arxiv](https://arxiv.org/abs/2305.18723)] Towards Accurate Data-free Quantization for Diffusion Models
- [[arxiv](https://arxiv.org/abs/2309.15505)] Finite Scalar Quantization: VQ-VAE Made Simple [[code](https://github.com/google-research/google-research/tree/master/fsq)]
- [[arxiv](https://arxiv.org/abs/2310.03270)] EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models

**Pruning and Sparsity**

- [[TPAMI](https://arxiv.org/abs/2211.02048)] Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models [[code](https://github.com/lmxyy/sige)]
- [[arxiv](https://arxiv.org/abs/2305.10924)] Structural Pruning for Diffusion Models [[code](https://github.com/VainF/Diff-Pruning)]

**Distillation**

- [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)] On Distillation of Guided Diffusion Models
- [[ICME](https://arxiv.org/abs/2211.12039)] Accelerating Diffusion Sampling with Classifier-based Feature Distillation [[code](https://github.com/zju-SWJ/RCFD)]
- [[ICML](https://arxiv.org/abs/2308.06644)] Accelerating Diffusion-based Combinatorial Optimization Solvers by Progressive Distillation [[code](https://github.com/jwrh/Accelerating-Diffusion-based-Combinatorial-Optimization-Solvers-by-Progressive-Distillation)]
- [[ICML](https://arxiv.org/abs/2307.05977)] Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models [[code](https://github.com/nannullna/safe-diffusion)]
- [[arxiv](https://arxiv.org/abs/2306.05544)] BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping
- [[arxiv](https://arxiv.org/abs/2306.00980)] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds
- [[arxiv](https://arxiv.org/abs/2305.10769)] Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling [[code](https://anonymous.4open.science/r/Catch-Up-Distillation-E31F)]
- [[arxiv](https://arxiv.org/abs/2101.02388)] Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed
- [[arxiv](https://arxiv.org/abs/2305.15798)] On Architectural Compression of Text-to-Image Diffusion Models
- [[arxiv](https://arxiv.org/abs/2202.00512)] Progressive Distillation for Fast Sampling of Diffusion Models

**Other**

- [[arxiv](https://arxiv.org/abs/2309.10438)] AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration
- [[arxiv](https://arxiv.org/abs/2308.10187)] Spiking-Diffusion: Vector Quantized Discrete Diffusion Model with Spiking Neural Networks [[code](https://github.com/Arktis2022/Spiking-Diffusion)]