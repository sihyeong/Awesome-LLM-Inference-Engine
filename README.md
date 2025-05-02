# Awesome-LLM-Inference-Engine

Welcome to the **Awesome-LLM-Inference-Engine** repository!

A curated list of LLM inference engines, system architectures, and optimization techniques for efficient large language model serving. This repository complements our survey paper analyzing 25 inference engines, both open-source and commercial. It aims to provide practical insights for researchers, system designers, and engineers building LLM inference infrastructure.

Our work is based on the following paper:
**Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency**

## 🗂 Table of Contents

- [🧠 Overview](#overview)
- [📊 Taxonomy](#taxonomy)
- [🛠 Optimization Techniques](#optimization-techniques)
- [🔓 Open Source Inference Engines](#open-source-inference-engines)
- [💼 Commercial Solutions](#commercial-solutions)
- [🧮 Comparison Table](#comparison-table)
- [🔭 Future Directions](#future-directions)
- [🤝 Contributing](#contributing)
- [⚖️ License](#license)
- [📝 Citation](#citation)

---

## 🧠 Overview

LLM services are evolving rapidly to support complex tasks such as chain-of-thought (CoT), reasoning, AI Agent workflows. These workloads significantly increase inference cost and system complexity.

This repository categorizes and compares LLM inference engines by:

- 🖧 **Deployment type** (single-node vs multi-node)
- ⚙️ **Hardware diversity** (homogeneous vs heterogeneous)

## 📊 Taxonomy

We classify LLM inference engines along the following dimensions:

- 🧑‍💻 **Ease-of-Use:** Assesses documentation quality and community activity. Higher scores indicate better developer experience and community support.
- ⚙️ **Ease-of-Deployment:** Measures the simplicity and speed of installation using tools like pip, APT, Homebrew, Conda, Docker, source builds, or prebuilt binaries.
- 🌐 **General-purpose support:** Reflects the range of supported LLM models and hardware platforms. Higher values indicate broader compatibility across diverse model families and execution environments.
- 🏗 **Scalability:** Indicates the engine’s ability to operate effectively across edge devices, servers, and multi-node deployments. Higher scores denote readiness for large-scale or distributed workloads.
- 📈 **Throughput-aware:** Captures the presence of optimization techniques focused on maximizing throughput, such as continuous batching, parallelism, and cache reuse.
- ⚡ **Latency-aware:** Captures support for techniques targeting low latency, including stall-free scheduling, chunked prefill, and priority-aware execution.

## 🔓 Open Source Inference Engines

- [bitnet.cpp](https://github.com/microsoft/BitNet)
- [DeepSpeed-FastGen](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen) 🌐 [Webpage](https://www.deepspeed.ai/) 📄 [Paper](https://arxiv.org/abs/2401.08671)
- [DistServe](https://github.com/LLMServe/DistServe) 📄 [Paper](https://arxiv.org/abs/2401.09670)
- [LightLLM](https://github.com/ModelTC/lightllm) 🌐 [Webpage](https://www.light-ai.top/lightllm-blog/blog/)
- [LitGPT](https://github.com/Lightning-AI/litgpt) 🌐 [Webpage](https://lightning.ai/)
- [LMDeploy](https://github.com/InternLM/lmdeploy) 🌐 [Webpage](https://lmdeploy.readthedocs.io/en/latest/)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MAX](https://github.com/modular/modular) 🌐 [Webpage](https://www.modular.com/max/solutions/ai-inference)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) 🌐 [Webpage](https://llm.mlc.ai/)
- [NanoFlow](https://github.com/efeslab/Nanoflow) 📄 [Paper](https://arxiv.org/abs/2408.12757)
- [Ollama](https://github.com/ollama/ollama) 🌐 [Webpage](https://ollama.com/)
- [OpenLLM](https://github.com/bentoml/OpenLLM) 🌐 [Webpage](https://www.bentoml.com/)
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) 📄 [Paper1](https://arxiv.org/abs/2312.12456), 📄 [Paper2](https://arxiv.org/abs/2406.06282)
- [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) 📄 [Paper](https://arxiv.org/abs/2403.02310)
- [SGLang](https://github.com/sgl-project/sglang) 🌐 [Webpage](https://docs.sglang.ai/) 📄 [Paper](https://arxiv.org/abs/2312.07104)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 🌐 [Webpage](https://docs.nvidia.com/tensorrt-llm/index.html)
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) 🌐 [Webpage](https://huggingface.co/docs/text-generation-inference/index)
- [Unsloth](https://github.com/unslothai/unsloth) 🌐 [Webpage](https://unsloth.ai/)
- [vAttention](https://github.com/microsoft/vattention) 📄 [Paper](https://arxiv.org/abs/2405.04437)
- [vLLM](https://github.com/vllm-project/vllm) 🌐 [Webpage](https://docs.vllm.ai/en/latest/) 📄 [Paper](https://arxiv.org/abs/2309.06180)

## 💼 Commercial Inference Engines

- 🌐 [Fireworks AI](https://fireworks.ai/)
- 🌐 [Friendli Inference](https://friendli.ai/)
- 🌐 [GroqCloud](https://groq.com/groqcloud/)
- 🌐 [Together Inference](https://www.together.ai/)

## 📋 Overview of LLM Inference Engines
The following table compares 25 open-source and commercial LLM inference engines along multiple dimensions including organization, release status, GitHub trends, documentation maturity, model support, and community presence.

| Framework | Organization | Release Date | Open Source | GitHub Stars | Docs | SNS | Forum | Meetup |
|-----------|--------------|---------------|--------------|----------------|------|------|--------|--------|
| Ollama | Community (Ollama) | Jun. 2023 | ✅ | 136K | 🟠 | ✅ | ❌ | ✅ |
| llama.cpp | Community (ggml.ai) | Mar. 2023 | ✅ | 77.6K | 🟡 | ❌ | ❌ | ❌ |
| vLLM | Academic (vLLM Team) | Feb. 2023 | ✅ | 43.4K | ✅ | ✅ | ✅ | ✅ |
| DeepSpeed-FastGen | Big Tech (Microsoft) | Nov. 2023 | ✅ | 37.7K | ✅ | ❌ | ❌ | ✅ |
| Unsloth | Startup (Unsloth AI) | Nov. 2023 | 🔷 | 36.5K | 🟡 | ✅ | ✅ | ❌ |
| MAX | Startup (Modular Inc.) | Apr. 2023 | 🔷 | 23.8K | 🟠 | ✅ | ✅ | ✅ |
| MLC LLM | Community (MLC-AI) | Apr. 2023 | ✅ | 20.3K | 🟠 | ✅ | ❌ | ❌ |
| llama2.c | Community (Andrej Karpathy) | Jul. 2023 | ✅ | 18.3K | ❌ | ✅ | ❌ | ❌ |
| bitnet.cpp | Big Tech (Microsoft) | Oct. 2024 | ✅ | 13.6K | ❌ | ❌ | ❌ | ❌ |
| SGLang | Academic (SGLang Team) | Jan. 2024 | ✅ | 12.8K | 🟠 | ✅ | ❌ | ✅ |
| LitGPT | Startup (Lightning AI) | Jun. 2024 | ✅ | 12.0K | 🟡 | ✅ | ❌ | ✅ |
| OpenLLM | Startup (BentoML) | Apr. 2023 | 🔷 | 11.1K | ❌ | ✅ | ❌ | ❌ |
| TensorRT-LLM | Big Tech (NVIDIA) | Aug. 2023 | 🔷 | 10.1K | ✅ | ❌ | ✅ | ✅ |
| TGI | Startup (Hugging Face) | Oct. 2022 | ✅ | 10.0K | 🟠 | ❌ | ✅ | ❌ |
| PowerInfer | Academic (SJTU-IPADS) | Dec. 2023 | ✅ | 8.2K | ❌ | ❌ | ❌ | ❌ |
| LMDeploy | Startup (MMDeploy) | Jun. 2023 | ✅ | 6.0K | 🟠 | ✅ | ❌ | ❌ |
| LightLLM | Academic (Lightllm Team) | Jul. 2023 | ✅ | 3.1K | 🟠 | ✅ | ❌ | ❌ |
| NanoFlow | Academic (UW Efeslab) | Aug. 2024 | ✅ | 0.7K | ❌ | ❌ | ❌ | ❌ |
| DistServe | Academic (PKU) | Jan. 2024 | ✅ | 0.5K | ❌ | ❌ | ❌ | ❌ |
| vAttention | Big Tech (Microsoft) | May. 2024 | ✅ | 0.3K | ❌ | ❌ | ❌ | ❌ |
| Sarathi-Serve | Big Tech (Microsoft) | Nov. 2023 | ✅ | 0.3K | ❌ | ❌ | ❌ | ❌ |
| Friendli Inference | Startup (FriendliAI Inc.) | Nov. 2023 | ❌ | -- | 🟡 | ❌ | ❌ | ✅ |
| Fireworks AI | Startup (Fireworks AI Inc.) | Jul. 2023 | ❌ | -- | 🟡 | ✅ | ❌ | ❌ |
| GroqCloud | Startup (Groq Inc.) | Feb. 2024 | ❌ | -- | ❌ | ✅ | ❌ | ✅ |
| Together Inference | Startup (together.ai) | Nov. 2023 | ❌ | -- | 🟡 | ✅ | ❌ | ❌ |

*Legend:*
- Open Source: ✅ = yes, 🔷 = partial, ❌ = closed
- Docs: ✅ = detailed, 🟠 = moderate, 🟡 = simple, ❌ = missing
- SNS / Forum / Meetup: presence of Discord/Slack, forum, or events

## 🛠 Optimization Techniques

We classify LLM inference optimization techniques into several major categories based on their target performance metrics, including latency, throughput, memory, and scalability. Each category includes representative methods and corresponding research publications.

## 🧩 Batch Optimization


| Technique           | Description                                                                                             | References                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Dynamic Batching    | Collects user requests over a short time window to process them together, improving hardware efficiency | [Crankshaw et al. (2017)](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw), [Ali et al. (2020)](https://ieeexplore.ieee.org/abstract/document/9355312/) |
| Continuous Batching | Forms batches incrementally based on arrival time to minimize latency                                   | [Yu et al. (2022)](https://www.usenix.org/conference/osdi22/presentation/yu), [He et al. (2024)](https://dl.acm.org/doi/abs/10.1145/3642970.3655835?casa_token=1NiNcQd9abkAAAAA:fyj6qosTlTWjIEPXBWkEoBbKuRZgHufVqlNur_4DL3M5dfla-ZRnh8JDdBCB5Nx1k0pZX15UbIo)         |
| Nano Batching       | Extremely fine-grained batching for ultra-low latency inference                                         | [Zhu et al. (2024)](https://arxiv.org/abs/2408.12757)                          |
| Chunked-prefills    | Splits prefill into chunks for parallel decoding                                                        | [Agrawal et al. (2023)](https://arxiv.org/abs/2308.16369)                      |

## 🕸 Parallelism


| Technique                            | Description                                                                         | References                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Data Parallelism (DP)                     | Copies the same model to multiple GPUs and splits input data for parallel execution | [Rajbhandari et al. (2020)](https://ieeexplore.ieee.org/abstract/document/9355301)                                                   |
| Fully Shared Data Parallelism (FSDP) | Shards model parameters across GPUs for memory-efficient training                   | [Zhao et al. (2023)](https://arxiv.org/abs/2304.11277)                                                          |
| Tensor Parallelism (TP)                   | Splits model tensors across devices for parallel computation                        | [Stojkovic et al. (2024)](https://arxiv.org/abs/2403.20306), [Prabhakar et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f4d1fc085b7504c140e66bb26ed8842-Abstract-Conference.html)                            |
| Pipeline Parallelism (PP)                | Divides model layers across devices and executes micro-batches sequentially         | [Agrawal et al. (2023)](https://arxiv.org/abs/2308.16369), [Hu et al. (2021)](https://arxiv.org/abs/2110.14895), [Ma et al. (2024)](https://aclanthology.org/2024.naacl-industry.1/), [Yu et al. (2024)](https://dl.acm.org/doi/abs/10.1145/3688351.3689164) |

## 📦 Compression

### Quantization


| Technique             | Description                                         | References                              |
| ----------------------- | ----------------------------------------------------- | ----------------------------------------- |
| PTQ                   | Applies quantization after training                 | [Li et al. (2023)](https://arxiv.org/abs/2308.15987)                      |
| QAT                   | Retrains with quantization awareness                | [Chen et al. (2024)](https://arxiv.org/abs/2407.11062), [Liu et al. (2023)](https://arxiv.org/abs/2305.17888)   |
| AQLM                  | Maintains performance at extremely low precision    | [Egiazarian et al. (2024)](https://arxiv.org/abs/2401.06118)                |
| SmoothQuant           | Uses scale folding for normalization                | [Xiao et al. (2023)](https://proceedings.mlr.press/v202/xiao23c.html)                      |
| KV Cache Quantization | Quantizes KV cache to reduce memory usage           | [Hooper et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html), [Liu et al. (2024)](https://arxiv.org/abs/2402.02750) |
| EXL2                  | Implements efficient quantization format            | [EXL2](https://github.com/turboderp-org/exllamav2)                          |
| EETQ                  | Inference-friendly quantization method              | [EETQ](https://github.com/NetEase-FuXi/EETQ)                           |
| LLM Compressor        | Unified framework for quantization and pruning      | [LLM Compressor](https://github.com/vllm-project/llm-compressor)                    |
| GPTQ                  | Hessian-aware quantization minimizing accuracy loss | [Frantar et al. (2022)](https://arxiv.org/abs/2210.17323)                   |
| Marlin                | Fused quantization kernels for performance          | [Frantar et al. (2025)](https://dl.acm.org/doi/abs/10.1145/3710848.3710871)                   |
| Microscaling Format   | Compact format for fine-grained quantization        | [Rouhani et al. (2023)](https://arxiv.org/abs/2310.10537)                   |

### Pruning


| Technique             | Description                                       | References              |
| ----------------------- | --------------------------------------------------- | ------------------------- |
| cuSPARSE              | NVIDIA-optimized sparse matrix library            | [NVIDIA cuSPARSE](https://developer.nvidia.com/cusparse)         |
| Wanda                 | Importance-based weight pruning                   | [Sun et al. (2023)](https://arxiv.org/abs/2306.11695)       |
| Mini-GPTs             | Efficient inference with reduced compute          | [Valicenti et al. (2023)](https://arxiv.org/abs/2312.12682) |
| Token pruning         | Skips decoding of unimportant tokens              | [Fu et al. (2024)](https://arxiv.org/abs/2407.14057)        |
| Post-Training Pruning | Prunes weights based on importance after training | [Zhao et al. (2024)](https://arxiv.org/abs/2410.15567)      |

### Sparsity Optimization


| Technique              | Description                               | References                                                                      |
| ------------------------ | ------------------------------------------- | --------------------------------------------------------------------------------- |
| Structured Sparsity    | Removes weights in fixed patterns         | [Zheng et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b8f10193cab43d45df9bb810637333fd-Abstract-Conference.html), [Dong et al. (2023)](https://openreview.net/forum?id=c4m0BkO4OL)                                         |
| Dynamic Sparsity       | Applies sparsity dynamically at runtime   | [Zhang et al. (2023)](https://arxiv.org/abs/2310.08915)                                                             |
| Kernel-level Sparsity  | Optimizations at CUDA kernel level        | [Xia et al. (2023)](https://arxiv.org/abs/2309.10285), [Borstnik et al. (2014)](https://doi.org/10.1016/j.parco.2014.03.012), [xFormers (2022)](https://github.com/facebookresearch/xformers), [Xiang et al. (2025)](https://arxiv.org/abs/2504.06443) |
| Block Sparsity         | Removes weights in block structures       | [Gao et al. (2024)](https://arxiv.org/abs/2410.13276)                                                               |
| N:M Sparsity           | Maintains sparsity in fixed N:M ratios    | [Zhang et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/06589ec9d86876508600a678f9c8f51d-Abstract-Conference.html)                                                             |
| MoE / Sparse MoE       | Activates only a subset of experts        | [Cai et al. (2024)](https://arxiv.org/abs/2407.06204), [Fedus et al. (2022)](https://www.jmlr.org/papers/v23/21-0998.html), [Du et al. (2022)](https://proceedings.mlr.press/v162/du22c.html)                        |
| Dynamic Token Sparsity | Prunes tokens based on dynamic importance | [Yang et al. (2024)](https://arxiv.org/abs/2408.07092), [Fu et al. (2024)](https://arxiv.org/abs/2407.14057)                                            |
| Contextual Sparsity    | Applies sparsity based on context         | [Liu et al. (2023)](https://proceedings.mlr.press/v202/liu23am.html), [Akhauri et al. (2024)](https://arxiv.org/abs/2406.16635)                                        |

## 🛠 Fine-Tuning


| Technique             | Description                                     | References                                  |
| ----------------------- | ------------------------------------------------- | --------------------------------------------- |
| Full-Parameter Tuning | Updates all model parameters                    | [Lv et al. (2023)](https://arxiv.org/abs/2306.09782)                            |
| LoRA                  | Injects low-rank matrices for efficient updates | [Hu et al. (2022)](https://openreview.net/forum?id=nZeVKeeFYf9), [Sheng et al. (2023) ](https://arxiv.org/abs/2311.03285)      |
| QLoRA                 | Combines LoRA with quantized weights            | [Dettmers et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html), [Zhang et al. (2023)](https://aclanthology.org/2023.wmt-1.43/) |

## 💾 Caching


| Technique      | Description                           | References                           |
| ---------------- | --------------------------------------- | -------------------------------------- |
| Prompt Caching | Caches responses to identical prompts | [Zhu et al. (2024)](https://arxiv.org/abs/2402.01173)                    |
| Prefix Caching | Reuses common prefix computations     | [Liu et al. (2024)](https://arxiv.org/abs/2403.05821), [Pan et al. (2024)](https://arxiv.org/abs/2411.19379) |
| KV Caching     | Stores KV pairs for reuse in decoding | [Pope et al. (2023)](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html)                   |

## 🔍 Attention Optimization


| Technique        | Description                                         | References                                  |
| ------------------ | ----------------------------------------------------- | --------------------------------------------- |
| PagedAttention   | Partitions KV cache into memory-efficient pages     | [Kwon et al. (2023)](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)                         |
| TokenAttention   | Selects tokens dynamically for attention            | [LightLLM](https://lightllm-en.readthedocs.io/en/latest/dev/token_attention.html#tokenattention)                          |
| ChunkedAttention | Divides attention into chunks for better scheduling | [Ye et al. (2024)](https://arxiv.org/abs/2402.15220)                            |
| FlashAttention   | High-speed kernel for attention                     | [Dao et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html),[Dao et al. (2023)](https://arxiv.org/abs/2307.08691), [Shah et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html) |
| RadixAttention   | Merges tokens to reuse KV cache                     | [Zheng et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)                         |
| FlexAttention    | Configurable attention via DSL                      | [Dong et al. (2024)](https://arxiv.org/abs/2412.05496)                         |
| FireAttention    | Optimized for MQA and fused heads                   | [Fireworks AI](https://fireworks.ai/)                        |

## 🎲 Sampling Optimization


| Technique | Description                                    | References                     |
| ----------- | ------------------------------------------------ | -------------------------------- |
| EAGLE     | Multi-token speculative decoding               | [Li et al. (2024a)](https://arxiv.org/abs/2401.15077), [Li et al.  (2024b)](https://arxiv.org/abs/2406.16858), [Li et al.  (2025)](https://arxiv.org/abs/2503.01840) |
| Medusa    | Tree-based multi-head decoding                 | [Cai et al. (2024)](https://arxiv.org/abs/2401.10774)             |
| ReDrafter | Regenerates output based on long-range context | [Cheng et al. (2024)](https://arxiv.org/abs/2403.09919)            |

## 🧾 Structured Outputs


| Technique                 | Description                            | References                                                     |
| --------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| FSM / CFG                 | Rule-based decoding constraints        | [Willard et al. (2023)](https://arxiv.org/abs/2307.09702), [Geng et al. (2023)](https://arxiv.org/abs/2305.13971), [Barke et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1c9c85bae6161d52182d0fe2f3640512-Abstract-Conference.html) |
| Outlines / XGrammar       | Token-level structural constraints     | [Wilard et al. (2023)](https://arxiv.org/abs/2307.09702), [Dong et al. (2024)](https://arxiv.org/abs/2411.15100) |
| LM Format Enforcer        | Enforces output to follow JSON schemas | [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer)                                                  |
| llguidance / GBNF         | Lightweight grammar-based decoding     | [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), [llguidance](https://github.com/guidance-ai/llguidance)                    |
| OpenAI Structured Outputs | API-supported structured outputs       | [OpenAI](https://platform.openai.com/docs/guides/structured-outputs)                                                  |
| JSONSchemaBench           | Benchmark for structured decoding      | [Geng et al. (2025)](https://arxiv.org/abs/2501.10868)                                            |
| StructTest / SoEval       | Tools for structured output validation | [Chen et al. (2024)](https://arxiv.org/abs/2412.18011), [Liu et al. (2024)](https://doi.org/10.1016/j.ipm.2024.103809)                          |


## 📚 Comparison Table

⚠️ Due to GitHub Markdown limitations, only a summarized Markdown version is available here. Please refer to the LaTeX version in the survey paper for full detail.

### 💻 Hardware Support Matrix
| Framework         | Linux | Windows | macOS | Web/API | x86-64 | ARM64/Apple Silicon | NVIDIA GPU (CUDA) | AMD GPU (ROCm/HIP) | Intel GPU (SYCL) | Google TPU | AMD Instinct | Intel Gaudi | Huawei Ascend | AWS Inferentia | Mobile / Edge                            | ETC                   |
|-------------------|--------|---------|--------|---------|--------|----------------------|--------------------|---------------------|------------------|-------------|---------------|--------------|----------------|----------------|------------------------------------------|------------------------|
| Ollama            | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ✅               | ❌          | ✅            | ❌           | ❌             | ❌             | ✅ (NVIDIA Jetson)                         | ❌                      |
| LLaMA.cpp         | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ✅               | ❌          | ✅            | ❌           | ✅             | ❌             | ✅ (Qualcomm Adreno)                      | Moore Threads MTT      |
| vLLM              | ✅     | ❌      | ❌     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ✅               | ✅          | ✅            | ✅           | ❌             | ✅             | ✅ (NVIDIA Jetson)                         | ❌                      |
| DeepSpeed-FastGen | ✅     | ✅      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ✅               | ❌          | ✅            | ✅           | ✅             | ❌             | ❌                                        | Tecorigin SDAA         |
| unsloth           | ✅     | ✅      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| MAX               | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| MLC-LLM           | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ✅               | ❌          | ❌            | ❌           | ❌             | ❌             | ✅ (Qualcomm Adreno, ARM Mali, Apple)     | ❌                      |
| llama2.c          | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ❌                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| bitnet.cpp        | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ❌                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| SGLang            | ✅     | ❌      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ✅               | ❌          | ✅            | ✅           | ❌             | ❌             | ✅ (NVIDIA Jetson)                         | ❌                      |
| LitGPT            | ✅     | ❌      | ✅     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ✅          | ✅            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| OpenLLM           | ✅     | ❌      | ❌     | ❌      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| TensorRT-LLM      | ✅     | ✅      | ❌     | ❌      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ✅ (NVIDIA Jetson)                         | ❌                      |
| TGI               | ✅     | ❌      | ❌     | ❌      | ✅     | ✅                   | ✅                 | ❌                  | ✅               | ✅          | ✅            | ✅           | ❌             | ✅             | ❌                                        | ❌                      |
| PowerInfer        | ✅     | ✅      | ✅     | ❌      | ✅     | ✅                   | ✅                 | ✅                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ✅ (Qualcomm Snapdragon 8)                | ❌                      |
| LMDeploy          | ✅     | ✅      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ✅             | ❌             | ✅ (NVIDIA Jetson)                         | ❌                      |
| LightLLM          | ✅     | ❌      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| NanoFlow          | ✅     | ❌      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| DistServe         | ✅     | ❌      | ❌     | ❌      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| vAttention        | ✅     | ❌      | ❌     | ❌      | ✅     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| Sarathi-Serve     | ✅     | ❌      | ❌     | ❌      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| Friendli Inference| ❌     | ❌      | ❌     | ✅      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| Fireworks AI      | ❌     | ❌      | ❌     | ✅      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ✅            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |
| GroqCloud         | ❌     | ❌      | ❌     | ✅      | ❌     | ❌                   | ❌                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | Groq LPU               |
| Together Inference| ❌     | ❌      | ❌     | ✅      | ❌     | ❌                   | ✅                 | ❌                  | ❌               | ❌          | ❌            | ❌           | ❌             | ❌             | ❌                                        | ❌                      |

- **NVIDIA GPU**: [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/), [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/?ncid=no-ncid), [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/?ncid=no-ncid) etc.
- **AMD GPU**: [AMD Radeon](https://www.amd.com/en/products/graphics/desktops/radeon.html), etc.
- **Intel GPU**: [Intel Arc](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html), etc.
- **Google TPU**: [TPU v4](https://cloud.google.com/tpu/docs/v4), [TPU v5e](https://cloud.google.com/tpu/docs/v5e), [TPU v5p](https://cloud.google.com/tpu/docs/v5p), etc.
- **AMD Instinct**: [Instinct MI200](https://www.amd.com/en/products/accelerators/instinct/mi200.html), [Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html), etc.
- **Intel Gaudi**: [Intel Gaudi 2](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi2.html), Intel [Gaudi 3](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html)
- **Huawei Ascend**: [Ascend series](https://e.huawei.com/ph/products/computing/ascend)
- **AWS Inferentia**: [Inferentia](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia.html), [Inferentia 2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia2.html)
- **Mobile/Edge**: [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/), [Qualcomm Snapdragon](https://www.qualcomm.com/snapdragon/overview), etc.
- **ETC**: [Moore Threads MTT](https://en.mthreads.com/), [Tecorigin SDAA](http://www.tecorigin.com/), [Groq LPU](https://groq.com/the-groq-lpu-explained/)

### 🧭 Deployment Scalability vs. Hardware Diversity

|                        | 🧩 Heterogeneous Devices                                                                                                  | ⚙️ Homogeneous Devices                                                                                                   |
|------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| 🖥 **Single-Node**      | llama.cpp, MAX, MLC LLM, Ollama, PowerInfer, TGI                                                                          | bitnet.cpp, LightLLM, llama2.c, NanoFlow, OpenLLM, Sarathi-Serve, Unsloth, vAttention, Friendli Inference                |
| 🖧 **Multi-Node**       | DeepSpeed-FastGen, LitGPT, LMDeploy, SGLang, vLLM, Fireworks AI, Together Inference                                       | DistServe, TensorRT-LLM, GroqCloud                                                                                       |

*Legend:*
- **🖥 Single-Node**: Designed for single-device execution
- **🖧 Multi-Node**: Supports distributed or multi-host serving
- **🧩 Heterogeneous Devices**: Supports diverse hardware (CPU, GPU, accelerators)
- **⚙️ Homogeneous Devices**: Optimized for a single hardware class

### 📌 Optimization Coverage Matrix

| Framework              | Dynamic Batching | Continuous Batching | Nano Batching | Chunked-prefills | Data Parallelism | FSDP | Tensor Parallelism | Pipeline Parallelism | Quantization | Pruning | Sparsity | LoRA | Prompt Caching | Prefix Caching | KV Caching | PagedAttention | vAttention | FlashAttention | RadixAttention | FlexAttention | FireAttention | Speculative Decoding | Guided Decoding |
|------------------------|------------------|----------------------|----------------|------------------|------------------|------|---------------------|------------------------|--------------|---------|----------|------|----------------|----------------|------------|----------------|-------------|----------------|----------------|----------------|----------------|------------------------|------------------|
| Ollama                 | ❌               | ❌                   | ❌             | ❌               | ❌               | ❌   | ✅                  | ✅                     | ✅           | ✅      | ✅       | ✅   | ✅             | ❌             | ✅         | ❌             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| LLaMA.cpp              | ❌               | ✅                   | ❌             | ❌               | ❌               | ❌   | ✅                  | ✅                     | ✅           | ❌      | ✅       | ✅   | ✅             | ❌             | ✅         | ❌             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| vLLM                   | ❌               | ✅                   | ❌             | ✅               | ✅               | ✅   | ✅                  | ✅                     | ✅           | ✅      | ✅       | ✅   | ❌             | ✅             | ✅         | ✅             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| DeepSpeed-FastGen      | ❌               | ✅                   | ❌             | ✅               | ✅               | ✅   | ✅                  | ✅                     | ✅           | ✅      | ✅       | ✅   | ❌             | ❌             | ✅         | ✅             | ❌          | ✅             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| unsloth                | ❌               | ❌                   | ❌             | ❌               | ❌               | ❌   | ❌                  | ❌                     | ✅           | ❌      | ❌       | ✅   | ❌             | ❌             | ✅         | ❌             | ❌          | ✅             | ❌             | ✅             | ❌             | ❌                     | ❌               |
| MAX                    | ❌               | ✅                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ❌                     | ✅           | ❌      | ✅       | ✅   | ❌             | ✅             | ✅         | ✅             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| MLC-LLM                | ❌               | ✅                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ✅                     | ✅           | ❌      | ✅       | ❌   | ❌             | ✅             | ✅         | ✅             | ❌          | ❌             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| llama2.c               | ❌               | ❌                   | ❌             | ❌               | ❌               | ❌   | ❌                  | ❌                     | ✅           | ❌      | ❌       | ❌   | ❌             | ❌             | ✅         | ❌             | ❌          | ❌             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| bitnet.cpp             | ❌               | ❌                   | ❌             | ❌               | ❌               | ❌   | ❌                  | ❌                     | ✅           | ❌      | ✅       | ❌   | ❌             | ❌             | ✅         | ❌             | ❌          | ❌             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| SGLang                 | ❌               | ✅                   | ❌             | ✅               | ✅               | ✅   | ✅                  | ❌                     | ✅           | ✅      | ✅       | ✅   | ❌             | ✅             | ✅         | ✅             | ❌          | ❌             | ✅             | ❌             | ✅             | ✅                     | ✅               |
| LitGPT                 | ❌               | ✅                   | ❌             | ❌               | ✅               | ✅   | ✅                  | ❌                     | ✅           | ❌      | ✅       | ✅   | ❌             | ❌             | ✅         | ❌             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ❌               |
| OpenLLM                | ❌               | ✅                   | ❌             | ❌               | ✅               | ❌   | ❌                  | ❌                     | ✅           | ❌      | ❌       | ❌   | ❌             | ❌             | ❌         | ❌             | ❌          | ❌             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| TensorRT-LLM           | ✅               | ✅                   | ❌             | ✅               | ✅               | ❌   | ✅                  | ✅                     | ✅           | ✅      | ✅       | ✅   | ✅             | ❌             | ✅         | ✅             | ❌          | ❌             | ❌             | ❌             | ✅             | ✅                     | ✅               |
| TGI                    | ❌               | ✅                   | ❌             | ❌               | ❌               | ❌   | ✅                  | ❌                     | ✅           | ✅      | ✅       | ✅   | ❌             | ✅             | ✅         | ✅             | ❌          | ✅             | ❌             | ❌             | ✅             | ✅                     | ✅               |
| PowerInfer             | ❌               | ✅                   | ❌             | ❌               | ✅               | ❌   | ❌                  | ✅                     | ✅           | ❌      | ✅       | ✅   | ❌             | ✅             | ❌         | ❌             | ✅          | ❌             | ❌             | ❌             | ✅             | ✅                     | ✅               |
| LMDeploy               | ❌               | ✅                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ❌                     | ✅           | ✅      | ✅       | ✅   | ❌             | ✅             | ✅         | ✅             | ❌          | ❌             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| LightLLM               | ✅               | ❌                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ❌                     | ✅           | ❌      | ✅       | ❌   | ✅             | ❌             | ✅         | ❌             | ❌          | ✅             | ❌             | ❌             | ❌             | ✅                     | ✅               |
| NanoFlow               | ❌               | ✅                   | ✅             | ✅               | ✅               | ❌   | ❌                  | ❌                     | ❌           | ❌      | ❌       | ❌   | ❌             | ❌             | ✅         | ❌             | ❌          | ❌             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| DistServe              | ✅               | ✅                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ✅                     | ❌           | ❌      | ❌       | ❌   | ❌             | ✅             | ✅         | ❌             | ✅          | ❌             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| vAttention             | ❌               | ✅                   | ❌             | ❌               | ✅               | ❌   | ✅                  | ✅                     | ✅           | ✅      | ✅       | ✅   | ❌             | ❌             | ✅         | ✅             | ✅          | ✅             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| Sarathi-Serve          | ❌               | ❌                   | ❌             | ✅               | ❌               | ❌   | ✅                  | ✅                     | ❌           | ❌      | ✅       | ❌   | ❌             | ✅             | ✅         | ✅             | ❌          | ✅             | ❌             | ❌             | ❌             | ❌                     | ❌               |
| Friendli Inference     | -                | ✅                   | -              | -                | -                | -    | ✅                  | ✅                     | ✅           | -       | ✅       | ✅   | -              | -              | -          | -              | ❌          | -              | -              | ❌             | ✅             | ✅                     | ✅               |
| Fireworks AI           | -                | ✅                   | -              | -                | -                | -    | -                   | -                      | ✅           | ✅      | ✅       | ✅   | ✅             | -              | ✅         | -              | ❌          | -              | -              | ❌             | ✅             | ✅                     | ✅               |
| GroqCloud              | -                | -                    | -              | -                | ✅               | -    | ✅                  | ✅                     | ✅           | ✅      | ✅       | -    | -              | -              | -          | -              | ❌          | -              | -              | ❌             | ✅             | ✅                     | ✅               |
| Together Inference     | -                | -                    | -              | -                | -                | ✅   | -                   | -                      | ✅           | -       | ✅       | ✅   | ✅             | -              | -          | -              | ❌          | ✅             | -              | ❌             | ✅             | ✅                     | ✅               |

### 🧮 Numeric Precision Support Matrix

| Framework           | FP32 | FP16 | FP8  | FP4  | NF4  | BF16 | INT8 | INT4 | MXFP8 | MXFP6 | MXFP4 | MXINT8 |
|---------------------|------|------|------|------|------|------|------|------|--------|--------|--------|---------|
| Ollama              | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| LLaMA.cpp           | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| vLLM                | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| DeepSpeed-FastGen   | ✅   | ✅   | ❌   | ✅   | ❌   | ❌   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| unsloth             | ✅   | ✅   | ✅   | ❌   | ✅   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| MAX                 | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| MLC-LLM             | ✅   | ✅   | ✅   | ❌   | ❌   | ❌   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| llama2.c            | ✅   | ❌   | ❌   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| bitnet.cpp          | ✅   | ✅   | ❌   | ❌   | ❌   | ✅   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| SGLang              | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| LitGPT              | ✅   | ✅   | ❌   | ✅   | ✅   | ❌   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| OpenLLM             | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| TensorRT-LLM        | ✅   | ✅   | ✅   | ✅   | ❌   | ✅   | ✅   | ✅   | ✅     | ❌     | ✅     | ❌      |
| TGI                 | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   | ❌   | ❌     | ❌     | ❌     | ❌      |
| PowerInfer          | ✅   | ✅   | ❌   | ❌   | ❌   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| LMDeploy            | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| LightLLM            | ✅   | ✅   | ❌   | ❌   | ❌   | ✅   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| NanoFlow            | ❌   | ✅   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌   | ❌     | ❌     | ❌     | ❌      |
| DistServe           | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌     | ❌     | ❌     | ❌      |
| vAttention          | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| Sarathi-Serve       | ✅   | ✅   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌   | ❌     | ❌     | ❌     | ❌      |
| Friendli Inference  | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   | ✅   | ❌     | ❌     | ❌     | ❌      |
| Fireworks AI        | ❌   | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌     | ❌     | ❌     | ❌      |
| GroqCloud           | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌     | ❌     | ❌     | ❌      |
| Together Inference  | ❌   | ✅   | ✅   | ❌   | ❌   | ❌   | ❌   | ✅   | ❌     | ❌     | ❌     | ❌      |


### 🧭 Radar Chart: Multi-Dimensional Evaluation of LLM Inference Engines

This radar chart compares 25 inference engines across six key dimensions: general-purpose support, ease of use, ease of deployment, latency awareness, throughput awareness, and scalability.

![Six-Dimension Evaluation](assets/six_dimension_graph.png)

### 📈 Commercial Inference Engine Performance Comparison
![Inference Throughput and Latency](assets/inference_throughput_latency.png)
- Source: [Artificial Analysis](https://artificialanalysis.ai/)

### 💲 Commercial Inference Engine Pricing by Model (USD per 1M tokens)

| Model | Friendli AI† | Fireworks AI | GroqCloud | Together AI‡ |
|-------|--------------|--------------|-----------|---------------|
| DeepSeek-R1 | 3.00 / 7.00 | 3.00 / 8.00 | 0.75* / 0.99* | 3.00 / 7.00 |
| DeepSeek-V3 | - / - | 0.90 / 0.90 | - / - | 1.25 / 1.25 |
| Llama 3.3 70B | 0.60 / 0.60 | - / - | 0.59 / 0.79 | 0.88 / 0.88 |
| Llama 3.1 405B | - / - | 3.00 / 3.00 | - / - | 3.50 / 3.50 |
| Llama 3.1 70B | 0.60 / 0.60 | - / - | - / - | 0.88 / 0.88 |
| Llama 3.1 8B | 0.10 / 0.10 | - / - | 0.05 / 0.08 | 0.18 / 0.18 |
| Qwen 2.5 Coder 32B | - / - | - / - | 0.79 / 0.79 | 0.80 / 0.80 |
| Qwen QwQ Preview 32B | - / - | - / - | 0.29 / 0.39 | 1.20 / 1.20 |

- † Llama is Instruct model 
- ‡ Turbo mode price   
- * DeepSeek-R1 Distill Llama 70B

### 💲 Commercial Inference Engine Pricing by Hardware Type (USD per hour per device)

| Hardware | Friendli AI | Fireworks AI | GroqCloud | Together AI |
|----------|-------------|--------------|-----------|---------------|
| NVIDIA A100 80GB | 2.9 | 2.9 | - | 2.56 |
| NVIDIA H100 80GB | 5.6 | 5.8 | - | 3.36 |
| NVIDIA H200 141GB | - | 9.99 | - | 4.99 |
| AMD MI300X | - | 4.99 | - | - |
| Groq LPU | - | - | - | - |


## 🔭 Future Directions

Recent advancements in LLM inference engines reveal several open challenges and research opportunities:

- **Multimodal Support:** As multimodal models like [Qwen2-VL](https://arxiv.org/abs/2409.12191) and [LLaVA-1.5](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html) emerge, inference engines must support efficient handling of image, audio, and video modalities. This includes multimodal preprocessing, M-RoPE position embedding, and modality-preserving quantization.

- **Beyond Transformers:** Emerging architectures such as [RetNet](https://arxiv.org/abs/2307.08621), [RWKV](https://arxiv.org/abs/2305.13048), and [Mamba](https://openreview.net/forum?id=tEYskw1VY2#discussion) challenge the dominance of Transformers. Engines must adapt to hybrid models like [Jamba](https://arxiv.org/abs/2403.19887) that mix Mamba and Transformer components, including MoE.

- **Hardware-Aware Optimization:** Efficient operator fusion (e.g., [FlashAttention-3](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)) and mixed-precision kernels are needed for specialized accelerators like H100, NPUs, or PIMs. These require advanced tiling strategies and memory alignment.

- **Extended Context Windows:** Models now support up to 10M tokens. This creates significant pressure on KV cache management, requiring hierarchical caching, CPU offloading, and memory-efficient attention.

- **Complex Reasoning:** Support for multi-step [CoT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html?ref=https://githubhelp.com), tool usage, and [multi-turn dialogs](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) is growing. Engines must manage long token sequences and optimize session continuity and streaming outputs.

- **Application-Driven Tradeoffs:** Real-time systems (e.g., chatbots) prioritize latency, while backend systems (e.g., batch translation) prioritize throughput. Engines must offer tunable optimization profiles.

- **Security & Robustness:** Prompt injection, jailbreaks, and data leakage risks necessitate runtime moderation (e.g., [OpenAI Moderation](https://ojs.aaai.org/index.php/AAAI/article/view/26752)), input sanitization, and access control.

- **On-Device Inference:** With compact models like [Gemma](https://arxiv.org/abs/2403.08295) and [Phi-3](https://arxiv.org/abs/2404.14219), edge inference is becoming viable. This requires compression, chunk scheduling, offloading, and collaboration across devices.

- **Heterogeneous Hardware:** Support for TPUs, NPUs, AMD MI300X, and custom AI chips demands hardware-aware partitioning, adaptive quantization, and load balancing.

- **Cloud Orchestration:** Inference systems must integrate with serving stacks like [Ray](https://github.com/ray-project/ray), [Kubernetes](https://kubernetes.io/), [Triton](https://github.com/triton-inference-server/server), and [Hugging Face Spaces](https://huggingface.co/spaces) to scale reliably.


## 🤝 Contributing

We welcome community contributions! Feel free to:

- Add new inference engines or papers
- Update benchmarks or hardware support
- Submit PRs for engine usage examples or tutorials

## ⚖️ License

MIT License. See `LICENSE` for details.


## 📝 Citation

```

```