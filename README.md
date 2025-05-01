# Awesome-LLM-Inference-Engine

Welcome to the **Awesome-LLM-Inference-Engine** repository!
A curated list of LLM inference engines, optimization techniques, and deployment strategies. This repository is based on the findings of our research paper analyzing **25 open-source and commercial LLM inference engines**. It aims to help practitioners and researchers select, compare, and design efficient LLM inference infrastructure.

Our work is based on the following paper:
 **Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency**

## Table of Contents

- [Overview](#overview)
- [Taxonomy](#taxonomy)
- [Open Source Inference Engines](#open-source-inference-engines)
- [Commercial Solutions](#commercial-solutions)
- [Optimization Techniques](#optimization-techniques)
- [Comparison Table](#comparison-table)
- [Future Directions](#future-directions)
- [Contributing](#contributing)
- [License](#license)

---


## Overview

LLM services are evolving rapidly to support complex tasks such as chain-of-thought (CoT), reasoning, AI Agent workflows. These workloads significantly increase inference cost and system complexity.

This repository categorizes and compares LLM inference engines by:
- **Deployment type** (single-node vs multi-node)
- **Hardware diversity** (homogeneous vs heterogeneous)


## Taxonomy

We classify LLM inference engines along the following dimensions:
- **Ease-of-Use:** Assesses documentation quality and community activity. Higher scores indicate better developer experience and community support.
- **Ease-of-Deployment:** Measures the simplicity and speed of installation using tools like pip, APT, Homebrew, Conda, Docker, source builds, or prebuilt binaries.
- **General-purpose support:** Reflects the range of supported LLM models and hardware platforms. Higher values indicate broader compatibility across diverse model families and execution environments.
- **Scalability:** Indicates the engine’s ability to operate effectively across edge devices, servers, and multi-node deployments. Higher scores denote readiness for large-scale or distributed workloads.
- **Throughput-aware:** Captures the presence of optimization techniques focused on maximizing throughput, such as continuous batching, parallelism, and cache reuse.
- **Latency-aware:** Captures support for techniques targeting low latency, including stall-free scheduling, chunked prefill, and priority-aware execution.

## Open Source Inference Engines

- [Ollama](https://github.com/ollama/ollama) ([Webpage](https://ollama.com/))
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MAX](https://github.com/modular/modular) ([Webpage](https://www.modular.com/max/solutions/ai-inference))
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) ([Webpage](https://llm.mlc.ai/))
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) ([Paper](https://arxiv.org/abs/2312.12456), [Paper](https://arxiv.org/abs/2406.06282))
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) ([Webpage](https://huggingface.co/docs/text-generation-inference/index))
- [Unsloth](https://github.com/unslothai/unsloth) ([Webpage](https://unsloth.ai/))
- [llama2.c](https://github.com/karpathy/llama2.c)
- [bitnet.cpp](https://github.com/microsoft/BitNet)
- [OpenLLM](https://github.com/bentoml/OpenLLM) ([Webpage](https://www.bentoml.com/))
- [LightLLM](https://github.com/ModelTC/lightllm) ([Webpage](https://www.light-ai.top/lightllm-blog/blog/))
- [NanoFlow](https://github.com/efeslab/Nanoflow) ([Paper](https://arxiv.org/abs/2408.12757))
- [vAttention](https://github.com/microsoft/vattention) ([Paper](https://arxiv.org/abs/2405.04437))
- [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) ([Paper](https://arxiv.org/abs/2403.02310))
- [vLLM](https://github.com/vllm-project/vllm) ([Webpage](https://docs.vllm.ai/en/latest/)) ([Paper](https://arxiv.org/abs/2309.06180))
- [DeepSpeed-FastGen](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen)  ([Webpage](https://www.deepspeed.ai/)) ([Paper](https://arxiv.org/abs/2401.08671))
- [SGLang](https://github.com/sgl-project/sglang) ([Webpage](https://docs.sglang.ai//)) ([arXiv](https://arxiv.org/abs/2312.07104))
- [LitGPT](https://github.com/Lightning-AI/litgpt) ([Webpage](https://lightning.ai/))
- [LMDeploy](https://github.com/InternLM/lmdeploy) ([Webpage](https://lmdeploy.readthedocs.io/en/latest/))
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) ([Webpage](https://docs.nvidia.com/tensorrt-llm/index.html))
- [DistServe](https://github.com/LLMServe/DistServe) ([Paper](https://arxiv.org/abs/2401.09670))

## Commercial Inference Engines

- [Together Inference](https://www.together.ai/)
- [GroqCloud](https://groq.com/groqcloud/)
- [Fireworks AI](https://fireworks.ai/)
- [Friendli Inference](https://friendli.ai/)

## Optimization Techniques

## Comparison Table

## Future Directions

## Contributing

We welcome community contributions! Feel free to:
- Add new inference engines or papers
- Update benchmarks or hardware support
- Submit PRs for engine usage examples or tutorials

## Contributing
```
```

## License

MIT License. See `LICENSE` for details.
