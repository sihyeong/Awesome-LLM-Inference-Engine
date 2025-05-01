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

## Commercial Solutions

## Optimization Techniques

## Comparison Table

## Future Directions

## Contributing

We welcome community contributions! Feel free to:
- Add new inference engines or papers
- Update benchmarks or hardware support
- Submit PRs for engine usage examples or tutorials

## License

MIT License. See `LICENSE` for details.
