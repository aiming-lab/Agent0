# Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2511.16043-b31b1b.svg)](https://arxiv.org/abs/2511.16043)

This repository is the official implementation of **Agent 0**.

![](./figs/fig1.png)

**Agent0** is a fully autonomous framework that evolves high-performing agents from scratch without relying on any human-curated data. It employs a symbiotic competition between a **Curriculum Agent** (proposing tasks) and an **Executor Agent** (solving tasks with tools), driving a self-reinforcing cycle of improvement.

## 🔥 Key Features
* **Zero Data:** Eliminates dependency on external data or human annotations.
* **Co-Evolution:** Establishes a competition where the curriculum agent creates increasingly difficult tasks and the executor agent learns to solve them.
* **Tool Integration:** Integrates external tools (e.g., code interpreter) to enhance problem-solving and pressure the curriculum agent to generate complex, tool-aware tasks.

## 📊 Results
Empirically, Agent 0 substantially boosts reasoning capabilities on the Qwen3-8B-Base model:
* **+18%** improvement on Mathematical Reasoning benchmarks.
* **+24%** improvement on General Reasoning benchmarks.

## 🚀 Code
**Coming soon.**

## 🖊️ Citation
If you find this work helpful, please consider citing our paper:

```bibtex
@article{xia2025agent0,
  title={Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning},
  author={Xia, Peng and Zeng, Kaide and Liu, Jiaqi and Qin, Can and Wu, Fang and Zhou, Yiyang and Xiong, Caiming and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.16043},
  year={2025}
}
