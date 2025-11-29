# Agent0 Series: Self-Evolving Agents from Zero Data

<div align="center">

[![Website](https://img.shields.io/badge/🌐-Website-blue)](https://aiming-lab.github.io/Agent0)
[![Agent0 Paper](https://img.shields.io/badge/📄-Agent0%20Paper-b31b1b)](https://arxiv.org/abs/2511.16043)
[![Agent0-VL Paper](https://img.shields.io/badge/📄-Agent0--VL%20Paper-b31b1b)](https://arxiv.org/abs/2511.19900)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Unleashing Autonomous Agent Evolution via Tool-Integrated Reasoning**

[UNC-Chapel Hill](https://cs.unc.edu/) · [Salesforce Research](https://www.salesforceairesearch.com/) · [Stanford University](https://cs.stanford.edu/)

</div>
<p align="center">
  <img src="figs/logo.png" width="60%">
</p>



## 🔥 News

- **[11/29/2025]** The code of Agent0 was released!
- **[11/26/2025]** We’ve set up a Discord server and WeChat group to make it easier to collaborate and exchange ideas on this project. Welcome to join the Group to share your thoughts, ask questions, or contribute your ideas! 🔥 Join our [Discord](https://discord.gg/WDjBNRbu) and [WeChat Group](figs/wechat_group.jpg) Now!
- **[11/25/2025]** Agent0-VL was released on [arXiv](https://arxiv.org/abs/2511.19900)!
- **[11/20/2025]** Agent0 paper was released on [arXiv](https://arxiv.org/abs/2511.16043)!

---

## 📖 Overview

The **Agent0 Series** explores a new direction for autonomous agent development, showing that capable agents can improve and evolve without relying on human-curated datasets or handcrafted supervision. This repository brings together two complementary studies that advance self-improving agents through tool-integrated reasoning.

### 🤖 [Agent0](./Agent0): Self-Evolving Language Agents

**Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning**

A fully autonomous framework that evolves high-performing language agents through multi-step co-evolution and seamless tool integration. Agent0 establishes a symbiotic competition between two agents:

- **Curriculum Agent**: Proposes increasingly challenging frontier tasks
- **Executor Agent**: Learns to solve them using external tools

**Key Results:**

- ✅ **+18%** improvement on mathematical reasoning benchmarks
- ✅ **+24%** improvement on general reasoning benchmarks
- ✅ Zero external data required for training
- ✅ Multi-turn interaction support

[📄 Paper](https://arxiv.org/abs/2511.16043) | [📁 Code](./Agent0) | [🔗 Details](./Agent0/README.md)

---

### 👁️ [Agent0-VL](./Agent0-VL): Self-Evolving Vision-Language Agents

**Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning**

A self-evolving vision-language agent that extends the Agent0 paradigm to multimodal reasoning tasks. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair through a dual-role architecture:

- **Solver**: Performs multi-turn tool-integrated reasoning
- **Verifier**: Generates structured feedback and fine-grained self-rewards

**Key Results:**

- ✅ **+12.5%** average improvement on visual reasoning benchmarks
- ✅ **+7.3%** improvement in test-time scaling performance
- ✅ State-of-the-art among open-source vision-language models
- ✅ Zero external reward for self-evolution

[📄 Paper](https://arxiv.org/abs/2511.19900) | [📁 Code](./Agent0-VL) | [🔗 Details](./Agent0-VL/README.md)

---

## 🎯 Key Features

### Shared Philosophy

Both Agent0 and Agent0-VL are built on the principle of **zero-data self-evolution**:

- **No Human Annotations**: Completely eliminates dependency on external data or human supervision
- **Tool-Integrated Reasoning**: Leverages external tools to enhance problem-solving capabilities
- **Autonomous Evolution**: Self-generates training data through intelligent exploration

---

## 📊 Results Summary

### Agent0: Language Reasoning

#### Mathematical Reasoning Benchmarks (Qwen3-8B-Base)

Complete comparison with state-of-the-art self-evolving methods:

| Model              | AVG            | AMC            | Minerva        | MATH           | GSM8K          | Olympiad       | AIME25         | AIME24         |
| ------------------ | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Base Model         | 49.2           | 52.0           | 50.0           | 78.0           | 89.1           | 44.7           | 16.7           | 13.9           |
| Base Model w/ Tool | 53.2           | 60.3           | 54.9           | 79.2           | 90.7           | 47.9           | 18.7           | 20.9           |
| + Absolute Zero    | 52.6           | 62.5           | 52.9           | 76.6           | 92.0           | 47.8           | 18.2           | 18.4           |
| + R-Zero           | 54.7           | 61.7           | 60.7           | 82.0           | 94.1           | 48.9           | 19.2           | 16.4           |
| + Socratic-Zero    | 56.1           | **63.7** | 52.4           | 81.2           | 87.3           | **55.1** | 24.5           | **28.4** |
| **+ Agent0** | **58.2** | 62.4           | **61.3** | **82.4** | **94.5** | 54.0           | **24.8** | 28.0           |

**Key Improvements:**

- 📈 **+18.3%** over base model (49.2 → 58.2)
- 🎯 **+6.4%** over R-Zero (54.7 → 58.2)
- 🔥 **+3.7%** over Socratic-Zero (56.1 → 58.2)

#### General Reasoning Benchmarks (Qwen3-8B-Base)

| Model              | Overall AVG    | MATH AVG       | SuperGPQA      | MMLU-Pro       | BBEH           |
| ------------------ | -------------- | -------------- | -------------- | -------------- | -------------- |
| Base Model         | 34.5           | 49.2           | 28.3           | 51.8           | 8.6            |
| Base Model w/ Tool | 36.7           | 53.2           | 29.5           | 54.8           | 9.37           |
| + Absolute Zero    | 39.9           | 52.6           | **33.5** | 62.5           | 10.8           |
| + R-Zero           | 38.7           | 54.7           | 31.4           | 58.2           | 10.6           |
| + Socratic-Zero    | 39.2           | 56.1           | 30.1           | 60.9           | 9.5            |
| **+ Agent0** | **42.1** | **58.2** | 33.0           | **63.4** | **13.7** |

**Key Improvements:**

- 📈 **+22.0%** over base model (34.5 → 42.1)
- 🎯 **+5.5%** over Absolute Zero (39.9 → 42.1)
- 🔥 Highest overall performance among all self-evolving methods

---

### Agent0-VL: Visual Reasoning

#### Main Results on Visual Reasoning Benchmarks

Comprehensive comparison with closed-source and open-source models:

| Model Category           | Model                  | MathVerse      | MathVision     | MathVista      | WeMath         | HallBench      | ChartQA        | MMMU           | **Avg.** |
| ------------------------ | ---------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| **Closed-Source**  | GPT-4o                 | 50.8           | 30.4           | 63.8           | 68.8           | 55.0           | 85.7           | 69.1           | 60.5           |
|                          | OpenAI-o1              | 57.0           | 60.3           | 73.9           | -              | -              | 83.1           | 77.6           | -              |
|                          | Claude-3.7-Sonnet      | 52.0           | 41.3           | 66.8           | 72.6           | 55.4           | 56.5           | 75.0           | 59.9           |
| **Open General**   | InternVL-2.5-8B        | 39.5           | 19.7           | 64.4           | 53.5           | 61.7           | 79.1           | 62.7           | 54.4           |
|                          | InternVL-3-8B          | 39.8           | 29.3           | 71.6           | 58.1           | 64.3           | 85.9           | 60.7           | 58.5           |
|                          | Qwen2.5-VL-7B          | 46.3           | 25.1           | 67.8           | 62.1           | 65.0           | 83.5           | 58.6           | 58.3           |
|                          | Qwen2.5-VL-7B-TIR      | 47.2           | 26.3           | 68.1           | 63.7           | 67.2           | 84.1           | 59.6           | 59.5           |
|                          | Qwen3-VL-8B            | 62.1           | 53.9           | 77.2           | 72.5           | 72.1           | 84.6           | 69.6           | 70.3           |
|                          | Qwen3-VL-8B-TIR        | 63.1           | 54.7           | 79.4           | 73.1           | 72.8           | 85.4           | 70.9           | 71.3           |
| **Open Reasoning** | Vision-R1-7B           | 51.9           | 30.7           | 73.5           | 73.9           | 68.8           | 79.8           | 50.5           | 61.3           |
|                          | OpenVLThinker-7B       | 45.7           | 26.3           | 71.2           | 66.7           | 70.2           | 78.4           | -              | -              |
|                          | MM-Eureka-7B           | 50.5           | 27.9           | 73.6           | 67.4           | 66.9           | 82.1           | 52.7           | 60.2           |
|                          | ThinkLite-VL-7B        | 52.1           | 32.9           | 75.1           | 69.3           | 70.9           | 84.8           | 55.5           | 62.9           |
|                          | Thyme-VL-7B            | 51.3           | 27.6           | 70.0           | -              | 71.0           | 86.1           | -              | -              |
| **Ours**           | **Agent0-VL-7B** | **53.1** | **37.3** | **75.6** | **71.7** | **72.9** | **87.3** | **61.1** | **65.6** |
|                          | **Agent0-VL-8B** | **65.5** | **56.2** | **83.7** | **79.6** | **74.3** | **89.7** | **73.4** | **74.6** |

**Key Improvements (Agent0-VL-7B):**

- 📈 **+12.5%** over Qwen2.5-VL-7B base (58.3 → 65.6)
- 🎯 **+10.3%** over Qwen2.5-VL-7B-TIR (59.5 → 65.6)
- 🔥 **+4.3%** over ThinkLite-VL-7B (62.9 → 65.6)
- 🏆 **Best among all open-source 7B models**

**Key Improvements (Agent0-VL-8B):**

- 📈 **+6.1%** over Qwen3-VL-8B base (70.3 → 74.6)
- 🎯 **+4.6%** over Qwen3-VL-8B-TIR (71.3 → 74.6)
- 🔥 Outperforms GPT-4o on MathVista, HallBench, and ChartQA
- 🏆 **State-of-the-art among all open-source models**

#### Iterative Self-Evolution Performance (Agent0-VL-7B)

| Stage                 | MathVerse      | MathVision     | MathVista      | WeMath         | HallBench      | ChartQA        | MME-Real       | MMMU           | **Avg.** |
| --------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Base Model            | 46.3           | 25.1           | 67.8           | 62.1           | 65.0           | 83.5           | 58.3           | 50.6           | 57.3           |
| Iteration 1           | 48.4           | 29.6           | 69.2           | 66.8           | 67.9           | 84.7           | 63.9           | 53.7           | 60.5           |
| Iteration 2           | 51.1           | 35.3           | 72.8           | 70.1           | 70.3           | 86.1           | 64.7           | 58.3           | 63.6           |
| **Iteration 3** | **53.1** | **37.3** | **75.6** | **71.7** | **72.9** | **87.3** | **65.3** | **61.1** | **65.5** |

**Evolution Progress:**

- 🔄 Iter 1: **+5.2%** improvement (57.3 → 60.5)
- 🔄 Iter 2: **+4.0%** additional gain (60.5 → 63.6)
- 🔄 Iter 3: **+2.8%** further improvement (63.6 → 65.5)
- ✅ **+8.2%** cumulative gain over base model

---

## 📚 Citation

If you find our work helpful, please consider citing:

### Agent0

```bibtex
@article{xia2025agent0,
  title={Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning},
  author={Xia, Peng and Zeng, Kaide and Liu, Jiaqi and Qin, Can and Wu, Fang and Zhou, Yiyang and Xiong, Caiming and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.16043},
  year={2025}
}
```

### Agent0-VL

```bibtex
@article{liu2025agent0vl,
  title={Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning},
  author={Liu, Jiaqi and Xiong, Kaiwen and Xia, Peng and Zhou, Yiyang and Ji, Haonian and Feng, Lu and Han, Siwei and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.19900},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the open-source community for their foundational work that made this research possible. Special thanks to:

- The teams behind Qwen, InternVL, and other base models
- The VeRL team for their excellent RL framework
- All the benchmark creators and maintainers

[![✨Star History Chart](https://api.star-history.com/svg?repos=aiming-lab/Agent0&type=Date)](https://star-history.com/#bytebase/star-history&Date)
