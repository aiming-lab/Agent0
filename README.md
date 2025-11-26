# Agent0 Series: Self-Evolving Agents from Zero Data

<div align="center">

[![Website](https://img.shields.io/badge/🌐-Website-blue)](https://aiming-lab.github.io/Agent0)
[![Agent0 Paper](https://img.shields.io/badge/📄-Agent0%20Paper-b31b1b)](https://arxiv.org/abs/2511.16043)
[![Agent0-VL Paper](https://img.shields.io/badge/📄-Agent0--VL%20Paper-b31b1b)](https://arxiv.org/abs/2511.19900)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Unleashing Autonomous Agent Evolution via Tool-Integrated Reasoning**

[UNC-Chapel Hill](https://cs.unc.edu/) · [Salesforce Research](https://www.salesforceairesearch.com/) · [Stanford University](https://cs.stanford.edu/)

</div>

---

## 🔥 News

- **[2025/11/25]** Agent0-VL is released on [arXiv](https://arxiv.org/abs/2511.19900)!
- **[2025/11]** Agent0 paper was released on [arXiv](https://arxiv.org/abs/2511.16043)!

---

## 📖 Overview

The **Agent0 Series** represents a breakthrough in autonomous agent development, demonstrating that high-performing agents can evolve entirely from scratch without relying on any human-curated data or external supervision. This repository contains two complementary works that push the boundaries of self-evolving agents through tool-integrated reasoning:

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

**Mathematical Reasoning** (Qwen3-8B-Base):
| Benchmark | Base Model | Agent0 | Improvement |
|-----------|------------|--------|-------------|
| AVG | 49.2 | **58.2** | +18.3% |
| MATH | 78.0 | **82.4** | +5.6% |
| GSM8K | 89.1 | **94.5** | +6.1% |

**General Reasoning** (Qwen3-8B-Base):
| Benchmark | Base Model | Agent0 | Improvement |
|-----------|------------|--------|-------------|
| Overall AVG | 34.5 | **42.1** | +22.0% |
| MMLU-Pro | 51.8 | **63.4** | +22.4% |

### Agent0-VL: Visual Reasoning

**Visual Reasoning Benchmarks**:
| Model | MathVerse | MathVision | MathVista | WeMath | Avg. |
|-------|-----------|------------|-----------|---------|------|
| Qwen2.5-VL-7B | 46.3 | 25.1 | 67.8 | 62.1 | 58.3 |
| **Agent0-VL-7B** | **53.1** | **37.3** | **75.6** | **71.7** | **65.6** |
| **Agent0-VL-8B** | **65.5** | **56.2** | **83.7** | **79.6** | **74.6** |

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

