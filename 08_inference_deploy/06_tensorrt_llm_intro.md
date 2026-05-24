# 06 · TensorRT-LLM(概念笔记)

> **为什么不强求本地跑**:TensorRT-LLM 依赖特定版本 CUDA + NVIDIA Container,Windows 原生坑深;国内云上 H800 才有 license。
> 本节只过概念,知道它做了什么、和 vLLM 区别在哪。

## 与 vLLM 的对比

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| 后端 | PagedAttention + CUDA kernel | NVIDIA TensorRT 优化图 |
| 易用性 | `pip install`,Python 友好 | C++/Triton Server 部署,陡 |
| 性能(B200/H100) | 90% TRT 水平 | 100%,FP8/FP4 全 fuse |
| 适用场景 | 研发 / 中小规模线上 | 大规模线上 + NVIDIA 专属优化 |

## 关键概念

- **Plugin**:GPT/Llama/Qwen 的 attention/MLP 都被替换成手写 plugin,绕开 TensorRT 通用算子的开销
- **In-flight batching**:类似 vLLM 的 continuous batching
- **量化**:FP8 / SmoothQuant / AWQ-Marlin / FP4(B200),全栈 fuse 到 kernel

## 推荐阅读

- [TensorRT-LLM 官方](https://github.com/NVIDIA/TensorRT-LLM)
- 国内观察:`llm-action/llm-inference/` 有专章
