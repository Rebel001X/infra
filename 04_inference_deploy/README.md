## 第 4 章 · 推理与部署

目标:把同一个 1.5B 模型用 4 条路跑一遍,看清楚每条路在做什么优化。

### 本章 notebook

- [`01_hf_baseline.ipynb`](01_hf_baseline.ipynb) — HuggingFace baseline:transformers.generate
- [`02_vllm_paged_attention.ipynb`](02_vllm_paged_attention.ipynb) — vLLM:PagedAttention + continuous batching
- [`03_gptq_quantize.ipynb`](03_gptq_quantize.ipynb) — GPTQ 4/8bit 量化
- [`04_llama_cpp_gguf.ipynb`](04_llama_cpp_gguf.ipynb) — 转 GGUF + 纯 CPU 推理
