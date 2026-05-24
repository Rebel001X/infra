## 第 4 章 · 推理与部署

目标:把同一个 1.5B 模型用 4 条路跑一遍,看清楚每条路在做什么优化。

### 本章 notebook

- [`01_hf_baseline.ipynb`](01_hf_baseline.ipynb) — HuggingFace baseline:transformers.generate
- [`02_vllm_paged_attention.ipynb`](02_vllm_paged_attention.ipynb) — vLLM:PagedAttention + continuous batching
- [`03_gptq_quantize.ipynb`](03_gptq_quantize.ipynb) — GPTQ 4/8bit 量化
- [`04_llama_cpp_gguf.ipynb`](04_llama_cpp_gguf.ipynb) — 转 GGUF + 纯 CPU 推理


---

## 📝 本章面试题

**Q**:vLLM 的 PagedAttention 到底解决了什么?和传统 KV cache 实现的区别在哪?为什么能把吞吐拉高这么多?

<details>
<summary>答题要点(点开)</summary>

- **传统 KV cache 的浪费**:为每个 request 预分配 `max_seq_len × hidden_dim` 的连续显存,实际只用一小段 → 内部碎片 + 早期 OOM
- **PagedAttention 的核心**:借鉴 OS 虚拟内存,把 KV cache 切成固定大小 block(16/32 tokens 一块),用 block table 索引,**按需分配**
- **直接收益**:
  - 显存利用率从 ~20% 拉到 ~96%(论文 2309.06180)
  - 同显存能塞 5-10x 并发 request
- **配套**:**continuous batching**(每个 step 都重新组 batch,完成的 request 立刻让位)+ **prefix sharing**(多个 request 共享 prompt 的 KV)
- **后续演进**:Mooncake 把 prefill 和 decode 拆机器(P/D 分离),DistServe 类似思路;DeepSeek-V3 用 MLA 把 KV cache 体积本身砍 90%

</details>

## ⚠️ 本章踩坑实录

**vLLM 0.6+ 启动 OOM,降到 0.5 才能跑**

vLLM 0.6 引入了 chunked prefill 默认开,预分配显存更激进。12GB 卡跑 1.5B 时直接 OOM。
修复:
1. `gpu_memory_utilization=0.85`(默认 0.9 太狠)
2. `max_model_len=4096`(默认会查 model config,Qwen2.5 可以到 32k,太大不需要)
3. `enable_chunked_prefill=False`(降一档吞吐换稳定)
