# 与 llm-action 八股对照

`liguodongiot/llm-action` 是国内 AI-Infra 面试最完整的中文八股库(本机克隆在 `C:\Users\jianm\Desktop\llm-action`)。本 repo 每个 notebook **跑完后**,对照下面这张表回去读对应的八股 md,把"我跑过"升级成"我能背"。

## 01 章 · Pretrain

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_tokenizer.ipynb` | `llm-data-engineering/` + `llm-interview/base.md` (Tokenizer 部分) | BPE / BBPE / SentencePiece 区别 |
| `02_data_pipeline.ipynb` | `llm-data-engineering/` | sequence packing、streaming、去重 |
| `03_nano_decoder.ipynb` | `llm-interview/llm-algo.md` | RoPE / RMSNorm / SwiGLU / GQA |
| `04_train_loop.ipynb` | `llm-interview/llm-train.md` | AMP / 梯度累积 / checkpoint |

## 02 章 · SFT

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_dataset_alpaca_zh.ipynb` | `llm-interview/llm-ft.md` (数据章) | chat template、ChatML、损失 mask |
| `02_full_param_sft_1_5b.ipynb` | `llm-interview/llm-ft.md` | 全参 vs PEFT 的工程取舍 |
| `03_lora_sft.ipynb` | `llm-interview/llm-ft.md` (LoRA) | r / alpha / target_modules |
| `04_qlora_3b.ipynb` | `llm-interview/llm-ft.md` (QLoRA) + `llm-compression/` | NF4 / double quant / 显存账 |

## 03 章 · 对齐

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_dpo_理论.ipynb` | `llm-interview/llm-rlhf.md` (DPO) | Bradley-Terry / DPO loss 推导 |
| `02_dpo_train.ipynb` | `llm-interview/llm-rlhf.md` | reference model / beta 参数 |
| `03_grpo_deepseek_style.ipynb` | `llm-interview/llm-rlhf.md` (GRPO) + `llm-alignment/` | GRPO vs PPO、reward shaping |

## 04 章 · 推理部署

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_hf_baseline.ipynb` | `llm-inference/` | KV cache 基本原理 |
| `02_vllm_paged_attention.ipynb` | `llm-inference/Mooncake.md` + `llm-inference/PD分离.md` + 论文 [[Papers/arxiv-2309.06180]] | PagedAttention / continuous batching |
| `03_gptq_quantize.ipynb` | `llm-compression/` + `llm-interview/llm-compress.md` | GPTQ / AWQ / SmoothQuant |
| `04_llama_cpp_gguf.ipynb` | `llm-inference/` (llama.cpp 节) | GGUF 格式 / k-quants |

## 跨章 / 进阶论文(用 `/paper fetch <arxiv-id>` 拉到本地)

| 主题 | arxiv |
|---|---|
| FlashAttention v1/v2/v3 | 2205.14135 / 2307.08691 / 2407.08608 |
| vLLM PagedAttention | 2309.06180 |
| Mooncake P/D 分离 | 2407.00079 |
| DeepSeek V3(MLA / FP8 / DualPipe) | 2412.19437 |
| MegaScale | 2401.02385 |
| DistServe | 2401.09670 |
| LoRA | 2106.09685 |
| QLoRA | 2305.14314 |
| DPO | 2305.18290 |
| GRPO(DeepSeekMath) | 2402.03300 |
