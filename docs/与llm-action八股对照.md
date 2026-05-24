# 与 llm-action 八股对照(v2,9 章)

`liguodongiot/llm-action` 是国内 AI-Infra 面试最完整的中文八股库(本机克隆在 `C:\Users\jianm\Desktop\llm-action`)。本 repo 每个 notebook **跑完后**,对照下面这张表回去读对应的八股 md,把「我跑过」升级成「我能背」。

## 01 章 · Pretrain

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_tokenizer.ipynb` | `llm-data-engineering/` + `llm-interview/base.md` (Tokenizer 部分) | BPE / BBPE / SentencePiece 区别 |
| `02_data_pipeline.ipynb` | `llm-data-engineering/` | sequence packing、streaming、去重 |
| `03_nano_decoder.ipynb` | `llm-interview/llm-algo.md` | RoPE / RMSNorm / SwiGLU / GQA |
| `04_train_loop.ipynb` | `llm-interview/llm-train.md` | AMP / 梯度累积 / checkpoint |

## 02 章 · 数据工程 ★

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_cc_cleaning.ipynb` | `llm-data-engineering/` | datatrove pipeline、fastText 语言识别、Gopher 质量过滤 |
| `02_dedup_minhash.ipynb` | `llm-data-engineering/` | MinHash + LSH、Jaccard,去重对模型记忆的影响 |

## 03 章 · 评测前置 ★

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_lm_eval_harness.ipynb` | `llm-interview/llm-eval.md` | lm-eval-harness、MMLU、`acc_norm` |
| `02_cmmlu_ceval_zh.ipynb` | `llm-interview/llm-eval.md` | CMMLU / C-Eval 差异、letter bias、contamination |

## 04 章 · Midtrain ★

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_smoltalk_zh_injection.ipynb` | 暂无对应(2025+ 新概念) | base → midtrain → SFT 三阶段、catastrophic forgetting |

## 05 章 · SFT

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_dataset_alpaca_zh.ipynb` | `llm-interview/llm-ft.md` (数据章) | chat template、ChatML、损失 mask |
| `02_full_param_sft_liger.ipynb` | `llm-interview/llm-ft.md` | 全参 vs PEFT、**Liger Kernel 加速原理** |
| `03_unsloth_lora.ipynb` | `llm-interview/llm-ft.md` (LoRA) | r / alpha / target_modules、**Unsloth 怎么省显存** |
| `04_unsloth_qlora_3b.ipynb` | `llm-interview/llm-ft.md` (QLoRA) + `llm-compression/` | NF4 / double quant / 显存账 |

## 06 章 · 对齐

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_dpo_理论.ipynb` | `llm-interview/llm-rlhf.md` (DPO) | Bradley-Terry / DPO loss 推导 |
| `02_dpo_unsloth.ipynb` | `llm-interview/llm-rlhf.md` | reference model / beta 参数 |
| `03_grpo_verifiable_reward.ipynb` | `llm-interview/llm-rlhf.md` (GRPO) + `llm-alignment/` | **verifiable reward** vs reward model、GRPO vs PPO |

## 07 章 · Systems ★

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_triton_flashattn2_mini.ipynb` | `llm-interview/llm-algo.md` + `ai-infra/` | tiling、online softmax、SRAM-HBM、FA v1/v2/v3 |
| `02_torchrun_ddp_demo.ipynb` | `llm-interview/llm-train.md` (分布式) | DDP 三件套、AllReduce、ring-tree |
| `03_fsdp_demo.ipynb` | `llm-interview/llm-train.md` (FSDP) | ZeRO-1/2/3 取舍、参数/梯度/optimizer 分片 |
| `04_pipeline_parallel.md` | `llm-train/分布式训练.md` | GPipe / 1F1B / DualPipe |

## 08 章 · 推理部署

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `01_hf_baseline.ipynb` | `llm-inference/` | KV cache、prefill vs decode |
| `02_vllm_paged_attention.ipynb` | `llm-inference/Mooncake.md` + `llm-inference/PD分离.md` | PagedAttention、continuous batching |
| `03_speculative_decoding.ipynb` ★ | `llm-inference/` | 推测解码:draft + target、acceptance rate |
| `04_gptq_awq_smoothquant_compare.ipynb` | `llm-compression/` + `llm-interview/llm-compress.md` | 三方法对比 |
| `05_gguf_llamacpp.ipynb` | `llm-inference/` (llama.cpp 节) | GGUF / k-quants |
| `06_tensorrt_llm_intro.md` ★ | `llm-inference/` | TRT-LLM 与 vLLM 差异 |

## 09 章 · Capstone ★

| 本仓 notebook | llm-action 对应 | 关键点 |
|---|---|---|
| `mini_r1_gsm8k_zh.ipynb` | `llm-interview/llm-rlhf.md` + `llm-alignment/` | DeepSeek-R1 路线、verifiable reward、面试 5 分钟讲解 |

## 跨章 / 进阶论文(用 `/paper fetch <arxiv-id>` 拉到本地)

| 主题 | arxiv |
|---|---|
| FlashAttention v1/v2/v3 | 2205.14135 / 2307.08691 / 2407.08608 |
| vLLM PagedAttention | 2309.06180 |
| Mooncake P/D 分离 | 2407.00079 |
| DeepSeek V3(MLA / FP8 / DualPipe) | 2412.19437 |
| **DeepSeek-R1** | 2501.12948 |
| MegaScale | 2401.02385 |
| DistServe | 2401.09670 |
| LoRA | 2106.09685 |
| QLoRA | 2305.14314 |
| DPO | 2305.18290 |
| GRPO(DeepSeekMath) | 2402.03300 |
| ZeRO | 1910.02054 |
| Speculative Decoding | 2302.01318 |
| FineWeb | 2406.17557 |
| Deduplicating Training Data | 2107.06499 |
