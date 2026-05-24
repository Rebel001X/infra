# 文档索引

- [学习路径](学习路径.md) — 8 周节奏 + 速通模式
- [硬件与环境](硬件与环境.md) — CUDA / 驱动 / bitsandbytes / Unsloth / HF 镜像踩坑
- [工具栈选型](工具栈选型.md) — Unsloth vs TRL、Liger Kernel、verifiable reward GRPO 的取舍
- [与 llm-action 八股对照](与llm-action八股对照.md) — 每个 notebook 对应中文八股的哪一篇

## 章节快速跳转(9 章)

| # | 目录 | 主题 | 备注 |
|---|---|---|---|
| 01 | [`../01_pretrain_toy/`](../01_pretrain_toy/) | toy nanoGPT pretrain | 风格样板 |
| 02 | [`../02_data/`](../02_data/) | CC 清洗 + MinHash 去重 | ★ v2 新增 |
| 03 | [`../03_eval/`](../03_eval/) | lm-eval + CMMLU + C-Eval | ★ v2 新增,前置到 SFT 之前 |
| 04 | [`../04_midtrain/`](../04_midtrain/) | SmolTalk 对话格式注入 | ★ v2 新增,nanochat 启发 |
| 05 | [`../05_sft/`](../05_sft/) | 全参 + Liger + Unsloth LoRA / QLoRA | 工具栈升级 |
| 06 | [`../06_alignment/`](../06_alignment/) | DPO + **verifiable reward GRPO** | GRPO 改用规则化 reward |
| 07 | [`../07_systems/`](../07_systems/) | Triton FlashAttn2 + DDP + FSDP | ★ v2 新增,面试核心 |
| 08 | [`../08_inference_deploy/`](../08_inference_deploy/) | vLLM + 推测解码 + 量化对比 + GGUF | 加 speculative + AWQ |
| 09 | [`../09_capstone/`](../09_capstone/) | **Capstone:mini-R1 复现** | ★ v2 新增,面试压轴 |
