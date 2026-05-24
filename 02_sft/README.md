## 第 2 章 · 监督微调(SFT)

目标:把 Qwen2.5-1.5B 微调成一个能听话的中文助手,顺便对比全参 / LoRA / QLoRA 三条路的工程权衡。

### 本章 notebook

- [`01_dataset_alpaca_zh.ipynb`](01_dataset_alpaca_zh.ipynb) — SFT 数据:Alpaca-zh + chat template + 损失 mask
- [`02_full_param_sft_1_5b.ipynb`](02_full_param_sft_1_5b.ipynb) — Qwen2.5-1.5B 全参 SFT(主推路径)
- [`03_lora_sft.ipynb`](03_lora_sft.ipynb) — LoRA SFT 对比
- [`04_qlora_3b.ipynb`](04_qlora_3b.ipynb) — QLoRA:4bit 把 3B 塞进 12GB
