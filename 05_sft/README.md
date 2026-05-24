## 第 2 章 · 监督微调(SFT)

目标:把 Qwen2.5-1.5B 微调成一个能听话的中文助手,顺便对比全参 / LoRA / QLoRA 三条路的工程权衡。

### 本章 notebook

- [`01_dataset_alpaca_zh.ipynb`](01_dataset_alpaca_zh.ipynb) — SFT 数据:Alpaca-zh + chat template + 损失 mask
- [`02_full_param_sft_1_5b.ipynb`](02_full_param_sft_1_5b.ipynb) — Qwen2.5-1.5B 全参 SFT(主推路径)
- [`03_lora_sft.ipynb`](03_lora_sft.ipynb) — LoRA SFT 对比
- [`04_qlora_3b.ipynb`](04_qlora_3b.ipynb) — QLoRA:4bit 把 3B 塞进 12GB


---

## 📝 本章面试题

**Q**:LoRA / QLoRA / 全参 SFT 三条路,各自的工程权衡是什么?面试官问"为什么不全用 LoRA"你怎么答?

<details>
<summary>答题要点(点开)</summary>

- **全参**:效果上限最高,但显存大(Qwen-1.5B bf16 至少 6GB 参数 + 6GB 梯度 + 12GB 优化器 = 24GB,grad ckpt 后能压到 10-12GB)
- **LoRA**:只训 0.5-2% 参数,显存大降,但**有效秩受限**(r 太小学不到、r 大显存又上去)
- **QLoRA**:4bit base + LoRA,8GB 卡能跑 7B,但 NF4 反量化会损失推理精度,合并 adapter 后效果略低于纯 LoRA
- **何时选什么**:
  - 小数据(<10k)+ 大模型:LoRA(不容易过拟合)
  - 大数据(>100k)+ 小模型:全参(LoRA 容量不够)
  - 卡不够:QLoRA(降一档质量换显存)
- **本仓:Liger Kernel 让全参 1.5B 能在 12GB 卡上跑,改写了"小卡只能 LoRA"的旧规则**

</details>

## ⚠️ 本章踩坑实录

**SFT 时忘 mask prompt 部分的 loss,效果跑废**

`SFTTrainer` 默认对整段 token 算 loss(包括 system + user prompt),这会让模型也学着复读问题。
修复:用 `DataCollatorForCompletionOnlyLM` 或在 `formatting_func` 里把 prompt token 的 label 设为 `-100`。
确认方法:训练前打一个 batch,看 `labels` 里 prompt 部分是不是 `-100`。
