## 第 3 章 · 评测前置(lm-eval-harness + 中文 benchmark)

目标:**先建 baseline,再做优化**。这章把 Qwen2.5-1.5B 在英文 + 中文 benchmark 上的初始分数跑出来,
后续 SFT/对齐/量化每章都用同一份脚本对照,避免"感觉变好了"式的幻觉评估。

参考 Stanford CS336 把 Eval 前置的设计 + HF smol-course 单元 2。

### 本章 notebook

- [`01_lm_eval_harness.ipynb`](01_lm_eval_harness.ipynb) — lm-evaluation-harness 跑英文 benchmark
- [`02_cmmlu_ceval_zh.ipynb`](02_cmmlu_ceval_zh.ipynb) — 中文 benchmark:CMMLU + C-Eval


---

## 📝 本章面试题

**Q**:做 LLM 评测时,为什么 perplexity 不够?面试常聊的 generative benchmark 有哪些坑?

<details>
<summary>答题要点(点开)</summary>

- **Perplexity 的问题**:只衡量"模型对训练分布的拟合度",不衡量"指令跟随能力"。SFT 后 PPL 可能变差但 chat 能力变好
- **Multiple-choice 偏差**:
  - **Letter bias**:Qwen 系喜欢选 A,Llama 系喜欢选 D,要用 `acc_norm`(per-token 长度归一化)抑制
  - **Calibration**:5-shot vs 0-shot 分数差异大,benchmark 要 lock 配置
  - **Contamination**:很多 base model 训练时已经吃过 MMLU,分数虚高(看 SOLAR 论文 contamination 章节)
- **生成式评测的痛**:GSM8K 要做 chain-of-thought 提取 + 数字精确匹配,正则要写好;HumanEval 要起 Docker sandbox 跑代码

</details>

## ⚠️ 本章踩坑实录

**lm-eval-harness 在 Windows 跑不通**

`vllm` backend 在 Windows 不支持,只能用 `hf` (transformers) backend 走 `.generate()`,慢但能跑。
若有 WSL2,直接装 `lm-eval[vllm]`,推理速度差 5-10 倍。

