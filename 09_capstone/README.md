## 第 9 章 · Capstone:mini-R1 复现(GSM8K-zh)

目标:**面试压轴故事**。把第 5-6 章串起来,从 Qwen2.5-1.5B SFT 出发,用第 6 章的 verifiable reward GRPO 流程,
在 GSM8K-zh 上拿到**真实可衡量的提升**(目标:~10% → ~25%)。完整复现一次 nanochat / mini-R1 / DeepSeek-R1-Zero 路线。

讲这个比讲 100 行 LoRA 代码有用 10 倍。

### 本章 notebook

- [`mini_r1_gsm8k_zh.ipynb`](mini_r1_gsm8k_zh.ipynb) — Capstone:在 GSM8K-zh 上跑通 SFT + GRPO mini-R1


---

## 📝 本章面试题

**Q**:你怎么向 HR/面试官讲这个 mini-R1 项目?要凸显什么、回避什么?

<details>
<summary>答题要点(点开)</summary>

**5 分钟版本**:
1. **目标(15s)**:在 12GB 卡上,用 Qwen2.5-1.5B 复现 DeepSeek-R1-Zero 的 GRPO + verifiable reward 路线,在中文数学题上拿到可衡量提升
2. **关键决策**(60s):
   - 选 GSM8K-zh 因为 reward 完全规则化(答案精确匹配),不需要训 reward model,可在单卡完成
   - 选 GRPO 不选 PPO 因为 PPO 需要 value model,12GB 跑不下;GRPO 用 group baseline 替代
3. **数字**(60s):baseline 10.3% → SFT 后 14.1% → GRPO 后 24.7%,reward 曲线在 100 步后开始稳定上升,`<think>...</think>` 在 80 步左右涌现
4. **踩坑**(60s):一开始 reward 给得太稀疏(只看答案对),模型不学 reasoning;改成 format reward 0.5 + answer reward 1.0 后曲线立刻上升
5. **凸显**:对 GRPO 原理(group baseline vs PPO value model)+ verifiable reward 设计 + 单卡工程化的理解
6. **回避**:不要扯"全量 R1 复现"(R1 是 671B,本项目是 1.5B 教学版),诚实说是 mini-R1

</details>

## ⚠️ 本章踩坑实录

**GRPO 跑着跑着 reward 崩了(reward hacking)**

常见症状:reward 曲线突然飙到天花板,但生成出来的全是格式垃圾(模型学会了刷 format reward 而不解题)。
**修复**:把 format reward 的权重从 0.5 降到 0.1,或改成"format 错则 -1 但 format 对不加分",让 answer 成为唯一正反馈。

