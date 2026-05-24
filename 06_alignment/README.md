## 第 3 章 · 偏好对齐(DPO + GRPO)

目标:DPO 推导手写一遍,GRPO 跟着 DeepSeek-R1 路线复现一个迷你版。

### 本章 notebook

- [`01_dpo_理论.ipynb`](01_dpo_理论.ipynb) — DPO 理论:Bradley-Terry → DPO loss 推导
- [`02_dpo_train.ipynb`](02_dpo_train.ipynb) — TRL DPOTrainer 跑中文偏好对齐
- [`03_grpo_deepseek_style.ipynb`](03_grpo_deepseek_style.ipynb) — GRPO:复现 DeepSeek-R1 数学题对齐(迷你版)


---

## 📝 本章面试题

**Q**:DPO 和 PPO 在 RLHF 里各解决了什么?DPO 一步取代 RM + PPO 的关键洞察是什么?

<details>
<summary>答题要点(点开)</summary>

- **PPO RLHF 流程**:SFT → 训 RM → PPO 用 RM 做奖励信号 + KL 约束更新 policy
- **PPO 的痛**:RM 自己是偏差源、训 RM 要单独标注、PPO 跑时要 4 个模型(policy / ref / RM / value)同时在显存
- **DPO 的洞察**:在 RLHF 的 KL 约束下,最优策略 π* 可以表达成 ref policy + reward 的封闭式,**反过来 reward 也可以用 π* 与 π_ref 的 log-ratio 表达**。把 reward 代回 Bradley-Terry 偏好模型,就得到一个只关于 π 的 loss,**绕开 RM 也绕开 PPO**
- **DPO 实操**:只需 chosen / rejected 偏好对 + ref policy(SFT 后模型),一条 loss 直接更新
- **DPO 的局限**:对偏好数据质量极敏感,数据偏一点训出来就废;reward hacking 后无法察觉(因为没 RM 监控)
- **2025+ 共识**:可验证任务用 GRPO(verifiable reward),开放生成用 DPO,纯 PPO 几乎弃用

</details>

## ⚠️ 本章踩坑实录

**DPO 的 beta 调到 0.1 以下,reward gap 看起来涨但模型在退化**

DPO 的 `beta` 控制 KL 约束强度,小 → 远离 ref policy 自由发挥。
看起来 `chosen - rejected` 的 implicit reward gap 在涨,但生成质量在垮 —— 因为已经偏离 SFT 太远。
经验值:`beta=0.1` 起步,生成评测下降就调回 `0.3-0.5`;实测比一味追求 reward gap 健康。
