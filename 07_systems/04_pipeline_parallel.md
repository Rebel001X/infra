# 04 · Pipeline Parallel(概念笔记)

> **为什么不做 notebook**:PP 真要跑得开心至少 2 张 GPU,单卡 mock 起来意义有限。这里把核心概念过一遍,留作面试谈资。

## 三种主流 PP 调度

| 调度 | 思路 | 取舍 |
|---|---|---|
| **GPipe** (Google, 2018) | micro-batch 切片,前向全跑完再回传 | bubble 大、显存高 |
| **1F1B / PipeDream** (MSR, 2019) | 一前向一反向交错,bubble 缩小 | 实现复杂、warmup/cooldown 阶段需特殊处理 |
| **DualPipe** (DeepSeek-V3, 2024) | 双向调度,bubble 接近 0 | 需要 ALL-to-ALL 通信,只在 H100/MI300 等高带宽互联上划算 |

## 与 TP / DP / SP 怎么组合

- 单机 8 卡:TP=8,不需要 PP
- 多机 64 卡:DP × PP × TP 三维并行,Megatron 经典配置 `dp=8 pp=4 tp=2`
- 训 671B(DeepSeek-V3):DualPipe + EP(MoE expert parallel)

## 推荐阅读

- 论文:`/paper fetch 1811.06965` (GPipe)
- 论文:`/paper fetch 1806.03377` (PipeDream)
- DeepSeek-V3 paper 的 DualPipe 图(`/paper fetch 2412.19437`)
- `liguodongiot/llm-action` 里的 `llm-train/分布式训练.md`
