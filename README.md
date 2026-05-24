# Enigneer-infra · 小模型 AI-Infra 实战(v2)

> 用一颗 **Qwen2.5-1.5B** 把 LLM 从原始数据到推理部署整条链路跑一遍。
> 9 章 Jupyter Notebook,每章可独立读、可独立跑;**全部对齐 2026 SOTA**(Stanford CS336 + Karpathy nanochat + HF smol-course + Unsloth + Liger Kernel)。

这是我自己准备国内 AI-Infra 工程师面试时的动手实验区,刻意做成「教学风格」——
关键算子手写、关键数字现场跑、踩坑全程记录、每章一道面试题对齐国内面经。

## 你需要什么

- **硬件**:消费级 NVIDIA GPU,**8 GB 显存起步**(3060 / 4060 / 4070 都行);3090 / 4090 跑 Capstone 更舒服
- **系统**:**Linux / WSL2**(强烈推荐,Unsloth / Liger Kernel / vLLM / datatrove / Triton 都只在 Linux 跑得舒服);macOS 大部分章节能跑;Windows 原生部分章节要绕开
- **Python**:3.11

详细安装、踩坑、镜像配置见 [`docs/硬件与环境.md`](docs/硬件与环境.md) 和 [`docs/工具栈选型.md`](docs/工具栈选型.md)。

## 快速开始

```bash
git clone https://github.com/Rebel001X/infra.git enigneer-infra
cd enigneer-infra

uv sync                              # 推荐
# 或 pip install -r requirements.txt

python scripts/env_check.py          # 自检
python scripts/download_models.py    # 拉 Qwen2.5-1.5B / 1.5B-Instruct
python scripts/download_datasets.py  # 拉 alpaca_zh / gsm8k_zh / preference_zh

jupyter lab
```

## 章节地图(9 章)

| # | 章节 | 主题 | 模型 | 显存 |
|---|---|---|---|---|
| 01 | [`01_pretrain_toy/`](01_pretrain_toy/) | 从零写 nanoGPT,看穿训练循环 | 自定义 ~5M | 2-4 GB |
| 02 | [`02_data/`](02_data/) ★ | CC 清洗 + MinHash 去重(对齐 CS336/FineWeb) | — | 2 GB |
| 03 | [`03_eval/`](03_eval/) ★ | lm-eval-harness + CMMLU + C-Eval baseline | Qwen2.5-1.5B | 4 GB |
| 04 | [`04_midtrain/`](04_midtrain/) ★ | SmolTalk-zh 对话格式注入(nanochat 风格) | Qwen2.5-1.5B | 10 GB |
| 05 | [`05_sft/`](05_sft/) | 全参 + Liger / Unsloth LoRA / QLoRA 三路对比 | Qwen2.5-1.5B / 3B | 8-14 GB |
| 06 | [`06_alignment/`](06_alignment/) | DPO 推导 + **verifiable reward GRPO** | Qwen2.5-1.5B | 10-14 GB |
| 07 | [`07_systems/`](07_systems/) ★ | Triton FlashAttn2 + mini-DDP + mini-FSDP(面试核心) | — | 4 GB |
| 08 | [`08_inference_deploy/`](08_inference_deploy/) | vLLM + 推测解码 + GPTQ/AWQ + GGUF + TensorRT-LLM | Qwen2.5-0.5B/1.5B | 4-12 GB |
| 09 | [`09_capstone/`](09_capstone/) ★ | **Capstone:mini-R1 复现** GSM8K-zh,SFT+GRPO 真实提升 | Qwen2.5-1.5B | 12-14 GB |

★ = v2 升级时新增的章节。完整学习路径(8 周节奏)在 [`docs/学习路径.md`](docs/学习路径.md)。

## 文档

- [`docs/_index.md`](docs/_index.md) — 整站章节地图
- [`docs/学习路径.md`](docs/学习路径.md) — 8 周节奏 + 速通模式
- [`docs/硬件与环境.md`](docs/硬件与环境.md) — CUDA / bitsandbytes / Unsloth / HF 镜像踩坑
- [`docs/工具栈选型.md`](docs/工具栈选型.md) ★ — Unsloth vs TRL、Liger Kernel 价值、为什么 verifiable reward GRPO
- [`docs/与llm-action八股对照.md`](docs/与llm-action八股对照.md) — 每章对应 [面试八股](https://github.com/liguodongiot/llm-action) 哪一篇

## 设计原则

1. **每个 notebook 自洽** —— 顶部一个 `env_check` cell 验证依赖,中间一段原理,后面才是代码
2. **关键算子不调库** —— RoPE / RMSNorm / DPO loss / FlashAttention 都手写一遍
3. **数字现场跑** —— loss / 显存 / 吞吐 / accuracy 都贴实测值
4. **每章一道面试题 + 一条踩坑实录** —— 直接对齐国内面经格式(看每章 `README.md` 末尾)
5. **现代 best practice 优先** —— SFT 用 Liger,GRPO 用 verifiable reward,SFT 前必 midtrain

## 工具栈(2026 SOTA)

- **训练加速**:`unsloth` + `liger-kernel`(单卡省显存提速事实标准)
- **微调**:`trl` + `peft` + `bitsandbytes`(QLoRA)
- **数据**:`datatrove` + `datasketch` + `datasets`
- **评测**:`lm-eval` + CMMLU / C-Eval
- **推理**:`vllm` + `llama-cpp-python` + `auto-gptq`
- **系统**:`torch.distributed` + Triton

## 不做的事

- ❌ 不做完整 PPO RLHF —— 业界已弃用,verifiable GRPO 覆盖现代场景
- ❌ 不做 multi-node 训练 —— 单机 8-16GB 场景跑不动,07 章只 mock 多卡
- ❌ 不在 repo 里存模型权重 —— HF 缓存在用户机器
- ❌ 不一次性把所有 notebook 都填满 —— 骨架先立,按章 `/longrun` 迭代

## License

Apache-2.0
