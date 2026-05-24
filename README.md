# Enigneer-infra · 小模型 AI-Infra 实战

> 用一颗 **Qwen2.5-1.5B** 把 LLM 全生命周期跑一遍 —— 预训练 → SFT → 偏好对齐 → 推理部署。
> 4 章 Jupyter Notebook,每章可独立读、可独立跑。

这是我自己准备国内 AI-Infra 面试时的动手实验区,刻意做成"教学风格" —— 关键算子手写、关键数字现场跑、踩坑全程记录。不是工程模板,是**学习材料**。

---

## 你需要什么

- **硬件**:消费级 NVIDIA GPU,**8 GB 显存起步**(3060 / 4060 / 4070 都行);3090 / 4090 跑 04 章 QLoRA 3B 更舒服
- **系统**:Linux / WSL2 / macOS(Windows 原生也行,vLLM 章节会绕开)
- **Python**:3.11

详细安装、踩坑、镜像配置见 [`docs/硬件与环境.md`](docs/硬件与环境.md)。

## 快速开始

```bash
git clone https://github.com/Rebel001X/infra.git enigneer-infra
cd enigneer-infra

# 装依赖(任选其一)
uv sync                              # 推荐
pip install -r requirements.txt      # 备用

# 自检环境(应该全 [OK])
python scripts/env_check.py

# 预拉模型到 HuggingFace 缓存(可选,后续 notebook 会用到)
python scripts/download_models.py 1.5b

# 启动 JupyterLab
jupyter lab
```

## 章节地图

| 章节 | 主题 | 模型 | 显存 | 预计时长 |
|---|---|---|---|---|
| [`01_pretrain_toy/`](01_pretrain_toy/) | 从零写一个 nanoGPT 风格小模型,跑通整条 pretrain 链路 | 自定义 ~25M | 2-4 GB | 1 天 |
| [`02_sft/`](02_sft/) | 监督微调:全参 / LoRA / QLoRA 三条路对比 | Qwen2.5-1.5B / 3B | 8-14 GB | 1-2 天 |
| [`03_alignment/`](03_alignment/) | 偏好对齐:DPO 入门 → GRPO 复现 DeepSeek-R1 思路 | Qwen2.5-1.5B | 10-14 GB | 2 天 |
| [`04_inference_deploy/`](04_inference_deploy/) | 推理优化:HF baseline → vLLM → GPTQ → llama.cpp | Qwen2.5-1.5B | 4-12 GB | 1 天 |

完整学习路径(4 周节奏)在 [`docs/学习路径.md`](docs/学习路径.md)。

## 文档

- [`docs/_index.md`](docs/_index.md) — 整站章节地图
- [`docs/学习路径.md`](docs/学习路径.md) — 建议顺序 + 每章用时
- [`docs/硬件与环境.md`](docs/硬件与环境.md) — CUDA/驱动/bitsandbytes/HF 镜像踩坑
- [`docs/与llm-action八股对照.md`](docs/与llm-action八股对照.md) — 每个 notebook 对应面试八股库哪一篇

## 设计原则

1. **每个 notebook 自洽** —— 顶部一个 `env_check` cell 验证依赖,中间一段原理,后面才是代码
2. **关键算子不调库** —— RoPE / RMSNorm / DPO loss 都手写一遍,看完心里有数
3. **数字现场跑** —— loss 曲线、显存占用、吞吐延迟,都贴实测值不是教科书数字
4. **踩坑全记录** —— OOM、版本冲突、bitsandbytes 装不上,都写在 notebook 末尾的"踩坑"格

## 不做的事

- ❌ 不引入 DeepSpeed / Megatron(单卡 8-16GB 场景用不上)
- ❌ 不预先填满所有 notebook(骨架先立,内容按章迭代,避免堆一堆跑不通的代码)
- ❌ 不做大规模 eval(每章自带 sanity check 够用,真做 eval 是另一个独立子项目)

## License

Apache-2.0
