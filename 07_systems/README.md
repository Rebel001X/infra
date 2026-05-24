## 第 7 章 · Systems(Triton + 分布式)

目标:**国内 AI-Infra 面试核心考点**。哪怕你只有一张卡也要写 mini-DDP / mini-FSDP / Triton FlashAttention2,
能讲清楚就能压 5-10% offer 涨幅。参考 Stanford CS336 A2 + DeepSpeed/FSDP 官方教程。

### 本章 notebook

- [`01_triton_flashattn2_mini.ipynb`](01_triton_flashattn2_mini.ipynb) — 手写迷你 FlashAttention2(Triton)
- [`02_torchrun_ddp_demo.ipynb`](02_torchrun_ddp_demo.ipynb) — mini-DDP:torchrun + AllReduce
- [`03_fsdp_demo.ipynb`](03_fsdp_demo.ipynb) — mini-FSDP:参数分片 + sharded optimizer


---

## 📝 本章面试题

**Q**:FlashAttention 为什么快?具体改了哪些算子?vs FlashDecoding 的差异?(2024+ 面试高频)

<details>
<summary>答题要点(点开)</summary>

- **FA v1 核心**:把 attention 的 `Q @ K.T → softmax → @ V` 三步融合成一个 kernel,**避免实例化 N×N 中间矩阵**(N=2048 时 N×N×bf16 = 8MB,塞不进 SRAM)
- **关键技术**:**Online softmax**(增量计算 max/sum,避免回扫)+ **tiling**(把 Q/K/V 分块塞进 SRAM,每块算完再写回 HBM)
- **FA v2 改了什么**:更少的 non-matmul FLOPs(把 softmax rescale 推到外面)+ 更好的 GPU occupancy(沿序列维并行而不是只沿 batch 维)
- **FA v3 改了什么**:H100 专属,用 WGMMA + async memory + FP8
- **FlashDecoding 的差异**:decode 阶段 batch=1 时 FA v2 占用率低(只能用 1-2 个 SM),FlashDecoding 沿 KV 维拆分,占用率拉满,推理快 8x

</details>

## ⚠️ 本章踩坑实录

**Triton 在 Windows 装不上**

Triton 官方 Linux only。Windows 想跑要么 WSL2,要么用 PyTorch 内建的 SDPA(`F.scaled_dot_product_attention`),后者实际就是封装的 FA v2 / mem-efficient,正确性等价但少了"自己写"的教学价值。
本章在 Windows 上至少能跑 02 / 03 节(DDP/FSDP 用 gloo backend + CPU 多进程)。

