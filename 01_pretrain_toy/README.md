## 第 1 章 · 从零做 Pretrain(toy 模型)

目标:**看穿** transformer 训练循环每一步发生了什么。不追性能,只追「看懂」。

### 本章 notebook

- [`01_tokenizer.ipynb`](01_tokenizer.ipynb) — 训练自己的 BPE tokenizer,并与 Qwen tokenizer 对照
- [`02_data_pipeline.ipynb`](02_data_pipeline.ipynb) — 数据流:streaming + sequence packing
- [`03_nano_decoder.ipynb`](03_nano_decoder.ipynb) — 手写一个对齐 Qwen 架构的 decoder-only
- [`04_train_loop.ipynb`](04_train_loop.ipynb) — 完整训练循环:AMP + grad accum + ckpt + wandb


---

## 📝 本章面试题

**Q**:为什么现代 LLM 都用 RoPE 而不是 absolute / learned position embedding?具体的相对位置编码是怎么实现的?

<details>
<summary>答题要点(点开)</summary>

- **Learned absolute** 的问题:训练时 max_len=2048,推到 32k 直接废
- **Sinusoidal** 的问题:虽然能外推一点,但 Q/K 的内积不直接含相对位置信息
- **RoPE 的优势**:旋转矩阵 R(m)·R(n)ᵀ = R(m-n),Q/K 在做点积时**天然带相对位置**
- **实现**:对 head_dim 每两个维度看作复平面向量,按位置 m 旋转角度 `m × inv_freq`(`inv_freq = 1 / 10000^(2i/d)`);Q/K 都旋转后再点积
- **长度外推**:NTK-aware scaling / YaRN 等都是在 RoPE 基础上调 `base`(10000) 或 `inv_freq`,Linear interpolation 也是 RoPE 才能玩

</details>

## ⚠️ 本章踩坑实录

**Embedding 初始化不做 std=0.02,初始 loss 会爆炸到 100+**

PyTorch `nn.Embedding` 默认 `N(0, 1)`,tied embedding 时 logits std ≈ 11,首次 CE ≈ 100 而不是理论的 `ln(vocab_size) ≈ 4.7`。
修复:`nn.init.normal_(self.embed.weight, std=0.02)`(GPT 默认)。看 `04_train_loop.ipynb` 第 3.6 节。
