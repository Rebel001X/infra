## 第 1 章 · 从零做 Pretrain(toy 模型)

目标:**看穿** transformer 训练循环每一步发生了什么。不追性能,只追「看懂」。

### 本章 notebook

- [`01_tokenizer.ipynb`](01_tokenizer.ipynb) — 训练自己的 BPE tokenizer,并与 Qwen tokenizer 对照
- [`02_data_pipeline.ipynb`](02_data_pipeline.ipynb) — 数据流:streaming + sequence packing
- [`03_nano_decoder.ipynb`](03_nano_decoder.ipynb) — 手写一个对齐 Qwen 架构的 decoder-only
- [`04_train_loop.ipynb`](04_train_loop.ipynb) — 完整训练循环:AMP + grad accum + ckpt + wandb
