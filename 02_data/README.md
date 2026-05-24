## 第 2 章 · 数据工程(CC 清洗 + 去重)

目标:理解一份 pretrain/SFT 数据从原始爬取到能喂训练之间的工程链路。第 1 章里我们 hardcode 了一段中文,
这章演示**工业界做法**:CommonCrawl 子集 → 语言识别 → URL/质量过滤 → MinHash 去重 → 输出可训样本。

参考 Stanford CS336 A4 + HF datatrove pipeline + DCLM。

### 本章 notebook

- [`01_cc_cleaning.ipynb`](01_cc_cleaning.ipynb) — CommonCrawl 微缩子集清洗(datatrove pipeline)
- [`02_dedup_minhash.ipynb`](02_dedup_minhash.ipynb) — MinHash + LSH 去重


---

## 📝 本章面试题

**Q**:为什么 LLM 预训练前必须做大规模去重?除了 MinHash,工业界还有哪些去重方法,各自的取舍?

<details>
<summary>答题要点(点开)</summary>

- **为什么去重**:重复样本会让模型在测试集上"假装"泛化(论文 2107.06499 证明 1 个 sample 出现 64 次后会被精确记忆),同时浪费算力
- **方法对照**:
  - **Exact dedup(URL/document hash)**:最快、最严格,只去字面重复
  - **Near-dedup(MinHash + LSH)**:能去 paraphrase,工业界标准,FineWeb / RedPajama 都用
  - **Semantic dedup(embedding + clustering)**:最贵但能去掉"换种说法的同一篇文章"
- **顺序**:先 exact 再 minhash 再 semantic,逐级筛
- **超参**:MinHash num_perm=128 / Jaccard threshold=0.8 是 FineWeb 配置

</details>

## ⚠️ 本章踩坑实录

**datatrove 在 Windows 跑不动**

datatrove 依赖 fastText / pyspark,Windows 原生跑不起来。**两个绕开方案**:
1. 在 WSL2 里跑(推荐,5 分钟搞定)
2. 用 `datasketch` + 手写 fastText replacement,功能阉割版,够教学但工业上别这么干

