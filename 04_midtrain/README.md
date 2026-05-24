## 第 4 章 · Midtrain(对话格式注入)

目标:**SFT 之前先做 midtrain**(nanochat / SOLAR 的关键一步,2024-2025 才成为共识)。
直接在 base 模型上 SFT 会出现"格式没学会,答案也没学好"的双输,因为 base 从没见过 `<|im_start|>user`、`<|im_end|>` 这类特殊 token。
Midtrain 用 SmolTalk(或中文等价物)做几千步快速适配,让 base 先"知道有对话格式",再做正经 SFT 才稳。

参考 Karpathy nanochat 的 midtrain 章节。

### 本章 notebook

- [`01_smoltalk_zh_injection.ipynb`](01_smoltalk_zh_injection.ipynb) — 用 SmolTalk-zh 注入对话格式


---

## 📝 本章面试题

**Q**:什么是 midtrain?为什么 2025 年开始 nanochat / SOLAR 这类项目都强调 midtrain 不能省?和 SFT、continued pretrain 的区别是什么?

<details>
<summary>答题要点(点开)</summary>

- **定义**:midtrain 是 base 模型 → instruct 模型之间的**小步快跑**,用对话格式数据(几千-几万步)让模型"建立角色概念",再做正经 SFT
- **解决的问题**:base 模型从没见过 chat template,直接 SFT 会让模型既要学格式又要学内容,样本效率极低;midtrain 先把格式问题解决
- **与 continued pretrain 的区别**:CPT 学的是新领域知识(数据多、loss 全长上算);midtrain 学的是格式与角色(数据少、可以只在 response 部分算 loss)
- **与 SFT 的区别**:midtrain 数据混杂(对话 + 短文档),用以"看见"格式;SFT 数据精炼、追求指令跟随能力

</details>

## ⚠️ 本章踩坑实录

**midtrain 数据混入比例搞错全废**

nanochat 的经验:midtrain 数据 = 30% 对话 + 70% 高质量 pretrain 文本(保留通用能力)。
如果只用对话数据,base 的世界知识会快速退化(catastrophic forgetting),SFT 后表现还不如不做 midtrain。

