"""提前把课程要用到的数据集拉到本地 HuggingFace 缓存,避免 notebook 里中断。

用法
----
    python scripts/download_datasets.py              # 拉默认列表
    python scripts/download_datasets.py alpaca_zh    # 只拉某一个
    python scripts/download_datasets.py --list       # 看可选

国内访问慢的话,设环境变量用镜像(注意 dataset 走的也是 HF endpoint):
    set HF_ENDPOINT=https://hf-mirror.com   (PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com")
"""
from __future__ import annotations

import os
import sys

# 名称 → (repo_id, 用途, 章节)
DATASETS = {
    # 02_data
    "fineweb_sample":   ("HuggingFaceFW/fineweb",        "微缩 CC 子集",                "02_data"),
    # 03_eval
    "cmmlu":            ("haonanli/cmmlu",               "中文 MMLU",                  "03_eval"),
    "ceval":            ("ceval/ceval-exam",             "中文 C-Eval",                "03_eval"),
    # 04_midtrain
    "smoltalk":         ("HuggingFaceTB/smoltalk",       "英文对话格式(midtrain)",     "04_midtrain"),
    "belle_zh":         ("BelleGroup/train_1M_CN",       "中文对话(midtrain/SFT)",     "04_midtrain/05_sft"),
    # 05_sft
    "alpaca_zh":        ("shibing624/alpaca-zh",         "中文 SFT 指令数据",          "05_sft"),
    # 06_alignment / 09_capstone
    "preference_zh":    ("hiyouga/DPO-En-Zh-20k",        "中文 DPO 偏好对",            "06_alignment"),
    "gsm8k":            ("openai/gsm8k",                 "英文数学推理(GRPO baseline)", "06_alignment/09_capstone"),
    "gsm8k_zh":         ("meta-math/GSM8K_zh",           "中文数学推理(Capstone 主力)", "09_capstone"),
}


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print(f"{'key':<18} {'repo':<40} {'用途':<24} 章节")
        print("-" * 100)
        for k, (repo, desc, chap) in DATASETS.items():
            print(f"{k:<18} {repo:<40} {desc:<24} {chap}")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("缺少 datasets:pip install datasets", file=sys.stderr)
        return 1

    args = [a for a in argv[1:] if not a.startswith("--")]
    keys = args or ["alpaca_zh", "gsm8k_zh", "preference_zh"]  # 默认:最常用的三件
    bad = [k for k in keys if k not in DATASETS]
    if bad:
        print(f"未知 key:{bad}。看可选:python {argv[0]} --list", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF endpoint: {endpoint}\n")

    for k in keys:
        repo, desc, chap = DATASETS[k]
        print(f">>> 拉取 [{k}] {repo}  ({desc}, 用于 {chap}) ...")
        try:
            ds = load_dataset(repo, split=None, streaming=False, trust_remote_code=False)
            n = sum(len(v) for v in ds.values()) if hasattr(ds, "values") else len(ds)
            print(f"    -> 成功,样本数 ≈ {n}")
        except Exception as e:
            print(f"    -> 失败:{type(e).__name__}: {e}")
            print("    (若是网络问题,设 HF_ENDPOINT=https://hf-mirror.com 重试)")

    print("\n全部完成。后续 notebook 里 load_dataset(repo_id) 会直接命中缓存。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
