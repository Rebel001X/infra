"""提前把 Qwen2.5 模型拉到本地 HuggingFace 缓存,避免在 notebook 里因网络中断卡住。

用法
----
    python scripts/download_models.py             # 拉默认列表
    python scripts/download_models.py 1.5b        # 只拉 1.5B
    python scripts/download_models.py 1.5b 3b     # 都拉

国内访问慢的话,设环境变量用镜像:
    set HF_ENDPOINT=https://hf-mirror.com   (PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com")
"""
from __future__ import annotations

import os
import sys

MODELS = {
    "1.5b-base":     "Qwen/Qwen2.5-1.5B",
    "1.5b":          "Qwen/Qwen2.5-1.5B-Instruct",
    "3b-base":       "Qwen/Qwen2.5-3B",
    "3b":            "Qwen/Qwen2.5-3B-Instruct",
}


def main(argv: list[str]) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub:pip install huggingface-hub", file=sys.stderr)
        return 1

    keys = argv[1:] or ["1.5b-base", "1.5b"]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        print(f"未知 key:{bad}。可选:{list(MODELS)}", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF endpoint: {endpoint}")

    for k in keys:
        repo = MODELS[k]
        print(f"\n>>> 拉取 {repo} ...")
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "tokenizer*",
                "merges.txt",
                "vocab.json",
            ],
        )
        print(f"    -> {path}")

    print("\n全部完成。后续 notebook 里 from_pretrained(repo_id) 会直接命中缓存。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
