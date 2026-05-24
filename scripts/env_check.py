"""环境自检:在每个 notebook 顶部 import,一行确认依赖齐备 + GPU 可用。

用法
----
    from scripts.env_check import check
    check()

输出形如:
    [OK] Python 3.11.9
    [OK] torch 2.5.1 (CUDA 12.4, GPU: NVIDIA GeForce RTX 4070, 12.0 GB)
    [OK] transformers 4.46.3
    [OK] bitsandbytes 0.44.1 (4bit 可用)
    [WARN] vllm 未安装(Windows 正常,Linux 才需要)
"""
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


def _check_pkg(mod: str, *, version_attr: str = "__version__", fatal: bool = True) -> Result:
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        return Result(mod, False, f"未安装({type(e).__name__})", fatal=fatal)
    v = getattr(m, version_attr, "?")
    return Result(mod, True, str(v))


def _check_torch() -> Result:
    try:
        import torch
    except Exception as e:
        return Result("torch", False, f"未安装({type(e).__name__})", fatal=True)
    detail = f"{torch.__version__}"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        detail += f" (CUDA {torch.version.cuda}, GPU: {gpu}, {mem:.1f} GB)"
    else:
        detail += " (CPU only — 训练章节会很慢,建议有 CUDA GPU)"
    return Result("torch", True, detail)


def _check_bnb() -> Result:
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as e:
        return Result("bitsandbytes", False, f"未安装({type(e).__name__})", fatal=False)
    try:
        import bitsandbytes as bnb
        return Result("bitsandbytes", True, f"{bnb.__version__} (4bit 可用)")
    except Exception as e:
        return Result("bitsandbytes", False, f"导入失败:{e}", fatal=False)


def check(extras: tuple[str, ...] = ()) -> list[Result]:
    """跑一遍环境自检。extras 里可以多塞章节专属包,如 ('vllm', 'auto_gptq')。"""
    results: list[Result] = []
    results.append(Result("Python", True, platform.python_version()))

    results.append(_check_torch())
    for pkg in ("transformers", "accelerate", "datasets", "peft", "trl"):
        results.append(_check_pkg(pkg))
    results.append(_check_bnb())

    # v2 工具栈:Unsloth + Liger Kernel + datatrove + lm-eval(非致命,缺了对应章节才用)
    for pkg in ("unsloth", "liger_kernel", "datatrove", "datasketch", "lm_eval"):
        results.append(_check_pkg(pkg, fatal=False))

    for extra in extras:
        results.append(_check_pkg(extra, fatal=False))

    for r in results:
        tag = "[OK]  " if r.ok else ("[FAIL]" if r.fatal else "[WARN]")
        print(f"{tag} {r.name:18s} {r.detail}")

    fatal_fails = [r for r in results if not r.ok and r.fatal]
    if fatal_fails:
        names = ", ".join(r.name for r in fatal_fails)
        raise RuntimeError(f"关键依赖缺失:{names}。先 `pip install -r requirements.txt`")
    return results


if __name__ == "__main__":
    check(extras=("vllm", "auto_gptq", "llama_cpp"))
    sys.exit(0)
