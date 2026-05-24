"""每个训练 notebook 都应该能跑通 1 个 mini-batch。这里只 stub,等具体 notebook 填完再加用例。

跑法:`pytest tests/ -v`
"""
import pytest


def test_env_check_importable() -> None:
    from scripts.env_check import check
    assert callable(check)


@pytest.mark.skip(reason="01_pretrain_toy 内容填完后启用")
def test_pretrain_one_step() -> None:
    ...


@pytest.mark.skip(reason="02_sft 内容填完后启用")
def test_sft_one_step() -> None:
    ...
