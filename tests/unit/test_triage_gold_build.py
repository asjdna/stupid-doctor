"""`scripts/build_triage_gold.py` 的质量闸回归测试。

产物是分诊评测的金标，金标答案必须唯一：伪变体一旦混进去，"分诊准确率"会给出
看起来精确、实际是假的数字且不报错。因此每道闸都配正反用例：

- 部位冲突：`乳头疼`、`腿疼` 不得挂到别的规范症状上；
- 起首总括/否定：`全部痛`、`无痛` 不是 `头痛` 的口语说法；
- 单字规范症状（`眼`、`咽`）语义太宽，其变体一律拒绝；
- 变体必须对规范症状本身成立，匹配只认前缀：`绞痛` 不得命中 `心绞痛`；
- 多科室归属或互为子串的关键词（`胸痛`、`腹痛`、`下腹痛`）一律丢弃。
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from kubedocs.config import repo_root

SCRIPT = repo_root() / "scripts" / "build_triage_gold.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """按路径加载脚本模块（`scripts/` 不是包，没有 `__init__.py`）。"""
    spec = importlib.util.spec_from_file_location("_build_triage_gold", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("乳头疼", "头痛"),  # 部位冲突：乳 ≠ 头
        ("腿疼", "腰痛"),  # 部位冲突：腿 ≠ 腰
        ("胃热", "口臭"),  # 实义字零交集
        ("喉咙有痰", "口臭"),
        ("一侧疼", "腹痛"),
        ("吐好多", "咳痰"),
    ],
)
def test_implausible_variants_are_rejected(script: ModuleType, variant: str, canonical: str) -> None:
    assert script._plausible_variant(variant, canonical) is False


@pytest.mark.parametrize("variant", ["全部痛", "所有地方痛", "无痛", "不疼"])
def test_lead_stopwords_are_rejected(script: ModuleType, variant: str) -> None:
    """「全部痛」不是「头痛」的口语说法。"""
    assert script._plausible_variant(variant, "头痛") is False


@pytest.mark.parametrize("canonical", ["眼", "耳", "咽", "牙"])
def test_single_char_canonicals_are_rejected(script: ModuleType, canonical: str) -> None:
    """单字规范症状语义太宽：其变体容易造出根本没提部位的假题。"""
    assert script._plausible_variant("干涩感", canonical) is False


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("一直咳", "咳嗽"),
        ("头也疼", "头痛"),
        ("小腹胀", "腹胀"),  # 起首非部位字（小），不该被部位冲突闸误杀
        ("尿也多", "多尿"),  # 规范症状起首非部位字，同样不该误杀
        ("咳出血", "咯血"),
        ("听力下降", "听力下降"),
    ],
)
def test_plausible_variants_survive(script: ModuleType, variant: str, canonical: str) -> None:
    """闸要有区分度：**不能靠一律拒绝来"通过测试"**。"""
    assert script._plausible_variant(variant, canonical) is True


def test_canonical_matching_is_prefix_only(script: ModuleType) -> None:
    """匹配只认前缀：子串包含会让 `绞痛` 命中 `心绞痛`，挂到错误的科室上。"""
    only_suffix = {"心绞痛": ["胸口疼"]}
    assert script._match_canonical("绞痛", only_suffix) == []  # 只有"绞痛"结尾的 ⇒ 不建用例
    assert script._match_canonical("心绞痛", only_suffix) == ["心绞痛"]
    assert script._match_canonical("绞痛", {"绞痛样痛": []}) == ["绞痛样痛"]  # 前缀命中仍然要能用


def test_conflicting_keywords_are_dropped(script: ModuleType) -> None:
    """互为子串（`腹痛`/`下腹痛`）或多科室归属（`胸痛`）的关键词都必须丢弃。"""
    bad = script._conflicting_keywords()
    assert {"腹痛", "下腹痛"} <= bad
    # 同一科室内部互为子串不算冲突（牙 / 牙龈 都归口腔科）
    assert not ({"牙", "牙龈"} & bad)
