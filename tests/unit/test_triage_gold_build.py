"""`scripts/build_triage_gold.py` 的质量闸回归测试。

为什么单独测这个脚本：它的产物是**分诊评测的金标**。金标一旦被伪变体污染，
所有引用"分诊准确率"的地方都会得到**看起来精确、实际是假**的数字——
而且不会报错，只会安静地给出错答案。所以三道闸各钉一条真实错例：

| 闸 | 真实错例（都曾生成出来过） |
|---|---|
| 部位冲突 | `头痛 -> 乳头疼`（造出「乳头疼 -> 神经内科」） |
| 起首总括/否定 | `头痛 -> 全部痛` |
| 单字规范症状 | `眼 -> 干涩感`、`咽 -> 有异物`、`口腔 -> 溃疡面` |
| 变体须对标准关键词成立 | `月经量少 -> 怕量少`（曾被记成「月经→妇产科」） |
| 前缀匹配（非子串） | `绞痛` 子串匹配到 `心绞痛`，造出「心绞痛 -> 泌尿科」 |

另外钉住"多科室归属 / 互为子串"的关键词一律丢弃（`胸痛`、`腹痛`、`下腹痛`）——
金标答案必须唯一，否则测出来的错分不清是模型错还是题错。
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
        ("乳头疼", "头痛"),  # 部位冲突：乳 ≠ 头（这条曾造出「乳头疼 -> 神经内科」）
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
    """单字规范症状语义太宽：其变体曾造出文本里根本没提部位的假题。"""
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
    """**核心回归**：子串包含匹配曾让 `绞痛` 命中 `心绞痛`，造出「心绞痛 -> 泌尿科」。"""
    only_suffix = {"心绞痛": ["胸口疼"]}
    assert script._match_canonical("绞痛", only_suffix) == []  # 只有"绞痛"结尾的 ⇒ 不建用例
    assert script._match_canonical("心绞痛", only_suffix) == ["心绞痛"]
    assert script._match_canonical("绞痛", {"绞痛样痛": []}) == ["绞痛样痛"]  # 前缀命中仍然要能用


def test_conflicting_keywords_are_dropped(script: ModuleType) -> None:
    """`腹痛`(消化内科) 与 `下腹痛`(妇产科) 互为子串 ⇒ 两个都丢；`胸痛` 多科室 ⇒ 丢。

    金标答案必须唯一，否则错分不清是模型错还是题错。
    """
    bad = script._conflicting_keywords()
    assert {"腹痛", "下腹痛"} <= bad
    # 同一科室内部互为子串不算冲突（牙 / 牙龈 都归口腔科）
    assert not ({"牙", "牙龈"} & bad)
