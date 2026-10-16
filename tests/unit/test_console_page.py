"""`console.html`（运维控制台）的**静态契约测试**：钉住它到底展示了什么。

这一页没有构建步骤，FastAPI 直接按路径读它返回；也没有模板层可以断言形状，所以"控制台
真的把这一轮的可观测读数摆出来了、而且没写与后端相反的话"只能靠读文件来验。这里同样
**不引入浏览器**：断言的是"页面脚本里存在这条协议"，渲染效果由人工演示与端到端走查负责。

每条断言防的是哪种回归，都写在各自 docstring 里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/unit/test_console_page.py → parents[2] 是仓库根
PAGE_PATH = Path(__file__).resolve().parents[2] / "data" / "web" / "console.html"


@pytest.fixture(scope="module")
def html() -> str:
    """页面原文（UTF-8）。读不出来就说明页面被写坏了，后面所有断言都失去意义。"""
    return PAGE_PATH.read_text(encoding="utf-8")


class TestPageFileIsWellFormed:
    def test_page_exists(self) -> None:
        assert PAGE_PATH.is_file(), f"页面不在预期位置：{PAGE_PATH}"

    def test_is_valid_utf8_with_chinese_copy(self) -> None:
        """文案全是简体中文：非法字节在这里就抛 UnicodeDecodeError，不要拖到浏览器里才发现。"""
        text = PAGE_PATH.read_bytes().decode("utf-8")
        assert "问诊 Agent" in text

    def test_starts_with_doctype(self, html: str) -> None:
        """没有 doctype 的页面会落进 quirks mode，深色配色下的盒模型会整体走样。"""
        assert html.lstrip().lower().startswith("<!doctype html>")


class TestTheChatStreamIsObservable:
    """`/chat` 的三个可观测读数必须在控制台上看得见。

    控制台是运维侧唯一的实时窗口：这一轮"交给模型的是什么""它哪里越界了"如果不在这里
    露出来，就只能事后翻日志。
    """

    def test_message_text_is_rendered(self, html: str) -> None:
        """回复正文来自 `obj.text`；**被约束层拦下的那段散文不会出现在这里**（服务端已换掉）。"""
        assert "obj.text" in html

    def test_reference_panel_is_traced(self, html: str) -> None:
        """参考标准面板留档要留痕：依据编号与挂号事实的条数是"它手上有什么"的最小读数。"""
        assert "obj.reference" in html
        assert "evidence_ids" in html
        assert "booking" in html

    def test_contract_violations_are_traced(self, html: str) -> None:
        """输出约束层的违规要按 `kind/severity` 列出来（模型有没有在编造，这是唯一读数）。"""
        assert "obj.contract_violations" in html
        assert "kind" in html and "severity" in html

    def test_tool_trace_and_cost_trace_remain(self, html: str) -> None:
        """工具轨迹是这一页的既有价值：工具名、参数、急诊横幅、算不出价的调用都要能看见。"""
        assert "obj.tool" in html
        assert "obj.arguments" in html
        assert "obj.status" in html
        assert "obj.unpriced_llm_calls" in html


class TestNoClaimContradictsTheBackend:
    """页面文案不许出现与后端相反的判断归属。

    红线是不是急症由模型给出，规则资产只是线索；页面上任何"词表命中就确定性触发"的说法
    都在把判断权写回规则层，读到它的人会据此理解错整条链路。
    """

    @pytest.mark.parametrize("stale", ["确定性触发", "兜底提示"])
    def test_stale_wording_is_absent(self, html: str, stale: str) -> None:
        assert stale not in html, f"控制台里仍写着已废弃的说法：{stale}"


class TestNoInlineEventHandlers:
    """一律 `addEventListener`，不用内联处理器（那会把页面变成拼字符串的地方）。"""

    @pytest.mark.parametrize("attr", ["onclick=", "onchange=", "onsubmit=", "oninput=", "onload="])
    def test_inline_handler_is_absent(self, html: str, attr: str) -> None:
        assert attr not in html, f"页面里不应出现内联事件处理器 {attr}"

    def test_buttons_are_bound_in_script(self, html: str) -> None:
        assert "addEventListener" in html
