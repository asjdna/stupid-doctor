"""`consult.html`（在线问诊页）的**静态契约测试**：钉住"界面 ↔ 挂号端点"这条窄接口。

这一页没有构建步骤，FastAPI 直接按路径读它返回（见 `api/consult.py` 的 `CONSULT_HTML`），
也没有模板层可以断言形状。所以"页面真的接上了后端、而且没有偷偷把自己变成一个写入口"
这件事，只能靠读文件来验。这里刻意**不引入浏览器**（不装 playwright / selenium）：
断言的是"页面脚本里存在这条协议"，不是"点下去屏幕上会出现什么"——渲染效果由人工演示与
端到端走查负责，本地文件这一层的回归由这里挡住。

每条断言防的是哪种回归，都写在各自 docstring 里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/unit/test_consult_page.py → parents[2] 是仓库根
PAGE_PATH = Path(__file__).resolve().parents[2] / "data" / "web" / "consult.html"


@pytest.fixture(scope="module")
def html() -> str:
    """页面原文（UTF-8）。读不出来就说明页面被写坏了，后面所有断言都失去意义。"""
    return PAGE_PATH.read_text(encoding="utf-8")


def _confirm_body_source(html: str) -> str:
    """切出页面里构造"确认挂号"请求体的那段函数源码。

    定位：从 `function confirmBody` 起，到下一个顶格的 `}`（函数结尾）为止。
    这段里出现的字段名，就是前端回传给 `/consult/booking/confirm` 的**全部内容**。
    比"在整个文件里搜有没有 fee"精确得多：页面别处（渲染卡片时）当然要读那些字段，
    只有这里不允许把它们发回去。
    """
    start = html.index("function confirmBody")
    end = html.index("\n}", start)  # 函数体内部的对象字面量是缩进的，不会先撞上
    return html[start:end]


class TestPageFileIsWellFormed:
    """页面文件本身：编码与开头。防的是"顺手用某个工具改写后中文变乱码/丢了 doctype"。"""

    def test_page_exists(self) -> None:
        assert PAGE_PATH.is_file(), f"页面不在预期位置：{PAGE_PATH}"

    def test_is_valid_utf8_with_chinese_copy(self) -> None:
        """页面文案全是简体中文：非法字节在这里就抛 UnicodeDecodeError，不要拖到浏览器里才发现。"""
        text = PAGE_PATH.read_bytes().decode("utf-8")
        assert "在线问诊" in text

    def test_starts_with_doctype(self, html: str) -> None:
        """没有 doctype 的页面会落进 quirks mode，深色配色下的盒模型会整体走样。"""
        assert html.lstrip().startswith("<!DOCTYPE html>")


class TestBookingEndpointsAreWired:
    """三个新端点的路径字符串必须在页面里出现。

    防的回归：路由改了名/换了前缀，后端测试照绿，而页面上那个按钮默默打向 404。
    """

    @pytest.mark.parametrize(
        "endpoint",
        ["/consult/booking/confirm", "/consult/booking/cancel", "/consult/appointments"],
    )
    def test_endpoint_path_appears(self, html: str, endpoint: str) -> None:
        assert endpoint in html, f"页面没有调用 {endpoint}"


class TestPatientIdentity:
    """患者身份 = 浏览器本地 UUID：存 localStorage，且每处需要它的请求都要带上。"""

    def test_localstorage_key_exists(self, html: str) -> None:
        """防"身份只放在内存里"：刷新一次就换人，挂号记录全部看不见。"""
        assert "kubedocs.consult.patient" in html
        assert "localStorage" in html

    def test_session_key_still_uses_sessionstorage(self, html: str) -> None:
        """会话仍走 sessionStorage：两个 key 混用会出现"刷新就换人"或"换标签页还是同一次问诊"。"""
        assert "kubedocs.consult.sid" in html
        assert "sessionStorage" in html

    def test_randomuuid_has_a_fallback(self, html: str) -> None:
        """`crypto.randomUUID` 在非安全上下文里不存在：没有兜底就是一行 TypeError 白屏。"""
        assert "crypto.randomUUID" in html
        assert "Date.now()" in html

    def test_patient_id_is_sent_on_every_request(self, html: str) -> None:
        """至少四处要带 `patient_id`：问诊、确认、退号、拉记录。

        少一处的后果不是报错而是"悄悄查不到/挂到别人名下"，所以这里按出现次数兜一下；
        精确到每一处的形状由下面几个类分别断言。
        """
        assert html.count("patient_id") >= 4, "有请求没带 patient_id"

    def test_appointments_query_carries_patient_id(self, html: str) -> None:
        start = html.index("/consult/appointments")
        assert "patient_id" in html[start : start + 220]

    def test_cancel_request_carries_patient_id_and_ticket_no(self, html: str) -> None:
        """退号要"凭据号 + 就诊人"：少了 patient_id，后端就没法核对是不是本人的号。"""
        start = html.index("/consult/booking/cancel")
        region = html[start : start + 320]
        assert "ticket_no" in region
        assert "patient_id" in region

    def test_switch_patient_button_exists(self, html: str) -> None:
        """「换个人」是演示时干净切换身份的入口：清 patient id + 重置会话，两个都要做。"""
        assert 'id="switch"' in html
        assert "换个人" in html

    def test_reset_button_is_kept(self, html: str) -> None:
        """「重新开始」（只重置会话）必须还在，且与「换个人」是两个按钮。"""
        assert 'id="reset"' in html
        assert "重新开始" in html


class TestConfirmSendsOnlyTheProposalId:
    """**确认挂号只发单号**——这是产品锁定的安全设计，也是本文件最该防的回归。

    科室/日期/时段/费用留在服务端那张单子上；前端一旦把 `department` / `date` / `fee`
    也回传，"改单"的口子就开了（哪怕本意只是"顺手带上"）。后端只读 `proposal_id`，
    所以这种回传在功能上毫无收益，却会把"服务端说了算"变成"客户端说了算"。
    """

    def test_confirm_body_source_is_found(self, html: str) -> None:
        """先证明切出来的区域非空，避免下面的"不存在"断言变成永真的空断言。"""
        body = _confirm_body_source(html)
        assert body.startswith("function confirmBody")
        assert "return" in body
        assert len(body) > 40

    def test_body_carries_proposal_id(self, html: str) -> None:
        """确认请求必须带 `proposal_id`：不带就无从确认（后端只认这个单号）。"""
        assert "proposal_id" in _confirm_body_source(html)

    @pytest.mark.parametrize("field", ["fee", "time_slot", "department", "date"])
    def test_body_does_not_carry_server_side_fields(self, html: str, field: str) -> None:
        """`fee` / `time_slot` / `department` / `date` 不得出现在确认请求体里。"""
        body = _confirm_body_source(html)
        assert field not in body, f"确认请求体里不该出现 {field}（那些字段留在服务端那张单子上）"


class TestBookingCardsAreRendered:
    """两类卡片都要真的被读到并渲染：`pending_action`（待确认单）与 `appointments`（记录）。"""

    @pytest.mark.parametrize("field", ["pending_action", "appointments"])
    def test_response_field_is_read(self, html: str, field: str) -> None:
        """防"后端回了、前端没读"：卡片从此不再出现，而所有后端测试仍然全绿。"""
        assert field in html, f"页面没有读响应里的 {field}"

    @pytest.mark.parametrize(
        "field",
        ["proposal_id", "department", "date", "time_slot", "fee", "remaining", "expires_at"],
    )
    def test_pending_card_uses_server_values(self, html: str, field: str) -> None:
        """待确认卡片上的每个值都来自 `pending_action`：前端不自己拼、不自己算。"""
        assert field in html, f"待确认卡片缺少服务端字段 {field}"

    def test_appointment_record_fields_are_rendered(self, html: str) -> None:
        for field in ("ticket_no", "queue_no", "status", "created_at"):
            assert field in html, f"挂号记录缺少字段 {field}"

    def test_trace_id_is_shown(self, html: str) -> None:
        """追踪号要能显示出来：这是"这一轮到底发生了什么"对账用的。"""
        assert "trace_id" in html

    def test_cancelled_records_are_kept_and_labelled(self, html: str) -> None:
        """退掉的号**留在列表里**并显示成"已退号"：它是历史，删掉就没法回看。"""
        assert '"cancelled"' in html
        assert "已退号" in html

    def test_business_failure_surfaces_the_server_sentence(self, html: str) -> None:
        """409 的 `detail` 是一句给患者看的人话，必须原样展示（并引导换时段/日期）。"""
        assert "409" in html
        assert "detail" in html


class TestNoInlineEventHandlers:
    """一律 `addEventListener`，不用内联处理器。

    防的是"顺手写个 onclick=…"：那会把页面变成拼字符串的地方，也是 XSS 的常见入口。
    """

    @pytest.mark.parametrize("attr", ["onclick=", "onchange=", "onsubmit=", "oninput=", "onload="])
    def test_inline_handler_is_absent(self, html: str, attr: str) -> None:
        assert attr not in html, f"页面里不应出现内联事件处理器 {attr}"

    def test_buttons_are_bound_in_script(self, html: str) -> None:
        assert "addEventListener" in html


class TestExistingBehaviourIsIntact:
    """老功能一个都不能坏（这些是原页面的可见契约，删了就少一个能用的键位）。"""

    @pytest.mark.parametrize(
        "needle",
        [
            'id="stream"',
            'id="q"',
            'id="send"',
            'id="emg"',
            "esc(",  # 动态文本仍然走同一个转义函数
            "Shift+Enter",  # 换行提示还在
            "ev.shiftKey",  # Enter 发送 / Shift+Enter 换行
            "正在整理…",  # 占位气泡
        ],
    )
    def test_dialogue_essentials_remain(self, html: str, needle: str) -> None:
        assert needle in html, f"问诊页缺少 {needle}"
