"""clinic 工具装配：工具必须真的注册进注册表，且语义正确。

只测 `schema()` 与 handler 本身不算接线，所以第一条断言就是注册之后
`ToolRegistry.for_domain("clinic")` 里能看到它们。
"""

from __future__ import annotations

import pytest

from kubedocs.agent.bootstrap import register_domain_tools, registered_for, unregister_domain
from kubedocs.agent.tools import ToolError, ToolRegistry
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.mock import MockAppointmentProvider


@pytest.fixture
def service() -> BookingService:
    return BookingService(MockAppointmentProvider())


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个用例前后都清空注册表，避免用例之间通过全局状态互相影响。"""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


class TestRegistrationActuallyHappens:
    def test_clinic_tools_are_visible_in_the_registry(self, service):
        """装配之后工具必须出现在注册表里：这是"接线生效"的直接证据。"""
        register_domain_tools("clinic", service=service)
        names = [s.name for s in ToolRegistry.for_domain("clinic")]
        assert "list_departments" in names
        assert "get_availability" in names
        assert "book_appointment" in names

    def test_registration_is_idempotent_and_replaces_instances(self, service):
        """反复装配不能抛"已注册"且必须换成新实例：工具闭包捕获具体 service，跳过就等于继续用旧号源库。"""
        register_domain_tools("clinic", service=service)
        first = ToolRegistry.get("list_departments")
        register_domain_tools("clinic", service=BookingService(MockAppointmentProvider()))
        second = ToolRegistry.get("list_departments")
        assert first is not second

    def test_retriever_tools_are_registered_when_provided(self, service):
        class _R:
            def search(self, q, top_k=5):  # noqa: ANN001, ANN201
                return []

        register_domain_tools("clinic", service=service, retriever=_R())
        names = [s.name for s in ToolRegistry.for_domain("clinic")]
        assert "search_docs" in names

    def test_retriever_is_optional(self, service):
        """没有检索器时只注册号源与引用核对——不能因为缺检索器就整体失败。"""
        tools = register_domain_tools("clinic", service=service)
        assert tools and all(s.name != "search_docs" for s in tools)

    def test_clinic_without_service_is_a_clear_error(self):
        with pytest.raises(ValueError, match="BookingService"):
            register_domain_tools("clinic")

    def test_unregister_removes_only_that_domain(self, service):
        register_domain_tools("clinic", service=service)
        n = unregister_domain("clinic")
        assert n >= 3
        assert ToolRegistry.for_domain("clinic") == []

    def test_registered_for_returns_descriptions(self, service):
        register_domain_tools("clinic", service=service)
        items = registered_for("clinic")
        assert items
        assert {"name", "description", "when_not", "domain", "side_effect"} <= set(items[0])


class TestToolSemantics:
    def test_list_departments_returns_rows(self, service):
        register_domain_tools("clinic", service=service)
        out = ToolRegistry.get("list_departments").handler(contains="")
        assert out["n"] > 0 and "name" in out["departments"][0]

    def test_list_departments_filter(self, service):
        register_domain_tools("clinic", service=service)
        out = ToolRegistry.get("list_departments").handler(contains="心内")
        assert out["n"] >= 1
        assert all("心内" in d["name"] for d in out["departments"])

    def test_availability_reports_slots(self, service):
        register_domain_tools("clinic", service=service)
        out = ToolRegistry.get("get_availability").handler(department="心内科", date="")
        assert out["slots"] and out["department"]

    def test_unknown_department_raises_tool_error(self, service):
        """业务错误必须是 ToolError（可回灌给模型），不是裸 Exception。"""
        register_domain_tools("clinic", service=service)
        with pytest.raises(ToolError):
            ToolRegistry.get("get_availability").handler(department="火星科", date="")

    def test_book_tool_marks_side_effect(self, service):
        """取号不可逆，必须标记 side_effect，工具循环才会要求确认。"""
        register_domain_tools("clinic", service=service)
        assert ToolRegistry.get("book_appointment").side_effect is True
        # 只读工具不该带 side_effect，否则每次查询都要用户确认一遍
        assert ToolRegistry.get("list_departments").side_effect is False
        assert ToolRegistry.get("get_availability").side_effect is False

    def test_book_tool_actually_books(self, service):
        register_domain_tools("clinic", service=service)
        ticket = ToolRegistry.get("book_appointment").handler(
            department="心内科", date="", time_slot="上午", patient_id="u1"
        )
        assert ticket.get("ticket_no")
        assert service.my_appointments("u1")

    def test_every_tool_declares_when_not(self, service):
        """`when_not` 是约束模型的主要手段，注册时会强制校验，这里再确认一遍。"""
        register_domain_tools("clinic", service=service)
        for spec in ToolRegistry.for_domain("clinic"):
            assert spec.when_not.strip(), f"{spec.name} 缺少 when_not"


class TestRelativeDates:
    """患者说的是"明天"，模型会把「明天」原样传进 `date`（实测）。

    号源层对日期是严格校验的，所以这一层不收口的话，那次工具调用会失败、整轮挂号提议
    根本不会发生——界面上只看到"模型没提议挂号"，而原因藏在工具结果里。
    """

    def test_relative_words_are_folded_to_iso(self, service):
        import datetime as dt

        from kubedocs.domains.clinic.tools import _normalize_date

        today = dt.date.today()
        assert _normalize_date("今天") == today.isoformat()
        assert _normalize_date("明天") == (today + dt.timedelta(days=1)).isoformat()
        assert _normalize_date("后天") == (today + dt.timedelta(days=2)).isoformat()
        assert _normalize_date("大后天") == (today + dt.timedelta(days=3)).isoformat()
        # 留空仍然是留空（"默认今天"由号源层决定，不在这一层替它做主）
        assert _normalize_date("") == ""
        assert _normalize_date(None) == ""

    def test_an_iso_date_passes_through_untouched(self):
        from kubedocs.domains.clinic.tools import _normalize_date

        assert _normalize_date("2026-05-06") == "2026-05-06"

    def test_unrecognised_wording_is_left_for_the_provider_to_reject(self):
        """ "下周三"**故意不猜**：猜错一个日期等于给患者挂错天，交给号源层回一句人话。"""
        from kubedocs.domains.clinic.tools import _normalize_date

        assert _normalize_date("下周三") == "下周三"

    def test_get_availability_accepts_the_relative_word(self, service):
        register_domain_tools("clinic", service=service)
        out = ToolRegistry.get("get_availability").handler(department="心内科", date="明天")
        assert out["slots"], "「明天」应当被折成 ISO 并真的查到号源，而不是报日期格式错"
