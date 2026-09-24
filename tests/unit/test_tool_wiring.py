"""clinic 工具装配测试：确认工具**真的注册进了注册表**、且语义正确。

这组测试针对的是一类具体的历史缺陷：**写好了但从未接线**。
在此之前 `ToolRegistry.register` 全仓库无人调用，工具链路是死的，而
35 项测试仍然全绿——因为那些测试只测了 `schema()` 与 handler 本身，
没有任何一条断言"它出现在注册表里"。

所以本文件的第一条断言就是：注册之后，`ToolRegistry.for_domain("clinic")` 里
必须能看到它们。
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
        """**这条就是历史上缺失的那个断言**：注册之后必须在注册表里看得到。"""
        register_domain_tools("clinic", service=service)
        names = [s.name for s in ToolRegistry.for_domain("clinic")]
        assert "list_departments" in names
        assert "get_availability" in names
        assert "book_appointment" in names

    def test_registration_is_idempotent_and_replaces_instances(self, service):
        """反复装配不能抛"已注册"，而且要换成新实例。

        工具是**闭包捕获具体 service** 的；若只做"已存在就跳过"，
        第二装配会继续用旧号源库（可能已经关闭），这是很难查的错。
        """
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
            department="心内科", date="", time_slot="上午", patient_id="u1")
        assert ticket.get("ticket_no")
        assert service.my_appointments("u1")

    def test_every_tool_declares_when_not(self, service):
        """`when_not` 是约束模型的主要手段，注册时会强制校验，这里再确认一遍。"""
        register_domain_tools("clinic", service=service)
        for spec in ToolRegistry.for_domain("clinic"):
            assert spec.when_not.strip(), f"{spec.name} 缺少 when_not"
