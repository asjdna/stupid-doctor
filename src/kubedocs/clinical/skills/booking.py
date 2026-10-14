"""挂号 skill：把挂号从"几个零散工具"变成**一条有顺序、有确认、有收据的流程**。

## 流程（方案 §6）

```
resolve_department  解析目标科室（把口语名收敛到系统里的科室）   无副作用
list_candidates     查号源，拿到可约时段                       无副作用
confirm             复述给患者并等待显式确认                   无副作用
commit              落库取号                                  **唯一副作用步**
receipt             生成可核对凭据                             无副作用
```

## 两条必须成立的性质

1. **未确认绝不取号**。`confirm` 步在 `request.confirmed=False` 时把流程停在这里，
   返回 `needs_confirmation=True` 与一句可念给患者听的复述。调用方拿到患者答复后，
   用 `confirmed=True` **重跑同一条 skill**——流程无状态，重跑是安全的（见第 2 条）。
2. **重跑是幂等的**：`resolve_department` 与 `list_candidates` 只读；
   `commit` 之前的所有步骤都不改外部世界。这就是"副作用只许出现在最后一步"的用处。

## 为什么不用模型直接调 `book_appointment` 工具

因为那会跳过 `confirm`。挂号占用的是**真实号源**，占错一次就要退号；
把确认做成流程里绕不过去的一步，比在提示词里写"请先确认"可靠得多。
"""

from __future__ import annotations

from typing import Any, Protocol

from kubedocs.clinical.skills.base import SkillOutcome, SkillRequest, SkillStep


class BookingBackend(Protocol):
    """挂号后端。与 `domains.clinic.booking.BookingService` 形状一致（可直接注入）。"""

    def resolve_department(self, name: str) -> str: ...

    def availability(self, department: str, date: str | None = None) -> Any: ...

    def first_available_slot(self, department: str, date: str | None = None) -> str | None: ...

    def book(
        self,
        department: str,
        patient_id: str,
        *,
        date: str | None = None,
        time_slot: str | None = None,
        patient_name: str = "",
    ) -> dict: ...


class BookingSkill:
    """挂号 skill。构造时注入后端；`run` 是纯流程，不持有状态。"""

    name = "book_appointment"
    description = "按科室与时段为患者挂号；必须先经患者显式确认才会占号。"
    # 显式标注成 `tuple[SkillStep, ...]`：`Skill` 协议里的 `steps` 是可变属性（不变型），
    # 固定长度元组虽然更精确，却会让 mypy 判定"不满足协议"。这里以协议为准。
    steps: tuple[SkillStep, ...] = (
        SkillStep("resolve_department", "把患者说的科室名解析成系统内的科室"),
        SkillStep("list_candidates", "查询该科室的可约时段"),
        SkillStep("confirm", "向患者复述并等待明确同意"),
        SkillStep("commit", "取号（唯一副作用步骤）", side_effect=True),
        SkillStep("receipt", "生成可核对凭据"),
    )

    def __init__(self, backend: BookingBackend) -> None:
        self._backend = backend

    def run(self, request: SkillRequest) -> SkillOutcome:
        done: list[str] = []
        params = request.params
        raw_department = str(params.get("department") or "").strip()
        date = params.get("date") or None

        # --- 1) 解析科室 -------------------------------------------------
        if not raw_department:
            return SkillOutcome(
                ok=False,
                stopped_at="resolve_department",
                message="还不知道要挂哪个科，请先确定科室。",
                steps_done=done,
            )
        try:
            department = self._backend.resolve_department(raw_department)
        except Exception as e:  # noqa: BLE001 - 后端异常一律转成流程内可读的失败
            return SkillOutcome(
                ok=False,
                stopped_at="resolve_department",
                message=f"无法识别科室「{raw_department}」：{e}",
                steps_done=done,
            )
        done.append("resolve_department")

        # --- 2) 查号源 ---------------------------------------------------
        try:
            avail = self._backend.availability(department, date)
        except Exception as e:  # noqa: BLE001
            return SkillOutcome(
                ok=False,
                stopped_at="list_candidates",
                message=f"查询号源失败（{department}）：{e}",
                steps_done=done,
            )
        remaining = getattr(avail, "total_remaining", lambda: 0)()
        if not remaining:
            return SkillOutcome(
                ok=False,
                stopped_at="list_candidates",
                message=f"{department}在 {date or '近期'} 没有余号了，可以换日期或换科室。",
                data={"department": department},
                steps_done=done,
            )
        slot = self._backend.first_available_slot(department, date)
        done.append("list_candidates")

        # --- 3) 确认（未确认就在这里停住，绝不往下走）---------------------
        if not request.confirmed:
            return SkillOutcome(
                ok=False,
                stopped_at="confirm",
                needs_confirmation=True,
                message=(
                    f"为你预约 {department}，时间是 {slot}（{date or '最近可约日期'}）。确认要挂这个号吗？"
                ),
                data={"department": department, "time_slot": slot, "date": date},
                steps_done=done,
            )
        done.append("confirm")

        # --- 4) 取号（唯一副作用）----------------------------------------
        try:
            ticket = self._backend.book(
                department,
                request.patient_id,
                date=date,
                time_slot=str(slot) if slot else None,
                patient_name=str(params.get("patient_name") or ""),
            )
        except Exception as e:  # noqa: BLE001
            return SkillOutcome(
                ok=False,
                stopped_at="commit",
                message=f"取号失败：{e}",
                data={"department": department, "time_slot": slot},
                steps_done=done,
            )
        done.append("commit")

        # --- 5) 收据 -----------------------------------------------------
        # 收据只放患者能核对的东西，不放内部 id（对患者无用，且是多余的信息暴露）。
        receipt = {
            "department": str(ticket.get("department") or department),
            "date": str(ticket.get("date") or date or ""),
            "time_slot": str(ticket.get("time_slot") or slot or ""),
            "ticket_no": str(ticket.get("ticket_no") or ""),
        }
        done.append("receipt")
        return SkillOutcome(
            ok=True,
            stopped_at="receipt",
            message=(
                f"已预约 {receipt['department']}，{receipt['date']} {receipt['time_slot']}，"
                f"取号凭据 {receipt['ticket_no']}。请按时就诊。"
            ),
            data=receipt,
            steps_done=done,
        )
