"""患者数据的持久化接缝（**当前只有占位实现**）。

患者数据要进系统、多机构是刚需，但当前不深挖实现。这里只做一件正确的事——
**把接缝定下来**，让上层调用点先长成将来的形状，将来接真实存储时不必回头改业务代码。

刻意**不含**的东西：

- 机构隔离（行级安全 / 连接级 schema / 分库）：隔离策略取决于部署形态与合规要求；
- 字段级加密与脱敏；
- 任何真实患者数据：`NoopPersistence` 会把写入丢掉，这样"跑起来的东西"不会
  意外把测试期数据留在磁盘上。

`institution` 参数现在就加上，是因为它是"多机构"这条需求唯一**现在就能确定**的形状：
先放上去，将来真实实现出现时不必改所有调用点的签名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kubedocs.clinical.contracts import ClinicalAssessment


@dataclass
class PatientRecord:
    """患者主体。`institution` 是机构标识（多机构隔离的锚点）。"""

    patient_id: str
    institution: str = ""
    display_name: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class EncounterRecord:
    """一次就诊（一轮问诊）。"""

    encounter_id: str
    patient_id: str
    institution: str = ""
    chief_complaint: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    assessment: ClinicalAssessment | None = None


class ClinicalStore(Protocol):
    """产品层需要的持久化能力。实现可以是 SQLite、Postgres，也可以先不实现。"""

    def upsert_patient(self, record: PatientRecord) -> None: ...

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None: ...

    def save_encounter(self, record: EncounterRecord) -> None: ...

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None: ...


class NoopPersistence:
    """默认实现：**什么都不存**。

    选它当默认而不是"顺手用内存字典"是有意的：内存字典会给人"数据存住了"的错觉，
    重启才发现没有；`NoopPersistence` 从命名到行为都不含糊——**现在没有持久化**。
    """

    def upsert_patient(self, record: PatientRecord) -> None:
        del record

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None:
        del encounter_id, role, text, institution

    def save_encounter(self, record: EncounterRecord) -> None:
        del record

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None:
        del patient_id, institution
        return None
