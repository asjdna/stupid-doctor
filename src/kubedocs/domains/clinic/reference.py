"""clinic 域的参考标准装配：把域内规则资产跑成一份 `FactSet` 交给大模型。

## 为什么装配放在域层，而不是 `clinical/reference.py`

`FactSet` 是通用形状（`clinical/reference.py`，不含任何医学知识）；但"用哪张症状词表、
怎么把口语折到语料键、红线词表长什么样"是 **clinic 域的知识**。所以：

    clinical/reference.py   形状 + 渲染 + 查询（谁知道怎么校验）
    domains/clinic/reference.py（本模块）  把域资产跑成那个形状（谁知道医学）

两条请求路径（`/consult` 的临床管线、`/chat` 的工具循环）都要这份面板。**只能有一份装配**：
各写一遍的结果是两条路给模型的线索不一样，而它们本该看到同样的事实。

## 本模块不做判断

它只产出线索：命中了哪些症状词、折到了哪些语料键、红线三维门控怎么说、规则票数给了哪些
候选科室、服务端有哪些号源事实。**一条结论都不下**——急症与否、挂哪个科，都由模型给出。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from kubedocs.agent import hypothesis
from kubedocs.clinical.evidence import EvidenceBundle, Guard, red_line_from_guard
from kubedocs.clinical.reference import (
    BookingFact,
    DepartmentSignal,
    FactSet,
    SymptomSignal,
    build_facts,
)
from kubedocs.domains.clinic.data.departments import normalize_department
from kubedocs.domains.clinic.pack import extract_evidence, match_symptoms
from kubedocs.domains.clinic.polarity import is_negated

SOURCE_LINKS = "症状关联票数"
"""规则票数的来源标签。写进面板是为了让模型知道"这条线索是怎么来的"。"""


def symptom_signals(question: str, links: Mapping[str, Sequence[str]] | None) -> list[SymptomSignal]:
    """患者原话 -> 症状归一线索：**原话词 → 标准词 → 它展开出的语料键**。

    三段都要保留，不能只留首尾：面板上写"原话「低烧」→ 语料键「术后发热」"是错的
    （"术后发热"不是标准词，它是语料里更细的症状名之一），而"低烧 → 发热 → 术后发热/
    癌性发热…"才是模型能据以判断的形状。
    """
    from kubedocs.domains.clinic.data.symptom_dict import match_symptom_in_text

    canonical_of = {matched: canonical for canonical, matched in match_symptom_in_text(question)}
    grouped: dict[str, list[str]] = {}
    for key, matched in match_symptoms(question, links):
        grouped.setdefault(matched, []).append(key)
    out: list[SymptomSignal] = []
    for matched, keys in grouped.items():
        out.append(
            SymptomSignal(
                matched=matched,
                # 逐字扫到的规范词没有中间层：标准词就是它自己。
                canonical=canonical_of.get(matched, matched),
                corpus_keys=list(dict.fromkeys(keys)),
                negated=is_negated(question, matched),
            )
        )
    return out


def department_signals(
    question: str,
    links: Mapping[str, Sequence[str]] | None,
    *,
    turn: int = 1,
    top: int = 8,
) -> list[DepartmentSignal]:
    """规则票数给出的候选科室。**只是候选**，最终由模型选。"""
    if not links:
        return []
    evidence = extract_evidence(question, turn, links=links)
    scores = hypothesis.score(evidence, links)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [DepartmentSignal(name=name, score=score, source=SOURCE_LINKS) for name, score in ranked]


def clinic_reference(
    question: str,
    *,
    links: Mapping[str, Sequence[str]] | None = None,
    guard: Guard | None = None,
    evidence: EvidenceBundle | None = None,
    booking: Sequence[BookingFact] = (),
    known_departments: Sequence[str] = (),
    turn: int = 1,
    notes: Sequence[str] = (),
) -> FactSet:
    """装配 clinic 域的参考标准面板。

    `evidence` 与 `guard` 都是可选的：`/chat` 的路径上检索是模型自己调工具做的
    （沿途才拿到资料），所以装配时它没有检索事实——面板照实写"知识依据为空"，
    而**校验发生在工具循环跑完之后**（那时证据已经拿到）。
    """
    problems = list(notes)
    bundle = evidence or EvidenceBundle()
    if guard is not None and bundle.red_line is None:
        degraded: list[str] = []
        red_line = red_line_from_guard(guard, question, degraded=degraded)
        problems.extend(degraded)
        if red_line is not None:
            bundle = bundle.model_copy(update={"red_line": red_line})
    if links:
        problems.extend([])  # 预留：关联表为空时在这里给一句可读说明
    else:
        problems.append("症状→科室关联表为空（语料缺失）：科室候选线索缺失，请仅依据知识与你的判断")
    return build_facts(
        evidence=bundle,
        symptoms=symptom_signals(question, links),
        departments=department_signals(question, links, turn=turn),
        booking=list(booking),
        known_departments=known_departments,
        # 校验器要按同一套规则归一化后再比较（"呼吸内科门诊" = "呼吸内科"）；
        # 规则本身属于本域，所以由这里注入，通用层不猜。
        normalize_department=normalize_department,
        notes=problems,
    )


def make_reference_builder(
    *,
    service: Any = None,
    links: Mapping[str, Sequence[str]] | None = None,
    known_departments: Sequence[str] | None = None,
    booking_history_limit: int = 5,
) -> Callable[[str, EvidenceBundle | None, str], FactSet]:
    """构造**参考标准面板建造器**（管线与评测共用的那一个）。

    面板是"模型看到的依据"，所以"产品路径"与"评测路径"必须是同一份装配：各写一遍的结果
    是评测在测一个产品不会走的形状（线索少了、写法不一样），而分数看起来仍然合理。

    - `service`（号源服务）给了就读"这个人已经确认的挂号/退号"，读不到时留一条 note
      而不是当成"没有挂号"（见下面 `booking_facts`）；
    - `links` 不给则惰性取域包的关联表并缓存一次（8MB 卡片语料解析一次是秒级，
      但每次请求都解析就不行）；
    - `known_departments` 不给则用 `DEPARTMENTS`（可挂号科室清单）。

    返回的建造器签名与管线要求的 `reference_builder(question, evidence, patient_id)` 一致。
    """
    from kubedocs.domains.clinic.data.departments import DEPARTMENTS
    from kubedocs.domains.clinic.redline import RiskGuard

    departments: Sequence[str] = DEPARTMENTS if known_departments is None else known_departments
    cached_links: list[Mapping[str, Sequence[str]] | None] = [links]

    def _links() -> Mapping[str, Sequence[str]] | None:
        if cached_links[0] is None:
            from kubedocs.domains.clinic.pack import link_vocabulary

            if link_vocabulary():
                from kubedocs.domains.clinic.domain import ClinicPack

                cached_links[0] = ClinicPack().triage_links()
            else:
                cached_links[0] = {}
        return cached_links[0]

    def builder(question: str, evidence: EvidenceBundle | None, patient_id: str) -> FactSet:
        resolved = _links()
        notes: list[str] = []
        if not resolved:
            notes.append("症状→科室关联表为空（卡片语料缺失）")
        booking, booking_notes = booking_facts(service, patient_id, limit=booking_history_limit)
        notes.extend(booking_notes)
        return clinic_reference(
            question,
            links=resolved,
            guard=RiskGuard(),
            evidence=evidence,
            booking=booking,
            known_departments=departments,
            notes=notes,
        )

    return builder


def booking_facts(service: Any, patient_id: str, *, limit: int = 5) -> tuple[list[BookingFact], list[str]]:
    """把**服务端已经确认的挂号/退号**读成面板事实。返回 `(事实, 装配问题)`。

    这是防"没挂上却说挂上"最直接的一条：模型手里有一份"本轮到底发生过什么"的清单。
    读不到时**不猜**——留一条 note 让模型知道这份事实这次不可用（而不是让它以为"没有挂号"）。
    `service` 为 None（纯离线评测）时不给事实、也不报问题：那一轮本来就没有号源服务。
    """
    if service is None:
        return [], []
    notes: list[str] = []
    try:
        rows = service.my_appointments(patient_id, limit)
    except Exception as e:  # noqa: BLE001 - 号源不可读不该让整轮问诊失败
        notes.append(f"挂号记录本次读不到（{type(e).__name__}）：请勿声称任何挂号结果")
        return [], notes
    out: list[BookingFact] = []
    for row in rows:
        status = str(row.get("status") or "")
        out.append(
            BookingFact(
                kind="cancelled" if status == "cancelled" else "confirmed",
                department=str(row.get("department") or ""),
                date=str(row.get("date") or ""),
                time_slot=str(row.get("time") or ""),
                fee=int(row.get("fee") or 0),
                ticket_no=str(row.get("ticket_no") or ""),
                queue_no=int(row.get("queue_no") or 0),
                status=status,
            )
        )
    return out, notes
