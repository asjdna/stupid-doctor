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

from collections.abc import Mapping, Sequence

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
