"""临床路径的**唯一装配入口**：检索后端 + 域工具 + 工具循环 + 管线。

## 为什么要有这个模块

`/consult` 与评测的 LLM 臂必须跑**同一条路**。以前的做法是两边各写一遍装配，结果是
评测测的是一个"产品不会走的形状"——工具没注册全、面板来自另一套装配、预算账本各记一份，
而分数看起来仍然合理。**分数可信的前提是它测的是产品路径**，所以装配只能有一处。

装配需要的东西全部**注入**：检索器、号源服务、待确认单、就诊记录落点、面板建造器、
审计落点。这样产品注入真实实现，评测注入 `MockAppointmentProvider` 与临时落点，
而**代码路径完全一致**。

## 两个入口

- `open_domain_index(settings)`：按 `settings.domain` 打开索引并构造检索器
  （产品由 `api/app.py` 的运行时持有索引；离线评测用这个自己开、自己关）。
- `build_clinical_assembly(...)`：把上一步的检索器接成一条可用的临床管线，
  返回管线与它用到的部件（预算账本、工具循环、单发 reasoner）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kubedocs.clinical.pipeline import ClinicalPipeline


@dataclass
class ClinicalAssembly:
    """装配好的产品路径。`budget` / `tool_loop` / `reasoner` 一起返回是为了**复用**：

    - `budget`：本次花了多少 = 调用前后的账本差（评测与运维都读它）；
    - `tool_loop`：真实判断路径（模型自己决定调哪些工具）；
    - `reasoner`：不带工具的一条判断路线，只在对照与单发评测里用。
    """

    pipeline: ClinicalPipeline
    budget: Any
    tool_loop: Any
    reasoner: Any
    settings: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def open_domain_index(settings: Any) -> tuple[Any, Any]:
    """按域打开索引并构造检索器，返回 `(store, retriever)`。**调用方负责关闭 store**。

    索引为空时抛 `RuntimeError`（而不是返回一个"检索永远为空"的检索器）：那会让
    "这次依据不全"看起来像"模型判断有问题"，是最难查的一类假象。
    """
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry
    from kubedocs.embed import build_embedder
    from kubedocs.store import from_settings

    store = from_settings(settings)
    stats = store.stats()
    if stats.n_chunks == 0:
        store.close()
        raise RuntimeError(
            f"{settings.domain} 域的索引为空（store={settings.store}）⇒ 无法装配检索器。"
            "先跑 `python -m kubedocs.cli ingest`。"
        )
    load_all()
    pack = DomainRegistry.get(settings.domain)
    retriever = pack.retriever(store, build_embedder(settings), settings)
    return store, retriever


def build_clinical_assembly(
    *,
    settings: Any,
    retriever: Any,
    booking: Any,
    pending: Any = None,
    store: Any = None,
    reference_builder: Any = None,
    has_llm: Any = None,
    index_root: str | Path | None = None,
    top_k: int = 10,
    budget: Any = None,
) -> ClinicalAssembly:
    """把产品路径装配起来。**这是 `/consult` 与评测 LLM 臂共用的那一份**。

    `retriever` 必须是已构造好的检索器（见 `open_domain_index`）：装配本身不做 IO，
    否则"导入即加载嵌入模型"这种副作用会跑到测试与 CLI 的启动路径上。

    `reference_builder` 由**域层**提供（`domains.clinic.reference.make_reference_builder`）：
    面板要用的症状词表、科室表、号源事实都是域知识，管线这一层不该知道。
    """
    from kubedocs.agent.bootstrap import register_domain_tools
    from kubedocs.clinical.orchestration import ClinicalToolLoop
    from kubedocs.clinical.reasoning import BudgetedClient, ClinicalReasoner, LiteLLMAdapter
    from kubedocs.clinical.triage import RetrievalTriage, load_department_map
    from kubedocs.domains.clinic.redline import RiskGuard
    from kubedocs.generate import DailyBudget, LiteLLMClient

    if budget is None:
        budget = DailyBudget(
            settings.llm_spend_path,
            settings.llm_daily_budget_usd,
            unpriced_usd=settings.llm_unpriced_call_usd,
        )

    # **把 `search_docs` 注册进当前域**——这一步是工具循环能不能工作的前提。漏了它，
    # `tool_schemas(domain=...)` 在过滤掉副作用工具后可能是空的，模型一个工具都没得调，
    # 于是退回用自己的通用知识作答：表面上"有回答"，实际上"检索不到就不给"被架空了。
    register_domain_tools(
        settings.domain,
        service=booking,
        retriever=retriever,
        # 没有 `pending` 就不会注册 `propose_booking`：挂号的入口是"提议一张待确认单"，
        # 不是"直接取号"。
        pending=pending,
    )

    # 同一个 BudgetedClient ⇒ 循环里每一次调用都过同一本日预算账。
    budgeted = BudgetedClient(LiteLLMClient(settings), budget)
    tool_loop = ClinicalToolLoop(budgeted, domain=settings.domain)
    reasoner = ClinicalReasoner(LiteLLMAdapter(budgeted))

    # 科室补全映射表：语料里近半卡片的科室是「未标注」，靠这张表把票补回来。
    # 只信 high 档（图谱直接关系 + SameAs 别名）；含模糊归并的 medium 档准确率更差。
    dep_map = load_department_map(min_confidence="high")

    pipeline = ClinicalPipeline(
        backend=_backend_for(settings, retriever),
        reasoner=reasoner,
        guard=RiskGuard(),
        triage=RetrievalTriage(department_map=dep_map),
        tool_loop=tool_loop,
        store=store,
        top_k=top_k,
        reference_builder=reference_builder,
        has_llm=has_llm,
        index_root=index_root,
    )
    return ClinicalAssembly(
        pipeline=pipeline,
        budget=budget,
        tool_loop=tool_loop,
        reasoner=reasoner,
        settings=settings,
        extras={"department_map_entries": len(dep_map)},
    )


def _backend_for(settings: Any, retriever: Any) -> Any:
    """把检索器包成管线要的 `RetrievalBackend`。

    `k8s_docs` 与 `clinic` 用同一个适配器：管线只认 `search(query, top_k)` 这个形状，
    域差异在检索器内部（域包自己决定候选池与精排）。
    """
    from kubedocs.clinical.rag import LocalIndexBackend

    return LocalIndexBackend(retriever)
