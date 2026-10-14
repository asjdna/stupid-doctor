"""提示词的加载与装配。

## 为什么提示词放在 `.md` 文件里而不是 Python 字符串里

1. **可 diff**：改一个字的语气，`git diff` 里就是那一行，而不是"某个三引号块变了"；
2. **可版本化**：`PROMPT_VERSION` 是内容的 sha256 前 12 位。评测报告里记上它，
   才能回答"这次的分数是哪版提示词跑出来的"——不然提示词一改，所有历史数字都失去可比性；
3. **可被测试锁死**：硬性条款（不给剂量、红线必触发紧急指引、依据缺失就不给建议）
   在单测里逐条断言存在。**提示词是产品契约的一部分，不该只靠人读。**

## 装配约定

系统提示词是**静态**的（`system.md`），每次请求都把动态内容（证据、已采集信息、
患者原话）放进 **user 消息**。这样系统提示词可以被缓存、被审计，也便于对照两份报告。
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from kubedocs.clinical.evidence import EvidenceBundle
from kubedocs.clinical.intake import IntakeState

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system.md")


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """读一次并缓存。文件缺失时**大声失败**：没有提示词的产品不该启动。"""
    if not SYSTEM_PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"系统提示词缺失：{SYSTEM_PROMPT_PATH}。"
            "提示词是产品契约的一部分（硬性条款在单测里被逐条断言），不能可选。"
        )
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def prompt_version() -> str:
    """提示词内容指纹（sha256 前 12 位）。评测报告与日志里都该带上它。"""
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()[:12]


def build_user_message(question: str, evidence: EvidenceBundle, intake: IntakeState) -> str:
    """把这一轮的全部动态输入拼成一条 user 消息。

    顺序是刻意的：**证据 → 已采集 → 患者原话**。
    证据在前，因为红线信号必须在模型读到患者描述之前就进入视野。
    """
    return "\n\n".join(
        [
            evidence.render(),
            intake.render(),
            f"【患者这一轮说的话】\n{question}",
        ]
    )
