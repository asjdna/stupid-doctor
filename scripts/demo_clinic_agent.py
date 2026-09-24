"""真实 LLM 走工具循环的演示：把"LLM 自己决定调什么工具"的证据打出来。

这是本项目唯一能证明"它是 agent 而不是流水线"的运行记录：
调用顺序不是代码排的，是模型通过 function calling 自己选的。

用法：
    python scripts/demo_clinic_agent.py            # 默认两轮：描述症状 → 要求挂号
    python scripts/demo_clinic_agent.py --turn "我咳嗽三天了，帮我挂呼吸内科"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.agent.clinic_react import ClinicReactRuntime  # noqa: E402
from kubedocs.config import Settings, load_env_file  # noqa: E402
from kubedocs.domains.clinic.booking import BookingService  # noqa: E402
from kubedocs.domains.clinic.mock import MockAppointmentProvider  # noqa: E402


def _show(i: int, message: str, turn: object) -> None:
    print(f"\n{'=' * 70}\n第 {i} 轮  用户：{message}")
    if getattr(turn, "emergency", False):
        print("  ⚠ 红线命中（确定性规则，LLM 未被调用）")
    for tc in getattr(turn, "tool_calls", []):
        args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
        print(f"  🔧 模型选择调用 {tc['tool']}({args})")
        if tc.get("error"):
            print(f"       ✗ 被拒/失败：{tc['error']}")
        else:
            result = json.dumps(tc.get("result"), ensure_ascii=False, default=str)
            print(f"       ✓ 返回：{result[:180]}")
    print(f"  助手：{getattr(turn, 'text', '')[:400]}")
    print(f"  （本轮 LLM 调用 {getattr(turn, 'llm_calls', 0)} 次，"
          f"步数 {getattr(turn, 'steps', 0)}）")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="真实 LLM 的工具循环演示")
    p.add_argument("--turn", action="append", default=[],
                   help="指定一轮用户消息（可重复）；不给则用默认两轮")
    args = p.parse_args(argv)

    added = load_env_file()
    settings = Settings(domain="clinic")
    rt = ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()))
    print(f"模型：{settings.llm_model}")
    print(f"密钥：{'已从 .env 载入 ' + str(added) if added else '环境变量中已有'}")
    print(f"可用工具：{[t.name for t in rt.tools]}")
    if not rt.has_llm():
        print("\n没有可用密钥，无法演示工具循环（这本身就说明产品会走降级路径）",
              file=sys.stderr)
        return 3

    messages = args.turn or ["我最近老咳嗽，还有点发烧", "帮我挂个号"]
    total_calls = 0
    for i, msg in enumerate(messages, start=1):
        turn = rt.turn(msg)
        total_calls += turn.llm_calls
        _show(i, msg, turn)

    print(f"\n{'=' * 70}")
    print(f"合计 LLM 调用 {total_calls} 次")
    print("注意：**工具是被模型选中的**——上面的调用顺序不是代码写死的。"
          "这正是它与确定性流水线的区别。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
