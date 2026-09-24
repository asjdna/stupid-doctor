"""CLI 测试：参数位置、退出码、子命令串起来能跑。

单测覆盖的是各模块内部逻辑；这里覆盖的是**接线**——参数名写错、退出码写错、
子命令之间不兼容这类问题，只有从命令行入口跑一遍才会暴露。
全部走哈希向量，不下载模型、不联网。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kubedocs.cli import EXIT_GATE_FAIL, EXIT_OK, EXIT_PRECONDITION, main

BODY = (
    "## Limits\n\nThe kubelet enforces memory limits for each container and the "
    "scheduler uses requests to decide placement.\n\n"
    "## Requests\n\nRequests affect scheduling decisions while limits are enforced "
    "by the container runtime.\n"
)


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    doc = corpus / "concepts" / "limits.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(f"---\ntitle: Resource Management\n---\n\n{BODY}\n", encoding="utf-8")
    return corpus


def _args(tmp_path: Path, *rest: str) -> list[str]:
    return ["--provider", "hashing", "--index-dir", str(tmp_path / "index"), *rest]


def _ingest(tmp_path: Path, corpus: Path) -> int:
    return main(_args(tmp_path, "--corpus-dir", str(corpus), "ingest", "--batch-files", "10"))


class TestGlobalArgs:
    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
        assert "ingest" in capsys.readouterr().out

    def test_globals_before_subcommand(self, tmp_path, capsys):
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK

    def test_globals_after_subcommand(self, tmp_path, capsys):
        """用户几乎必然这么写一次；只挂在主解析器上时这里会报"无法识别的参数"。"""
        corpus = _corpus(tmp_path)
        rc = main(["ingest", "--batch-files", "10", "--provider", "hashing",
                   "--index-dir", str(tmp_path / "index"), "--corpus-dir", str(corpus)])
        assert rc == EXIT_OK
        # 索引落在 <index_dir>/<domain>/ 下（按域分目录）
        assert (tmp_path / "index" / "k8s_docs" / "kubedocs.db").exists()

    def test_subcommand_after_flag_does_not_clobber_global(self, tmp_path):
        """子解析器上的同名参数若用普通 default，会把这个值覆盖成 None。"""
        corpus = _corpus(tmp_path)
        index = tmp_path / "index"
        assert main(["ingest", "--index-dir", str(index), "--provider", "hashing",
                     "--corpus-dir", str(corpus)]) == EXIT_OK
        assert (index / "k8s_docs" / "kubedocs.db").exists()
        # 索引确实建在指定目录 -> 用该目录能查到
        assert main(["--index-dir", str(index), "--provider", "hashing",
                     "stats", "--json"]) == EXIT_OK

    def test_missing_subcommand_is_an_error(self):
        with pytest.raises(SystemExit) as e:
            main([])
        assert e.value.code != 0

    def test_unknown_flag_is_an_error(self):
        with pytest.raises(SystemExit) as e:
            main(["--nope"])
        assert e.value.code != 0


class TestExitCodes:
    def test_stats_on_empty_index_is_precondition_failure(self, tmp_path, capsys):
        assert main(_args(tmp_path, "stats")) == EXIT_PRECONDITION
        assert "索引为空" in capsys.readouterr().err

    def test_stats_on_missing_corpus_dir_is_fine(self, tmp_path):
        # stats 不依赖语料，只依赖索引
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        assert main(_args(tmp_path, "stats", "--verify")) == EXIT_OK

    def test_search_on_empty_index_is_precondition_failure(self, tmp_path):
        assert main(_args(tmp_path, "search", "pod limits")) == EXIT_PRECONDITION

    def test_eval_without_gold_is_precondition_failure(self, tmp_path):
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        rc = main(_args(tmp_path, "eval", "--gold", str(tmp_path / "nope.jsonl")))
        assert rc == EXIT_PRECONDITION

    def test_eval_gate_without_baseline_fails(self, tmp_path):
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "id": "g1", "kind": "concept", "q": "who enforces memory limits",
            "must_contain": {"concepts/limits": "kubelet enforces memory"},
        }) + "\n", encoding="utf-8")
        rc = main(_args(tmp_path, "eval", "--gold", str(gold),
                        "--baseline", str(tmp_path / "base.json"), "--gate"))
        assert rc == EXIT_GATE_FAIL

    def test_eval_baseline_then_gate_passes(self, tmp_path):
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "id": "g1", "kind": "concept", "q": "who enforces memory limits",
            "must_contain": {"concepts/limits": "kubelet enforces memory"},
        }) + "\n", encoding="utf-8")
        base = tmp_path / "base.json"
        assert main(_args(tmp_path, "eval", "--gold", str(gold), "--baseline", str(base),
                        "--write-baseline")) == EXIT_OK
        assert base.exists()
        assert main(_args(tmp_path, "eval", "--gold", str(gold), "--baseline", str(base),
                        "--gate")) == EXIT_OK


class TestFullPipeline:
    """一条命令串起来：摄取 -> 体检 -> 检索 -> 回答（无密钥应降级而不是崩）。"""

    def test_pipeline_runs_end_to_end(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        capsys.readouterr()          # 丢掉摄取阶段的输出，下面只解析本条命令的 JSON

        assert main(_args(tmp_path, "stats", "--verify", "--json")) == EXIT_OK
        payload = json.loads(capsys.readouterr().out.split("完成")[0])
        assert payload["n_chunks"] > 0
        assert payload["fts_integrity"] == "ok"
        assert payload["embedding_model"] == "hashing-384"

        assert main(_args(tmp_path, "search", "memory limits", "--json")) == EXIT_OK
        results = json.loads(capsys.readouterr().out.split("完成")[0])["results"]
        assert results and results[0]["source_url"].startswith("https://kubernetes.io/docs/")

        # ask：没有密钥 -> 明确拒答 + 仍返回检索来源，而不是 traceback
        assert main(_args(tmp_path, "ask", "how are memory limits enforced?", "--json")) == EXIT_OK
        answer = json.loads(capsys.readouterr().out.split("完成")[0])
        assert answer["refused"] is True
        assert answer["citations"], "降级时也要给出检索来源"
        assert "DEEPSEEK_API_KEY" in answer["refusal_reason"]

    def test_ask_show_context_prints_the_prompt_context(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        capsys.readouterr()
        assert main(_args(tmp_path, "ask", "memory limits", "--show-context")) == EXIT_OK
        out = capsys.readouterr().out
        assert "提供给模型的上下文" in out
        assert "[1] Resource Management" in out
