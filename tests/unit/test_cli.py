"""CLI 契约测试：全局参数位置、退出码、子命令能否串起来跑。

参数名写错、退出码写错、子命令不兼容这类接线问题，只有从命令行入口跑一遍才暴露。
全部走哈希向量，不下载模型、不联网。

`ask` 的契约（新架构）：**没有大模型密钥时只回一句话**（`NO_LLM_MESSAGE`），退出码走
`EXIT_PRECONDITION`（缺配置 ≠ 用户参数错），**不打印检索结果**——没有判断者就没有结论。
检索不需要大模型，走 `kubedocs search`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kubedocs.cli import EXIT_GATE_FAIL, EXIT_OK, EXIT_PRECONDITION, main
from kubedocs.llm_status import NO_LLM_MESSAGE

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
    """CLI 参数基线：**显式指定本地后端**。

    `--index-dir` 只对本地后端有意义：`pgvector` 的索引在 Postgres 的 schema 里，
    目录参数对它没有作用。默认后端是 `pgvector`，所以这些用例如果不写
    `--store sqlite-vec`，就会连到真实的 Postgres 去读真实索引——测试于是既不是
    隔离的、结果也取决于那台机器上有什么。要测 pgvector 的用例自己显式传 store。
    """
    return [
        "--store",
        "sqlite-vec",
        "--provider",
        "hashing",
        "--index-dir",
        str(tmp_path / "index"),
        *rest,
    ]


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
        rc = main(
            [
                "ingest",
                "--batch-files",
                "10",
                "--store",
                "sqlite-vec",
                "--provider",
                "hashing",
                "--index-dir",
                str(tmp_path / "index"),
                "--corpus-dir",
                str(corpus),
            ]
        )
        assert rc == EXIT_OK
        # 索引落在 <index_dir>/<domain>/ 下（按域分目录）
        assert (tmp_path / "index" / "k8s_docs" / "kubedocs.db").exists()

    def test_subcommand_after_flag_does_not_clobber_global(self, tmp_path):
        """子解析器上的同名参数若用普通 default，会把这个值覆盖成 None。"""
        corpus = _corpus(tmp_path)
        index = tmp_path / "index"
        assert (
            main(
                [
                    "ingest",
                    "--store",
                    "sqlite-vec",
                    "--index-dir",
                    str(index),
                    "--provider",
                    "hashing",
                    "--corpus-dir",
                    str(corpus),
                ]
            )
            == EXIT_OK
        )
        assert (index / "k8s_docs" / "kubedocs.db").exists()
        # 索引确实建在指定目录 -> 用该目录能查到
        assert (
            main(
                [
                    "--store",
                    "sqlite-vec",
                    "--index-dir",
                    str(index),
                    "--provider",
                    "hashing",
                    "stats",
                    "--json",
                ]
            )
            == EXIT_OK
        )

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

    def test_ask_without_key_is_precondition_failure_even_with_an_empty_index(self, tmp_path, capsys):
        """`ask` 的前置条件是"有判断者"（密钥），排在索引检查之前：退出码同为 3。

        所以空索引 + 无密钥时给的是"未连接大模型"那句话，而不是"索引为空"——
        两件事都能让人拿不到回答，但要做的事完全不同。
        """
        assert main(_args(tmp_path, "ask", "pod limits")) == EXIT_PRECONDITION
        captured = capsys.readouterr()
        assert NO_LLM_MESSAGE in captured.err
        assert "索引为空" not in captured.err

    def test_eval_without_gold_is_precondition_failure(self, tmp_path):
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        rc = main(_args(tmp_path, "eval", "--gold", str(tmp_path / "nope.jsonl")))
        assert rc == EXIT_PRECONDITION

    def test_eval_gate_without_baseline_fails(self, tmp_path):
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "id": "g1",
                    "kind": "concept",
                    "q": "who enforces memory limits",
                    "must_contain": {"concepts/limits": "kubelet enforces memory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rc = main(
            _args(tmp_path, "eval", "--gold", str(gold), "--baseline", str(tmp_path / "base.json"), "--gate")
        )
        assert rc == EXIT_GATE_FAIL

    def test_eval_baseline_then_gate_passes(self, tmp_path):
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "id": "g1",
                    "kind": "concept",
                    "q": "who enforces memory limits",
                    "must_contain": {"concepts/limits": "kubelet enforces memory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        base = tmp_path / "base.json"
        assert (
            main(_args(tmp_path, "eval", "--gold", str(gold), "--baseline", str(base), "--write-baseline"))
            == EXIT_OK
        )
        assert base.exists()
        assert (
            main(_args(tmp_path, "eval", "--gold", str(gold), "--baseline", str(base), "--gate")) == EXIT_OK
        )


class TestFullPipeline:
    """一条命令串起来：摄取 -> 体检 -> 检索 -> 回答（无密钥只回一句话，不返回检索兜底）。"""

    def test_pipeline_runs_end_to_end(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        corpus = _corpus(tmp_path)
        assert _ingest(tmp_path, corpus) == EXIT_OK
        capsys.readouterr()  # 丢掉摄取阶段的输出，下面只解析本条命令的 JSON

        assert main(_args(tmp_path, "stats", "--verify", "--json")) == EXIT_OK
        captured = capsys.readouterr()
        # 收尾旁白按约定走 stderr，所以断言写成"stdout 能被 json.loads 直接吃掉"
        assert "完成，用时" in captured.err
        payload = json.loads(captured.out)
        assert payload["n_chunks"] > 0
        assert payload["fts_integrity"] == "ok"
        assert payload["embedding_model"] == "hashing-384"

        # 检索不需要大模型：照常返回结果（`ask` 的收口不影响这条路）
        assert main(_args(tmp_path, "search", "memory limits", "--json")) == EXIT_OK
        results = json.loads(capsys.readouterr().out)["results"]
        assert results and results[0]["source_url"].startswith("https://kubernetes.io/docs/")

        # ask：没有密钥 -> 只回一句话 + 前置条件失败退出码，**stdout 上一个检索结果都没有**
        assert main(_args(tmp_path, "ask", "how are memory limits enforced?", "--json")) == EXIT_PRECONDITION
        captured = capsys.readouterr()
        assert NO_LLM_MESSAGE in captured.err
        assert captured.out == "", "无密钥时不许把检索结果当兜底打印出来"

    def test_ask_without_key_says_so_even_on_a_non_empty_index(self, tmp_path, capsys, monkeypatch):
        """**新契约**：无密钥时的那句话是"未连接大模型"，而不是索引为空。

        密钥检查排在 `open_store` 之前：没有判断者时索引里有什么都不改变结论，
        也就不该为这一条命令加载嵌入器、跑一遍注定不用的检索。
        """
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        capsys.readouterr()

        rc = main(_args(tmp_path, "ask", "memory limits"))

        assert rc == EXIT_PRECONDITION
        captured = capsys.readouterr()
        assert captured.err.strip() == NO_LLM_MESSAGE
        assert "索引为空" not in captured.err
        # 检索结果（标题、来源 URL）一个都不许出现：没有兜底层
        assert "Resource Management" not in captured.out + captured.err
        assert "https://" not in captured.out + captured.err

    def test_ask_show_context_prints_the_prompt_context(self, tmp_path, capsys, monkeypatch):
        """`--show-context` 打印喂给模型的上下文：它需要**有密钥**才走得到生成分支。

        打桩打在 `LiteLLMClient` 与 `has_api_key` 上（CLI 里是函数内 import，
        所以补丁打在被导入的模块属性上），绝不联网。
        """
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        capsys.readouterr()
        monkeypatch.setattr("kubedocs.generate.has_api_key", lambda _model: True)

        class _StubClient:
            def __init__(self, _settings) -> None:
                pass

            def complete(self, system: str, user: str):
                from kubedocs.generate import LLMResult

                return LLMResult(text="答案 [1]")

        monkeypatch.setattr("kubedocs.generate.LiteLLMClient", _StubClient)

        assert main(_args(tmp_path, "ask", "memory limits", "--show-context")) == EXIT_OK
        out = capsys.readouterr().out
        assert "提供给模型的上下文" in out
        assert "[1] Resource Management" in out


class TestJsonStdoutIsPureJson:
    """`--json` 时 stdout 上只允许有 JSON，`main()` 的收尾旁白一律走 stderr。

    否则 `... | jq`（或 PowerShell 的 `ConvertFrom-Json`）整体解析失败，
    所有 `--json` 子命令的机器可读输出都报废。
    """

    def test_timing_line_goes_to_stderr_in_json_mode(self, tmp_path, capsys):
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        capsys.readouterr()
        assert main(_args(tmp_path, "stats", "--json")) == EXIT_OK

        captured = capsys.readouterr()
        json.loads(captured.out)  # 旁白混进 stdout 就会在这里炸
        assert "完成，用时" in captured.err

    def test_timing_line_stays_on_stdout_without_json(self, tmp_path, capsys):
        """非 `--json` 时位置不变：人读的输出仍在 stdout，不制造无谓的破坏性改动。"""
        assert _ingest(tmp_path, _corpus(tmp_path)) == EXIT_OK
        capsys.readouterr()
        assert main(_args(tmp_path, "stats")) == EXIT_OK

        captured = capsys.readouterr()
        assert "完成，用时" in captured.out
        assert captured.err == ""


class TestStatsVerifyCallsTheVectorChecker:
    """`stats --verify` 必须真的调用后端的向量体检方法，且两种方法名都要认。

    只认 `verify_vec` 会让 `verify_hnsw` 那样的后端永远体检不到，`stats --verify` 照样打印"通过"。
    """

    class _FakeStore:
        """只实现 `stats --verify` 用到的方法；`method=None` 表示后端没有向量体检方法。"""

        def __init__(self, method: str | None) -> None:
            self._method = method

        def verify_fts(self) -> str:
            return "ok"

        def __getattr__(self, name: str):
            if self._method is not None and name == self._method:
                return lambda: "ok"
            raise AttributeError(name)

        def stats(self):
            class _S:
                n_docs = 1
                n_chunks = 1
                # 父子切割之后"块数"与"父块数"是两个数，stats 两个都要读；
                # 这个替身缺字段会让 `kubedocs stats` 直接 AttributeError。
                n_parents = 1
                mean_chars = 10.0
                n_with_code = 0
                embedding_model = "fake"
                embedding_dim = 8
                indexed_at = "now"
                normalizer = "identity"

            return _S()

        def meta_dump(self) -> dict[str, str]:
            return {}

        def close(self) -> None:
            pass

    @pytest.mark.parametrize("method", ["verify_vec", "verify_hnsw"])
    def test_each_backend_method_name_is_recognised(self, tmp_path, capsys, monkeypatch, method):
        import kubedocs.store as store_mod

        monkeypatch.setattr(store_mod, "from_settings", lambda _settings: self._FakeStore(method))

        rc = main(_args(tmp_path, "stats", "--verify"))
        out = capsys.readouterr().out

        assert rc == EXIT_OK
        assert "向量一致性 : ok" in out, f"{method} 没有被调用"

    def test_missing_checker_is_not_an_error(self, tmp_path, capsys, monkeypatch):
        """后端没有任何向量体检方法时不该崩（只是不打印那一行）。"""
        import kubedocs.store as store_mod

        monkeypatch.setattr(store_mod, "from_settings", lambda _settings: self._FakeStore(None))

        rc = main(_args(tmp_path, "stats", "--verify"))
        out = capsys.readouterr().out

        assert rc == EXIT_OK
        assert "向量一致性" not in out
        assert "FTS 一致性 : ok" in out
