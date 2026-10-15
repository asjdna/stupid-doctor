"""评测器测试：标注集自检、闸门判定、端到端跑通。"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from kubedocs.config import Settings
from kubedocs.embed import HashingEmbedder
from kubedocs.evaluation.harness import (
    BASELINE_SCHEMA,
    GATE_TOL_HIT,
    GoldRow,
    check_gate,
    gold_sha256,
    load_gold,
    run_eval,
    validate_gold,
)
from kubedocs.ingest import ingest

BODY = (
    "## Limits\n\nThe kubelet enforces memory limits per container, and the "
    "scheduler uses requests to place Pods on nodes.\n"
)

DOC = "concepts/limits"
DISTRACTOR = "concepts/networking"


def _corpus_with_one_doc(settings: Settings) -> None:
    p = settings.corpus_path / f"{DOC}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\ntitle: Resource Management\n---\n\n{BODY}\n", encoding="utf-8")


def _corpus_with_two_docs(settings: Settings) -> None:
    """加一篇主题完全无关的干扰文档，这样 top_k=1 时才有可能"问不到"。"""
    _corpus_with_one_doc(settings)
    p = settings.corpus_path / f"{DISTRACTOR}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\ntitle: Network Policies\n---\n\n## Policy\n\nNetwork policy ingress and "
        "egress rules select pods by label and control which connections are allowed.\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------- 标注集加载
class TestLoadGold:
    def test_missing_must_contain_is_rejected(self, tmp_path):
        f = tmp_path / "g.jsonl"
        f.write_text(json.dumps({"id": "x", "q": "q"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must_contain"):
            load_gold(f)

    def test_duplicate_ids_rejected(self, tmp_path):
        f = tmp_path / "g.jsonl"
        row = {"id": "x", "q": "q", "must_contain": {"d": "t"}}
        f.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="重复"):
            load_gold(f)

    def test_bad_json_reports_line_number(self, tmp_path):
        f = tmp_path / "g.jsonl"
        f.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r":1"):
            load_gold(f)

    def test_empty_file_rejected(self, tmp_path):
        f = tmp_path / "g.jsonl"
        f.write_text("\n# 只有注释\n", encoding="utf-8")
        with pytest.raises(ValueError, match="没有任何标注行"):
            load_gold(f)

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        f = tmp_path / "g.jsonl"
        f.write_text(
            "# 注释\n\n" + json.dumps({"id": "x", "q": "q", "must_contain": {"d": "t"}}) + "\n",
            encoding="utf-8",
        )
        assert len(load_gold(f)) == 1


class TestValidateGold:
    def test_good_row_passes(self, settings, store, embedder):
        _corpus_with_one_doc(settings)
        ingest(store, embedder, settings)
        rows = [GoldRow(id="g1", q="q", kind="concept", must_contain={DOC: "kubelet enforces memory"})]
        assert validate_gold(rows, store) == []

    def test_unknown_doc_is_reported(self, settings, store, embedder):
        _corpus_with_one_doc(settings)
        ingest(store, embedder, settings)
        rows = [GoldRow(id="g1", q="q", kind="concept", must_contain={"nope/none": "x"})]
        problems = validate_gold(rows, store)
        assert problems and "不在索引中" in problems[0].issue

    def test_wrong_term_is_reported(self, settings, store, embedder):
        """这是最关键的一道：标注了一篇不含答案的文档，指标会静默变成噪声。"""
        _corpus_with_one_doc(settings)
        ingest(store, embedder, settings)
        rows = [GoldRow(id="g1", q="q", kind="concept", must_contain={DOC: "this phrase is not there"})]
        problems = validate_gold(rows, store)
        assert problems and "找不到" in problems[0].issue


# --------------------------------------------------------------------- 闸门
def _base(**kw) -> dict:
    base = {
        "schema": BASELINE_SCHEMA,
        "gold_sha256": "abc",
        "corpus_sha": "c0ffee",
        "embedding_model": "m1",
        "n_chunks": 1000,
        "top_k": 10,
        "metrics": {"hit": 0.80, "mrr": 0.60},
    }
    base.update(kw)
    return base


def _current(**kw) -> dict:
    cur = {
        "gold_sha256": "abc",
        "corpus_sha": "c0ffee",
        "embedding_model": "m1",
        "n_chunks": 1000,
        "top_k": 10,
        "metrics": {"hit": 0.80, "mrr": 0.60},
        "n_citation_violations": 0,
    }
    cur.update(kw)
    return cur


def _hard(fails: list[str]) -> list[str]:
    return [f for f in fails if not f.startswith("警告")]


class TestGate:
    def test_identical_metrics_pass(self):
        assert _hard(check_gate(_base(), _current(), allow_drift=False)) == []

    def test_small_drop_within_tolerance_passes(self):
        cur = _current(metrics={"hit": 0.80 - GATE_TOL_HIT / 2, "mrr": 0.60})
        assert _hard(check_gate(_base(), cur, allow_drift=False)) == []

    def test_hit_drop_fails(self):
        cur = _current(metrics={"hit": 0.70, "mrr": 0.60})
        fails = _hard(check_gate(_base(), cur, allow_drift=False))
        assert any("hit@10" in f for f in fails)

    def test_mrr_drop_fails(self):
        cur = _current(metrics={"hit": 0.80, "mrr": 0.40})
        fails = _hard(check_gate(_base(), cur, allow_drift=False))
        assert any("mrr@10" in f for f in fails)

    def test_gold_change_refuses_comparison(self):
        """改了问题却拿旧基线比，是自欺——必须拒绝，而不是给出一个数字。"""
        fails = _hard(check_gate(_base(), _current(gold_sha256="zzz"), allow_drift=False))
        assert any("标注集已变更" in f for f in fails)

    def test_model_change_refuses_comparison(self):
        fails = _hard(check_gate(_base(), _current(embedding_model="m2"), allow_drift=False))
        assert any("嵌入模型变了" in f for f in fails)

    def test_top_k_change_refuses_comparison(self):
        """hit 随 k 单调上升，拿 hit@20 比基线的 hit@10 会变绿，等于留了后门。"""
        fails = _hard(check_gate(_base(), _current(top_k=20), allow_drift=False))
        assert any("top_k 不一致" in f for f in fails)

    def test_same_top_k_still_passes(self):
        """上一条不能把正常路径一起打死。"""
        assert _hard(check_gate(_base(), _current(top_k=10), allow_drift=False)) == []

    def test_corpus_sha_change_refuses_comparison(self):
        """语料换了就必须拒绝比较，不能因为块数差在容差内就照比。

        只比对块数容差时，换语料会静默通过；指纹变了就一律拒绝。
        """
        fails = _hard(check_gate(_base(), _current(corpus_sha="deadbeef"), allow_drift=False))
        assert any("语料指纹变了" in f for f in fails)

    def test_corpus_sha_change_allowed_with_flag_but_warns(self):
        """逃生口仍然存在，但必须留下痕迹（而不是静默）。"""
        fails = check_gate(_base(), _current(corpus_sha="deadbeef"), allow_drift=True)
        assert _hard(fails) == []
        assert any(f.startswith("警告") and "语料指纹" in f for f in fails)

    def test_corpus_sha_mismatch_is_detected_even_when_chunk_count_is_close(self):
        """**这就是那条静默路径**：块数只差一点点（容差内），指纹仍然必须拦住它。"""
        cur = _current(corpus_sha="deadbeef", n_chunks=1002)  # +0.2%，远在 2% 容差内
        fails = _hard(check_gate(_base(), cur, allow_drift=False))
        assert any("语料指纹变了" in f for f in fails), "容差内的语料变化被静默放过了"

    def test_corpus_drift_refused_by_default(self):
        fails = _hard(check_gate(_base(), _current(n_chunks=2000), allow_drift=False))
        assert any("索引规模变化" in f for f in fails)

    def test_corpus_drift_allowed_with_flag_but_warns(self):
        fails = check_gate(_base(), _current(n_chunks=2000), allow_drift=True)
        assert _hard(fails) == []
        assert any(f.startswith("警告") for f in fails)

    def test_citation_violation_fails(self):
        fails = _hard(check_gate(_base(), _current(n_citation_violations=3), allow_drift=False))
        assert any("引用完整性" in f for f in fails)

    def test_schema_mismatch_refuses(self):
        fails = _hard(check_gate(_base(schema=99), _current(), allow_drift=False))
        assert any("schema" in f for f in fails)


def test_gold_sha_is_stable_and_content_sensitive():
    a = [GoldRow(id="1", q="q", kind="concept", must_contain={"d": "t"})]
    b = [GoldRow(id="1", q="q2", kind="concept", must_contain={"d": "t"})]
    assert gold_sha256(a) == gold_sha256(list(a))
    assert gold_sha256(a) != gold_sha256(b)


# --------------------------------------------------------------------- 端到端
def _eval_args(gold: Path, baseline: Path, **kw) -> Namespace:
    args = {
        "gold": str(gold),
        "baseline": str(baseline),
        "top_k": 5,
        "gate": False,
        "write_baseline": False,
        "allow_corpus_drift": False,
        "no_ablation": False,
        "json": False,
    }
    args.update(kw)
    return Namespace(**args)


def test_run_eval_end_to_end_writes_baseline(settings, store, embedder, tmp_path):
    _corpus_with_one_doc(settings)
    ingest(store, embedder, settings)

    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "g1",
                "kind": "concept",
                "q": "who enforces memory limits",
                "must_contain": {DOC: "kubelet enforces memory"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "base.json"

    rc = run_eval(_eval_args(gold, baseline, write_baseline=True), settings, store)
    assert rc == 0
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["schema"] == BASELINE_SCHEMA
    assert payload["n_gold"] == 1
    assert payload["metrics"]["hit"] == pytest.approx(1.0)
    assert payload["ablation"]["hybrid"]["overall"]["hit"] == pytest.approx(1.0)
    assert payload["n_citation_violations"] == 0


def test_run_eval_gate_passes_then_fails_on_regression(settings, store, embedder, tmp_path):
    """端到端验证闸门真的会红：抬高基线后当前分数必须被判为回归。

    标注集刻意含一条问不到的查询（top_k=1 时目标文档挤不进结果），实际 hit=0.5，
    否则当前分数已是 1.0，基线怎么抬都触发不了下降判定。
    """
    _corpus_with_two_docs(settings)
    ingest(store, embedder, settings)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "g1",
                "kind": "concept",
                "q": "who enforces memory limits",
                "must_contain": {DOC: "kubelet enforces memory"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "g2",
                "kind": "concept",
                "q": "network policy ingress egress label selector",
                "must_contain": {DOC: "kubelet enforces memory"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "base.json"
    args = {"top_k": 1}

    assert run_eval(_eval_args(gold, baseline, write_baseline=True, **args), settings, store) == 0
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["n_gold"] == 2
    assert payload["metrics"]["hit"] == pytest.approx(0.5)

    # 同一份标注集、同一个索引 -> 闸门应通过
    assert run_eval(_eval_args(gold, baseline, gate=True, **args), settings, store) == 0

    # 把基线抬高到当前达不到的水平 -> 必须判为回归（退出码 2）
    payload["metrics"]["hit"] = 1.0
    payload["metrics"]["mrr"] = 1.0
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    assert run_eval(_eval_args(gold, baseline, gate=True, **args), settings, store) == 2


def test_run_eval_refuses_when_gold_invalid(settings, store, embedder, tmp_path):
    """标注集自检失败时不应产出任何分数（退出码 3）。"""
    _corpus_with_one_doc(settings)
    ingest(store, embedder, settings)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "g1",
                "kind": "concept",
                "q": "q",
                "must_contain": {DOC: "phrase that does not exist"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_eval(_eval_args(gold, tmp_path / "b.json", write_baseline=True), settings, store) == 3


def test_run_eval_without_baseline_fails_gate(settings, store, embedder, tmp_path):
    _corpus_with_one_doc(settings)
    ingest(store, embedder, settings)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "id": "g1",
                "kind": "concept",
                "q": "who enforces memory limits",
                "must_contain": {DOC: "kubelet enforces memory"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_eval(_eval_args(gold, tmp_path / "missing.json", gate=True), settings, store) == 2


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=16)
    a = e.encode(["pod resource limits"])
    b = e.encode(["pod resource limits"])
    assert (a == b).all()
    assert float((a**2).sum()) == pytest.approx(1.0)


# ------------------------------------------------- `--json` 不得绕过闸门（回归测试）
class TestJsonDoesNotBypassGate:
    """`--json` 不得绕过 `--gate`：退出码必须逐位相同，否则自动化在看假绿灯。"""

    @staticmethod
    def _fixture(settings, store, embedder, tmp_path):
        _corpus_with_two_docs(settings)
        ingest(store, embedder, settings)
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "id": "g1",
                    "kind": "concept",
                    "q": "who enforces memory limits",
                    "must_contain": {DOC: "kubelet enforces memory"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "g2",
                    "kind": "concept",
                    "q": "network policy ingress egress label selector",
                    "must_contain": {DOC: "kubelet enforces memory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        baseline = tmp_path / "base.json"
        # top_k=1 + 一条问不到的查询 -> hit=0.5，给"抬高基线触发回归"留出空间
        assert run_eval(_eval_args(gold, baseline, write_baseline=True, top_k=1), settings, store) == 0
        return gold, baseline

    def test_gate_exit_code_matches_with_and_without_json(self, settings, store, embedder, tmp_path, capsys):
        """同一输入下，`--gate --json` 与 `--gate` 的退出码必须**逐位相同**。"""
        gold, baseline = self._fixture(settings, store, embedder, tmp_path)
        args = {"top_k": 1, "gate": True}

        # 1) 基线 == 当前 -> 两个变体都应通过（0）
        assert run_eval(_eval_args(gold, baseline, **args), settings, store) == 0
        capsys.readouterr()
        assert run_eval(_eval_args(gold, baseline, json=True, **args), settings, store) == 0

        # 2) 抬高基线到当前达不到的水平 -> 两个变体都必须判为回归（2）
        payload = json.loads(baseline.read_text(encoding="utf-8"))
        payload["metrics"]["hit"] = 1.0
        payload["metrics"]["mrr"] = 1.0
        baseline.write_text(json.dumps(payload), encoding="utf-8")

        rc_plain = run_eval(_eval_args(gold, baseline, **args), settings, store)
        capsys.readouterr()
        rc_json = run_eval(_eval_args(gold, baseline, json=True, **args), settings, store)

        assert rc_plain == 2, "不带 --json 时闸门应当判为不通过"
        assert rc_json == rc_plain, f"--json 改变了退出码（{rc_json} != {rc_plain}）——闸门被静默跳过"

    def test_json_output_is_pure_json_on_stdout_gate_noise_on_stderr(
        self, settings, store, embedder, tmp_path, capsys
    ):
        """`--json` 时 stdout 必须仍是**可解析的纯 JSON**：闸门旁白走 stderr。"""
        gold, baseline = self._fixture(settings, store, embedder, tmp_path)
        capsys.readouterr()  # 清掉 fixture 里那次 run_eval 的 stdout
        payload = json.loads(baseline.read_text(encoding="utf-8"))
        payload["metrics"]["hit"] = 1.0
        payload["metrics"]["mrr"] = 1.0
        baseline.write_text(json.dumps(payload), encoding="utf-8")

        rc = run_eval(_eval_args(gold, baseline, top_k=1, gate=True, json=True), settings, store)
        assert rc == 2
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)  # 解析失败即测试失败
        assert parsed["n_gold"] == 2
        assert parsed["metrics"]["hit"] == pytest.approx(0.5)
        # 闸门结论没被丢掉，只是在 stderr 上
        assert "闸门：不通过" in captured.err

    def test_write_baseline_with_json_still_writes_file(self, settings, store, embedder, tmp_path, capsys):
        """`--write-baseline --json`：JSON 只改输出格式，写盘动作不能被动掉。"""
        _corpus_with_one_doc(settings)
        ingest(store, embedder, settings)
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "id": "g1",
                    "kind": "concept",
                    "q": "who enforces memory limits",
                    "must_contain": {DOC: "kubelet enforces memory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        baseline = tmp_path / "base.json"

        rc = run_eval(_eval_args(gold, baseline, write_baseline=True, json=True), settings, store)
        assert rc == 0
        assert baseline.exists()
        assert json.loads(baseline.read_text(encoding="utf-8"))["n_gold"] == 1
        capsys.readouterr()


# ------------------------------------------- 人读路径的输出不得重复（回归测试）
class TestHumanReadableOutputIsNotDuplicated:
    """人读输出不得重复：表格与"未命中"列表在 stdout 各只出现一次。"""

    @staticmethod
    def _fixture(settings, store, embedder, tmp_path):
        _corpus_with_two_docs(settings)
        ingest(store, embedder, settings)
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "id": "g1",
                    "kind": "concept",
                    "q": "who enforces memory limits",
                    "must_contain": {DOC: "kubelet enforces memory"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "g2",
                    "kind": "concept",
                    "q": "network policy ingress egress label selector",
                    "must_contain": {DOC: "kubelet enforces memory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return gold

    def test_table_and_miss_list_appear_exactly_once(self, settings, store, embedder, tmp_path, capsys):
        gold = self._fixture(settings, store, embedder, tmp_path)

        # top_k=1 + 一条问不到的查询 -> 必然有未命中，两段输出都会出现
        rc = run_eval(_eval_args(gold, tmp_path / "b.json", top_k=1), settings, store)
        assert rc == 0

        out = capsys.readouterr().out
        assert out.count("hit@1") == 1, "表格被打印了多遍"
        assert out.count("未命中") == 1, "未命中列表被打印了多遍"

    def test_json_mode_emits_no_human_readable_output_on_stdout(
        self, settings, store, embedder, tmp_path, capsys
    ):
        """`--json` 时 stdout 上连着"表格/未命中"的旁白都不该有。"""
        gold = self._fixture(settings, store, embedder, tmp_path)

        rc = run_eval(_eval_args(gold, tmp_path / "b.json", top_k=1, json=True), settings, store)
        assert rc == 0

        captured = capsys.readouterr()
        assert json.loads(captured.out)["n_gold"] == 2  # 解析失败即回归
        assert "hit@1" not in captured.out
        assert "hit@1" not in captured.err
