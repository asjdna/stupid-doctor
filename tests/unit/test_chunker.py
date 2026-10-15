"""切块器契约测试：每条测试钉住一条会直接影响检索质量的切块规则。

覆盖 front matter 剥离、shortcode 与 HTML 清洗、代码块与 YAML 清单不被切断、
面包屑与上下文头、稳定的 `chunk_id` 与内容指纹。语料是 Hugo markdown。
"""

from kubedocs.chunker import (
    chunk_document,
    clean_body,
    parse_front_matter,
    source_url_for,
    split_sections,
)

FRONT_MATTER_DOC = """---
title: 使用 cgroups 限制资源
weight: 20
description: 一段会被剥离的元数据
---

# 使用 cgroups 限制资源

正文第一段，讲的是怎么给 Pod 设 CPU 上限。

## 配置示例

下面是一段清单：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: limited
spec:
  containers:
    - name: app
      resources:
        limits:
          cpu: "500m"
```

## 注意事项

{{< note >}}
设置了 limits 之后，容器可能被 OOMKilled。
{{< /note >}}
"""


def test_front_matter_剥离且用作标题():
    meta, body = parse_front_matter(FRONT_MATTER_DOC)
    assert meta["title"] == "使用 cgroups 限制资源"
    assert meta["weight"] == "20"
    assert "description" not in body  # 元数据不得出现在正文里
    assert body.lstrip().startswith("# 使用 cgroups")


def test_shortcode_标记被剥掉但正文保留():
    _, body = parse_front_matter(FRONT_MATTER_DOC)
    cleaned = clean_body(body)
    assert "{{< note >}}" not in cleaned
    assert "{{< /note >}}" not in cleaned
    assert "OOMKilled" in cleaned, "提示框里的正文是有效信息，不能一起丢掉"


def test_html_注释被删除():
    assert "隐藏内容" not in clean_body("正常文字\n<!-- 隐藏内容 -->\n结尾")


def test_面包屑按层级维护():
    body = "## A\n\n内容 a\n\n### A1\n\n内容 a1\n\n## B\n\n内容 b\n"
    sections = split_sections(body)
    crumbs = [c for c, _ in sections]
    assert crumbs == [["A"], ["A", "A1"], ["B"]], f"层级回退不对：{crumbs}"


def test_代码围栏内的井号不算标题():
    body = "## 真标题\n\n```bash\n# 这是注释不是标题\necho hi\n```\n\n正文\n"
    sections = split_sections(body)
    assert len(sections) == 1, f"围栏内的 # 被误判为标题：{[c for c, _ in sections]}"


def test_yaml清单不被切断():
    """超长代码块必须整体保留——半截 kubectl/YAML 是不可用的。"""
    lines = "\n".join(f"  - item{i}: value{i}" for i in range(120))
    doc = f"## 清单\n\n```yaml\n{lines}\n```\n\n## 之后\n\n结尾说明文字。\n"
    chunks = chunk_document("concepts/big.md", doc, max_chars=400, overlap=50, min_chars=10)
    code_chunks = [c for c in chunks if "```yaml" in c.text]
    assert code_chunks, "应当存在包含代码块的块"
    joined = "\n".join(c.text for c in code_chunks)
    assert joined.count("```") % 2 == 0 or joined.count("```") >= 2, "代码块围栏不配对"
    assert "item0" in joined and "item119" in joined, "代码块首尾内容都在同一块里"


def test_超长正文按段落切且带重叠():
    para = "这是一段用于测量长度的正文。" * 8
    doc = "## 段\n\n" + "\n\n".join(para for _ in range(12))
    chunks = chunk_document("concepts/long.md", doc, max_chars=600, overlap=100, min_chars=10)
    assert len(chunks) >= 2, "超长正文没有被切开"
    assert all(len(c.text) <= 600 + 200 for c in chunks), "块长度失控"
    assert any(chunks[i].text[-50:] in chunks[i + 1].text for i in range(len(chunks) - 1)), (
        "块之间没有重叠，跨块上下文会丢"
    )


def test_过短碎片并入前一块():
    doc = "## 章\n\n" + ("正文" * 200) + "\n\n### 小节\n\n短\n"
    chunks = chunk_document("concepts/m.md", doc, max_chars=2000, min_chars=50)
    assert len(chunks) == 1, "过短小节应被并入前一块，而不是单独成块污染检索"


def test_chunk_id稳定且内容指纹随内容变化():
    a = chunk_document("concepts/x.md", "## 标题\n\n内容 A 足够长" * 3)
    b = chunk_document("concepts/x.md", "## 标题\n\n内容 A 足够长" * 3)
    c = chunk_document("concepts/x.md", "## 标题\n\n内容 B 足够长" * 3)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b], "同内容 id 必须稳定（供 upsert）"
    assert [x.content_hash for x in a] != [x.content_hash for x in c], "内容变了指纹必须变"


def test_source_url可点击():
    assert (
        source_url_for("concepts/architecture/cgroups.md", "https://kubernetes.io/docs/")
        == "https://kubernetes.io/docs/concepts/architecture/cgroups/"
    )
    assert (
        source_url_for("concepts/workloads/_index.md", "https://kubernetes.io/docs/")
        == "https://kubernetes.io/docs/concepts/workloads/"
    )
    assert source_url_for("_index.md", "https://kubernetes.io/docs/") == "https://kubernetes.io/docs/"


def test_空文档返回空列表():
    assert chunk_document("a.md", "") == []
    assert chunk_document("a.md", "---\ntitle: 只有元数据\n---\n") == []


def test_标题优先级_front_matter_优先于_h1():
    doc = "---\ntitle: 元数据标题\n---\n\n# h1 标题\n\n正文内容足够长一些。\n"
    chunks = chunk_document("concepts/t.md", doc)
    assert chunks[0].title == "元数据标题"


def test_无标题文档回退为文件名():
    doc = "这里没有任何标题，只有一段足够长的正文内容用于测试回退逻辑。" * 3
    chunks = chunk_document("concepts/some-topic.md", doc)
    assert chunks[0].title == "Some Topic", chunks[0].title


def test_真实文档端到端不丢正文():
    """整篇解析后，关键内容点必须都能在某个块里找到。"""
    chunks = chunk_document("concepts/cgroups.md", FRONT_MATTER_DOC)
    blob = "\n".join(c.text for c in chunks)
    for must in ("CPU 上限", "apiVersion: v1", "OOMKilled", "注意事项"):
        assert must in blob, f"正文点了 {must!r} 在切块后丢失"
    assert "description" not in blob
    assert all(c.source_url == "https://kubernetes.io/docs/concepts/cgroups/" for c in chunks)


# HTML 清洗：K8s API reference 用 HTML 表格排版，占真实语料近六成块；
# 标签不剥离就会变成检索词，把 BM25 与嵌入都带偏。
def test_HTML_标签剥离但保留文字():
    from kubedocs.chunker import strip_html

    raw = "<p>参数 <code>pretty</code> 用于美化输出。</p>"
    out = strip_html(raw)
    assert "<" not in out and ">" not in out
    assert "pretty" in out and "美化输出" in out, out


def test_HTML_表格变成可读行文本():
    from kubedocs.chunker import strip_html

    raw = (
        "<table><thead><tr><th>Name</th><th>Type</th></tr></thead>"
        "<tbody><tr><td>pretty</td><td>string</td></tr></tbody></table>"
    )
    out = strip_html(raw)
    assert "Name" in out and "pretty" in out
    assert "|" in out, f"单元格应变成可读分隔而不是粘在一起：{out!r}"
    assert "<td>" not in out and "<tr>" not in out


def test_HTML_实体被解码():
    from kubedocs.chunker import strip_html

    out = strip_html("A &lt; B &amp;&amp; C &nbsp;D")
    assert "A < B" in out and "&&" in out


def test_HTML_script_style_整块删除():
    from kubedocs.chunker import strip_html

    out = strip_html("正文<script>var x = 1;</script>结尾<style>p{color:red}</style>")
    assert "var x" not in out and "color:red" not in out
    assert "正文" in out and "结尾" in out


def test_HTML_不碰代码围栏内的尖括号():
    """`if [[ $x < 5 ]]` 是合法代码，剥离 HTML 时绝不能动它。"""
    from kubedocs.chunker import strip_html

    raw = "```bash\nif [[ $x < 5 ]]; then echo '<ok>'; fi\n```\n"
    out = strip_html(raw)
    assert "$x < 5" in out and "<ok>" in out, out


def test_重叠对齐到安全边界不切词():
    """按字符硬切会切出从词中间开始的块，跨块上下文因此失真。"""
    from kubedocs.chunker import _safe_overlap_tail

    block = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    tail = _safe_overlap_tail(block, 20)
    assert tail, "应当能对齐到一个词边界"
    assert block.endswith(tail)
    assert block[-len(tail) - 1] in " \n", f"重叠起点落在词中间：{tail!r}"
    assert _safe_overlap_tail(block, 0) == ""
    assert _safe_overlap_tail("", 10) == ""
    # 一整块没有空白时放弃重叠（宁可少上下文，也不给半截内容）
    assert _safe_overlap_tail("a" * 50, 10) == ""


# 上下文头（add_context）：不加时模板化小节会跨文档逐字重复（约占三成块），
# 加上 `[文档 › 小节]` 头后内容指纹才区分得开。
def test_上下文头出现在块首():
    doc = (
        "---\ntitle: DeviceClass\n---\n\n## Operations\n\n### Query Parameters\n\n"
        + ("说明文字。" * 30)
        + "\n"
    )
    chunks = chunk_document("reference/device-class-v1.md", doc)
    # `## Operations` 自身成一块（它下面没有正文），完整面包屑在子节那块
    heads = [c.text.splitlines()[0] for c in chunks]
    assert "[DeviceClass › Operations]" in heads, heads
    assert "[DeviceClass › Operations › Query Parameters]" in heads, heads


def test_上下文头解决跨文档同形块():
    """同一个小节在两份文档里必须**不同形**，否则会产生大量重复块。"""
    body = "## Operations\n\n### Query Parameters\n\n" + ("| Name | Type |\n| pretty | string |\n" * 20)
    a = chunk_document("reference/res-a.md", f"---\ntitle: ResourceA\n---\n\n{body}")
    b = chunk_document("reference/res-b.md", f"---\ntitle: ResourceB\n---\n\n{body}")
    assert a[0].content_hash != b[0].content_hash, "同名小节跨文档仍同形 —— 上下文头没生效"
    assert "ResourceA" in a[0].text and "ResourceB" in b[0].text


def test_标题不重复出现在上下文头里():
    """面包屑首项已等于文档标题时不得重复（`[T › T › X]` 是噪声）。"""
    doc = "---\ntitle: 我的标题\n---\n\n## 我的标题\n\n### 小节\n\n" + ("内容。" * 40)
    chunks = chunk_document("concepts/t.md", doc)
    head = chunks[0].text.splitlines()[0]
    assert head.count("我的标题") == 1, head


def test_可关闭上下文头():
    doc = "---\ntitle: T\n---\n\n## A\n\n" + ("内容。" * 40)
    chunks = chunk_document("x.md", doc, add_context=False)
    assert not chunks[0].text.startswith("[T"), chunks[0].text[:40]
