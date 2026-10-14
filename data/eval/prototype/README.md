# 从原型抽取的测试用例

## 出处

全部从 `E:\Users\86189\medical-agent`（原型，**已冻结不再开发**）复制而来，
抽取日期 2026-10-14。**逐字复制，未做任何修改**——改了就不再是原型里那份，
也就失去了"对照"的意义。

| 文件 | 原型路径 | 条数 | 用途 |
|---|---|---|---|
| `symptom_status_gold.jsonl` | `data/eval/` | 35 | 症状实体的**三维判定**（status / experiencer / temporality）+ 难例标签（`fake-negation`、`hypothetical`、`scope-reset`…） |
| `noise_ood_gold.jsonl` | `data/eval/` | 60 | 非医疗噪音文本（`label: non_medical`）：不得触发急诊或医学建议 |
| `rag_gold.jsonl` | `data/eval/` | 121（meta 1 + 三类各 40） | 检索 / 卡内容 / 卡类型。**⚠️ 见下方"不可迁移"一节** |
| `departments.json` | `data/knowledge/` | 54 个科室名 | 分诊的科室词表（含"减肥""其他科室"这类真实脏数据） |

## 为什么 `symptom_status_gold` 最有价值

它测的是**"说不清"的那些情况**，而这些与语料无关、可以直接迁移到本项目：

| 标签 | 条数 | 意思是 |
|---|---|---|
| `negation` | 9 | 否定（"没发烧"） |
| `experiencer` | 6 | 主体不是患者本人（"我邻居胸痛"） |
| `temporality` | 6 | 时间性（"上周疼过，现在不疼"） |
| `hypothetical` | 3 | 假设（"如果胸痛怎么办"） |
| `fake-negation` | 1 | 假否定（"无痛性"）——不能当否定 |
| `uncertain` | 4 | 不确定（"不能排除"） |
| `scope-reset` / `comma-scope` | 3 | 否定作用域（"没有胸痛，就是胸闷"——后半句不能被前半句否定掉） |

**最有价值的一条**：`ss-003 「没有胸痛，就是有点胸闷」`。
把这类句子判成急诊，就是**假阳性**；而假阳性会让人不再相信红线提示。

## ⚠️ 不可迁移的部分：`rag_gold.jsonl`

原型自己的 meta 行就写明了：

- `human_verified: false`；
- `disclaimer`：**"本文件是冻结基线：用于检测语料变更，不判定医学正确性"**；
- `expect_titles` 指的是**原型那套 OpenCMKG 派生卡片**的标题
  （如 `"丘脑出血"` → `expect_titles: ["丘脑出血"]`），而本项目的 clinic 语料是
  另一份卡片（`data/clinic/cards.json`，31,592 张）。

所以它在本项目里**只能当"检索健壮性"检查**（这些查询能不能召回**任何**东西），
**不能**当作检索质量指标，也不该进任何闸门的基线。

## 许可

原型的知识库来自 **OpenCMKG**（上游标注"仅用于学术研究"）。本目录的文件
**只用于本项目的内部测试**，不随产品分发；与本项目 `data/clinic/` 的处理方式一致
（见 `SECURITY.md` 第 6 节关于数据合规的说明）。
