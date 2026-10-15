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
| `department_scope.md` | `data/docs/科室导诊范围.md` | 1 份 | **科室导诊范围标准**：显式给出「症状 → 科室」判据，是 C2 金标的答案依据 |
| `phenomenon_to_symptom.json` | `data/knowledge/` | 300 个规范症状 | 口语现象 → 规范症状；C2 的**口语查询文本**来源。⚠️ 挖掘而来，含伪变体，见下 |
| `disease_to_department.json` | `data/knowledge/` | 8,819 条 | 疾病 → 科室（C2 用于交叉校验） |
| `manual_annotation.csv` | `data/raw/OpenCMKG/` | 853 行（552 条判 1） | 人工实体等价标注。**实测对本环节无用**：装的是疾病名等价（`妊娠合并糖尿病`↔`妊娠期糖尿病`），不含口语症状，给 C2 的背书数为 **0** |

## ⚠️ `phenomenon_to_symptom.json` 是**挖掘**来的，不能直接当同义词典

C2 金标用它作口语查询来源，但**必须过质量闸**才可用。实测见过的伪变体：

- `腰痛 -> 腿疼`（部位就不同）
- `头痛 -> 乳头疼`、`头痛 -> 全部痛`
- `口臭 -> 胃热`、`口臭 -> 喉咙有痰`
- `咳痰 -> 吐好多`、`腹痛 -> 一侧疼`
- `通痛`、`拉胡弦`、`啦肚子` 这类 OCR/输入噪声

所以 `scripts/build_triage_gold.py` 要过三道闸（部位冲突、起首总括/否定词、
单字规范症状不建、变体须对标准关键词也成立），并且**默认只信 `high` 档的科室补全**。
把这张表直接当同义词表用，会把假答案写进金标。

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

## 许可（**发布前必读**）

原型的知识库来自 **OpenCMKG**（上游标注"仅用于学术研究，不得用于商用"）。

本目录的文件**已入库**（`git ls-files data/eval/prototype` 可核对），所以：

- **推到公开仓库 = 分发**，不再只是"内部测试"。这句话是写给准备发布的人的，
  不是免责套话——本目录里确实有 OpenCMKG 的**派生数据**。
- 各类文件的性质不同，风险也不同：

  | 文件 | 是什么 | 风险 |
  |---|---|---|
  | `disease_to_department.json` | **OpenCMKG 关系的直接导出**（8,819 条疾病→科室） | **最直接**：这就是那把三元组导出来的 |
  | `phenomenon_to_symptom.json` | 口语现象 → 规范症状（300 个规范症状及变体） | 派生自同一套实体 |
  | `patients.json` | 原型用 LLM 生成的 500 例**病例文本**（含金标科室/疾病方向） | 文本是生成的，但疾病名来自同一知识体系 |
  | `manual_annotation.csv` | 853 行人工实体等价标注 | 派生 |
  | `rag_gold.jsonl` / `symptom_status_gold.jsonl` / `noise_ood_gold.jsonl` | 原型自建的评测用例 | 主要是原型**自己写的**用例，风险最低 |
  | `department_scope.md` / `departments.json` | 科室导诊范围说明 + 科室词表 | 原型自己整理的 |

- **发布者要做的选择（二选一，别不做）**：
  1. **保留**：在仓库的 NOTICE / README 里显式写清"本仓库包含 OpenCMKG 派生数据，
     仅用于研究与评测"，并自行确认上游许可是否允许再分发；
  2. **移出**：把上面风险最高的几项加进 `.gitignore` 并写清获取/重建方式
     （`scripts/build_triage_gold.py` 与 `scripts/eval_prototype_patients.py` 需要它们，
     移出后这两个脚本要另附数据准备说明）。

`data/clinic/`（知识卡片）与 `data/raw/`（原始语料）**都没有入库**，这一点没变。

