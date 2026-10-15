# NOTICE —— 第三方数据与许可边界

本仓库的**软件代码**以 MIT 许可发布（见 `LICENSE`）。
**但 `LICENSE` 不覆盖下面列出的第三方派生数据**，它们的许可以上游为准。

## 1. 明确**没有**包含的内容

| 目录 | 状态 |
|---|---|
| `data/clinic/` | **未入库**（`.gitignore` 排除）。医疗知识卡片，上游标注"仅用于学术研究" |
| `data/raw/` | **未入库**。OpenCMKG 原始三元组等语料 |

核对方式：`git ls-files data/clinic data/raw` 输出为空。

## 2. 明确**包含**的第三方派生数据

`data/eval/prototype/` 下的文件来自原型项目（`medical-agent`，已冻结），
其知识库来自 **OpenCMKG**，上游标注：

> 仅用于学术研究，不得用于商用。如认为侵犯了您的版权，请联系删除

| 文件 | 性质 |
|---|---|
| `disease_to_department.json` | **OpenCMKG 关系的直接导出**（8,819 条 疾病→科室） |
| `phenomenon_to_symptom.json` | 口语现象 → 规范症状（300 个规范症状及变体） |
| `manual_annotation.csv` | 853 行人工实体等价标注 |
| `patients.json` | 原型用 LLM 生成的 500 例病例文本（含金标科室/疾病方向） |
| `rag_gold.jsonl`、`symptom_status_gold.jsonl`、`noise_ood_gold.jsonl` | 原型自建的评测用例 |
| `department_scope.md`、`departments.json` | 科室导诊范围说明与科室词表 |

**这些文件在本仓库中只用于研究与评测**（分诊金标、检索基线、否定/主体/时间性判定回归），
不参与任何产品运行时路径，也不随任何产品分发。

**使用/再分发前请自行确认上游许可。** 若上游权利人提出异议，删除本目录即可 ——
受影响的只有两个离线评测脚本（`scripts/build_triage_gold.py`、
`scripts/eval_prototype_patients.py`），核心功能与测试不受影响。

## 3. 依赖与模型

Python 依赖各自的许可见 `pyproject.toml` 与 `uv.lock`。
嵌入模型（`BAAI/bge-small-*`）与重排模型（`BAAI/bge-reranker-base`）在运行时从
Hugging Face 下载，**不随本仓库分发**，其许可以上游模型页为准。
