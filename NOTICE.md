# 第三方数据说明

本仓库的**代码**以 MIT 许可发布（见 [`LICENSE`](LICENSE)）。
`LICENSE` **不覆盖**下面列出的第三方数据，它们的许可以上游为准。

## 仓库携带的数据

`data/eval/prototype/` 下的文件来自一个已冻结的医疗 agent 原型项目，
其知识体系源自 **OpenCMKG**，上游标注：

> 仅用于学术研究，不得用于商用。如认为侵犯了您的版权，请联系删除

| 文件 | 内容 |
|---|---|
| `patients.json` | 500 例带标注的病例文本（含期望科室与疾病方向） |
| `disease_to_department.json` | 疾病 → 科室关系（源自 OpenCMKG） |
| `phenomenon_to_symptom.json` | 口语表述 → 规范症状 |
| `manual_annotation.csv` | 人工实体等价标注 |
| `department_scope.md` | 科室导诊范围说明 |

这些文件在本仓库中**只用于研究与评测**（分诊与急诊的评测用例、派生科室补全表），
不参与任何运行时路径，也不随任何产品分发。

使用或再分发前请自行确认上游许可。若上游权利人提出异议，删除该目录即可——
受影响的只有两个离线评测脚本（`scripts/build_triage_gold.py`、
`scripts/eval_prototype_patients.py`），核心功能与测试不受影响。

## 未携带的数据

| 目录 | 说明 |
|---|---|
| `data/clinic/` | 医疗知识卡片。体积较大且受上游许可约束，由使用者按需获取或重建 |
| `data/raw/` | 原始语料（Kubernetes 官方文档、OpenCMKG 三元组等） |

获取方式见 README 的「知识依据」一节。核对是否携带：`git ls-files data/clinic data/raw`。

## 依赖与模型

Python 依赖各自的许可记录在 `pyproject.toml` 中。
嵌入模型（`BAAI/bge-small-*`）与重排模型（`BAAI/bge-reranker-base`）在运行时下载，
**不随仓库分发**，其许可以上游模型页为准。
