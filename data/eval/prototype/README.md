# 评测用例的出处

本目录是从一个已冻结的医疗 agent 原型项目复制来的评测用例与对照数据，
**逐字复制，未做修改**。

## 文件

| 文件 | 用途 |
|---|---|
| `patients.json` | 500 例带标注病例（严重度 × 语言风格分层，含期望科室与疾病方向），用于分诊与急诊评测 |
| `department_scope.md` | 科室导诊范围标准：症状 → 科室的判据 |
| `phenomenon_to_symptom.json` | 口语表述 → 规范症状 |
| `disease_to_department.json` | 疾病 → 科室关系，用于交叉校验 |
| `manual_annotation.csv` | 人工实体等价标注 |

## 使用方式

```bash
python scripts/eval_prototype_patients.py    # 分诊（离线、免费）
```

## 许可

本目录包含第三方派生数据，**不在 [`LICENSE`](../../../LICENSE) 的覆盖范围内**，
只用于研究与评测。完整清单与边界见 [`NOTICE.md`](../../../NOTICE.md)。
