# 临床治理

这份文档回答一个具体问题：**改了哪条安全规则、谁改的、为什么、怎么证明它还有效**。

系统对患者的安全不是靠某一次评测分数保证的，而是靠三层防护（语义理解 / 主动问诊 /
确定性安全门，见 [`ARCHITECTURE.md`](ARCHITECTURE.md)）加上这里的治理机制。
治理之所以要落到文件与脚本上，是因为"我们会小心改"这种承诺没有任何可检查的形式。

---

## 一、安全资产清单（`data/governance/safety_assets.json`）

被认定为**安全资产**的东西，改动都要留痕（来源 / 版本 / 生效日期 / 负责人 / 理由）：

| 资产 | 类别 | 它为什么算安全资产 |
|---|---|---|
| `src/kubedocs/clinical/prompts/system.md` | `prompt` | 指令层：多角度扫描、不确定性交代、追问义务都写在里面，改一个字的语气都会改变判断行为 |
| `src/kubedocs/domains/clinic/redline.py` | `rule_asset` | 危险信号关键词 + 三维门控：它是**喂给模型的线索**，决定模型能看到哪些词汇层面的信号 |
| `src/kubedocs/domains/clinic/data/symptom_dict.py` | `rule_asset` | 口语主诉归一：患者怎么说与词表认不认，直接决定面板里给到模型什么 |
| `src/kubedocs/domains/clinic/data/departments.py` | `rule_asset` | 可挂号科室清单（"科室只能从这里选"的依据）与挂号费（金额校验的唯一真相源） |
| `data/eval/clinic_gold.jsonl` | `goldset` | 评测的尺子本身；换尺子后与历史分数不可比，所以要留痕 |

清单里每条资产都带 **sha256**。改动被钉住的资产而不更新清单，CI 会直接失败——
这就是变更控制：不是要求"改之前先审批"，而是要求**改了就有人能查出来**。

**只钉内容、不钉代码实现**：实现文件（`verify.py` / `safety.py` / `pipeline.py`）不在清单的
哈希里。它们本来就会被持续重构，钉住它们只会让每次重构都要改清单，最终没人再看清单。
代码侧的保障是另一条路——不变量注册表 + 测试覆盖（见下）。

---

## 二、确定性安全不变量（代码里的那三条）

`kubedocs/clinical/safety.py` 的 `INVARIANTS` 是**唯一来源**。每条不变量声明：
id、标题、理由、触发后的处置动作、实现位置、以及它在哪几条入口上成立。

| id | 一句话 | 触发后的处置 | 适用范围 |
|---|---|---|---|
| `no_medical_advice_without_evidence` | 没有检索到资料就不给医学内容 | 模型那段散文不上屏，改由服务端拼只含事实的回答 | `/consult` |
| `no_fabricated_booking_action` | 不得声称服务端没执行过的挂号动作 / 凭据号 | 同上（患者永远看不到被拦下的那句话） | `/consult` + `/chat` |
| `emergency_judgment_requires_directive` | 判出急症必须给出可执行的急救动作 | **补**一句固定指引（唯一"加内容"的不变量） | `/consult` |

增减一条不变量，**必须同时**改三处，否则 `scripts/check_governance.py` 会失败：

1. `safety.py` 的 `INVARIANTS`（代码）；
2. 清单里的 `invariants` 条目（治理记录：负责人 / 生效日期 / 理由 / 验证出处）；
3. `tests/` 里至少一处引用它的 id（**没有测试覆盖的不变量等于没人守**）。

`/chat` 的散文路径拿不到结构化判断，所以第三条在那里不成立——这个差别写在注册表的
`scope` 里，而不是省略不提。

---

## 三、运行期审计（`data/index/safety_audit.jsonl`）

每次不变量被触发，服务端追加一行 JSONL：

```json
{"ts": "2026-10-16T12:00:00+00:00", "domain": "clinic",
 "invariant": "emergency_judgment_requires_directive", "action": "append_directive",
 "detail": "急症判断（urgency=emergency 或 critical/likely 的危险信号）但正文没有可执行的急救动作，已补固定指引",
 "trace_id": "8f3c…"}
```

三条纪律：

1. **不记患者原话**。临床文本已经落在 `clinical` schema（有自己的访问控制），审计日志
   再抄一份只会扩大泄露面。`trace_id` 是 `encounter_id`，需要细节时按它去临床库取。
2. **写失败不影响这一轮应答**（与落库同一条纪律），但会记 warning——静默失败等于审计是假的。
3. **没有触发就不建文件**：空文件会被读成"跑过、没问题"，而实际是"从没触发过"。
   读回入口是 `GET /consult/status` 的 `safety_audit_recent`。

---

## 四、评测的角色：记录，不是门限

指标（危险信号召回、误报率、分诊命中率）来自**自建的小样本用例集**：它们能说明趋势与
回归，不能承担"这条回答安全"的责任。把它们写成门限只有两种结局——为了让闸门变绿而放宽
尺子，或者一个永远红着、没人看的闸门。

所以 `scripts/eval_clinic.py --gate` 现在只对两类事失败：

1. **这次运行不能作为证据**：一条可评测用例都没有，或有题目没拿到模型判断；
2. **触发了确定性安全不变量**（上表那三条）。

指标与参考目标（100% / ≤20% / ≥60%）照旧打印并进档案，由人看趋势；`--archive` 把整份
报告写成带时间戳的治理档案（`data/eval/clinic_archive/`），永不覆盖。

**校准**（`evaluation/calibration.py`）同样只报告：它把"模型自报的置信度"与"实际判对率"
摆在一起，给出 Brier 分数与可靠性分箱。`confidence` 这个字段本身也受行为约束
（低于 0.6 必须交代 `information_gaps`），但**没有任何阈值卡它**——样本量小到谈不了校准
结论时，报表会自己写明这一点。

---

## 五、已知缺口（写在清单里，不藏着）

- **尚无具名临床审阅人**：所有安全资产目前只经过代码评审，**不能声称做过临床审阅**。
- 关键词表与症状词表的来源是仓库内人工整理，没有可追溯的外部标准版本号。
- 校准样本量远小于能谈校准结论的量级，只用于看趋势。
- 前两条不变量的验证靠单测与 `/consult` 运行期执行；"判了急症必须给动作"这条在评测里也
  有覆盖，但评测那一臂不检索，所以"无依据给医学内容"那条**不在评测里验**（见清单的
  `verified_by`）。

新增缺口时**写进清单的 `governance_gaps`**，而不是留白：留白会被读成"没有问题"。

---

## 六、怎么跑

```bash
python scripts/check_governance.py                  # 门禁：资产哈希 / 不变量对应 / 测试覆盖 / 验证出处
python scripts/check_governance.py --update-hashes  # 改完资产后刷哈希（版本号与理由仍要人写）
python scripts/eval_clinic.py --arm both --archive data/eval/clinic_archive
```

`check_governance.py` 在 `scripts/ci_local.py` 的 quality job 里跑，与单元测试同级：
**它是唯一按内容卡 CI 的那道门**。
