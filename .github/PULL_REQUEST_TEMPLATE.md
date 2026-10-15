## 改了什么

<!-- 一句话说明改动内容。 -->

## 为什么

<!-- 解决的问题，或要达成的目标。 -->

## 怎么验证

<!-- 贴出可复现的命令与结果，例如： -->

```
python scripts/ci_local.py
python -m pytest tests/unit -q
```

## 检查清单

- [ ] `python -m pytest tests/unit -q` 通过
- [ ] `python -m ruff check .` 与 `python -m ruff format --check .` 通过
- [ ] `python -m mypy src scripts` 通过
- [ ] `python scripts/check_doc_refs.py` 与 `python scripts/check_deps.py` 通过
- [ ] 行为变更已同步更新对应的文档（README / `docs/`）
- [ ] 涉及输出契约或提示词时，已说明对评测基线的影响
