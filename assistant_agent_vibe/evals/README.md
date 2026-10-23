# 固定评估集

Q01–Q12 来自 `docs/02-implementation-steps.md`，用于比较检索与回答改动是否真的有效。
结果保存在 `results/`，每次运行都会生成 JSON 与 Markdown。

## 命令

```powershell
# 只评估检索（调用查询 Embedding，不调用对话模型）
python -m evals.run_eval --strategy vector --k 5

# 混合检索
python -m evals.run_eval --strategy hybrid --k 5

# LLM 重排（每题多一次对话模型调用）
python -m evals.run_eval --strategy vector --rerank llm --k 5

# 固定检索 + 答案
python -m evals.run_eval --strategy vector --answer --k 5

# 工具型 Agent（模型自主检索，调用次数与延迟会变化）
python -m evals.run_eval --agent tool --k 5

# 只跑指定题
python -m evals.run_eval --ids Q05,Q06

# 比较两份结果，不调用服务
python -m evals.run_eval --compare results/a.json results/b.json
```

## 指标

- 事实召回率：期望事实在召回文本/答案中的子串比例；
- 支持段落召回：命中父块的 `chapter / section` 是否覆盖标注段落；
- 上下文长度：原始命中文本与格式化 Prompt 分别记录；
- 延迟、模型调用次数、工具调用次数；
- Q11/Q12 拒答行为单独判定。

子串检查是启发式，措辞不同可能被计为未命中；关键事实和边界条件仍需人工核对。
新增问题应进入下一版评估集，不要只围绕已见题调参。
