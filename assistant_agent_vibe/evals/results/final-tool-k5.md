# 固定评估结果：tool / vector / rerank=none

- 时间: 2026-10-23T17:33:03
- Top-K: 5；父块补全: True
- Embedding: text-embedding-v4
- 对话模型: qwen3.8-max
- 检索参数: `{"rerank_strategy": "none"}`

## 汇总

| 指标 | 值 |
| --- | --- |
| 用例数 | 12 |
| 事实召回率 | 1.0000 |
| 全部事实命中用例 | 10/10 |
| 支持段落召回率 | 0.9167 |
| 平均召回原文长度 | 1159.8 |
| 平均格式化上下文长度 | 1159.8 |
| 平均检索延迟(ms) | 20783.26 |
| 检索错误 | 0 |
| 模型调用次数 | 34 |
| 检索工具调用次数 | 22 |
| 答案事实召回率 | 1.0000 |
| 拒答行为通过 | 2/2 |
| 答案错误 | 0 |
| 平均回答延迟(ms) | 20783.26 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 959 | 17854.55 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p011；knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p004；knowledge-8a0b667e89#p014 |
| Q02 | 3/3 | 1/1 | 1313 | 13229.26 | knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p011；knowledge-8a0b667e89#p012；knowledge-8a0b667e89#p017 |
| Q03 | 5/5 | 1/2 | 976 | 24688.59 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p021 |
| Q04 | 2/2 | 1/1 | 903 | 24533.97 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p020 |
| Q05 | 2/2 | 1/1 | 1286 | 29793.72 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p008；knowledge-8a0b667e89#p020 |
| Q06 | 1/1 | 1/1 | 1154 | 18962.61 | knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p023；knowledge-8a0b667e89#p024 |
| Q07 | 3/3 | 1/1 | 1025 | 14040.67 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p016 |
| Q08 | 2/2 | 1/1 | 954 | 21976.92 | knowledge-8a0b667e89#p008；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p010 |
| Q09 | 2/2 | 1/1 | 1290 | 14798.13 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p005；knowledge-8a0b667e89#p004 |
| Q10 | 1/1 | 1/1 | 1659 | 22724.24 | knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p020 |
| Q11 | 0/0 | 0/0 | 1220 | 23120.43 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p028 |
| Q12 | 0/0 | 0/0 | 1178 | 23676.01 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p024 |
