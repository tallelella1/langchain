# 固定评估结果：fixed / hybrid / rerank=llm

- 时间: 2026-10-23T17:42:09
- Top-K: 5；父块补全: True
- Embedding: text-embedding-v4
- 对话模型: 未调用
- 检索参数: `{"rerank_strategy": "llm"}`

## 汇总

| 指标 | 值 |
| --- | --- |
| 用例数 | 12 |
| 事实召回率 | 1.0000 |
| 全部事实命中用例 | 10/10 |
| 支持段落召回率 | 1.0000 |
| 平均召回原文长度 | 824.6 |
| 平均格式化上下文长度 | 1209.0 |
| 平均检索延迟(ms) | 13579.25 |
| 检索错误 | 0 |
| 模型调用次数 | 12 |
| 检索工具调用次数 | 0 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 859 | 18067.46 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p004；knowledge-8a0b667e89#p011；knowledge-8a0b667e89#p013 |
| Q02 | 3/3 | 1/1 | 584 | 5100.89 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p013 |
| Q03 | 5/5 | 2/2 | 995 | 13662.03 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p020 |
| Q04 | 2/2 | 1/1 | 337 | 21067.6 | knowledge-8a0b667e89#p019 |
| Q05 | 2/2 | 1/1 | 767 | 10257.66 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029 |
| Q06 | 1/1 | 1/1 | 890 | 12141.41 | knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029 |
| Q07 | 3/3 | 1/1 | 868 | 13036.98 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p015 |
| Q08 | 2/2 | 1/1 | 571 | 14031.65 | knowledge-8a0b667e89#p010；knowledge-8a0b667e89#p029 |
| Q09 | 2/2 | 1/1 | 791 | 11672.11 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p006 |
| Q10 | 1/1 | 1/1 | 1141 | 10927.99 | knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p026；knowledge-8a0b667e89#p009 |
| Q11 | 0/0 | 0/0 | 1097 | 19572.67 | knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p028 |
| Q12 | 0/0 | 0/0 | 995 | 13412.61 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p025 |
