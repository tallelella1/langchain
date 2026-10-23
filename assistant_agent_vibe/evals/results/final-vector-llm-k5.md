# 固定评估结果：fixed / vector / rerank=llm

- 时间: 2026-10-23T17:14:11
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
| 平均召回原文长度 | 876.5 |
| 平均格式化上下文长度 | 1272.1 |
| 平均检索延迟(ms) | 14069.81 |
| 检索错误 | 0 |
| 模型调用次数 | 12 |
| 检索工具调用次数 | 0 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 704 | 20613.21 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p004；knowledge-8a0b667e89#p011 |
| Q02 | 3/3 | 1/1 | 584 | 15421.79 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p013 |
| Q03 | 5/5 | 2/2 | 995 | 12126.55 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p020 |
| Q04 | 2/2 | 1/1 | 767 | 12299.36 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029 |
| Q05 | 2/2 | 1/1 | 767 | 12343.61 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029 |
| Q06 | 1/1 | 1/1 | 890 | 12068.69 | knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029 |
| Q07 | 3/3 | 1/1 | 868 | 7850.13 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p015 |
| Q08 | 2/2 | 1/1 | 954 | 17111.36 | knowledge-8a0b667e89#p010；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p008 |
| Q09 | 2/2 | 1/1 | 791 | 15371.55 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p006 |
| Q10 | 1/1 | 1/1 | 1106 | 14113.1 | knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p028；knowledge-8a0b667e89#p027 |
| Q11 | 0/0 | 0/0 | 1097 | 15829.29 | knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p028 |
| Q12 | 0/0 | 0/0 | 995 | 13689.13 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p025 |
