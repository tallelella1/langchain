# 固定评估结果：fixed / hybrid / rerank=none

- 时间: 2026-10-23T17:10:02
- Top-K: 5；父块补全: True
- Embedding: text-embedding-v4
- 对话模型: 未调用
- 检索参数: `{"rerank_strategy": "none"}`

## 汇总

| 指标 | 值 |
| --- | --- |
| 用例数 | 12 |
| 事实召回率 | 1.0000 |
| 全部事实命中用例 | 10/10 |
| 支持段落召回率 | 0.9167 |
| 平均召回原文长度 | 1077.3 |
| 平均格式化上下文长度 | 1600.6 |
| 平均检索延迟(ms) | 1816.9 |
| 检索错误 | 0 |
| 模型调用次数 | 0 |
| 检索工具调用次数 | 0 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 1298 | 9995.43 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p004；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p011 |
| Q02 | 3/3 | 1/1 | 1210 | 1049.32 | knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p001；knowledge-8a0b667e89#p006 |
| Q03 | 5/5 | 1/2 | 945 | 884.94 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p021；knowledge-8a0b667e89#p020 |
| Q04 | 2/2 | 1/1 | 882 | 852.5 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p007 |
| Q05 | 2/2 | 1/1 | 903 | 1513.04 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p020 |
| Q06 | 1/1 | 1/1 | 1085 | 927.89 | knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p023 |
| Q07 | 3/3 | 1/1 | 1068 | 850.91 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p018；knowledge-8a0b667e89#p003 |
| Q08 | 2/2 | 1/1 | 954 | 1423.67 | knowledge-8a0b667e89#p010；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p008 |
| Q09 | 2/2 | 1/1 | 1097 | 1309.16 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p028；knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p000 |
| Q10 | 1/1 | 1/1 | 1436 | 1089.02 | knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p015 |
| Q11 | 0/0 | 0/0 | 1160 | 1088.53 | knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p001 |
| Q12 | 0/0 | 0/0 | 890 | 818.37 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p025 |
