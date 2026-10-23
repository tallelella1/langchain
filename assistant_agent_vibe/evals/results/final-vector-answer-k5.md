# 固定评估结果：fixed / vector / rerank=none

- 时间: 2026-10-23T17:18:19
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
| 平均召回原文长度 | 937.3 |
| 平均格式化上下文长度 | 1440.2 |
| 平均检索延迟(ms) | 1323.3 |
| 检索错误 | 0 |
| 模型调用次数 | 12 |
| 检索工具调用次数 | 0 |
| 答案事实召回率 | 0.8696 |
| 拒答行为通过 | 2/2 |
| 答案错误 | 0 |
| 平均回答延迟(ms) | 15232.67 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 959 | 6353.12 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p011；knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p004；knowledge-8a0b667e89#p014 |
| Q02 | 3/3 | 1/1 | 943 | 771.32 | knowledge-8a0b667e89#p013；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p011；knowledge-8a0b667e89#p012；knowledge-8a0b667e89#p014 |
| Q03 | 5/5 | 1/2 | 945 | 651.83 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p021；knowledge-8a0b667e89#p020 |
| Q04 | 2/2 | 1/1 | 798 | 1861.73 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p018 |
| Q05 | 2/2 | 1/1 | 903 | 761.84 | knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p018 |
| Q06 | 1/1 | 1/1 | 724 | 794.16 | knowledge-8a0b667e89#p025；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p024；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p023 |
| Q07 | 3/3 | 1/1 | 1037 | 822.54 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p015；knowledge-8a0b667e89#p003 |
| Q08 | 2/2 | 1/1 | 954 | 736.14 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p010；knowledge-8a0b667e89#p008 |
| Q09 | 2/2 | 1/1 | 891 | 615.82 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p028；knowledge-8a0b667e89#p006 |
| Q10 | 1/1 | 1/1 | 1056 | 1080.56 | knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p012 |
| Q11 | 0/0 | 0/0 | 1097 | 730.77 | knowledge-8a0b667e89#p006；knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p027；knowledge-8a0b667e89#p000；knowledge-8a0b667e89#p028 |
| Q12 | 0/0 | 0/0 | 941 | 699.75 | knowledge-8a0b667e89#p029；knowledge-8a0b667e89#p019；knowledge-8a0b667e89#p020；knowledge-8a0b667e89#p024 |
