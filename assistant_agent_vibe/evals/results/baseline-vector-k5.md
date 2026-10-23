# 固定评估结果：vector

- 时间: 2026-10-23T16:00:26
- Top-K: 5；父块补全: True
- Embedding: text-embedding-v4
- 对话模型: 未调用
- 检索参数: `{}`

## 汇总

| 指标 | 值 |
| --- | --- |
| 用例数 | 12 |
| 事实召回率 | 1.0000 |
| 全部事实命中用例 | 10/10 |
| 支持段落召回率 | 0.9167 |
| 平均召回原文长度 | 937.3 |
| 平均格式化上下文长度 | 1316.6 |
| 平均检索延迟(ms) | 933.97 |
| 检索错误 | 0 |
| 模型调用次数 | 0 |

## 明细

| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 2/2 | 2/2 | 959 | 4775.09 | knowledge.txt#p029；knowledge.txt#p011；knowledge.txt#p013；knowledge.txt#p004；knowledge.txt#p014 |
| Q02 | 3/3 | 1/1 | 943 | 985.49 | knowledge.txt#p013；knowledge.txt#p029；knowledge.txt#p011；knowledge.txt#p012；knowledge.txt#p014 |
| Q03 | 5/5 | 1/2 | 945 | 501.39 | knowledge.txt#p029；knowledge.txt#p019；knowledge.txt#p021；knowledge.txt#p020 |
| Q04 | 2/2 | 1/1 | 798 | 718.31 | knowledge.txt#p019；knowledge.txt#p029；knowledge.txt#p018 |
| Q05 | 2/2 | 1/1 | 903 | 512.64 | knowledge.txt#p019；knowledge.txt#p029；knowledge.txt#p020；knowledge.txt#p018 |
| Q06 | 1/1 | 1/1 | 724 | 469.8 | knowledge.txt#p025；knowledge.txt#p020；knowledge.txt#p024；knowledge.txt#p019；knowledge.txt#p023 |
| Q07 | 3/3 | 1/1 | 1037 | 526.75 | knowledge.txt#p029；knowledge.txt#p015；knowledge.txt#p003 |
| Q08 | 2/2 | 1/1 | 954 | 509.74 | knowledge.txt#p029；knowledge.txt#p010；knowledge.txt#p008 |
| Q09 | 2/2 | 1/1 | 891 | 531.25 | knowledge.txt#p029；knowledge.txt#p027；knowledge.txt#p028；knowledge.txt#p006 |
| Q10 | 1/1 | 1/1 | 1056 | 601.34 | knowledge.txt#p006；knowledge.txt#p000；knowledge.txt#p029；knowledge.txt#p012 |
| Q11 | 0/0 | 0/0 | 1097 | 530.86 | knowledge.txt#p006；knowledge.txt#p029；knowledge.txt#p027；knowledge.txt#p000；knowledge.txt#p028 |
| Q12 | 0/0 | 0/0 | 941 | 544.93 | knowledge.txt#p029；knowledge.txt#p019；knowledge.txt#p020；knowledge.txt#p024 |
