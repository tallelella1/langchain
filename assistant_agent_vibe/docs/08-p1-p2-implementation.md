# P1 / P2 扩展实施记录

日期：2026-10-23。范围：`docs/03-extension-roadmap.md` 的 P1.2–P1.4、P2.1–P2.4。
P1.1（章节切分与父块补全）此前已完成，本次在其基础上继续。

默认策略保持“可解释、可验证”的首版基线：**向量检索 + 父块补全 + 固定检索 Agent**。
混合检索、LLM 重排、工具型 Agent、多轮与 API 均已实现并通过测试，但只有评估集
证明有明确收益的能力才进入默认路径。

## 1. 结论

| 能力 | 状态 | 默认是否启用 | 依据 |
| --- | --- | --- | --- |
| P1.2 混合检索（向量 + BM25 + RRF） | 已实现 | 否 | 12 题事实召回不变，支持段落召回 0.9167→0.9167，上下文 937→1077，延迟 1.14s→1.82s |
| P1.2 LLM 重排 | 已实现 | 否 | 支持段落 0.9167→1.0，上下文 937→876；每题多 1 次模型调用、约 +13s |
| P1.3 版本化索引/更新/删除 | 已实现 | 是 | 新集合构建后原子切换；未变化复用；失败保留旧版本；回滚/清理显式执行 |
| P1.4 固定评估集 | 已实现 | 是 | `evals/cases.json` + `evals/run_eval.py`，结果保存在 `evals/results/` |
| P2.1 多格式导入 | 已实现 | 按来源显式加入 | TXT/Markdown/PDF/DOCX；保留页码、标题层级、段落与表格列 |
| P2.2 多轮 + 流式 | 已实现 | 可选 | 指代改写、会话隔离、SSE/终端事件；流式中断不发 done |
| P2.3 工具型 Agent | 已实现 | 否 | 与固定基线同事实覆盖；22 次工具调用/34 次模型调用，平均 20.8s，固定基线 12 次/15.2s |
| P2.4 FastAPI + 页面 | 已实现 | 是（独立入口） | 问答/流式/健康检查/管理任务；与终端共用 `rag` 核心 |

当前活动集合：`rag_tutorial.docs_v3`（schema v2，38 子块/30 父块，index_version=3）；
`docs_v2` 保留用于对比。索引清单 `data/index_manifest.json`，旧清单归档在 `data/manifests/`。

## 2. P1.4 固定评估集

### 2.1 文件

- `evals/cases.json`：Q01–Q12，含期望事实、支持段落、可回答性与拒答行为。
- `evals/run_eval.py`：检索/答案评估、参数对比、Markdown 报告。
- `evals/results/`：本次真实运行结果。

指标：

- 事实召回率：期望事实在召回文本/答案中的子串命中比例；
- 支持段落召回：命中的父块 `chapter / section` 是否覆盖标注段落；
- 上下文长度：原始命中文本与 `format_context` 后的实际 Prompt 长度分开记录；
- 延迟、模型调用次数、工具调用次数；
- Q11/Q12 的拒答行为单独判定。

子串检查是启发式，不等于答案正确；Q01 的答案写成“10 个正式成员”而不是“10 人”、
Q03/Q08 省略非核心档位时会被计为未命中，需结合人工核对。

### 2.2 检索结果（`docs_v3`，Top-5）

| 策略 | 事实召回 | 支持段落 | 平均原文 | 平均格式化上下文 | 平均检索(ms) | 模型调用/12题 |
| --- | --- | --- | --- | --- | --- | --- |
| vector | 1.0000 | 0.9167 | 937.3 | 1440.2 | 1135.6 | 0 |
| hybrid（RRF） | 1.0000 | 0.9167 | 1077.3 | 1600.6 | 1816.9 | 0 |
| vector + LLM 重排 | 1.0000 | 1.0000 | 876.5 | 1272.1 | 14069.8 | 12 |
| hybrid + LLM 重排 | 1.0000 | 1.0000 | 824.6 | 1209.0 | 13579.3 | 12 |

`docs_v2` 上的 220/80 基线为：事实 1.0、支持段落 0.9167、平均原文 937.3、
格式化 1316.6、平均检索 934.0ms。`docs_v3` 与 `docs_v2` 的切分内容相同，
格式化上下文增加主要来自新增的定位信息。

结论：RRF 单独使用没有改善；LLM 重排改善了支持段落召回并缩短上下文，
但延迟增加约一个数量级。默认保持 `vector + rerank=none`；
需要时可显式使用 `--strategy hybrid`、`--rerank llm`。

本轮没有引入固定分数阈值：Q11/Q12 用于检查拒答行为，检索分数只用于排序，
不把余弦或 RRF 分数解释为“正确概率”。如果后续产品需要按分数拒答，
应使用已标注的可回答/不可回答集合单独校准，并分别处理重排分数与余弦分数。

### 2.3 答案结果

| 模式 | 答案事实召回 | 拒答行为 | 平均耗时(ms) | 模型调用/12题 | 工具调用/12题 |
| --- | --- | --- | --- | --- | --- |
| 固定检索 + 父块 | 0.8696 | 2/2 | 15232.7 | 12 | 0 |
| 工具型 Agent | 1.0000 | 2/2 | 20783.3 | 34 | 22 |

工具型 Agent 的答案事实召回更高，但每题平均多出约 2 次模型调用和 5.5 秒，
且检索调用次数由模型决定，稳定性低于固定检索。默认不启用；只有出现
“需要模型自行改写查询、多次检索或多个只读工具”的需求时再切换。

## 3. P1.2 混合检索与重排

- `rag/lexical.py`：无额外依赖的中文 BM25。中文按 bigram + trigram，英文/数字/百分号
  按连续串，标点作分隔；不使用 unigram，避免“火星基地”只凭单字“专”命中“专用发票”。
- `rag/reranking.py`：RRF 按名次融合（默认 `rrf_k=60`），分别保留 `semantic_score`
  与 `bm25_score`；LLM 重排是可选的一次 listwise 调用，输出 JSON `{"order": [...]}`，
  未知/重复 key 明确失败。
- `rag/retrieval.py`：`strategy=vector|hybrid`、`rerank_strategy=none|llm`；
  向量与 BM25 各取候选，按稳定 key 去重后融合，再按父块补全。
- 内存 BM25 是当前集合的只读快照；新版本集合构建后自然失效。超过
  `MAX_LEXICAL_RECORDS=10000` 时明确失败，不静默只索引一部分数据。
  更大规模应改用 Milvus 服务端全文检索，而不是扩大内存索引。

## 4. P1.3 版本化索引、更新与删除

### 4.1 数据模型

- 来源清单：`data/sources.json`（缺失时默认 `knowledge.txt`）；`asset/load` 不会自动入库。
- 清单 `manifest_version=2` 记录：
  `schema_version`、`strategy`、`collection`、`collection_prefix`、`index_version`、
  `previous_collection`、`chunk_config_hash`、`embedding_model/dim`、`documents`、`parents`、`children`。
- 每个文档：`doc_id`、`source`、`format`、`content_hash`、`document_version`、块数。
- 每个子块：稳定 `chunk_key`、全局 `chunk_id`、`doc_id`、`parent_id`、定位、原文；
- 每个父块：完整文本、来源定位、`child_ids`/`child_keys`。
- Milvus schema v2 增加 `chunk_key/doc_id/parent_id/content_hash/document_version/
  index_version/embedding_model/chunk_config_hash/location_type/location/page`。
  主键仍是全局唯一 INT64；`chunk_key` 是稳定逻辑键，不要求每份文件从 0 开始。

### 4.2 构建与切换

`rag/indexer.py`：

1. 读取来源并计算内容哈希；与活动清单比较来源集合、内容、切分配置与 Embedding 身份。
2. 未变化且未加 `--force` 时直接复用，不调用 Embedding。
3. 需要更新时选择下一个未被占用的 `docs_v{n}`，在**新集合**完成解析、切分、向量化、
   写入、读回核对、计数与自搜索。
4. 用固定问题 Q01/Q03/Q06 抽查新集合；缺事实则拒绝切换。
5. 归档旧清单，原子替换 `data/index_manifest.json`；查询下一次请求使用新集合。
   这里使用活动清单切换而不是 Milvus alias，避免依赖不同版本服务端的 alias 语义。
6. 失败/中断时新集合不被引用，旧清单和旧集合继续可用；失败留下的集合会在下一次
   构建时被跳过。

### 4.3 删除、回滚与清理

- 删除来源：`python ingest.py --remove-source <path>` 更新来源清单并重建新集合；
  新检索不再召回该文档。旧集合按版本保留，不会被原地修改。
- 回滚：`python ingest.py --rollback` 切回 `previous_collection`，清单标记
  `stale_sources=true`，避免源文件已变化时误判为当前版本。
- 清理：`python ingest.py --prune --keep 1 --yes`，只处理 `data/manifests/` 中
  有归档清单的旧集合，永不删除当前集合；默认不加 `--yes` 只打印计划。
- `--write-manifest` 已停用：清单必须与新集合一起原子切换，不能单独改清单。

### 4.4 验收覆盖

`tests/test_index_versioning.py`（真实 Milvus，使用随机前缀并在结束后删除）：

- 重复导入复用；文件变短生成新版本且旧片段不残留；
- 加入第三份来源、移除来源；Embedding 模型变化触发重建；
- 模拟清单切换失败：活动清单字节不变、旧集合仍存在；
- 失败构建留下的集合版本被跳过；回滚到上一版本；
- `--prune` 计划不包含当前集合。

## 5. P2.1 多格式知识源

`rag/loaders/`：

| 格式 | 保留信息 | 失败处理 |
| --- | --- | --- |
| TXT | 一级章节、编号小节、原文偏移 | 空文件/非 UTF-8 明确失败 |
| Markdown | 标题层级 `heading_path`、代码块内容；代码围栏内的 `#` 不当作标题 | 空文件明确失败 |
| PDF | 每页一个父块，`page`、`第 N 页` 定位；跳过空页 | 全空（疑似扫描件）明确提示需要 OCR |
| DOCX | 样式识别标题层级；段落范围；表格按 `列 | 列` 保留字段对应关系 | 无可用文本明确失败 |

`asset/load` 中的学习素材只有在被显式 `--add-source` 或 `--source` 指定时才进入索引；
解析失败不会作为空文本成功入库。扫描 PDF 的 OCR、复杂表格重建、版面识别不在本轮范围。

## 6. P2.2 多轮对话与流式输出

- `rag/conversation.py`：按 `session_id` 隔离的进程内会话（默认最近 6 轮）；
  有历史时调用一次模型把“那专业版呢”改写为独立检索问题；历史只含问题与检索问题，
  **不含历史回答**，避免历史答案被当作事实。
- 最终回答仍使用原始当前问题；`resolved_query` 单独返回，便于用户核对。
- `answering.stream_answer`：事件顺序 `start → delta… → sources → done`；
  检索/模型失败只发 `error`，不发 `done`；工具调用前的模型文本不会进入最终答案。
  本轮不实现自动重连；重连与重复计费由上游接口能力决定，客户端收到 `error`
  应显示失败而不是追加为正常完成。
- 终端：`python main.py --chat`、`--stream`；API SSE 使用同一事件结构。
- 实测：“基础版多少钱？” → “那专业版呢？” 改写为“专业版多少钱？”，回答 199 元。
- 会话保存在进程内；多进程部署需要外部会话存储（P3）。

## 7. P2.3 检索工具型 Agent

- `rag/tools.py`：`search_knowledge` 只读工具调用既有 `retrieve`；
  工具返回带 `[片段 N]` 的原文、来源与定位；无结果、执行失败、预算耗尽都明确返回。
- 引用编号按父块去重并全局编号；同一父块经不同子块重复命中不会重复占号。
- 调用预算 `TOOL_MAX_CALLS=3`（初次 + 最多两次补充/改写），
  `AGENT_RECURSION_LIMIT=8` 只防失控，不替代业务预算。
- Agent 未调用工具却给出知识性答案时明确失败；工具失败且无命中时用标准“材料不足”
  覆盖模型输出，避免编造。
- 评估结果见 2.3；默认入口仍是固定检索，工具模式通过 `--mode tool` 或 API 的
  `"mode": "tool"` 显式启用。

## 8. P2.4 FastAPI 与页面

### 8.1 启动

```powershell
# 项目目录
$env:ASSISTANT_ADMIN_TOKEN = "change-me"   # 不配置则管理接口关闭
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

页面：`http://127.0.0.1:8000/`（问答）；`http://127.0.0.1:8000/admin`
（知识管理：依赖就绪、当前索引/文件、任务状态与失败原因；令牌只存 sessionStorage）。
若命令行客户端经过系统代理访问 localhost 失败，
设置 `NO_PROXY=127.0.0.1` 或客户端关闭环境代理；浏览器通常直接绕过 localhost。

依赖：FastAPI/uvicorn 已在父工程 `requirements.txt` 中；PDF/DOCX 需要
`pypdf` 与 `python-docx`（父工程 `requirements_full.txt` 已列，当前环境已安装）。
未安装时对应 Loader 会明确失败，不会把文件当作空文本入库。

### 8.2 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/ask` | 单次问答；`session_id` 省略时自动生成 |
| POST | `/ask/stream` | SSE；事件 `start/delta/sources/done/error` |
| GET | `/health/live` | 进程存活 |
| GET | `/health/ready` | 配置、来源清单、来源与索引同步、索引清单、Milvus 加载状态；依赖不就绪返回 503 |
| POST | `/admin/ingest` | 创建入库任务；需要 `X-Admin-Token` |
| GET | `/admin/ingest/{task_id}` | 任务状态 |
| GET | `/admin/ingest` | 任务列表 |
| GET | `/admin/knowledge` | 当前集合、版本、文档块数；需要管理令牌 |

- 建索引不复用聊天入口权限；未配置 `ASSISTANT_ADMIN_TOKEN` 时管理接口返回 503。
  本轮未提供网页文件上传接口；新文件先用 `python ingest.py --add-source <path>`
  加入来源清单，或在管理页创建入库任务前由维护命令登记。
- 入库任务为单任务后台线程，状态保存在进程内；并发任务返回 409。
  持久队列、多 worker 一致性属于 P3。
- 页面最小功能：问题、回答、来源定位、错误；流式切换；模式/检索选择；新会话。

### 8.3 实测

- `/health/ready`：200，`docs_v3` Loaded，文档 1，子块 38。
- `/ask`：200，返回答案、来源、引用编号（固定检索模式）。
- `/ask/stream`：`start → delta×9 → sources → done`，引用 `[1, 4]`。
- `/`：200，问答页面可加载；`/admin`：200，知识管理页面可加载。
- 入库任务状态机使用假构建在测试中覆盖；真实集合构建由 `ingest.py` 的版本化流程
  与 `tests/test_index_versioning.py` 验证，避免在验收时反复产生收费向量。
- API 的 `mode=tool` 与 CLI/评估共用 `rag.tools`；本次真实工具调用在 CLI 与评估中执行，
  API 层由 `test_api.py` 的注入替身覆盖参数与错误映射。

## 9. 测试与验证

- 默认 `python -m pytest -q`：**105 passed, 1 skipped**（真实 Embedding 需显式启用）。
- 新增测试：
  - `test_loaders.py`：Markdown/PDF/DOCX 定位与失败路径；
  - `test_index_versioning.py`：真实 Milvus 的版本化构建、更新、删除、切换失败、回滚；
  - `test_lexical_reranking.py`：分词、BM25、RRF、LLM 重排解析、混合检索；
  - `test_conversation_streaming.py`：会话隔离、改写、流式事件与错误；
  - `test_tools.py`：工具去重、预算、失败、未调用工具、流式；
  - `test_api.py`：鉴权、任务状态、SSE、健康检查、页面；
  - `test_eval_metrics.py`：拒答判定、覆盖率和结果对比。
- 真实运行记录：`evals/results/final-*.json/md`；Q01–Q12 验收见
  [P1/P2 真实验收](09-p1-p2-live-acceptance.md)（若已生成）。

## 10. 已知限制与后续

- 内存 BM25 不支持增量更新；数据量超过 10000 子块需改用服务端全文检索。
- LLM 重排和工具型 Agent 增加模型调用与延迟，默认不启用。
- 扫描 PDF 需要 OCR；PDF 表格重建、版面分析未实现；DOCX 标题依赖样式，
  无样式文档会归入“正文”父块。
- 会话和入库任务状态均为进程内；多进程/多实例需要 P3 的外部存储与队列。
- 回滚清单标记 `stale_sources=true`；跨 Embedding 模型回滚需同时恢复旧模型配置。
- 源文件在活动索引之外被修改时，查询会明确要求重新入库，避免新旧知识混用。
- Windows 终端编码可能影响中文显示；项目文件统一 UTF-8。
