# Atguigu Assistant 客服知识库：设计文档入口

更新时间：2026-10-23。项目目录：`D:/grow/langchain/assistant_agent`。

## 1. 建议采用的整体方案

以课件 2.5.3 为基线，使用 **LangChain + PyMilvus + Milvus + Embedding 服务 + 对话模型**，先完成单份 TXT 客服知识库的检索问答。程序分为“知识入库”和“在线问答”两个入口；问答启动时只读取已有索引，不重复切分、向量化或清空集合。

2026-10-23 已按步骤 1–9 完成首版实现与验收：`config.py`、`rag/`、`ingest.py`、`main.py`、`tests/` 均已落地。
实际结果、已知问题与取舍见 [步骤 9 验收记录](05-acceptance-results.md)。

## 2. 阅读顺序

| 文档 | 解决的问题 |
| --- | --- |
| [整体架构与文件树](01-architecture.md) | 系统如何分层、数据如何流动、每个文件负责什么、配置与数据结构是什么 |
| [详细实施步骤与验收](02-implementation-steps.md) | 按什么顺序实现、每一步的输入输出、GitHub 参考与本项目落点、如何验证与定位失败 |
| [扩展路线与取舍](03-extension-roadmap.md) | 检索质量、知识更新、多轮对话、Web/API、多租户等能力如何逐步加入 |
| [GitHub 项目调研与来源](04-github-research.md) | 本次相关项目 Star 前五、筛选口径、可借鉴内容和官方技术资料 |

## 3. 课件依据

来源：`C:/Users/tallerella/Desktop/01-课件/尚硅谷-10-RAG.pdf`，2.5.3「案例：Atguigu Assistant 客服知识库」，PDF 文件第 **52–66 页**。页码从文件第一页起计数。已提取该节文本，并查看相关页面渲染结果。

| 课件步骤 | PDF 页码 | 规划落点 |
| --- | --- | --- |
| ① 全局配置 | 52–53 | `config.py` |
| ② 初始化 Milvus，创建 database/collection | 53–54 | `rag/vector_store.py`、`ingest.py` |
| ③ 初始化 Embedding 模型 | 54 | `rag/models.py` |
| ④ 文档加载与切分 | 54–61 | `rag/documents.py` |
| ⑤ 向量化与写入 | 61–62 | `ingest.py`、`rag/vector_store.py` |
| ⑥ 初始化模型与 Agent | 63 | `rag/models.py`、`rag/answering.py` |
| ⑦ 相似度检索 | 63–64 | `rag/retrieval.py` |
| ⑧ 上下文组装、生成与示例问答 | 64–66 | `rag/prompts.py`、`rag/answering.py`、`main.py` |

课件的 Agent 使用 `tools=[]`。这里保留“程序先检索，再把问题和上下文交给 Agent”的流程；让 Agent 自主调用检索工具属于后续扩展。LangChain 官方对两步检索与 Agent 检索的说明见 [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)。

## 4. 当前现场（2026-10-23 实施后）

**已核实：**

- 解释器 `D:/conda/envs/langchain/python.exe`（Python 3.13.12）；langchain 1.2.12、langchain-openai 1.1.11、pymilvus 2.6.12、pytest 9.0.3。
- Milvus 地址 `http://localhost:19530`，容器 `milvus-standalone` / `milvusdb/milvus:v3.0-beta`。
- 旧集合 `rag_tutorial.docs` 有 2 条早期实验记录，缺少 bloom filter 统计对象、无法加载；按用户决定保留不动。
- `rag_tutorial.docs_v1`：220/80 基线集合，43 条记录，保留用于对比。
- 当前集合 `rag_tutorial.docs_v2`：`section-parent-v1` 章节切分 + 父块补全，38 个子块、30 个父块；
  集合属性记录 `embedding_model=text-embedding-v4`、`embedding_dim=1024`。父块全文与子块映射写入
  `data/index_manifest.json`（由 `ingest.py` 生成并校验源文件哈希）。
- 问答/检查入口会拒绝标识缺失或与配置不符的集合。已有集合可用 `python ingest.py --stamp` 显式写入身份（不改数据）。
- Embedding：百炼 OpenAI 兼容接口 `text-embedding-v4`，实测文档 2 条/查询 1 条均为 1024 维，同一文本自相似度 1.0000。计划中的 SiliconFlow 凭据本机不存在，经用户确认替换；如需换回，修改 `config.py` 中 `EMBED_*` 并重新向量化全部文档。
- 对话模型：`qwen3.8-max`，通过 `BAILIAN_API_URL`；保留 `create_agent(tools=[])` 路径。
- 父工程 `.env` 只由 `config.py` 加载一次（进程环境变量优先）；运行记录与测试不输出密钥。
- `knowledge.txt` 的 220/80 基线为 43 块（`docs_v1`，见 [220/80 基线](06-baseline-220-80.md)）；
  当前章节策略为 38 个子块/30 个父块（`docs_v2`，见 [父块补全对比](07-parent-expansion-comparison.md)）。

**已知限制：**

- 旧 `docs` 集合不可用，且入库/问答入口均不执行 drop。
- 基线 `docs_v1` 存在标题单独成块（chunk 17/28/35）；`docs_v2` 通过章节/小节切分避免了纯标题子块。
  当前父块补全仍可能把同一父块内的无关内容一起带入，例如第九章问答父块；引用相关性仍需人工核对。
- 服务端 v3.0-beta 与 PyMilvus 2.6.12 不是官方推荐配对；新建集合的写入、读回、搜索已通过，升级服务端前需复测。
- 服务端未启用鉴权，鉴权失败路径未实测；错误地址与维度冲突已明确报错。

## 5. 首版可验收结果

用户提出“为什么我在 7 天内申请退款，还是被拒了？”后，能够看到：

1. 实际参与回答的知识片段、`source`、`chunk_id` 和相似度分数。
2. 根据知识片段解释使用量、订单升级、专票等条件；不会直接断言用户的实际订单满足某个拒绝条件。
3. 回答中的引用能对应到实际召回片段。
4. 无法从材料确定的内容明确说明不足，不编造套餐、合同或订单数据。

以上行为已在 Q01–Q12 中逐题核对，结果见 [步骤 9 验收记录](05-acceptance-results.md)。
2.1 父块补全的 Q03–Q06 对比见 [父块补全对比](07-parent-expansion-comparison.md)。
后续质量改进按 [扩展路线](03-extension-roadmap.md) 推进。

## 6. 运行方式（在 `assistant_agent` 目录）

| 目的 | 命令 | 是否收费 |
| --- | --- | --- |
| 切分预览 | `python -m rag.documents --limit 5` | 否 |
| Embedding 实测 | `python -m rag.models` | 是 |
| 集合只读检查 | `python ingest.py --inspect` | 否 |
| 加载已有集合 | `python ingest.py --load` | 否 |
| 写入集合 Embedding 标识 | `python ingest.py --stamp` | 否（仅改集合属性） |
| 重建父块清单 | `python ingest.py --write-manifest` | 否（只读集合并写清单） |
| 空集合读写验证 | `python ingest.py --verify` | 否（仅本地 Milvus） |
| 首次入库 | `python ingest.py` | 是（Embedding） |
| 单次问答 | `python main.py "问题"` | 是（Embedding + 对话） |
| 离线/集成测试 | `pytest -q` | 否 |
| 含真实 Embedding 的测试 | `RUN_LIVE_EMBEDDING=1 pytest -q` | 是 |
| Q01–Q12 验收 | `python tests/live_acceptance.py --strict --malicious` | 是 |

`main.py` 不触发入库；`ingest.py` 在目标集合非空时停止并显示集合信息。
`data/index_manifest.json` 由 `ingest.py` 生成；缺失、源文件变化或与集合/Embedding 配置不一致时，
检索会明确报错并要求重新入库，不会用错配的父块来源回答。
Milvus 可达时 `pytest -q` 会运行集成测试，并创建/删除一个随机命名的实验集合（不触碰 `docs`/`docs_v1`/`docs_v2`）。
`live_acceptance.py` 默认覆盖 `docs/05-acceptance-results.md`；重跑时建议用 `--out` 指向临时文件。
