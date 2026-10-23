# GitHub 项目调研与来源

检索日期：**2026-10-22，Asia/Hong_Kong（UTC+08:00）**。Star 采用本次成功返回的 GitHub 官方 Search API 精确整数，是不同请求的同日快照，不代表阅读时的实时值。

## 1. “相关项目 Star 前五”的筛选口径

本次范围：公开、未归档、非 fork，能运行的知识库／文档问答／RAG 应用或平台，文档知识问答是其明确能力之一。

排除纯资源清单、教学合集、通用框架、向量数据库、单独 OCR／爬虫、编码工具与会话记忆组件。先按 Star 找到高关注项目，再基于官方 README／产品资料判断是否符合范围。

因此，下表是**本次检索结果中，按上述应用范围筛选后的 Star 前五**，不宣称是 GitHub 全站所有“与 RAG 有关”仓库的绝对前五。GitHub Topic 由项目自行标注，描述和名称也会变化，检索存在覆盖边界。

### 已执行的官方检索

| 查询 | 排序与返回量 | 本次返回 |
| --- | --- | --- |
| `topic:rag stars:>10000 archived:false fork:false` | Star 降序，前 40 | 总匹配 76，`incomplete_results=false`；第 40 名为 26,806 Star，低于入选第五名，足以覆盖该查询中竞争前五的候选 |
| `topic:knowledge-base stars:>10000 archived:false fork:false` | Star 降序，最多 40 | 总匹配 16，全部返回，`incomplete_results=false` |
| `RAG in:name,description stars:>50000 archived:false fork:false` | Star 降序，最多 50 | 总匹配 9，全部返回；补充未打相同 Topic 的项目，如 Dify |

复核入口：

- [RAG Topic 的官方 Search API](https://api.github.com/search/repositories?q=topic%3Arag%20stars%3A%3E10000%20archived%3Afalse%20fork%3Afalse&sort=stars&order=desc&per_page=40)
- [Knowledge-base Topic 的官方 Search API](https://api.github.com/search/repositories?q=topic%3Aknowledge-base%20stars%3A%3E10000%20archived%3Afalse%20fork%3Afalse&sort=stars&order=desc&per_page=40)
- [名称／描述 RAG 的官方 Search API](https://api.github.com/search/repositories?q=RAG%20in%3Aname%2Cdescription%20stars%3A%3E50000%20archived%3Afalse%20fork%3Afalse&sort=stars&order=desc&per_page=50)

这些链接再次访问时会返回新的数值。后续逐仓库元数据补充请求遇到 GitHub 未认证限流，所以没有取得统一时刻的逐仓库元数据和 commit SHA；下表只使用已成功的搜索响应，不把网页约数补写成精确数。

## 2. 入选项目

| 排名 | 官方仓库 | Star 快照 | 本次数字来源 | 与本项目相关的能力 |
| --- | --- | ---: | --- | --- |
| 1 | [langgenius/dify](https://github.com/langgenius/dify) | 159,205 | 名称／描述 RAG 查询 | 文档入库与检索、RAG Pipeline、模型管理、工作流 |
| 2 | [open-webui/open-webui](https://github.com/open-webui/open-webui) | 156,402 | RAG Topic 查询 | 本地 RAG、混合检索、重排、对话中使用资料 |
| 3 | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 92,634 | RAG Topic 查询 | 文档理解、模板切分、切分预览、可追溯引用 |
| 4 | [lobehub/lobehub](https://github.com/lobehub/lobehub) | 83,406 | Knowledge-base Topic 查询 | 自托管应用、文件与知识库在对话中的使用 |
| 5 | [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) | 67,423 | RAG Topic 查询 | 文档处理、向量库与模型接入、问答来源展示 |

功能依据分别来自项目官方 README。LobeHub 当前 README 侧重 Agent 产品定位，其文件与知识库能力补充依据 [官方“文件与知识库”说明](https://lobehub.com/zh/pricing)；该产品页面用于确认能力存在，不作为开源版全部功能范围或具体技术实现的证据。

### 较高 Star 但未入选的候选

| 类型 | 本次发现的代表 | 排除理由 |
| --- | --- | --- |
| 框架与底层库 | `langchain-ai/langchain`、`mem0ai/mem0` | 分别是通用开发框架和记忆组件，不与可运行客服知识库应用混排 |
| 教程与资料集合 | `Shubhamsaboo/awesome-llm-apps`、`datawhalechina/hello-agents`、`dair-ai/Prompt-Engineering-Guide` | 用于学习，但不是同一类完整知识库应用 |
| 编码／Agent 上下文工具 | `Graphify-Labs/graphify`、`Egonex-AI/Understand-Anything`、`thedotmack/claude-mem`、`ruvnet/ruflo`、`headroomlabs-ai/headroom` | 主目标是代码理解、会话记忆、Agent 编排或上下文处理 |
| 文档处理与采集组件 | `PaddlePaddle/PaddleOCR`、`unclecode/crawl4ai` | 是数据准备环节的组件 |
| 综合知识工作空间 | `toeverything/AFFiNE` | 主定位是知识编辑与协作；本次限于明确的文档问答／RAG 应用范围 |

不将高 Star 当作架构优劣或适合本项目的证明。授权协议、部署成本和运行依赖如涉及后续复用，需要再核验；本轮没有对这些大型平台进行本地部署、完整源码审计或许可证审计。

## 3. 应该借鉴什么

以下“本项目采用方式”为设计推断，不代表上游采用完全相同模块边界。

### 3.1 Dify：将知识处理与问答编排分开

官方 README 明确介绍文档入库到检索的 RAG Pipeline，以及模型管理和工作流能力。

本项目采用方式：入库入口和问答入口分开，模型配置集中维护，后续 API 复用核心模块。首版不移植整套平台、前端或工作流引擎。

阅读入口：[官方仓库 README](https://github.com/langgenius/dify#readme)。

### 3.2 Open WebUI：检索策略是可评估的组合

官方 README 列出本地 RAG、BM25 与向量混合检索、重排和多种内容提取方式。

本项目采用方式：在 `retrieval.py` 中保留替换空间；先做单一向量基线，再用固定问题集比较混合检索。多格式解析按需求增加。

阅读入口：[官方仓库 README](https://github.com/open-webui/open-webui#readme)。

### 3.3 RAGFlow：先看解析与切片质量，再评价答案

官方 README 展示复杂文档理解、模板切分、切分可视化及可追溯引用。

本项目采用方式：开发阶段预览真实切片，使用来源和切片号追溯答案；后续 PDF 场景保留页码与文档结构。首版不因为输入是 TXT 就加入 OCR 系统。

阅读入口：[官方仓库 README](https://github.com/infiniflow/ragflow#readme)。当前分支结构可能持续调整，本次未验证内部实现路径，不把历史目录名写成当前源码入口。

### 3.4 LobeHub：知识使用范围应对用户可见

官方产品资料确认文件与知识库可在对话中使用；仓库提供自托管路径。

本项目采用方式：先展示本次答案实际获得的资料；以后加入多知识库或多轮对话时，再明确当前会话绑定的知识范围和会话边界。

阅读入口：[官方仓库 README](https://github.com/lobehub/lobehub#readme)、[官方文件与知识库能力说明](https://lobehub.com/zh/pricing)。本次未核实其检索算法和内部存储结构，不据此指定本项目的底层数据库。

### 3.5 AnythingLLM：文档处理独立于聊天

官方 README 说明文档上传、来源引用和本地／远端模型接入，并区分 `collector` 文档处理与 `server` 模型及向量库交互。

本项目采用方式：`documents.py` 处理文档，`vector_store.py` 管理向量访问，`answering.py` 组织回答；后续文档处理变慢时再独立为任务。

阅读入口：[官方仓库 README](https://github.com/Mintplex-Labs/anything-llm#readme)、[collector](https://github.com/Mintplex-Labs/anything-llm/tree/master/collector)、[server](https://github.com/Mintplex-Labs/anything-llm/tree/master/server)。

## 4. 对本学习工程的选择

保留课件技术栈，把上述项目当作能力与边界参考：借鉴 Dify 的入库／问答分工、Open WebUI 的检索改进思路、RAGFlow 的切片预览与证据追溯、LobeHub 的会话知识范围、AnythingLLM 的文档处理分工。

以上参考已逐项关联到 [实施步骤对应表](02-implementation-steps.md#github-参考与实施步骤对应表)。步骤 1–8 在相应位置列出阅读入口、本项目落点与检查方式；环境和 Milvus 集合操作以官方资料为准；步骤 9 给出本项目验收方式，步骤 10 将发现的问题连接到扩展路线。这里完成的是设计与能力映射，尚未完成上游源码级适配或运行验证。

首版只实施单文件入库、固定检索、无工具 Agent 和答案来源展示；这是基于学习范围的设计取舍。高 Star 平台不作为本项目必须安装的依赖。

## 5. 官方技术资料

| 来源 | 用途 | 应用边界 |
| --- | --- | --- |
| [LangChain Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval) | 两步 RAG、Agentic RAG 与混合流程 | 用于解释架构类型，具体 API 对照项目安装版本 |
| [RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) | 字符长度、目标重叠、中文分隔符 | 说明 220/80 基线的参数语义 |
| [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3) | 底层模型与 1024 维信息 | 不证明服务商某个 model ID 当前一定可用 |
| [SiliconFlow Embeddings](https://docs.siliconflow.cn/docs/api/embeddings-post) | 实施前核对接口、请求和响应 | 服务商当前支持参数须实际复核，不照搬 OpenAI 所有参数 |
| [Milvus Windows 部署](https://milvus.io/docs/install_standalone-windows.md) | Docker Desktop／WSL 2 部署入口 | 先核对现有服务与目标版本，不自动部署最新版本 |
| [Milvus Similarity Metrics](https://milvus.io/docs/metric.md) | COSINE、L2、IP 及索引选择 | 本项目使用 COSINE，分数不是正确概率 |
| [Milvus Consistency](https://milvus.io/docs/consistency.md) | 写入后的读取可见性 | 区分 flush、统计信息与一致性保障 |

课件与本地文件属于另外两类依据：课件提供教学基线；`config.py`、`knowledge.txt`、父工程 `requirements.txt` 提供当前现场事实。架构改进、模块划分、参数实验与扩展阶段均为本次设计建议，尚无运行验证结果。
