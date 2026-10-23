# Atguigu Assistant 客服知识库：设计文档入口

更新时间：2026-10-22。项目目录：`D:/grow/langchain/assistant_agent`。

## 1. 建议采用的整体方案

以课件 2.5.3 为基线，使用 **LangChain + PyMilvus + Milvus + Embedding 服务 + 对话模型**，先完成单份 TXT 客服知识库的检索问答。程序分为“知识入库”和“在线问答”两个入口；问答启动时只读取已有索引，不重复切分、向量化或清空集合。

本次只交付设计文档。文件树中的 Python 模块、测试、部署文件和扩展目录均为规划，没有创建或修改；没有安装依赖、启动数据库或调用模型。

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

## 4. 当前现场与设计假设

**已核实：**

- 当前目录已有 `config.py`、`knowledge.txt`、`asset/load/`；本次创建此前不存在的 `docs/`。
- `config.py` 已有 `MILVUS_URI`、`DB_NAME`、`COLLECTION_NAME`、`KNOWLEDGE_FILE`、`EMBED_MODEL_NAME`、`EMBED_DIM` 六个配置名。
- 现有配置值包含 `http://localhost:19530`、`rag_tutorial`、`docs`、`Pro/BAAI/bge-m3`、`1024`。这些是配置事实，不表示连接或模型可用性已经验证。
- `knowledge.txt` 位于项目当前目录，内容标注为 **2026 Q1 版教学知识库**。其中的套餐、退款和发票口径是本项目测试素材，不应表述为现实产品的现行规则。
- 父工程 `D:/grow/langchain/requirements.txt` 固定了 LangChain 等依赖；RAG 相关的 `pymilvus==2.6.12` 等条目仍处于注释区。注释不代表环境中一定没有安装，须在实施时检查。
- Git 工作区存在用户未提交的学习文件与其他改动；本次只新增本目录中的 Markdown 文档。

**规划采用的默认边界：**

- 首版单用户、单知识文件、单次提问，终端展示召回结果和最终回答。
- 保留课件切分基线 `chunk_size=220`、`chunk_overlap=80`、`top_k=5`，先观察效果再调参。
- 保留 Milvus 和原有主要命名，使用 PyMilvus 直接访问，便于对应课件。
- 首版只进行首次索引构建；已有非空集合时停止重复导入并给出说明。更新与重建作为显式维护操作设计。
- 对话模型的服务商、模型 ID、鉴权、Embedding 模型在服务商平台上的可用名称，以及 Milvus 实际运行状态，均留待实现前核验。本次不读取密钥值。

## 5. 首版可验收结果

用户提出“为什么我在 7 天内申请退款，还是被拒了？”后，能够看到：

1. 实际参与回答的知识片段、`source`、`chunk_id` 和相似度分数。
2. 根据知识片段解释使用量、订单升级、专票等条件；不会直接断言用户的实际订单满足某个拒绝条件。
3. 回答中的引用能对应到实际召回片段。
4. 无法从材料确定的内容明确说明不足，不编造套餐、合同或订单数据。

首版完成后，再按 [扩展路线](03-extension-roadmap.md) 引入质量改进。后续实施起点是 [步骤 0：核对环境与配置](02-implementation-steps.md#步骤-0核对环境与配置)，不是直接连接模型或覆盖现有集合。
