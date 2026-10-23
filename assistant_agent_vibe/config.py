"""Atguigu Assistant 统一配置。

- 路径全部以本文件位置解析，从父目录或项目目录启动都得到同一路径。
- 父工程 .env 只在这里加载一次：进程环境变量优先，.env 只补充缺失项。
- import 阶段只做离线校验，不连接 Milvus、不调用模型、不修改集合。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# =========================
# 1. 项目路径（与启动目录无关）
# =========================

PROJECT_DIR = Path(__file__).resolve().parent  # assistant_agent/
PARENT_DIR = PROJECT_DIR.parent  # D:/grow/langchain/
ENV_FILE = PARENT_DIR / ".env"
KNOWLEDGE_FILE = (PROJECT_DIR / "knowledge.txt").resolve()
LOAD_DIR = (PROJECT_DIR / "asset" / "load").resolve()

# 整个项目只在这里加载一次 .env；override=False 表示进程环境变量优先。
load_dotenv(dotenv_path=ENV_FILE, override=False)

# =========================
# 2. 基本配置（保留原有六个配置名）
# =========================

MILVUS_URI = "http://localhost:19530"  # Milvus 服务的连接地址
DB_NAME = "rag_tutorial"  # 自定义数据库名称
# 旧 docs 集合含 2 条早期实验记录且缺少 bloom filter 统计对象、无法加载；
# 经用户确认保留旧集合不动。docs_v1 是 220/80 基线，docs_v2 用于章节切分/父块补全。
COLLECTION_NAME = "docs_v2"  # 向量集合名称（类似于传统数据库的表）
# KNOWLEDGE_FILE 已在上方按项目位置解析为绝对路径
# 课件候选为 Pro/BAAI/bge-m3（SiliconFlow）。本地没有 SiliconFlow 凭据，
# 经用户确认改用百炼 OpenAI 兼容接口的 text-embedding-v4，复用 BAILIAN_*。
EMBED_MODEL_NAME = "text-embedding-v4"
EMBED_DIM = 1024  # 与 text-embedding-v4 的 dimensions 参数、Milvus schema 三者一致
EMBED_PROVIDER = "openai"  # init_embeddings 的 provider（OpenAI 兼容协议）
EMBED_SEND_DIMENSIONS = True  # text-embedding-v4 支持 dimensions；若换模型需核验
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))  # 百炼单次请求批量上限

# =========================
# 3. 切分与检索
# =========================

CHUNK_SIZE = 220
CHUNK_OVERLAP = 80
TOP_K = 5

# 2.1 章节切分与父块补全
CHUNKING_STRATEGY = "section-parent-v2"  # 多文档、稳定 chunk_key、来源定位
MIN_CHILD_CHARS = 5  # 去掉章节/小节标题后仍有少量有效文字即可入库
INDEX_MANIFEST_FILE = (PROJECT_DIR / "data" / "index_manifest.json").resolve()

# 2.3 知识更新、版本与删除
SCHEMA_VERSION = 2  # 版本 2 集合新增 doc_id/chunk_key/定位/版本字段
SOURCE_REGISTRY_FILE = (PROJECT_DIR / "data" / "sources.json").resolve()
MANIFEST_ARCHIVE_DIR = (PROJECT_DIR / "data" / "manifests").resolve()
COLLECTION_PREFIX = "docs_v"
DEFAULT_SOURCE = "knowledge.txt"
# 构建新集合后、切换前用固定问题抽查；任一问题缺少预期事实则拒绝切换。
BUILD_VERIFY_IDS = ("Q01", "Q03", "Q06")

# 2.2 混合检索与重排
# 默认仍为 vector，只有评估集证明混合检索改善后才切换为 hybrid。
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "vector")
HYBRID_SEMANTIC_K = int(os.getenv("HYBRID_SEMANTIC_K", "20"))
HYBRID_BM25_K = int(os.getenv("HYBRID_BM25_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
# 内存 BM25 需要一次性读取当前集合的全部子块；超过上限时明确失败，
# 不静默只索引一部分数据。更大数据量应改用服务端全文索引。
MAX_LEXICAL_RECORDS = int(os.getenv("MAX_LEXICAL_RECORDS", "10000"))
# none / llm；llm 会额外调用一次对话模型，先由评估集验证是否值得保留。
RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "none")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
RERANK_MAX_TEXT_CHARS = int(os.getenv("RERANK_MAX_TEXT_CHARS", "500"))

# 2.3 检索工具型 Agent
# 初次检索 + 最多两次重写/补充检索；预算来自产品对延迟与调用次数的取舍，
# 不是模型自行决定的无限循环。recursion_limit 只防失控，不替代业务预算。
TOOL_MAX_CALLS = int(os.getenv("TOOL_MAX_CALLS", "3"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "8"))

# 2.4 API 与页面
# 管理接口（上传/建索引）使用独立令牌；未配置时管理接口关闭，不复用聊天入口。
ADMIN_TOKEN_ENV = "ASSISTANT_ADMIN_TOKEN"
ADMIN_TOKEN = os.getenv(ADMIN_TOKEN_ENV)
API_MAX_INGEST_TASKS = int(os.getenv("API_MAX_INGEST_TASKS", "50"))
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "6"))


# =========================
# 4. 对话模型
# =========================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.8-max")
# 当前父工程实际使用百炼兼容接口；换成其他服务商时显式修改映射，不自动复用凭据。
LLM_API_KEY_ENV = "BAILIAN_API_KEY"
LLM_BASE_URL_ENV = "BAILIAN_API_URL"
LLM_API_KEY = os.getenv(LLM_API_KEY_ENV)
LLM_BASE_URL = os.getenv(LLM_BASE_URL_ENV)

# Embedding 与对话模型分开配置，不假设两者共用同一服务商或地址。
# 本机实际可用的 Embedding 凭据是百炼；变量名与对话模型相同，但模型配置分开维护。
EMBED_API_KEY_ENV = "BAILIAN_API_KEY"
EMBED_BASE_URL_ENV = "BAILIAN_API_URL"
EMBED_API_KEY = os.getenv(EMBED_API_KEY_ENV)
EMBED_BASE_URL = os.getenv(EMBED_BASE_URL_ENV)


# =========================
# 5. 启动校验
# =========================


def _validate_knowledge_file() -> None:
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"知识库文件不存在: {KNOWLEDGE_FILE}")
    try:
        KNOWLEDGE_FILE.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"知识库文件不是有效的 UTF-8: {KNOWLEDGE_FILE}") from exc
    except OSError as exc:
        raise OSError(f"知识库文件不可读: {KNOWLEDGE_FILE}") from exc


def _validate_params() -> None:
    problems = []
    if CHUNK_SIZE <= 0:
        problems.append("CHUNK_SIZE 必须大于 0")
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        problems.append("CHUNK_OVERLAP 必须满足 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
    if TOP_K <= 0:
        problems.append("TOP_K 必须大于 0")
    if EMBED_DIM <= 0:
        problems.append("EMBED_DIM 必须大于 0")
    if RETRIEVAL_STRATEGY not in {"vector", "hybrid"}:
        problems.append("RETRIEVAL_STRATEGY 必须为 vector 或 hybrid")
    if HYBRID_SEMANTIC_K <= 0 or HYBRID_BM25_K <= 0:
        problems.append("HYBRID_SEMANTIC_K/HYBRID_BM25_K 必须大于 0")
    if RRF_K < 0:
        problems.append("RRF_K 不能为负数")
    if BM25_K1 < 0:
        problems.append("BM25_K1 不能为负数")
    if not 0 <= BM25_B <= 1:
        problems.append("BM25_B 必须在 [0, 1] 之间")
    if MAX_LEXICAL_RECORDS <= 0:
        problems.append("MAX_LEXICAL_RECORDS 必须大于 0")
    if RERANK_STRATEGY not in {"none", "llm"}:
        problems.append("RERANK_STRATEGY 必须为 none 或 llm")
    if RERANK_CANDIDATES <= 0 or RERANK_MAX_TEXT_CHARS <= 0:
        problems.append("RERANK_CANDIDATES/RERANK_MAX_TEXT_CHARS 必须大于 0")
    if TOOL_MAX_CALLS <= 0 or AGENT_RECURSION_LIMIT <= 0:
        problems.append("TOOL_MAX_CALLS/AGENT_RECURSION_LIMIT 必须大于 0")
    if API_MAX_INGEST_TASKS <= 0 or SESSION_MAX_TURNS <= 0:
        problems.append("API_MAX_INGEST_TASKS/SESSION_MAX_TURNS 必须大于 0")
    if problems:
        raise ValueError("配置参数无效: " + "；".join(problems))


def missing_env_vars(*names: str) -> list[str]:
    """返回缺失的环境变量名，不返回值本身。"""
    return [name for name in names if not os.getenv(name)]


def validate_llm_config() -> None:
    """在真正初始化对话模型前调用；缺失时指出变量名。"""
    missing = missing_env_vars(LLM_API_KEY_ENV, LLM_BASE_URL_ENV)
    if not LLM_MODEL_NAME:
        missing.append("LLM_MODEL_NAME")
    if missing:
        raise RuntimeError("缺少对话模型配置: " + ", ".join(missing))


def validate_embedding_config() -> None:
    """在真正初始化 Embedding 前调用；缺失时指出变量名。"""
    missing = missing_env_vars(EMBED_API_KEY_ENV, EMBED_BASE_URL_ENV)
    if missing:
        raise RuntimeError("缺少 Embedding 配置: " + ", ".join(missing))


def validate_config(*, require_llm: bool = False, require_embedding: bool = False) -> None:
    """入口统一校验；默认只检查离线必需的路径与参数。"""
    _validate_knowledge_file()
    _validate_params()
    if require_llm:
        validate_llm_config()
    if require_embedding:
        validate_embedding_config()


# import 时完成离线校验：文件存在、UTF-8 可读、参数有效。
_validate_knowledge_file()
_validate_params()
