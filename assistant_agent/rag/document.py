import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 直接运行本文件时，先把项目目录加入 sys.path，才能导入统一配置。
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402  （必须在 sys.path 调整之后导入）

print("Python:", sys.executable)
print("langchain-community:", version("langchain-community"))
print("langchain-text-splitters:", version("langchain-text-splitters"))

# 入口统一校验：知识库存在、UTF-8 可读、参数有效。
config.validate_config()
print("知识库文件:", config.KNOWLEDGE_FILE)
print("父工程 .env:", config.ENV_FILE, "存在:", config.ENV_FILE.is_file())
print("资料目录存在:", config.LOAD_DIR.is_dir())


def normalize_source(source: str | Path) -> str:
    """把加载器给出的路径规范为项目内相对路径（统一使用 /）。"""
    path = Path(source).resolve()
    try:
        return path.relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


# 先加载 asset/load 下的示例文本，再加载 knowledge.txt；顺序固定，便于复现。
file_paths = [*sorted(config.LOAD_DIR.glob("*.txt")), config.KNOWLEDGE_FILE]

documents = []
for file_path in file_paths:
    # 已知示例中只有这个文件使用 GBK，其余文本使用 UTF-8。
    encoding = "gbk" if file_path.name == "01-langchain-gbk.txt" else "utf-8"
    loader = TextLoader(file_path=file_path, encoding=encoding)
    loaded = loader.load()
    documents.extend(loaded)
    print(f"已加载 {normalize_source(file_path)}: {len(loaded)} 个 Document")


print(
    f"切分参数: chunk_size={config.CHUNK_SIZE}, "
    f"chunk_overlap={config.CHUNK_OVERLAP}, top_k={config.TOP_K}"
)

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""],
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    keep_separator="end",
    add_start_index=True,
)

chunks = text_splitter.split_documents(documents)

# 先记录正文，之后只补元数据，确保原文和顺序不变。
original_texts = [chunk.page_content for chunk in chunks]

# 对每块补 source 与 chunk_id：source 为项目相对路径，chunk_id 为来源文件内的切片序号。
chunk_counts = Counter()
for chunk in chunks:
    source = chunk.metadata.get("source")
    if not source:
        raise ValueError("切片缺少 source 元数据")
    source = normalize_source(source)
    chunk.metadata["source"] = source
    chunk.metadata["chunk_id"] = chunk_counts[source]
    chunk_counts[source] += 1

assert [chunk.page_content for chunk in chunks] == original_texts, "切片正文或顺序被改动"

keys = [(chunk.metadata["source"], chunk.metadata["chunk_id"]) for chunk in chunks]
assert len(keys) == len(set(keys)), "(source, chunk_id) 存在重复"

empty_chunks = [chunk for chunk in chunks if not chunk.page_content.strip()]
lengths = [len(chunk.page_content) for chunk in chunks]

print(f"\n共切分出 {len(chunks)} 个 chunk，来源文件 {len(chunk_counts)} 个")
print(f"块长度 min/avg/max: {min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}")
print("各来源块数:")
for source, count in chunk_counts.items():
    print(f"  {source}: {count}")
print("空块数:", len(empty_chunks))

# 打印切分结果供调试；每块均带 source 与 chunk_id，正文保持原样。
print("\n=== 全部切分结果 ===")
for chunk in chunks:
    print(
        f"\n--- {chunk.metadata['source']} | chunk_id={chunk.metadata['chunk_id']} "
        f"| start_index={chunk.metadata.get('start_index')} "
        f"| len={len(chunk.page_content)} ---"
    )
    print(chunk.page_content)
