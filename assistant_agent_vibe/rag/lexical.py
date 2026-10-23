"""中文友好的 BM25 关键词索引（P1.2，纯离线）。

分词约定（不引入额外依赖）：
- 连续中文按 bigram + trigram 展开，覆盖“退款”“红字发票”“增值税”等术语；
  不使用 unigram，避免“火星基地”仅凭单个“专”字命中“专用发票”；
- 英文、数字、百分号按连续串保留，例如 ``API``、``50%``、``text-embedding-v4``；
- 标点与空白只作分隔符。

索引更新方式：BM25 是当前集合子块的内存快照，只在新集合构建完成后重建；
不支持在同一集合内增量更新。调用方按 ``collection``、``index_version`` 与
记录数缓存索引，索引切换后自然失效。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

CJK_RUN = re.compile(r"[㐀-䶿一-鿿]+")
ASCII_TOKEN = re.compile(r"[A-Za-z0-9%]+(?:[._+\-/][A-Za-z0-9%]+)*")


def tokenize(text: str) -> list[str]:
    """把文本切成检索词；保留重复词供 BM25 计算词频。"""
    if not isinstance(text, str):
        raise ValueError("待分词文本必须是字符串")
    tokens: list[str] = []
    for run in CJK_RUN.findall(text):
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
    for token in ASCII_TOKEN.findall(text):
        lowered = token.lower()
        if lowered:
            tokens.append(lowered)
    return tokens


@dataclass(frozen=True)
class LexicalHit:
    key: str
    score: float


class BM25Index:
    """小型 BM25 实现；数据量受 ``MAX_LEXICAL_RECORDS`` 限制。"""

    def __init__(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("BM25 k1 不能为负数")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b 必须在 [0, 1] 之间")
        keys: list[str] = []
        frequencies: list[Counter[str]] = []
        lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for key, text in documents:
            if not isinstance(key, str) or not key:
                raise ValueError("BM25 文档 key 不能为空")
            tokens = tokenize(text)
            counter = Counter(tokens)
            keys.append(key)
            frequencies.append(counter)
            lengths.append(len(tokens))
            document_frequency.update(counter.keys())

        self.keys = keys
        self._frequencies = frequencies
        self._lengths = lengths
        self._k1 = k1
        self._b = b
        self._count = len(keys)
        self._avg_length = (
            sum(lengths) / self._count if self._count else 0.0
        )
        self._idf = {
            token: math.log(1 + (self._count - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }

    @property
    def size(self) -> int:
        return self._count

    def search(self, query: str, k: int) -> list[LexicalHit]:
        """返回得分大于 0 的前 k 个文档；同分按文档原始顺序稳定排序。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("关键词查询不能为空")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k 必须是正整数，实际 {k!r}")
        query_tokens = tokenize(query)
        if not query_tokens or not self._count:
            return []

        scored: list[LexicalHit] = []
        for index, frequencies in enumerate(self._frequencies):
            length = self._lengths[index]
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / (self._avg_length or 1.0)
                )
                score += self._idf.get(token, 0.0) * (
                    frequency * (self._k1 + 1) / denominator
                )
            if score > 0:
                scored.append(LexicalHit(self.keys[index], score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]


def build_index(
    documents: Iterable[tuple[str, str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    return BM25Index(list(documents), k1=k1, b=b)
