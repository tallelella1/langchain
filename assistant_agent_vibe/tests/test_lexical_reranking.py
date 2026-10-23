"""P1.2：中文 BM25、RRF 融合、LLM 重排与混合检索（离线替身）。"""

import pytest

import config
from rag import lexical, reranking, retrieval

MANIFEST = {
    "collection": "test_collection",
    "index_version": 3,
    "chunk_count": 3,
    "parents": {
        "knowledge.txt#p000": {
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
            "chapter": "六、退款规则",
            "section": "2. 基础版与专业版",
            "title": "2. 基础版与专业版",
            "plan": "基础版、专业版",
            "text": "退款规则：超过 10% 未超过 50% 退还 50%。",
            "child_ids": [0, 1],
        },
        "knowledge.txt#p001": {
            "parent_id": "knowledge.txt#p001",
            "source": "knowledge.txt",
            "chapter": "七、发票规则",
            "section": "4. 专票补充规则",
            "title": "4. 专票补充规则",
            "plan": "",
            "text": "增值税专用发票需要先完成红字发票流程。",
            "child_ids": [2],
        },
    },
    "children": {
        "knowledge.txt#c0": {
            "chunk_key": "knowledge.txt#c0",
            "chunk_id": 0,
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
            "text": "退款规则：超过 10% 未超过 50% 退还 50%。",
        },
        "knowledge.txt#c1": {
            "chunk_key": "knowledge.txt#c1",
            "chunk_id": 1,
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
            "text": "套餐升级部分不适用首购 7 日无理由退款。",
        },
        "knowledge.txt#c2": {
            "chunk_key": "knowledge.txt#c2",
            "chunk_id": 2,
            "parent_id": "knowledge.txt#p001",
            "source": "knowledge.txt",
            "text": "增值税专用发票需要先完成红字发票流程。",
        },
    },
}


class FakeEmbedder:
    def embed_query(self, text):
        return [0.5] * config.EMBED_DIM


class HybridClient:
    def __init__(self):
        self.rows = [
            {
                "id": child["chunk_id"],
                "chunk_id": child["chunk_id"],
                "chunk_key": child["chunk_key"],
                "text": child["text"],
                "source": child["source"],
            }
            for child in MANIFEST["children"].values()
        ]
        self.searches = 0

    def search(self, **kwargs):
        self.searches += 1
        return [
            [
                {
                    "id": 0,
                    "distance": 0.9,
                    "entity": {
                        "chunk_key": "knowledge.txt#c0",
                        "chunk_id": 0,
                        "text": MANIFEST["children"]["knowledge.txt#c0"]["text"],
                        "source": "knowledge.txt",
                    },
                }
            ]
        ]

    def query(self, **kwargs):
        if "count(*)" in kwargs.get("output_fields", []):
            return [{"count(*)": len(self.rows)}]
        return list(self.rows)


def test_tokenize_keeps_terms_numbers_and_percent():
    tokens = lexical.tokenize("增值税专用发票 50% API 超额 text-embedding-v4")
    assert "增值" in tokens and "发票" in tokens
    assert "50%" in tokens and "api" in tokens and "text-embedding-v4" in tokens
    assert "增值税" in tokens


def test_bm25_ranks_exact_terms_and_returns_nothing_for_unrelated():
    index = lexical.build_index(
        [
            ("a", "增值税专用发票需要红字发票流程。"),
            ("b", "退款规则：超过 10% 未超过 50% 退还 50%。"),
            ("c", "成员人数上限为 10 人。"),
        ]
    )
    hits = index.search("专票 红字发票", 3)
    assert hits and hits[0].key == "a"
    assert index.search("火星基地专线", 3) == []
    with pytest.raises(ValueError, match="查询"):
        index.search("  ", 3)


def test_rrf_fuse_uses_ranks_and_dedups():
    left = [{"chunk_key": "a"}, {"chunk_key": "b"}]
    right = [{"chunk_key": "b"}, {"chunk_key": "c"}]
    fused = reranking.rrf_fuse(
        [("semantic", left), ("bm25", right)], key=retrieval.hit_key, rrf_k=60
    )
    assert [hit["chunk_key"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["ranks"] == {"semantic": 2, "bm25": 1}
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
    assert [hit["rrf_rank"] for hit in fused] == [1, 2, 3]


class FakeRerankModel:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return type("Message", (), {"content": self.content})()


def test_parse_rerank_order_rejects_invalid_output():
    candidates = ["a", "b"]
    assert reranking.parse_rerank_order('{"order": ["b", "a"]}', candidates) == [
        "b",
        "a",
    ]
    for text in ("不是 JSON", '{"order": []}', '{"order": ["x"]}', '{"order": ["a", "a"]}'):
        with pytest.raises(reranking.RerankError):
            reranking.parse_rerank_order(text, candidates)


def test_llm_reranker_selects_subset_and_reports_invalid():
    hits = [{"chunk_key": "a", "text": "A"}, {"chunk_key": "b", "text": "B"}]
    model = FakeRerankModel('{"order": ["b"]}')
    selected = reranking.LLMReranker(model).rerank("问题", hits, top_n=2)
    assert [hit["chunk_key"] for hit in selected] == ["b"]
    assert selected[0]["rerank_rank"] == 1
    assert model.calls == 1

    broken = reranking.LLMReranker(FakeRerankModel("{}"))
    with pytest.raises(reranking.RerankError, match="order"):
        broken.rerank("问题", hits, top_n=2)


def test_hybrid_retrieve_fuses_vector_and_bm25(monkeypatch):
    retrieval.clear_lexical_cache()
    monkeypatch.setattr(config, "RETRIEVAL_STRATEGY", "hybrid")
    client = HybridClient()
    hits = retrieval.retrieve(
        "增值税专用发票怎么退款？",
        2,
        embedder=FakeEmbedder(),
        client=client,
        manifest=MANIFEST,
    )
    assert client.searches == 1
    assert {hit["parent_id"] for hit in hits} == {
        "knowledge.txt#p000",
        "knowledge.txt#p001",
    }
    assert all(hit["metric"] == "RRF" for hit in hits)
    assert any(hit.get("bm25_score", 0) > 0 for hit in hits)
    assert all(hit["rrf_score"] > 0 for hit in hits)
    assert any("semantic_score" in hit for hit in hits)


def test_hybrid_rejects_oversized_lexical_corpus(monkeypatch):
    retrieval.clear_lexical_cache()

    def too_large(*args, **kwargs):
        raise retrieval.vector_store.VectorStoreError(
            f"集合记录数超过内存 BM25 上限 {config.MAX_LEXICAL_RECORDS}"
        )

    monkeypatch.setattr(retrieval.vector_store, "fetch_all_records", too_large)
    with pytest.raises(retrieval.RetrievalError, match="上限"):
        retrieval.retrieve(
            "问题",
            2,
            strategy="hybrid",
            embedder=FakeEmbedder(),
            client=HybridClient(),
            manifest=MANIFEST,
        )
