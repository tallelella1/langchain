"""P1.3：版本化索引、未变化复用、删除、切换失败与回滚。"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from pymilvus import MilvusClient

import config
from rag import documents, indexer, sources, vector_store


class FakeEmbedder:
    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % 32] / 255.0 for index in range(config.EMBED_DIM)]


class FakeClient:
    def __init__(self, existing=()):
        self.existing = set(existing)

    def has_collection(self, collection_name):
        return collection_name in self.existing


def test_next_index_version_skips_occupied_names():
    client = FakeClient({"docs_v3", "docs_v4"})
    version, collection = indexer.next_index_version(
        client, {"collection": "docs_v2", "index_version": 2}, prefix="docs_v"
    )
    assert (version, collection) == (5, "docs_v5")


def test_document_versions_and_unchanged_detection(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("第一段知识。\n第二段知识。\n", encoding="utf-8")
    entries = sources.entries_from_paths([source])
    loaded = documents.load_sources_documents(entries)
    _, _, manifest = documents.load_and_split(
        [source], document_versions={"a.txt": 1}, index_version=1
    )
    assert indexer.is_unchanged(loaded, manifest)
    versions = indexer.document_versions(loaded, manifest)
    assert versions == {loaded[0].source: 1}

    source.write_text("第一段知识。\n", encoding="utf-8")
    loaded = documents.load_sources_documents(entries)
    assert not indexer.is_unchanged(loaded, manifest)
    assert indexer.document_versions(loaded, manifest) == {loaded[0].source: 2}


def test_build_records_v2_contains_version_and_location():
    _, chunks, manifest = documents.load_and_split()
    vectors = [FakeEmbedder._vector(chunk.page_content) for chunk in chunks]
    records = indexer.build_records_v2(chunks, vectors, manifest)
    assert len(records) == len(chunks)
    first = records[0]
    assert first["chunk_key"] == chunks[0].metadata["chunk_key"]
    assert first["doc_id"] == chunks[0].metadata["doc_id"]
    assert first["content_hash"] == manifest["documents"][first["doc_id"]]["content_hash"]
    assert first["index_version"] == manifest["index_version"]
    assert first["page"] == chunks[0].metadata.get("page", -1)


def _milvus_available() -> bool:
    try:
        client = MilvusClient(uri=config.MILVUS_URI, timeout=3)
        client.list_databases()
        client.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_available(), reason=f"Milvus 不可达: {config.MILVUS_URI}"
)


def test_versioned_build_update_delete_switch_failure_and_rollback(tmp_path):
    prefix = f"itv_{uuid.uuid4().hex[:8]}"
    manifest_path = tmp_path / "index_manifest.json"
    archive_dir = tmp_path / "manifests"
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.md"
    source_c = tmp_path / "c.txt"
    source_a.write_text(
        "第一段：基础版成员上限 10 人。\n第二段：外部协作者不占名额。\n",
        encoding="utf-8",
    )
    source_b.write_text(
        "# 退款规则\n\n## 专票\n\n增值税专用发票需要先完成红字发票流程。\n",
        encoding="utf-8",
    )
    source_c.write_text("企业版私有化部署需要单独签约。\n", encoding="utf-8")

    client = vector_store.open_target(create_database=True)
    embedder = FakeEmbedder()
    entries = sources.entries_from_paths([source_a, source_b])
    created: list[str] = []

    def build(active_entries, *, force=False):
        result = indexer.build_index(
            active_entries,
            force=force,
            client=client,
            embedder=embedder,
            manifest_path=manifest_path,
            archive_dir=archive_dir,
            collection_prefix=prefix,
            verify=False,
        )
        created.append(result.collection)
        return result

    try:
        first = build(entries)
        assert not first.reused
        assert first.collection == f"{prefix}1"
        assert first.record_count == first.manifest["chunk_count"] > 0
        assert {doc["source"] for doc in first.manifest["documents"].values()} == {
            item["source"] for item in entries
        }
        assert documents.load_index_manifest(manifest_path)["collection"] == first.collection

        second = build(entries)
        assert second.reused
        assert second.collection == first.collection

        source_a.write_text("第一段：基础版成员上限 10 人。\n", encoding="utf-8")
        shorter = build(entries)
        assert not shorter.reused
        assert shorter.collection == f"{prefix}2"
        doc_a = next(
            doc
            for doc in shorter.manifest["documents"].values()
            if doc["source"].endswith("a.txt")
        )
        assert doc_a["document_version"] == 2
        old_text = "外部协作者不占名额"
        assert all(
            old_text not in child["text"]
            for child in shorter.manifest["children"].values()
        )
        assert vector_store.count_records(client, shorter.collection) == shorter.record_count

        entries_three = sources.entries_from_paths([source_a, source_b, source_c])
        third = build(entries_three)
        assert third.collection == f"{prefix}3"
        assert len(third.manifest["documents"]) == 3

        entries_two = sources.entries_from_paths([source_a, source_c])
        fourth = build(entries_two)
        assert fourth.collection == f"{prefix}4"
        assert len(fourth.manifest["documents"]) == 2
        assert all(
            not doc["source"].endswith("b.md")
            for doc in fourth.manifest["documents"].values()
        )

        original_model = config.EMBED_MODEL_NAME
        config.EMBED_MODEL_NAME = "fake-embed-v2"
        try:
            changed = build(entries_two)
        finally:
            config.EMBED_MODEL_NAME = original_model
        assert changed.collection == f"{prefix}5"
        assert changed.manifest["embedding_model"] == "fake-embed-v2"

        # 换回原模型再建一版，保证回滚目标与当前查询模型一致。
        restored = build(entries_two)
        assert restored.collection == f"{prefix}6"
        assert restored.manifest["embedding_model"] == original_model

        before = manifest_path.read_bytes()
        real_save = documents.save_index_manifest

        def fail_switch(manifest, path=None):
            target = Path(path or config.INDEX_MANIFEST_FILE)
            if target == manifest_path:
                raise OSError("simulated switch failure")
            return real_save(manifest, path)

        documents.save_index_manifest = fail_switch
        try:
            with pytest.raises(indexer.BuildError, match="切换失败"):
                build(entries_two, force=True)
        finally:
            documents.save_index_manifest = real_save
        assert manifest_path.read_bytes() == before
        active_after_failure = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert active_after_failure["collection"] == f"{prefix}6"

        retried = build(entries_two, force=True)
        assert retried.collection == f"{prefix}8"  # v7 是失败构建留下的未引用集合

        rolled_back = indexer.rollback(
            client=client,
            manifest_path=manifest_path,
            archive_dir=archive_dir,
        )
        assert rolled_back["collection"] == f"{prefix}6"
        assert rolled_back["stale_sources"] is True
        assert (
            json.loads(manifest_path.read_text(encoding="utf-8"))["collection"]
            == f"{prefix}6"
        )

        kept, candidates = indexer.prune_plan(
            keep=1, manifest_path=manifest_path, archive_dir=archive_dir
        )
        assert f"{prefix}6" in kept
        assert f"{prefix}6" not in candidates
    finally:
        for collection in client.list_collections():
            if collection.startswith(prefix):
                client.drop_collection(collection_name=collection)
        client.close()
