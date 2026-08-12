from app.pipelines import ingest
from app.splitter.chunker import Chunk


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"c{index}",
        doc_id="doc-batch",
        page=1,
        section_path="section",
        chunk_type="text",
        content=f"content {index}",
        token_count=2,
    )


def test_embedding_and_index_batches_are_bounded(monkeypatch):
    saved = {}
    calls = []

    class FakeEmbedder:
        def embed_texts_with_sparse(self, texts, text_type):
            calls.append(len(texts))
            return [[float(i)] for i in range(len(texts))], [
                {"indices": [i], "values": [1.0]} for i in range(len(texts))
            ]

    monkeypatch.setattr(ingest, "save_stage", lambda _doc, name, data: saved.__setitem__(name, data))
    monkeypatch.setattr(ingest, "load_stage", lambda _doc, name: saved.get(name))
    chunks = [_chunk(i) for i in range(ingest.VECTOR_BATCH_SIZE + 3)]

    manifest = ingest._embed_chunks_batched("doc-batch", chunks, "report", FakeEmbedder())
    batches = list(ingest._iter_vector_batches("doc-batch", chunks, manifest))

    assert calls == [ingest.VECTOR_BATCH_SIZE, 3]
    assert manifest["count"] == len(chunks)
    assert [len(batch[0]) for batch in batches] == [ingest.VECTOR_BATCH_SIZE, 3]
    assert sum(len(batch[1]) for batch in batches) == len(chunks)


def test_legacy_vector_stage_remains_readable():
    chunks = [_chunk(i) for i in range(3)]
    stage = {"vectors": [[1.0], [2.0], [3.0]], "sparse": None}

    batches = list(ingest._iter_vector_batches("doc-batch", chunks, stage))

    assert len(batches) == 1
    assert batches[0][1] == stage["vectors"]
