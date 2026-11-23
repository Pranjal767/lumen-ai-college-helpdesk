"""Tests for embeddings, persisted indexing and cosine search."""

import json

import numpy as np
import pytest

from backend.database.db import get_connection, initialize_database
from backend.database.models import (
    insert_document_version,
    insert_source,
    upsert_document,
)
from backend.services.chunker import chunk_document_version
from backend.services.embeddings import EmbeddingError, SentenceTransformerEmbedder
from backend.services.vector_store import (
    MAPPING_FILE_NAME,
    VectorIndexError,
    build_vector_index,
    is_vector_index_current,
    search_chunks,
)


class FakeEmbedder:
    model_name = "test-embedding-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "exam" in lowered:
                vectors.append([3.0, 0.0])
            elif "fee" in lowered:
                vectors.append([0.0, 4.0])
            else:
                vectors.append([1.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeSentenceModel:
    def __init__(self, values):
        self.values = values
        self.options = None

    def encode(
        self,
        sentences,
        *,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.options = (
            sentences,
            convert_to_numpy,
            normalize_embeddings,
            show_progress_bar,
        )
        return self.values


def create_chunked_corpus(database_path):
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        source_id = insert_source(
            connection,
            name="College notices",
            url="https://bkbck.edu.in/",
            category="notices",
        )
        document_specs = [
            (
                "Examination notice",
                "https://bkbck.edu.in/notices/examination.pdf",
                "pdf",
                "exam form deadline\fexam timetable",
            ),
            (
                "Fee schedule",
                "https://bkbck.edu.in/notices/fees.html",
                "html",
                "fee payment details and receipt",
            ),
        ]
        version_ids = []
        for title, url, document_type, text in document_specs:
            document_id = upsert_document(
                connection,
                source_id=source_id,
                title=title,
                source_url=url,
                document_type=document_type,
                category="notices",
                content_hash=f"hash-{title}",
                file_path=f"data/raw/{title}",
                fetched_at="2026-11-17T00:00:00+00:00",
                status="fetched",
            )
            version_ids.append(
                insert_document_version(
                    connection,
                    document_id=document_id,
                    version_number=1,
                    content_hash=f"hash-{title}",
                    file_path=f"data/raw/{title}",
                    extracted_text=text,
                    extraction_status="extracted",
                    created_at="2026-11-17T00:00:00+00:00",
                )
            )

    for version_id in version_ids:
        chunk_document_version(
            version_id,
            chunk_size=20,
            overlap=0,
            database_path=database_path,
        )
    return version_ids


def test_sentence_transformer_embedder_validates_and_normalizes_output_shape():
    model = FakeSentenceModel([[1.0, 2.0], [3.0, 4.0]])
    embedder = SentenceTransformerEmbedder("test-model", model=model)

    vectors = embedder.encode(["first text", "second text"])

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    assert model.options == (["first text", "second text"], True, True, False)


def test_sentence_transformer_embedder_rejects_bad_model_output():
    embedder = SentenceTransformerEmbedder(
        "test-model",
        model=FakeSentenceModel([[1.0, 2.0]]),
    )

    with pytest.raises(EmbeddingError, match="invalid shape"):
        embedder.encode(["first", "second"])


def test_build_and_search_returns_ranked_citations(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    create_chunked_corpus(database_path)

    summary = build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    results = search_chunks(
        "exam form",
        top_k=3,
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )

    assert summary.vector_count == 3
    assert summary.dimensions == 2
    assert summary.model_name == FakeEmbedder.model_name
    assert results[0].title == "Examination notice"
    assert results[0].source_url.endswith("examination.pdf")
    assert results[0].page_number == 1
    assert results[0].document_version == 1
    assert results[0].score == pytest.approx(1.0)
    assert results[1].title == "Examination notice"
    assert results[1].page_number == 2
    assert results[2].title == "Fee schedule"


def test_index_current_check_detects_missing_stale_and_wrong_model_indexes(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    version_ids = create_chunked_corpus(database_path)

    assert not is_vector_index_current(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    assert is_vector_index_current(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )

    different_embedder = FakeEmbedder()
    different_embedder.model_name = "different-model"
    assert not is_vector_index_current(
        database_path=database_path,
        index_directory=index_directory,
        embedder=different_embedder,
    )

    chunk_document_version(
        version_ids[0],
        chunk_size=2,
        overlap=0,
        database_path=database_path,
    )
    assert not is_vector_index_current(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )


def test_rebuild_keeps_a_stable_chunk_id_mapping(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    create_chunked_corpus(database_path)

    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    mapping_before = json.loads(
        (index_directory / MAPPING_FILE_NAME).read_text(encoding="utf-8")
    )["chunk_ids"]
    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    mapping_after = json.loads(
        (index_directory / MAPPING_FILE_NAME).read_text(encoding="utf-8")
    )["chunk_ids"]

    assert mapping_before == mapping_after
    assert mapping_before == sorted(mapping_before)


def test_search_rejects_an_index_after_chunk_data_changes(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    version_ids = create_chunked_corpus(database_path)
    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    chunk_document_version(
        version_ids[0],
        chunk_size=2,
        overlap=0,
        database_path=database_path,
    )

    with pytest.raises(VectorIndexError, match="Chunks changed"):
        search_chunks(
            "exam",
            database_path=database_path,
            index_directory=index_directory,
            embedder=FakeEmbedder(),
        )


def test_search_rejects_a_different_embedding_model(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    create_chunked_corpus(database_path)
    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    different_embedder = FakeEmbedder()
    different_embedder.model_name = "different-model"

    with pytest.raises(VectorIndexError, match="differs from the index model"):
        search_chunks(
            "exam",
            database_path=database_path,
            index_directory=index_directory,
            embedder=different_embedder,
        )


def test_search_rejects_corrupted_index_files(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    create_chunked_corpus(database_path)
    build_vector_index(
        database_path=database_path,
        index_directory=index_directory,
        embedder=FakeEmbedder(),
    )
    index_path = index_directory / "chunks.npz"
    index_path.write_bytes(index_path.read_bytes() + b"corruption")

    with pytest.raises(VectorIndexError, match="corrupt or incompatible"):
        search_chunks(
            "exam",
            database_path=database_path,
            index_directory=index_directory,
            embedder=FakeEmbedder(),
        )


def test_empty_corpus_and_invalid_search_arguments_are_reported(tmp_path):
    database_path = tmp_path / "data" / "test.db"
    index_directory = tmp_path / "vector_index"
    initialize_database(database_path)

    with pytest.raises(ValueError, match="No stored chunks"):
        build_vector_index(
            database_path=database_path,
            index_directory=index_directory,
            embedder=FakeEmbedder(),
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        search_chunks(
            "  ",
            database_path=database_path,
            index_directory=index_directory,
            embedder=FakeEmbedder(),
        )
