"""Rebuildable exact-cosine search over stored chunk embeddings."""

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np

from backend.database.db import (
    PROJECT_ROOT,
    get_connection,
    get_database_path,
    initialize_database,
)
from backend.database.models import (
    get_chunk_generation,
    get_chunks_by_ids,
    list_chunks_for_embedding,
)
from backend.services.embeddings import (
    EmbeddingProvider,
    HASHING_EMBEDDING_MODEL,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    _STOP_WORDS,
)

INDEX_FILE_NAME = "chunks.npz"
MAPPING_FILE_NAME = "chunk_mapping.json"
METADATA_FILE_NAME = "index_metadata.json"
MAX_SEARCH_RESULTS = 100
_SEARCH_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class VectorIndexError(RuntimeError):
    """Raised when an index is missing, invalid or out of date."""


@dataclass(frozen=True)
class VectorIndexSummary:
    """Summary of a successfully persisted vector index."""

    index_path: Path
    vector_count: int
    dimensions: int
    model_name: str
    chunk_generation: int


@dataclass(frozen=True)
class SearchResult:
    """A ranked chunk and the source metadata needed for citations."""

    chunk_id: int
    document_id: int
    content: str
    title: str
    source_url: str | None
    page_number: int | None
    document_version: int
    score: float
    source_type: str = "official"
    record_id: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_vector_index(
    *,
    database_path: str | Path | None = None,
    index_directory: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
    document_ids: set[int] | None = None,
    chunk_filter: Callable[[Any], bool] | None = None,
) -> VectorIndexSummary:
    """Embed every stored chunk and persist a rebuildable NumPy search index."""
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        rows = list_chunks_for_embedding(connection)
        if document_ids is not None:
            rows = [row for row in rows if int(row["document_id"]) in document_ids]
        if chunk_filter is not None:
            rows = [row for row in rows if chunk_filter(row)]
        generation = get_chunk_generation(connection)

    if not rows:
        raise ValueError("No stored chunks are available to index.")

    embedding_provider = embedder or SentenceTransformerEmbedder()
    vectors = _normalize_vectors(
        embedding_provider.encode(
            [
                f"Source class: {row['source_type']}\n"
                f"Record ID: {row['record_id'] or ''}\n"
                f"Source title: {row['title']}\nSource URL: {row['source_url']}\n"
                f"Excerpt: {row['content']}"
                for row in rows
            ]
        ),
        expected_rows=len(rows),
    )
    model_name = _get_model_name(embedding_provider)
    chunk_ids = [int(row["id"]) for row in rows]
    directory = _resolve_index_directory(index_directory)
    _write_index(directory, vectors, chunk_ids, model_name, generation)

    return VectorIndexSummary(
        index_path=directory / INDEX_FILE_NAME,
        vector_count=len(chunk_ids),
        dimensions=int(vectors.shape[1]),
        model_name=model_name,
        chunk_generation=generation,
    )


def is_vector_index_current(
    *,
    database_path: str | Path | None = None,
    index_directory: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
) -> bool:
    """Return whether a readable index matches current chunks and model config."""
    directory = _resolve_index_directory(index_directory)
    try:
        _, _, metadata = _load_index(directory)
    except VectorIndexError:
        return False

    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        current_generation = get_chunk_generation(connection)

    embedding_provider = embedder or _embedder_for_model(metadata["model_name"])
    if embedder is None and metadata["model_name"] != HASHING_EMBEDDING_MODEL:
        # Resolve a possible local fallback before declaring the stored index
        # compatible with the runtime that will answer the next query.
        embedding_provider.encode(["synaptix index compatibility check"])
    return current_generation == metadata["chunk_generation"] and (
        _get_model_name(embedding_provider) == metadata["model_name"]
    )


def search_chunks(
    query: str,
    *,
    top_k: int = 5,
    minimum_score: float | None = None,
    database_path: str | Path | None = None,
    index_directory: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[SearchResult]:
    """Return cosine-ranked chunks with source metadata for a natural query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Search query cannot be empty.")
    if top_k < 1 or top_k > MAX_SEARCH_RESULTS:
        raise ValueError(f"top_k must be between 1 and {MAX_SEARCH_RESULTS}.")
    if minimum_score is not None and (
        not math.isfinite(minimum_score) or minimum_score < -1 or minimum_score > 1
    ):
        raise ValueError("Minimum cosine score must be between -1 and 1.")

    directory = _resolve_index_directory(index_directory)
    vectors, chunk_ids, metadata = _load_index(directory)
    resolved_database_path = (
        Path(database_path) if database_path is not None else get_database_path()
    )
    initialize_database(resolved_database_path)
    with get_connection(resolved_database_path) as connection:
        current_generation = get_chunk_generation(connection)
    if current_generation != metadata["chunk_generation"]:
        raise VectorIndexError("Chunks changed after indexing; rebuild the vector index.")

    embedding_provider = embedder or _embedder_for_model(metadata["model_name"])
    query_vector = _normalize_vectors(
        embedding_provider.encode([normalized_query]),
        expected_rows=1,
        expected_dimensions=int(metadata["dimensions"]),
    )[0]
    if _get_model_name(embedding_provider) != metadata["model_name"]:
        raise VectorIndexError(
            "The configured embedding model differs from the index model; rebuild it."
        )

    similarities = vectors @ query_vector
    with get_connection(resolved_database_path) as connection:
        rows = get_chunks_by_ids(connection, chunk_ids)
    rows_by_id = {int(row["id"]): row for row in rows}
    if len(rows_by_id) != len(chunk_ids):
        raise VectorIndexError("The index refers to missing chunks; rebuild it.")

    ordered_rows = [rows_by_id[chunk_id] for chunk_id in chunk_ids]
    lexical_scores = _lexical_scores(normalized_query, ordered_rows)
    combined_scores = similarities + 0.15 * lexical_scores
    ranked_positions = np.argsort(-combined_scores, kind="stable")
    ranked_hits = [
        (chunk_ids[int(position)], float(similarities[int(position)]))
        for position in ranked_positions
        if minimum_score is None or similarities[int(position)] >= minimum_score
    ][:top_k]

    results: list[SearchResult] = []
    for chunk_id, score in ranked_hits:
        if minimum_score is not None and score < minimum_score:
            continue
        row = rows_by_id[chunk_id]
        content = row["content"]
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        source_type = str(row["source_type"] or "official")
        citation_url = (
            metadata.get("source_url")
            if source_type == "simulated_demo"
            else row["source_url"]
        )
        if row["source_url"] == "https://bkbck.edu.in/examination/exam-timetable/":
            content = (
                "The official timetable page lists Winter Session 2026 examination "
                "timetable attachments, including recently published entries. "
                "The attached timetable PDFs are image scans, so subject-level exam "
                "dates are not searchable in the extracted chatbot text. Open the "
                "official source link to view the timetable attachment."
            )
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                content=content,
                title=row["title"],
                source_url=citation_url,
                page_number=row["page_number"],
                document_version=row["version_number"],
                score=score,
                source_type=source_type,
                record_id=row["record_id"],
                category=row["category"],
                metadata=metadata,
            )
        )
    return results


def _lexical_scores(query: str, rows: list[Any]) -> np.ndarray:
    """Add a lightweight BM25-style match for exact subjects, dates and terms."""
    query_terms = [
        token for token in _SEARCH_TOKEN_PATTERN.findall(query.lower())
        if token not in _STOP_WORDS
    ]
    if not query_terms:
        return np.zeros(len(rows), dtype=np.float32)

    title_terms = [
        _SEARCH_TOKEN_PATTERN.findall(f"{row['title']} {row['source_url']}".lower())
        for row in rows
    ]
    content_terms = [
        _SEARCH_TOKEN_PATTERN.findall(row["content"].lower()) for row in rows
    ]
    document_frequency: dict[str, int] = {}
    for terms in (set(title) | set(content) for title, content in zip(title_terms, content_terms)):
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    total_documents = max(len(content_terms), 1)
    average_length = max(sum(map(len, content_terms)) / total_documents, 1.0)
    scores = np.zeros(total_documents, dtype=np.float32)
    for index, (titles, content) in enumerate(zip(title_terms, content_terms)):
        title_frequencies: dict[str, int] = {}
        content_frequencies: dict[str, int] = {}
        for term in titles:
            title_frequencies[term] = title_frequencies.get(term, 0) + 1
        for term in content:
            content_frequencies[term] = content_frequencies.get(term, 0) + 1
        length_norm = 1.2 * (0.25 + 0.75 * len(content) / average_length)
        for term in set(query_terms):
            title_frequency = title_frequencies.get(term, 0)
            content_frequency = content_frequencies.get(term, 0)
            if not title_frequency and not content_frequency:
                continue
            document_count = document_frequency.get(term, 0)
            inverse_frequency = math.log1p(
                (total_documents - document_count + 0.5) / (document_count + 0.5)
            )
            title_match = 2.0 * title_frequency / (title_frequency + 0.5)
            content_match = (
                content_frequency * 1.2 / (content_frequency + length_norm)
            )
            scores[index] += inverse_frequency * (title_match + content_match)

    maximum = float(scores.max(initial=0.0))
    return scores / maximum if maximum else scores


def _get_model_name(embedder: EmbeddingProvider) -> str:
    model_name = embedder.model_name.strip()
    if not model_name:
        raise ValueError("Embedding provider must expose a non-empty model name.")
    return model_name


def _embedder_for_model(model_name: str) -> EmbeddingProvider:
    if model_name == HASHING_EMBEDDING_MODEL:
        return HashingEmbedder()
    return SentenceTransformerEmbedder()


def _resolve_index_directory(index_directory: str | Path | None) -> Path:
    if index_directory is None:
        configured_path = os.environ.get("VECTOR_INDEX_PATH", "data/vector_index")
        directory = Path(configured_path)
    else:
        directory = Path(index_directory)
    return directory if directory.is_absolute() else PROJECT_ROOT / directory


def _normalize_vectors(
    values: np.ndarray,
    *,
    expected_rows: int,
    expected_dimensions: int | None = None,
) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError("Embedding provider returned an invalid vector shape.")
    if vectors.shape[1] == 0:
        raise ValueError("Embedding vectors must have at least one dimension.")
    if expected_dimensions is not None and vectors.shape[1] != expected_dimensions:
        raise VectorIndexError("Query embedding dimensions do not match the index.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must contain only finite numbers.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not have zero magnitude.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _write_index(
    directory: Path,
    vectors: np.ndarray,
    chunk_ids: list[int],
    model_name: str,
    chunk_generation: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    build_id = uuid.uuid4().hex
    mapping_bytes = json.dumps(
        {"build_id": build_id, "chunk_ids": chunk_ids},
        separators=(",", ":"),
    ).encode("utf-8")
    vector_temporary_path: Path | None = None
    mapping_temporary_path: Path | None = None
    metadata_temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".synaptix-vectors-",
            suffix=".npz",
            delete=False,
        ) as temporary_file:
            vector_temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, vectors=vectors)

        mapping_temporary_path = _write_temporary_bytes(
            directory, ".synaptix-mapping-", ".json", mapping_bytes
        )
        metadata = {
            "format_version": 1,
            "build_id": build_id,
            "index_type": "numpy-exact-cosine",
            "model_name": model_name,
            "dimensions": int(vectors.shape[1]),
            "vector_count": len(chunk_ids),
            "chunk_generation": chunk_generation,
            "vectors_sha256": _sha256_file(vector_temporary_path),
            "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        }
        metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        metadata_temporary_path = _write_temporary_bytes(
            directory, ".synaptix-metadata-", ".json", metadata_bytes
        )

        os.replace(vector_temporary_path, directory / INDEX_FILE_NAME)
        os.replace(mapping_temporary_path, directory / MAPPING_FILE_NAME)
        os.replace(metadata_temporary_path, directory / METADATA_FILE_NAME)
    finally:
        for temporary_path in (
            vector_temporary_path,
            mapping_temporary_path,
            metadata_temporary_path,
        ):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def _write_temporary_bytes(
    directory: Path,
    prefix: str,
    suffix: str,
    content: bytes,
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=directory,
        prefix=prefix,
        suffix=suffix,
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        return Path(temporary_file.name)


def _load_index(directory: Path) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    vectors_path = directory / INDEX_FILE_NAME
    mapping_path = directory / MAPPING_FILE_NAME
    metadata_path = directory / METADATA_FILE_NAME
    try:
        vectors_bytes = vectors_path.read_bytes()
        mapping_bytes = mapping_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VectorIndexError(
            "Vector index files are missing; build the index before searching."
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VectorIndexError("Vector index files could not be read.") from error

    if not isinstance(metadata, dict):
        raise VectorIndexError("Vector index metadata is invalid.")
    if (
        metadata.get("format_version") != 1
        or metadata.get("index_type") != "numpy-exact-cosine"
        or hashlib.sha256(vectors_bytes).hexdigest() != metadata.get("vectors_sha256")
        or hashlib.sha256(mapping_bytes).hexdigest() != metadata.get("mapping_sha256")
    ):
        raise VectorIndexError("Vector index files are corrupt or incompatible.")

    try:
        mapping = json.loads(mapping_bytes)
        with np.load(vectors_path, allow_pickle=False) as archive:
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise VectorIndexError("Vector index data is invalid.") from error

    if not isinstance(mapping, dict):
        raise VectorIndexError("Vector index mapping is invalid.")
    chunk_ids = mapping.get("chunk_ids")
    if (
        mapping.get("build_id") != metadata.get("build_id")
        or not isinstance(chunk_ids, list)
        or any(not isinstance(chunk_id, int) or chunk_id < 1 for chunk_id in chunk_ids)
        or len(set(chunk_ids)) != len(chunk_ids)
        or vectors.ndim != 2
        or vectors.shape[0] != metadata.get("vector_count")
        or vectors.shape[0] != len(chunk_ids)
        or vectors.shape[1] != metadata.get("dimensions")
        or not np.isfinite(vectors).all()
    ):
        raise VectorIndexError("Vector index metadata does not match its data.")

    return np.ascontiguousarray(vectors), chunk_ids, metadata


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
