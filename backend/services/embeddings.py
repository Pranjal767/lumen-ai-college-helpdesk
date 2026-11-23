"""Local sentence-transformer embeddings for document chunks and queries."""

from functools import lru_cache
import hashlib
from os import environ
import re
from typing import Protocol, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASHING_EMBEDDING_MODEL = "synaptix-feature-hash-v3"
HASHING_DIMENSIONS = 8192
_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an and are as at be been by for from had has have i in is it its of on or "
    "our that the their them there these this to was were will with you your "
    "what when where which who how can could would should do does did tell me "
    "please kindly thanks thank may i'm about find get give let know based what "
    "who why very grateful something good morning afternoon evening howdy yo".split()
)


class EmbeddingError(RuntimeError):
    """Raised when an embedding model cannot produce valid vectors."""


class SentenceModel(Protocol):
    """The Sentence Transformers model operation used by the embedder."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


class EmbeddingProvider(Protocol):
    """Minimal interface required by the vector index."""

    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> "SentenceTransformer":
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, OSError) as error:
        raise EmbeddingError(
            "Sentence Transformers is unavailable in this Python environment."
        ) from error
    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Load a local Sentence Transformer model lazily and encode text batches."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model: SentenceModel | None = None,
    ) -> None:
        self.model_name = model_name or environ.get(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self._model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required for embedding.")

        if self.model_name == HASHING_EMBEDDING_MODEL:
            return HashingEmbedder().encode(texts)
        try:
            model = self._model or _load_model(self.model_name)
        except EmbeddingError:
            # Keep retrieval available when native Sentence Transformers files
            # are blocked by a Windows Application Control policy.
            self.model_name = HASHING_EMBEDDING_MODEL
            return HashingEmbedder().encode(texts)
        encoded = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingError("The embedding model returned an invalid shape.")
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise EmbeddingError("The embedding model returned invalid vector values.")
        return np.ascontiguousarray(vectors)


class HashingEmbedder:
    """Dependency-free lexical vectors used when native embeddings are blocked."""

    model_name = HASHING_EMBEDDING_MODEL

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required for embedding.")

        vectors = np.zeros((len(texts), HASHING_DIMENSIONS), dtype=np.float32)
        for row, text in enumerate(texts):
            words = [word.lower() for word in _WORD_PATTERN.findall(text)]
            useful_words = [word for word in words if word not in _STOP_WORDS]
            features: dict[str, float] = {}
            for index, word in enumerate(useful_words):
                features[f"w:{word}"] = features.get(f"w:{word}", 0.0) + 1.0
                if index:
                    pair = f"b:{useful_words[index - 1]}_{word}"
                    features[pair] = features.get(pair, 0.0) + 0.55
                padded = f"^{word}$"
                for start in range(max(1, len(padded) - 2)):
                    trigram = f"c:{padded[start:start + 3]}"
                    features[trigram] = features.get(trigram, 0.0) + 0.2

            for feature, weight in features.items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % HASHING_DIMENSIONS
                vectors[row, bucket] += weight

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        vectors[nonzero] /= norms[nonzero]
        return vectors
