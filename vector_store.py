"""
Thin wrapper around Chroma so the rest of the app doesn't care about the
underlying vector DB implementation. Swap this file's internals for
Pinecone/Qdrant/Weaviate later without touching ingestion.py or retrieval.py.
"""
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from config import settings

_client = None
_collection = None


class _NoOpEmbeddingFunction(EmbeddingFunction):
    """
    We always supply our own embeddings (via NVIDIA NIM) on every add/query
    call, so Chroma never needs to embed anything itself. Without this,
    Chroma silently loads its default onnxruntime-based embedder even
    though it's unused - and onnxruntime's native DLL is a common source
    of silent, traceback-free crashes on Windows machines missing the
    Microsoft Visual C++ Redistributable. This avoids that dependency
    entirely.
    """
    def __call__(self, input):
        raise RuntimeError(
            "This collection requires embeddings to be supplied explicitly - "
            "the default embedder should never be invoked."
        )


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name="stock_documents",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_NoOpEmbeddingFunction(),
        )
    return _collection
