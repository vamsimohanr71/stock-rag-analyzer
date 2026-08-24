"""
Retrieval layer: embeds the query, does a metadata-filtered vector search
scoped to the requested ticker, and returns ranked chunks with their
source metadata (so the LLM output can be traced back to a URL/filing).
"""
from nvidia_embeddings import create_embeddings
from vector_store import get_collection


def embed_query(query: str) -> list[float]:
    embeddings = create_embeddings([query], input_type="query")
    return embeddings[0]


def retrieve_context(ticker: str, query: str, k: int = 5) -> list[dict]:
    """
    Returns a list of {text, source, date, url} dicts, most relevant first.
    Empty list means: no ingested data for this ticker yet - caller should
    trigger ingestion before generating an analysis.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"ticker": ticker.upper()},
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({
            "text": doc,
            "source": meta.get("source"),
            "date": meta.get("date"),
            "url": meta.get("url", ""),
            "relevance_score": round(1 - dist, 4),  # cosine distance -> similarity
        })
    return out
