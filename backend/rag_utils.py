"""
rag_utils.py
-------------
RAG retrieval utilities for Qwen2.5-VL backend.

Assumes you have already run build_fashion_rag_db.py
to create:
    ./chroma_fashion_db_hybrid/
    fashion_image_store_hybrid.pkl
"""

import os
import pickle
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image


# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
_chroma_client = None
_collection = None
_embedding_model = None
_tfidf_vectorizer = None
_image_store = None

_DEFAULT_DB_PATH = "./chroma_fashion_db_hybrid"
_DEFAULT_IMAGE_STORE_PATH = "fashion_image_store_hybrid.pkl"


# ---------------------------------------------------------------------
# Initialization (lightweight)
# ---------------------------------------------------------------------
def init_rag(
    db_path: str = _DEFAULT_DB_PATH,
    image_store_path: str = _DEFAULT_IMAGE_STORE_PATH,
) -> None:
    """Load existing Chroma collection and supporting models."""
    global _chroma_client, _collection, _embedding_model, _tfidf_vectorizer, _image_store

    if _chroma_client is None:
        if not os.path.exists(db_path):
            raise RuntimeError(
                f"ChromaDB not found at '{db_path}'. Run build_fashion_rag_db.py first."
            )
        _chroma_client = chromadb.PersistentClient(path=db_path)
        _collection = _chroma_client.get_collection("fashion_items_hybrid")

    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-mpnet-base-v2")

    if _tfidf_vectorizer is None:
        _tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        docs = _collection.get(limit=min(_collection.count(), 3000)).get("documents", [])
        if docs:
            _tfidf_vectorizer.fit(docs)

    if _image_store is None:
        if not os.path.exists(image_store_path):
            raise RuntimeError(
                f"Image store not found at '{image_store_path}'. Run build_fashion_rag_db.py first."
            )
        with open(image_store_path, "rb") as f:
            _image_store = pickle.load(f)


# ---------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------
def hybrid_retrieve_v2(
    query_text: str,
    top_k: int = 3,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.25,
    type_weight: float = 0.15,
) -> List[Dict[str, Any]]:
    """Hybrid semantic + keyword + type retrieval."""
    if _collection is None or _embedding_model is None:
        raise RuntimeError("RAG not initialized. Call init_rag() first.")

    query_embedding = _embedding_model.encode(query_text).tolist()
    semantic_results = _collection.query(query_embeddings=[query_embedding], n_results=top_k * 3)

    query_keywords = set([w for w in query_text.lower().split() if len(w) > 3])

    scored = []
    for idx, (iid, desc, meta) in enumerate(
        zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
        )
    ):
        semantic_score = 1.0 - semantic_results["distances"][0][idx]
        item_keywords = set((meta or {}).get("keywords", "").split(","))
        keyword_overlap = len(query_keywords & item_keywords) / (
            len(query_keywords) + len(item_keywords) + 1e-6
        )
        try:
            desc_words = set(desc.lower().split())
            q_words = set(query_text.lower().split())
            type_overlap = len(desc_words & q_words) / (len(q_words) + 1e-6)
        except Exception:
            type_overlap = 0.0

        hybrid_score = (
            semantic_weight * semantic_score
            + keyword_weight * keyword_overlap
            + type_weight * type_overlap
        )
        scored.append(
            {
                "item_id": iid,
                "description": desc,
                "score": hybrid_score,
                "semantic_score": semantic_score,
                "keyword_overlap": keyword_overlap,
                "type_overlap": type_overlap,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------
def retrieve_relevant_items_from_text(recommendation_text: str, top_k: int = 3):
    """Retrieve fashion images + metadata for model recommendation text."""
    init_rag()
    results = hybrid_retrieve_v2(recommendation_text, top_k=top_k)

    images = []
    for r in results:
        img = _image_store.get(r["item_id"]) if _image_store else None
        images.append(img if isinstance(img, Image.Image) else None)

    return {"rag_results": results, "images": images}
