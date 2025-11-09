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
from PIL import Image, ImageChops, ImageOps
import re

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
_chroma_client = None
_collection = None
_embedding_model = None
_tfidf_vectorizer = None
_image_store = None

from transformers import AutoTokenizer

# Load the base tokenizer (not the processor)
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# Configuration
BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
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

def extract_contextual_description(description: str, generate_response) -> str:
    """
    Uses an LLM to extract only the non-recommendation part of a fashion recommendation sentence.
    """
    prompt = f"""
    The following sentence contains a fashion recommendation.
    Your task is to extract only the CONTEXT — the part describing the user's existing outfit or
    what the recommendation is based on. Exclude the item being recommended.

    Example 1:
    Sentence: "I recommend black stiletto heels to complement your black dress."
    Output: "to complement your black dress."

    Example 2:
    Sentence: "For your white blouse and black skirt, I recommend heels."
    Output: "for your white blouse and black skirt."

    Example 3:
    Sentence: "Pairing your beige coat with a red scarf will look great."
    Output: "your beige coat"

    Sentence: "{description}"
    Output:
    """
    try:
        context = generate_response(prompt, None, TEMPERATURE=0.1).strip()
        return context
    except Exception as e:
        print(f"[WARN] Context extraction failed: {e}")
        return description  # fallback
    
# ---------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------
def hybrid_retrieve_v2(
    query_text: str,
    top_k: int = 3,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.25,
    type_weight: float = 0.15,
    generate_response=None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval that emphasizes the *non-recommendation* context.
    It extracts descriptive context using an LLM and re-embeds that portion
    for accurate semantic comparison.
    """
    if _collection is None or _embedding_model is None:
        raise RuntimeError("RAG not initialized. Call init_rag() first.")

    # Encode query
    query_embedding = _embedding_model.encode(query_text).tolist()
    # Fetch a larger pool for reranking
    semantic_results = _collection.query(query_embeddings=[query_embedding], n_results=top_k * 3)
    query_keywords = query_text.lower().split()

    scored = []
    for idx, (iid, desc, meta) in enumerate(
        zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
        )
    ):
        # 1️⃣ Extract only contextual (non-recommendation) text
        try:
            context_text = extract_contextual_description(desc, generate_response)
            if not context_text:
                context_text = desc  # fallback
        except Exception as e:
            print(f"[WARN] Context extraction failed: {e}")
            context_text = desc

        print(f"\n[DEBUG] Original desc: {desc}")
        print(f"[DEBUG] Extracted context: {context_text}")

        # 2️⃣ Compute new semantic score using context embedding
        context_embedding = _embedding_model.encode(context_text).tolist()
        # cosine similarity = 1 - distance
        semantic_score = 1.0 - float(
            _embedding_model.similarity(query_embedding, context_embedding)
            if hasattr(_embedding_model, "similarity")
            else semantic_results["distances"][0][idx]          
        )

        # 3️⃣ Keyword and type overlap scores
        item_keywords = set((meta or {}).get("keywords", "").split(","))
        keyword_overlap = len(query_keywords & item_keywords) / (
            len(query_keywords) + len(item_keywords) + 1e-6
        )

        desc_words = set(context_text.lower().split())
        q_words = set(query_text.lower().split())
        type_overlap = len(desc_words & q_words) / (len(q_words) + 1e-6)

        # 4️⃣ Weighted hybrid score
        hybrid_score = (
            semantic_weight * semantic_score
            + keyword_weight * keyword_overlap
            + type_weight * type_overlap
        )

        scored.append(
            {
                "item_id": iid,
                "description": desc,
                "context_used": context_text,
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
def retrieve_relevant_items_from_text(recommendation_text: str, user_query: str, top_k: int = 4, generate_response=None):
    """
    Use the Qwen model itself to extract the recommended item (color + type)
    from its own generated answer, then run RAG retrieval using that extracted phrase.
    Includes detailed debug prints to trace the flow.
    """
    init_rag()

    print("\n==============================")
    print(f"[DEBUG] Full model recommendation text:\n{recommendation_text}")
    print("==============================")

    # Step 1 — Ask Qwen to extract the item it recommended
    extraction_prompt = f"""Given this fashion recommendation, extract ONLY the item being RECOMMENDED (suggested), NOT the user's item.

        User Query: "{user_query}"
        Recommendation: "{recommendation_text}"

        What specific item is being RECOMMENDED? (answer with only the item name, e.g., "blue denim jacket" or "leather boots")
        Recommended item:"""

    try:
        extracted_item = generate_response(extraction_prompt, None, TEMPERATURE=0.1).strip()
    except Exception as e:
        print(f"[DEBUG] Extraction model failed → {e}")
        extracted_item = ""

    print(f"[DEBUG] Qwen-extracted item raw → '{extracted_item}'")    
    if not extracted_item:
        extracted_item = recommendation_text.strip().split('.')[0]  # fallback

    print(f"[DEBUG] Qwen-extracted item phrase → '{extracted_item}'")

    # Step 2 — Query RAG with that extracted phrase
    query_text = extracted_item
    results = hybrid_retrieve_v2(query_text, top_k=3, generate_response=generate_response)

    print(f"[DEBUG] RAG queried for → '{query_text}'")
    print(f"[DEBUG] Retrieved {len(results)} candidate items")
    print(f"[DEBUG] {results[0:top_k]}")
    # Step 3 — Print debug table of top matches
    for i, r in enumerate(results[:top_k]):
        desc = r['description'][:80].replace("\n", " ")
        score = round(r['score'], 3)
        print(f"[DEBUG] #{i+1} → '{desc}' (score={score})")

    # Step 4 — Collect images
    images = []
    for r in results[:top_k]:
        img = _image_store.get(r["item_id"]) if _image_store else None
        
        images.append(img)

    print(f"[DEBUG] Returning {len(images)} images\n==============================\n")

    return {"rag_results": results[:top_k], "images": images}
