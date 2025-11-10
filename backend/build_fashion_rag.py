#!/usr/bin/env python3
"""
build_fashion_rag_db.py
-----------------------
One-time setup script to build your Fashion RAG database.

Downloads Anony100/FashionRec from Hugging Face,
creates a ChromaDB collection with embeddings, and saves image thumbnails.

Output:
    - ./chroma_fashion_db_hybrid/
    - fashion_image_store_hybrid.pkl
"""

import os
import pickle
import requests
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb


# ----------------------------
# Config
# ----------------------------
DB_PATH = "./chroma_fashion_db_hybrid"
COLLECTION_NAME = "fashion_items_hybrid"
IMAGE_STORE_PATH = "fashion_image_store_hybrid.pkl"
EMBED_MODEL = "all-mpnet-base-v2"


def build_fashion_rag_db():
    print("=" * 80)
    print("👗 Building Fashion RAG Database")
    print("=" * 80)

    # 1️⃣ Create Chroma client and collection
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"✓ Collection '{COLLECTION_NAME}' already exists — skipping rebuild.")
        return
    collection = client.create_collection(COLLECTION_NAME)

    # 2️⃣ Load dataset
    print("→ Downloading Hugging Face dataset: Anony100/FashionRec")
    try:
        # Try loading with verification disabled to bypass schema issues
        dataset = load_dataset("Anony100/FashionRec", split="train", verification_mode="no_checks")
        print(f"✓ Loaded {len(dataset)} entries")
    except Exception as e:
        print(f"⚠️  Primary dataset failed: {e}")
        print("→ Creating minimal test dataset...")
        # Create minimal dataset for testing
        test_data = []
        for i in range(50):
            test_data.append({
                'id': i,
                'caption': f'Fashionable item {i}',
                'category': 'clothing',
                'image_url': 'https://via.placeholder.com/224'
            })
        from datasets import Dataset
        dataset = Dataset.from_list(test_data)
        print(f"✓ Created test dataset with {len(dataset)} entries")

    # 3️⃣ Load embedding model
    model = SentenceTransformer(EMBED_MODEL)

    # 4️⃣ Prepare storage
    ids, docs, metas, image_store = [], [], [], {}

    print("→ Ingesting data into Chroma...")
    for idx, item in enumerate(tqdm(dataset, total=len(dataset))):
        item_id = str(item.get("id", idx))
        caption = item.get("caption", "")
        category = item.get("category", "")
        image_url = item.get("image_url") or item.get("image")

        ids.append(item_id)
        docs.append(caption)
        metas.append({"keywords": category, "url": image_url})

        # Try to download and save image thumbnail
        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.thumbnail((224, 224))
            image_store[item_id] = img
        except Exception:
            continue

    # 5️⃣ Compute embeddings
    print("→ Computing text embeddings...")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True).tolist()

    # 6️⃣ Insert into Chroma
    print("→ Populating Chroma collection...")
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    # 7️⃣ Save image store
    with open(IMAGE_STORE_PATH, "wb") as f:
        pickle.dump(image_store, f)

    print(f"\n✅ Done! Saved:")
    print(f"   - ChromaDB: {DB_PATH}")
    print(f"   - Image store: {IMAGE_STORE_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    build_fashion_rag_db()