import torch
from sentence_transformers import util
from transformers import AutoTokenizer

# Load the base tokenizer (not the processor)
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def extract_recommended_item(recommendation_text, user_query):
    """
    Use the model to extract ONLY the recommended item (not the user's input item)
    Uses base tokenizer (text-only)
    """
    
    extraction_prompt = f"""Given this fashion recommendation, extract ONLY the item being RECOMMENDED (suggested), NOT the user's item.

User Query: "{user_query}"
Recommendation: "{recommendation_text}"

What specific item is being RECOMMENDED? (answer with only the item name, e.g., "blue denim jacket" or "leather boots")
Recommended item:"""
    
    # Use base tokenizer (text-only)
    inputs = base_tokenizer(extraction_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.1,
        )
    
    recommended_item = base_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    recommended_item = recommended_item.split('\n')[0].lower()
    
    return recommended_item

print("✅ extract_recommended_item loaded with base tokenizer")

def hybrid_retrieve_v2(query_text, top_k=3, semantic_weight=0.6, keyword_weight=0.25, type_weight=0.15):
    """
    Hybrid retrieval with semantic + keyword matching
    """
    # Semantic retrieval
    query_embedding = embedding_model.encode(query_text).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3
    )
    
    # Extract keywords
    query_keywords = set(query_text.lower().split())
    query_keywords = {w for w in query_keywords if len(w) > 3}
    
    # Score results
    scored_results = []
    for idx, (item_id, description, metadata) in enumerate(zip(
        semantic_results['ids'][0],
        semantic_results['documents'][0],
        semantic_results['metadatas'][0]
    )):
        # Semantic score
        semantic_score = 1.0 - semantic_results['distances'][0][idx]
        
        # Keyword score
        item_keywords = set(metadata.get('keywords', '').split(','))
        keyword_overlap = len(query_keywords & item_keywords) / (len(query_keywords) + len(item_keywords) + 1e-6)
        
        # Type score (extract nouns from description)
        type_overlap = 0.0
        try:
            description_words = set(description.lower().split())
            query_words = set(query_text.lower().split())
            type_overlap = len(description_words & query_words) / (len(query_words) + 1e-6)
        except:
            pass
        
        # Hybrid score
        hybrid_score = (
            semantic_weight * semantic_score +
            keyword_weight * keyword_overlap +
            type_weight * type_overlap
        )
        
        scored_results.append({
            'item_id': item_id,
            'description': description,
            'score': hybrid_score,
            'semantic_score': semantic_score,
            'keyword_overlap': keyword_overlap,
            'type_overlap': type_overlap
        })
    
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    return scored_results[:top_k]

print("✅ hybrid_retrieve_v2 function loaded")

import matplotlib.pyplot as plt
from datetime import datetime

def full_pipeline_with_smart_extraction(user_image, user_query, model, tokenizer, top_k=3):
    """
    Complete pipeline: Recommend → Extract recommended item → RAG retrieval
    """
    
    print("="*80)
    print("FASHION RECOMMENDATION WITH SMART EXTRACTION")
    print("="*80)
    print(f"\n📸 User Query: {user_query}")
    
    # Step 1: Get recommendation
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": user_image},
            {"type": "text", "text": user_query}
        ],
    }]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(user_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    
    print("\n🤖 Generating recommendation...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
    )
    
    recommendation = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print(recommendation)
    
    # Step 2: Extract ONLY the recommended item
    print("\n" + "="*80)
    print("EXTRACTING RECOMMENDED ITEM...")
    print("="*80)
    
    recommended_item = extract_recommended_item(recommendation, user_query)
    
    print(f"\n✅ Recommended item: {recommended_item}")
    
    # Step 3: Use recommended item for RAG
    print("\n" + "="*80)
    print("RETRIEVING SIMILAR ITEMS...")
    print("="*80)
    
    print(f"\n🔍 Searching for items similar to: {recommended_item}")
    
    rag_results = hybrid_retrieve_v2(recommended_item, top_k)
    
    # Step 4: Display results
    print("\n" + "="*80)
    print(f"TOP {top_k} MATCHING ITEMS:")
    print("="*80)
    
    fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
    
    # User's image
    axes[0].imshow(user_image)
    axes[0].set_title("User's Item", fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Retrieved matches
    for i, res in enumerate(rag_results):
        item_id = res['item_id']
        img = image_store.get(item_id)
        if img is not None:
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"Match {i+1}\n({res['score']:.3f})", fontweight='bold', fontsize=11)
            axes[i+1].axis('off')
        
        print(f"\n{i+1}. {res['description'][:200]}...")
        print(f"   Score: {res['score']:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'recommendation': recommendation,
        'recommended_item': recommended_item,
        'rag_results': rag_results
    }


def save_recommendation_result(result, filename=None):
    """Save results to JSON"""
    
    if filename is None:
        filename = f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    
    saved_data = {
        'recommendation': result['recommendation'],
        'recommended_item': result['recommended_item'],
        'rag_results': [
            {
                'item_id': r['item_id'],
                'description': r['description'][:200],
                'score': float(r['score'])
            }
            for r in result['rag_results']
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filename, 'w') as f:
        json.dump(saved_data, f, indent=2)
    
    print(f"\n✅ Saved to {filename}")
    return filename


# TEST IT
result = full_pipeline_with_smart_extraction(
    user_image=test_image,
    user_query="What would look good with my beige chinos?",
    model=model,
    tokenizer=tokenizer,
    top_k=3
)

# Optional: Save results
save_recommendation_result(result)
