from embeddings import text_embedding
import numpy as np
"""
Retrieval function to get similar images based on a text query.
"""
def retrieve_images(query, store, top_k=3):
    query_embedding = text_embedding(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape for compatibility
    results = store.search(query_embedding, top_k)
    return results