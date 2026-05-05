import faiss
import numpy as np
"""A simple vector store using FAISS for similarity search."""
class VectorStore:
    def __init__(self, dim):
        self.index=faiss.IndexFlatIP(dim)
        self.image_paths=[]
    def add_embeddings(self,embeddings,paths):
        self.index.add(embeddings)
        self.image_paths.extend(paths)
    def search(self,query_embedding,top_k=3):
        top_k=min(top_k,len(self.image_paths))
        query_embedding = query_embedding.reshape(1, -1)
        D,I=self.index.search(query_embedding,top_k)

        results=[]
        for idx,score in zip(I[0],D[0]):#idx is the index of the retrieved image, score is the similarity score
            if idx < len(self.image_paths):
             results.append((self.image_paths[idx],float(score)))

        return results