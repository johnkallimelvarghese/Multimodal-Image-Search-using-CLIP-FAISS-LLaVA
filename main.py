import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from embeddings import get_image_embedding, text_embedding
from store import VectorStore
from retrieve import retrieve_images
from llava_utils import ask_llava
import numpy as np
# Step 1: Create store
store=VectorStore(dim=512)
image_paths=["apple.png","banana.png"]
embeddings=[]
# Step 2: Generate embeddings
for path in image_paths:
    emb=get_image_embedding(path)
    embeddings.append(emb)
embeddings=np.array(embeddings)
# Step 3: Add to store
store.add_embeddings(embeddings,image_paths)
# Step 4: Query
query="a photo of a apple"
query_emb=text_embedding(query)

results=retrieve_images(query, store)
print(results)

# Step 5: LLaVA
llava_response=ask_llava(results[0][0], "What is in this image?")
print(llava_response)