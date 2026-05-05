import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
"""
This code defines two functions, `get_image_embedding` and `text_embedding`, which use the CLIP model to generate embeddings for images and text, respectively. The embeddings are normalized to have a unit norm, and the functions return the embeddings as NumPy arrays of type float32. The code also includes commented-out examples of how to use these functions to compute similarities between an image and various text descriptions.
"""
#Load onec
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    image=Image.open(image_path).convert("RGB")
    inputs=processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # embedding = model.get_image_features(**inputs)
        outputs = model.get_image_features(**inputs)
        embedding = outputs.pooler_output  #pooled_output is a single vector representation of the image
    embedding=embedding/embedding.norm(p=2, dim=-1, keepdim=True)
    # print(type(embedding))
    return embedding.squeeze().numpy().astype("float32")

def text_embedding(text):
    inputs=processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        # embedding = model.get_text_features(**inputs)
        outputs = model.get_text_features(**inputs)
        embedding = outputs.pooler_output
    embedding=embedding/embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.squeeze().numpy().astype("float32")

# img_emb = get_image_embedding("apple.png")

# texts = ["a photo of an apple", "banana", "car", "dog"]
# for t in texts:
#     txt_emd=text_embedding(t)
#     similarity=np.dot(img_emb, txt_emd)
#     print(f"Similarity between image and '{t}': {similarity}")

# print(img_emb[:10])
# print("Image embedding shape:", img_emb.shape)
