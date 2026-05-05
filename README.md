# 🔍 Image Search + Reasoning (CLIP + FAISS + LLaVA)

A multimodal image retrieval and reasoning system that combines:

- CLIP for image-text embeddings
- FAISS for fast vector similarity search
- LLaVA for visual understanding and explanation

---

## 🚀 Features

- Upload multiple images
- Search using natural language queries
- Retrieve similar images using embeddings
- Explain results using a vision-language model
- Confidence-aware retrieval

---

## 🧠 Architecture

User Query  
↓  
CLIP → Embedding  
↓  
FAISS → Retrieve Top-K Images  
↓  
LLaVA → Explain Results  

---

## 📸 Demo

![Demo](demo/demo.gif)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/image-search-reasoning.git
cd image-search-reasoning
### Installation

pip install -r requirements.txt

### Dev Setup

pip install -r requirements-dev.txt

streamlit run app.py