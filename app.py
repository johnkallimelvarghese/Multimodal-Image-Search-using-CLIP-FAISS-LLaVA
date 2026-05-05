# import streamlit as st
# import os
# import numpy as np

# from embeddings import get_image_embedding
# from store import VectorStore
# from retrieve import retrieve_images
# from llava_utils import ask_llava

# # Fix OpenMP issue (torch + faiss conflict)
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# st.set_page_config(page_title="Image Search + Reasoning", layout="wide")

# st.title("🔍 Image Search + Reasoning (CLIP + FAISS + LLaVA)")

# # Upload images
# uploaded_files = st.file_uploader(
#     "Upload images",
#     accept_multiple_files=True,
#     type=["png", "jpg", "jpeg"]
# )

# # Query input
# query = st.text_input("Enter your query")

# # Run only when both are provided
# if uploaded_files and query:

#     with st.spinner("Processing images..."):
#         image_paths = []

#         # Save uploaded images temporarily
#         for file in uploaded_files:
#             path = f"temp_{file.name}"
#             with open(path, "wb") as f:
#                 f.write(file.read())
#             image_paths.append(path)

#         # Step 1: Create vector store
#         store = VectorStore(dim=512)

#         # Step 2: Generate embeddings
#         embeddings = []
#         for path in image_paths:
#             emb = get_image_embedding(path)
#             embeddings.append(emb)

#         embeddings = np.array(embeddings)

#         # Step 3: Store embeddings
#         store.add_embeddings(embeddings, image_paths)

#         # Step 4: Retrieve similar images
#         results = retrieve_images(query, store, top_k=3)

#     # 🔎 Show results
#     st.subheader("🔎 Retrieval Results")

#     for path, score in results:
#         st.image(path)
#         st.caption(f"📌 {path} | similarity: {score:.3f}")

#     # ⚠️ Confidence check
#     if results[0][1] < 0.25:
#         st.warning("⚠️ Low confidence match. Results may not be accurate.")

#     # ✅ Best match highlight
#     st.success(f"✅ Best match: {results[0][0]} (score: {results[0][1]:.3f})")

#     # 🧠 LLaVA Explanation
#     st.subheader("🧠 Explanation")

#     for path, score in results:
#         st.markdown(f"### {path} (score: {score:.3f})")

#         prompt = f"""
#         You are a strict visual inspector.

#         Describe ONLY what is clearly visible in the image.
#         - Do NOT assume anything
#         - Do NOT add objects that are not visible
#         - Do NOT infer context or meaning
#         - If unsure, say "unclear"

#         Keep the description short and factual.

#         User query: {query}
#         """

#         with st.spinner(f"Analyzing {path}..."):
#             answer = ask_llava(path, prompt)

#         st.write(answer)



import streamlit as st
import os
import numpy as np

from embeddings import get_image_embedding
from store import VectorStore
from retrieve import retrieve_images
from llava_utils import ask_llava

# Fix OpenMP issue (torch + faiss conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Image Search + Reasoning", layout="wide")

st.title("🔍 Image Search + Reasoning (CLIP + FAISS + LLaVA)")

# Upload images
uploaded_files = st.file_uploader(
    "Upload images",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

# Query input
query = st.text_input("Enter your query")

# Cache embeddings (avoids recomputation)
@st.cache_data
def compute_embeddings(paths):
    return np.array([get_image_embedding(p) for p in paths])


if uploaded_files and query:

    with st.spinner("🔄 Processing images..."):

        image_paths = []

        # Save uploaded images temporarily
        for file in uploaded_files:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())
            image_paths.append(path)

        # Create vector store
        store = VectorStore(dim=512)

        # Generate embeddings (cached)
        embeddings = compute_embeddings(image_paths)

        # Add to FAISS
        store.add_embeddings(embeddings, image_paths)

        # Retrieve results
        results = retrieve_images(query, store, top_k=3)

    # 🔎 Retrieval Results
    st.subheader("🔎 Retrieval Results")

    for path, score in results:
        display_name = os.path.basename(path).replace("temp_", "")
        st.image(path)
        st.caption(f"📌 {display_name} | similarity: {score:.3f}")

    # ⚠️ Confidence check
    if results[0][1] < 0.25:
        st.warning("⚠️ Low confidence match. Results may not be accurate.")

    # ✅ Best match highlight
    best_name = os.path.basename(results[0][0]).replace("temp_", "")
    st.success(f"✅ Best match: {best_name} (score: {results[0][1]:.3f})")

    # 🧠 Explanation
    st.subheader("🧠 Explanation")

    for path, score in results:
        display_name = os.path.basename(path).replace("temp_", "")
        st.markdown(f"### {display_name} (score: {score:.3f})")

        # 🔥 Improved prompt (reduces hallucination)
        prompt = f"""
Describe what is visible in the image.

Rules:
- Only mention objects clearly visible
- Do NOT guess hidden or partially visible items
- Avoid speculation
- If unsure, describe conservatively

User query: {query}
"""

        with st.spinner(f"🧠 Analyzing {display_name}..."):
            answer = ask_llava(path, prompt)

        st.write(answer)