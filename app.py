import re
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# LOAD MODEL

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()


# TEXT CLEANING

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# LOAD DATASET + CORPUS

@st.cache_data
def load_corpus():
    dataset = load_dataset("AlekseyKorshuk/quora-question-pairs")
    df = pd.DataFrame(dataset["train"])

    corpus = pd.concat([df["question1"], df["question2"]], ignore_index=True)
    corpus = corpus.dropna()
    corpus = corpus.apply(clean_text)
    corpus = corpus.drop_duplicates()
    corpus = corpus.sample(100000, random_state=42)
    corpus = corpus.reset_index(drop=True)

    return corpus.tolist()


corpus = load_corpus()

# CREATE EMBEDDINGS

@st.cache_resource
def get_embeddings(_corpus_tuple):
    return model.encode(list(_corpus_tuple), show_progress_bar=True, convert_to_numpy=True)


corpus_embeddings = get_embeddings(tuple(corpus))


# BUILD FAISS INDEX

@st.cache_resource
def build_faiss_index(_embeddings):
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    normalized = _embeddings / norms
    index = faiss.IndexFlatIP(_embeddings.shape[1])
    index.add(normalized.astype(np.float32))
    return index, normalized


faiss_index, _ = build_faiss_index(corpus_embeddings)


# SEARCH FUNCTION

def semantic_search(query, top_k=5):
    query_embedding = model.encode([clean_text(query)], convert_to_numpy=True)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    scores, indices = faiss_index.search(query_norm.astype(np.float32), top_k)
    return [(corpus[idx], float(score)) for score, idx in zip(scores[0], indices[0])]


# -----------------------
# UI
# -----------------------

st.markdown("""
    <div style="text-align: center;">
        <h1>Semantic Search Engine</h1>
        <p style="color: gray;">Sentence-BERT + FAISS · Quora Question Pairs</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# 10 preset example queries
example_queries = [
    "How to learn programming?", "What is machine learning?",
    "How to improve English?", "How to lose weight fast?",
    "What is data science?", "How does deep learning work?",
    "Best career options in IT?", "How to study effectively?",
    "How to start investing money?", "What are the benefits of meditation?",
]


# Display example queries as clickable buttons in 2 columns
# Clicking a button populates the text input via session state
st.write("**Example Queries:**")
cols = st.columns(2)
for i, eq in enumerate(example_queries):
    if cols[i % 2].button(eq, use_container_width=True):
        st.session_state["query"] = eq

query = st.text_input("Enter Your Query:", value=st.session_state.get("query", ""))
top_k = st.slider("Top K Results", 1, 10, 5)

if st.button("Search"):
    if query.strip():
        with st.spinner("Searching..."):
            results = semantic_search(query, top_k)
        st.subheader("Results")
        for i, (text, score) in enumerate(results, 1):
            st.write(f"**{i}.** {text}")
            st.caption(f"Similarity: {score:.4f}")
            st.divider()
    else:
        st.warning("Please enter a query.")