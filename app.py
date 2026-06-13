import streamlit as st
import pandas as pd
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# LOAD MODEL
# -----------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()



# -----------------------
# LOAD DATASET + CORPUS
# -----------------------

@st.cache_data
def load_corpus():

    dataset = load_dataset(
        "AlekseyKorshuk/quora-question-pairs"
    )

    df = pd.DataFrame(dataset["train"])


    corpus = pd.concat(
        [
            df["question1"],
            df["question2"]
        ],
        ignore_index=True
    )


    # preprocessing

    corpus = corpus.dropna()

    corpus = corpus.str.lower()

    corpus = corpus.str.replace(
        r"[^a-z0-9\s]",
        "",
        regex=True
    )


    corpus = corpus.str.strip()


    # remove duplicates

    corpus = corpus.drop_duplicates()


    # sample 100k questions

    corpus = corpus.sample(
        100000,
        random_state=42
    )


    corpus = corpus.reset_index(drop=True)


    return corpus.tolist()



corpus = load_corpus()



# -----------------------
# CREATE EMBEDDINGS
# -----------------------

@st.cache_resource
def get_embeddings(corpus):

    return model.encode(
        corpus,
        show_progress_bar=True
    )


corpus_embeddings = get_embeddings(corpus)



# -----------------------
# SEARCH FUNCTION
# -----------------------

def semantic_search(query, corpus, corpus_embeddings, top_k=5):

    query_embedding = model.encode([query])


    similarities = cosine_similarity(
        query_embedding,
        corpus_embeddings
    )[0]


    top_k_indices = similarities.argsort()[-top_k:][::-1]


    results = []

    for idx in top_k_indices:

        results.append(
            (
                corpus[idx],
                similarities[idx]
            )
        )


    return results



# -----------------------
# UI DESIGN
# -----------------------

st.title(
    "Semantic Search Engine (SBERT + Cosine Similarity)"
)


st.write(
    "Ask a question and find similar questions from Quora dataset"
)



query = st.text_input(
    "Enter your query:"
)


top_k = st.slider(
    "Top K results",
    1,
    10,
    5
)



# -----------------------
# RUN SEARCH
# -----------------------

if st.button("Search"):

    if query:

        results = semantic_search(
            query,
            corpus,
            corpus_embeddings,
            top_k
        )


        st.subheader(
            "Top Results:"
        )


        for i, (text, score) in enumerate(results,1):

            st.write(
                f"### {i}. {text}"
            )

            st.write(
                f"Similarity Score: {score:.4f}"
            )

            st.divider()


    else:

        st.warning(
            "Please enter a query"
        )