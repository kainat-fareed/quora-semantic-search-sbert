# Semantic Search Engine using Sentence-BERT and Cosine Similarity

A semantic search engine that understands the **meaning** of user queries instead of relying on keyword matching — built with Sentence-BERT embeddings, FAISS indexing, and cosine similarity retrieval.

---

##  Project Overview

Traditional search systems fail when different words are used to express the same idea. This project solves that problem using pretrained **Sentence-BERT** embeddings and **cosine similarity-based** retrieval.

The system encodes all questions into dense vector representations and retrieves the most semantically relevant results for any natural language query — without any model training.

---

##  Pipeline

```
Raw Dataset (Quora Question Pairs)
        │
        ▼
Data Preprocessing
  ├── Drop missing values (question1, question2)
  ├── Merge question1 + question2 into one corpus
  └── Remove duplicate questions
        │
        ▼
Text Normalization
  ├── Lowercase
  ├── Remove punctuation & special characters
  └── Strip extra whitespace
        │
        ▼
Sample 10,000 Questions
        │
        ▼
Generate Embeddings  ←  all-MiniLM-L6-v2 (Sentence-BERT)
        │
        ▼
Store in FAISS Index  (IndexFlatL2, float32)
        │
        ▼
Semantic Search  ←  Cosine similarity on query embedding
        │
        ▼
Top-K Retrieved Results
```

---

##  Dataset

**Source:** [Quora Question Pairs — HuggingFace](https://huggingface.co/datasets/AlekseyKorshuk/quora-question-pairs)

Real user-generated questions from Quora, widely used for semantic similarity tasks.

| Feature | Description |
|---|---|
| `question1` | First question text | used |
| `question2` | Second question text | used |
| `id`, `qid1`, `qid2`, `is_duplicate` | Not used |

Both `question1` and `question2` are merged into a single corpus — using only one column would discard half the data.

---

##  Data Preprocessing

- Dropped rows with missing values in `question1` or `question2`
- Merged both question columns into one unified corpus
- Removed duplicate questions after merging

| Stage | Count |
|---|---|
| Total questions (before cleaning) | 808,574 |
| Duplicates removed | 271,215 |
| Final unique corpus size | 537,359 |
| Sampled for search | 10,000 |

---

##  Text Normalization

Before embedding generation, all text was normalized by converting to lowercase, removing punctuation and special characters, and stripping extra whitespace. This ensures consistency and improves embedding quality.

---

##  Embedding Model

**`all-MiniLM-L6-v2`** from the `sentence-transformers` library:

- Lightweight 6-layer MiniLM architecture
- Produces **384-dimensional** dense sentence embeddings
- Sentences with similar meaning are closer in vector space
- No fine-tuning required — works out of the box

---

##  FAISS Vector Index

All embeddings are stored in a FAISS `IndexFlatL2` index. Embeddings are converted to `float32` as required by FAISS for memory efficiency and fast similarity search.

---

##  Semantic Search

For any user query, the system encodes it into an embedding and computes cosine similarity against all stored corpus embeddings. The top-K most similar questions are returned ranked by similarity score — a higher score means stronger semantic relevance.

---

##  Test Queries

The system was evaluated using 10 diverse queries:

| # | Query |
|---|---|
| 1 | How can I learn programming? |
| 2 | What is machine learning? |
| 3 | Best way to get a job in tech? |
| 4 | How to improve English? |
| 5 | What is AI? |
| 6 | How does deep learning work? |
| 7 | How to lose weight fast? |
| 8 | Best career options in IT? |
| 9 | How to study effectively? |
| 10 | What is data science? |

Each query returns the **top-5 most semantically similar questions** with cosine similarity scores.

---

##  Tech Stack

| Library | Purpose |
|---|---|
| `sentence-transformers` | Pretrained Sentence-BERT embeddings |
| `faiss-cpu` | Fast vector indexing and storage |
| `scikit-learn` | Cosine similarity computation |
| `datasets` | Loading Quora Question Pairs from HuggingFace |
| `pandas` | Data manipulation and corpus management |
| `numpy` | Embedding matrix operations |
| `re` | Text normalization |

---

##  Installation

```bash
pip install sentence-transformers faiss-cpu scikit-learn datasets pandas numpy
```

> Use `faiss-gpu` instead of `faiss-cpu` if you have a CUDA-compatible GPU.

---


##  Project Structure

```
semantic-search-engine/
│
├── semantic_search_using_pretrained_embeddings.ipynb      # Main notebook with full pipeline
├── app.py               # Streamlit web app
├── requirements.txt     # Dependencies
└── README.md
```

---
##  How to Run

**Google Colab:**
1. Open `notebook.ipynb`
2. Run all cells in order
3. Test your own queries in the final cell

**Local (Jupyter):**
```bash
git clone https://github.com/kainat-fareed/quora-semantic-search-sbert
cd semantic-search-engine
pip install -r requirements.txt
jupyter notebook semantic_search_using_pretrained_embeddings.ipynb
```

**Streamlit App:**
```bash
streamlit run app.py
```

---

##  Conclusion

This project demonstrates a working semantic search engine using Sentence-BERT and cosine similarity. Unlike keyword-based search, it retrieves contextually relevant results even when the exact words differ.

**Suitable for:**
- Intelligent FAQ retrieval systems
- Chatbot query matching
- Duplicate question detection
- General information retrieval applications
