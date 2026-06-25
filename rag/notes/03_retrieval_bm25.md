# Sparse Retrieval with BM25

**File:** `03_retrieval/02_sparse_retrieval_bm25.py`

## Core Concept
Retrieve documents using keyword frequency (BM25/TF-IDF) rather than semantic embeddings. Excels at exact syntax matches — function names, error codes, config keys — where semantic search often fails.

## What You Learn
- Tokenize a corpus and build a BM25 index
- Score documents by term frequency and inverse document frequency
- Use BM25 for lexical retrieval alongside vector search (hybrid RAG)

## Key Constructs
```python
from rank_bm25 import BM25Okapi

corpus = ["Set connectTimeout to 5000ms.", "Booking BK-130 was cancelled.", ...]
tokenized = [doc.lower().split() for doc in corpus]

bm25 = BM25Okapi(tokenized)

query = "connectTimeout"
scores = bm25.get_scores(query.lower().split())
# → [0.0, 0.0, 0.93] — only the matching doc scores high
```

## When to Use BM25 vs. Vector Search
| | BM25 | Vector Search |
|---|---|---|
| Exact keyword match | Excellent | Poor |
| Semantic similarity | Poor | Excellent |
| Speed | Very fast | Slower (embedding needed) |
| Best for | Code, error codes, config keys | Natural language questions |

## In Hybrid RAG
Run both BM25 and vector search, then merge results (e.g., Reciprocal Rank Fusion) before reranking. BM25 catches what embeddings miss.

## Pitfalls
- BM25 requires exact token matches — `connectTimeout` and `connect_timeout` score differently
- Simple whitespace `.split()` is fine for demos; use NLTK/SpaCy tokenizer in production
- No semantic understanding — a synonym query returns zero score
