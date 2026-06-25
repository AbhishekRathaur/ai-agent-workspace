# Reranking

**Files:** `04_routing_reranking/reranking.ipynb` · `04_routing_reranking/03_cross_encoder_reranking.py`

## Core Concept
Two-pass retrieval: cast a wide net with vector search (high recall), then filter with a reranker (high precision). Only the top-ranked documents reach the LLM, reducing noise and "lost in the middle" failures.

## What You Learn
- Retrieve more documents than needed (k=4) in the first pass
- Score retrieved documents with a lightweight term-overlap reranker
- Use a simulated cross-encoder to score query-document pairs jointly
- Truncate to top-2 documents before building the LLM prompt

## Key Constructs — Lightweight Term-Overlap Reranker
```python
def rerank(query: str, docs: list) -> list:
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_terms = set(doc.page_content.lower().split())
        overlap = query_terms.intersection(doc_terms)
        score = len(overlap) / len(query_terms) if query_terms else 0.0
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]

# Two-pass pipeline
candidates = vector_store.similarity_search(query, k=4)   # wide net
ranked     = rerank(query, candidates)                     # filter
context    = "\n\n".join(d.page_content for d in ranked[:2])  # top-2 only
```

## Key Constructs — Cross-Encoder Pattern
```python
def cross_encoder_rerank(query: str, chunks: list) -> list:
    scored = []
    for chunk in chunks:
        score = 0.05  # baseline
        if "State 13" in chunk["content"] or "booking" in chunk["content"].lower():
            score = 0.95
        elif "administrative" in chunk["content"]:
            score = 0.40
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]
```

## Why Rerank?
Vector search ranks by embedding distance, which captures broad semantic similarity but misses exact keyword relevance. Reranking adds a focused pass that checks whether the actual query terms appear in the document — boosting precision without re-embedding.

## Pitfalls
- Term overlap misses synonyms ("DB" vs "database") — use a cross-encoder model in production
- Truncating to top-2 is arbitrary — tune based on context window budget
- Normalize scores by query length (not document length) to avoid biasing long documents
- Reranking adds a second pass of latency — profile on your data before adding it blindly
