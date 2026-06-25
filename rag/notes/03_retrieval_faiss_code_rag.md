# FAISS Code RAG

**File:** `03_retrieval/faiss_code_rag.py`

## Core Concept
Build a local RAG pipeline specifically for code retrieval using FAISS (Facebook AI Similarity Search). Each code snippet is stored as a Document with file metadata, retrieved by semantic similarity, and fed as structured context to the LLM.

## What You Learn
- Create `Document` objects with `metadata` (source path, type)
- Build a FAISS vector store from documents using Ollama embeddings
- Retrieve k=2 documents and loop through all results to build full context
- Inject code context into the system prompt using markdown fences

## Key Constructs
```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

docs = [
    Document(
        page_content="export function validateSession(token) { ... }",
        metadata={"source": "src/auth/sessionManager.ts", "type": "middleware"}
    ),
]

vector_store = FAISS.from_documents(docs, embeddings)
matches = vector_store.similarity_search(user_query, k=2)

# Build context from ALL matched chunks
context = ""
for doc in matches:
    context += f"File: {doc.metadata['source']}\n```typescript\n{doc.page_content}\n```\n\n"

# Inject as system context
system_prompt = f"Answer using ONLY this code:\n{context}\nQuery: {user_query}"
```

## Mental Model
Code snippets are vectors in semantic space. A question like "how does session validation work?" lands near the `validateSession` function vector. FAISS finds nearest neighbors in milliseconds without scanning every document.

## Pitfalls
- Retrieving only k=1 misses related files — use k=2+ for multi-file features
- Forgetting to loop through all `k` results wastes retrieval work
- Missing metadata keys raise `KeyError` — use `doc.metadata.get("source", "unknown")` defensively
- Always include `"INSUFFICIENT_DATA"` fallback instruction in the prompt to prevent hallucinated APIs
