# Parent-Child Retrieval

**File:** `03_retrieval/parent_child_retrieval.ipynb`

## Core Concept
A two-tier retrieval strategy: index small child chunks for precise vector search, but return the full parent chunk to the LLM for richer context. Solves the precision vs. context trade-off in standard RAG.

## What You Learn
- Split documents into large parent chunks (2000 tokens) and small child chunks (150 tokens)
- Store parents in a key-value `InMemoryStore`, index children in FAISS
- Link child chunks to their parent via `parent_id` in metadata
- Build a custom `BaseRetriever` subclass that searches children and returns parents
- Add multi-turn `ChatMessageHistory` for conversational follow-ups

## Key Constructs
```python
from langchain_core.retrievers import BaseRetriever
from langchain.storage import InMemoryStore

class ParentChildRetriever(BaseRetriever):
    vectorstore: FAISS
    docstore: InMemoryStore

    def add_documents(self, documents):
        for doc in documents:
            parent_chunks = parent_splitter.split_documents([doc])
            for i, parent in enumerate(parent_chunks):
                parent_id = f"{doc.metadata['source']}_p_{i}"
                self.docstore.mset([(parent_id, parent)])

                child_chunks = child_splitter.split_documents([parent])
                for child in child_chunks:
                    child.metadata["parent_id"] = parent_id
                self.vectorstore.add_documents(child_chunks)

    def _get_relevant_documents(self, query, **kwargs):
        children = self.vectorstore.similarity_search(query, k=2)
        parent_ids = list(set(c.metadata["parent_id"] for c in children))
        return [d for d in self.docstore.mget(parent_ids) if d]
```

## Why Parent-Child vs Standard RAG
| | Standard RAG | Parent-Child |
|---|---|---|
| Indexed chunk | Same chunk returned | Small child (precise match) |
| Returned to LLM | Same small chunk | Full parent (rich context) |
| Precision | Medium | High |
| Context richness | Low | High |

## Pitfalls
- `parent_id` generation must be deterministic — same doc always maps to same ID
- `InMemoryStore` is lost on restart — use Redis or Postgres for production
- Always dedup parent IDs before fetching (`set()`) to avoid returning duplicate context
- Save both user and AI messages to chat history or follow-up questions lose context
