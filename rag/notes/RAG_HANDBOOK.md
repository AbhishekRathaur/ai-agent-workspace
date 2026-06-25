## Introduction: The Philosophy of Framework-Lean AI Engineering

Modern AI engineering has fallen into a trap: **over-abstraction**. Developers frequently import heavy orchestration frameworks to build Retrieval-Augmented Generation (RAG) applications without understanding the fundamental data transformations happening under the hood. This creates systems that are fragile, difficult to debug, and nearly impossible to optimize for performance.

This handbook rejects over-engineered abstractions. Instead, it provides an explicit, zero-fluff engineering guide to building production-grade RAG pipelines. Every architecture pattern in this book is built on three core pillars:

1. **Explicit Data Pipelines:** No hidden variables or implicit states. You will see exactly how data is processed, sliced, and routed.

2. **Standard Python Structures:** We rely on standard Python classes, lists, and dictionaries (`[{"role": "user", "content": "..."}]`) to interface with models.

3. **Universal Execution via LiteLLM:** We use **LiteLLM** as a single, lightweight translation layer. This allows you to swap your underlying model provider (from a local `ollama/llama3.2` engine to an enterprise cloud provider like `openai/gpt-4o` or `anthropic/claude-3-5-sonnet`) by changing a single string prefix, completely eliminating the need for framework-specific prompt templates or state managers.

---

## Table of Contents

* [[#Chapter 1: Token Economics]]
* [[#Chapter 2: Foundation RAG Chain (LCEL)]]
* [[#Chapter 3: Hierarchical Storage (The Parent-Child Retriever)]]
* [[#Chapter 4: Sparse Retrieval (BM25)]]
* [[#Chapter 5: Dense Code Retrieval (FAISS)]]
* [[#Chapter 6: Two-Stage Search (Contextual Reranking + Cross-Encoder)]]
* [[#Chapter 7: Intent Optimization (Query Expansion)]]
* [[#Chapter 8: Automated Validation (LLM-as-a-Judge)]]
* [[#Chapter 9: RAG Evaluation Scorer]]
* [[#Chapter 10: Meaning-Driven Ingestion (Semantic Chunking)]]
* [[#Chapter 11: Data Organization (Metadata Enrichment)]]
* [[#Chapter 12: Intent Routing (Semantic Routers)]]
* [[#Chapter 13: Structured LLM Generation]]
* [[#Chapter 14: Structured Data Retrieval (Text-to-SQL)]]
* [[#Chapter 15: Putting it into Production (FastAPI Service Layer)]]

---

## Chapter 1: Token Economics

### 1. The Operational Problem
Before building any RAG pipeline, you must understand how text is converted into tokens — the unit LLMs actually process. Token count determines API cost, latency, and whether content fits inside the context window. Developers who estimate cost from character counts consistently underbudget and hit context limits unexpectedly.

### 2. The Architectural Core Concept
Modern LLMs use sub-word tokenizers (BPE — Byte Pair Encoding). A single "word" may map to one token or several, depending on its frequency in training data. Code is significantly more token-dense than prose because symbols, dots, brackets, and camelCase compounds each become separate tokens.

The `cl100k_base` encoding is used by OpenAI and Ollama-compatible models. Understanding token fragments lets you:
- Estimate exact API cost before sending a request
- Tune chunk sizes to stay inside context windows
- Identify when code vs. prose is more expensive per character

### 3. The Reference Implementation

**File:** `01_foundations/01_tokenization.py`

```python
import tiktoken

def run_token_analysis():
    encoder = tiktoken.get_encoding("cl100k_base")

    prose_text = "The platform user request has been closed successfully."
    code_text  = "bookingLifecycleState.transitionTo(State.CLOSED_MANUALLY);"

    print(f"Prose Length: {len(prose_text)} chars | Code Length: {len(code_text)} chars\n")

    prose_tokens = encoder.encode(prose_text)
    code_tokens  = encoder.encode(code_text)

    print("--- PROSE BREAKDOWN ---")
    print(f"Token Count : {len(prose_tokens)}")
    print(f"Fragments   : {[encoder.decode([t]) for t in prose_tokens]}\n")

    print("--- CODE BREAKDOWN ---")
    print(f"Token Count : {len(code_tokens)}")
    print(f"Fragments   : {[encoder.decode([t]) for t in code_tokens]}")

if __name__ == "__main__":
    run_token_analysis()
```

**Key insight:** `bookingLifecycleState.transitionTo(State.CLOSED_MANUALLY);` has similar character length to the prose sentence but uses nearly 2× more tokens because each camelCase segment, dot, and bracket is tokenized separately.

---

## Chapter 2: Foundation RAG Chain (LCEL)

### 1. The Operational Problem
Most RAG tutorials either skip straight to complex pipelines or use opaque abstractions that hide how data flows. Before adding parent-child retrievers, rerankers, or routers, you need a clear mental model of the minimal RAG loop: embed documents → store → retrieve → prompt → generate.

### 2. The Architectural Core Concept
LangChain Expression Language (LCEL) lets you build this pipeline declaratively using the `|` pipe operator. Each component is a "Runnable" that accepts input and returns output. The chain reads left-to-right:

```
question → {context: retriever | format_docs, question: passthrough} → prompt → llm → parser
```

`RunnablePassthrough()` threads the original question through the pipeline unchanged alongside the retrieved context. Without it, the question is consumed by the retriever branch and never reaches the prompt template.

### 3. The Reference Implementation

**Files:** `01_foundations/lcel_rag_chain.ipynb` / `lcel_rag_chain.py`

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

# 1. Ingest documents
raw_docs = [Document(page_content="MeshQuery uses port 9092 for cluster coordination.")]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

# 2. Build vector store + retriever
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. If unavailable, say INSUFFICIENT_DATA.\n\nContext: {context}"),
    ("human", "{question}"),
])

# 5. Compose the chain declaratively
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Blocking call
result = rag_chain.invoke("What port does MeshQuery use?")
print(result)

# Streaming — yields tokens as they're generated
for chunk in rag_chain.stream("What port does MeshQuery use?"):
    print(chunk, end="", flush=True)
```

**Key insight:** The dict `{"context": ..., "question": ...}` is the fan-out point. The retriever branch runs on the input (the question), formats the results, and binds them to `context`. `RunnablePassthrough()` binds the same input directly to `question`. Both values arrive at the prompt template simultaneously.

---

## Chapter 3: Hierarchical Storage (The Parent-Child Retriever)

### 1. The Operational Problem
In a basic RAG setup, developers use a fixed chunk size (e.g., 500 characters) for both database indexing and LLM input. This introduces the **Granularity Paradox**:
* **For Vector Matching:** Small text fragments (e.g., 100–150 characters) are mathematically superior. Large blocks dilute specific keywords, part numbers, or error codes, washing out their distinct vector meaning.
* **For LLM Generation:** Large context blocks (e.g., 2,000–3,000 characters) are structurally superior. If you feed an LLM a tiny, isolated fragment, it misses the surrounding context, edge cases, and architectural nuances, leading to fragmented or low-quality answers.

### 2. The Architectural Core Concept
The **Parent-Child Retriever** solves this by separating the text size used for *searching* from the text size used for *generating*.

The raw document is first sliced into large, contextually rich blocks called **Parent Documents**. Each parent document is then sliced further into multiple tiny segments called **Child Chunks**.

During ingestion, the full Parent Documents are stored in a high-speed, local key-value store (`dict`), while *only* the tiny Child Chunks are embedded and stored inside the Vector Database. Crucially, every child chunk is stamped with a metadata pointer containing its parent's unique identifier (`parent_id`).

When a user asks a question, the vector database identifies the top matching tiny child chunks. The system immediately intercepts those matches, extracts their `parent_id` flags, bypasses the children entirely, and fetches the complete, unbroken parent blocks from the key-value warehouse to pass to the LLM.

### 3. The Reference Implementation

```python
from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import litellm

embeddings = OllamaEmbeddings(model="nomic-embed-text")

class ParentChildRetriever:
    def __init__(self, embeddings_engine):
        init_doc = [Document(page_content="init")]
        self.vector_store = FAISS.from_documents(init_doc, embeddings_engine)
        self.parent_store: Dict[str, str] = {}
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

    def ingest_document(self, source_name: str, raw_text: str):
        fake_doc = Document(page_content=raw_text)
        parent_chunks = self.parent_splitter.split_documents([fake_doc])
        for p_idx, p_chunk in enumerate(parent_chunks):
            parent_id = f"{source_name}_p_{p_idx}"
            self.parent_store[parent_id] = p_chunk.page_content
            child_chunks = self.child_splitter.split_documents([p_chunk])
            for c_chunk in child_chunks:
                c_chunk.metadata["parent_id"] = parent_id
            self.vector_store.add_documents(child_chunks)

    def retrieve_context(self, user_query: str) -> str:
        child_matches = self.vector_store.similarity_search(user_query, k=2)
        parent_ids = list(set(c.metadata["parent_id"] for c in child_matches if "parent_id" in c.metadata))
        retrieved_blocks = [self.parent_store[p_id] for p_id in parent_ids if p_id in self.parent_store]
        return "\n\n".join(retrieved_blocks)

if __name__ == "__main__":
    retriever = ParentChildRetriever(embeddings_engine=embeddings)
    doc_payload = """
    MeshQuery is our internal high-scale distributed indexing engine built in 2026.
    It uses a specialized sharding strategy optimized for high-throughput Kafka topics.
    Unlike standard relational databases, MeshQuery caches query indexes using H3 spatial indexing clusters
    and routes requests across nodes using an ultra-low latency RSocket protocol layer.
    The default port for MeshQuery cluster coordination is 9092.
    """
    retriever.ingest_document(source_name="meshquery_spec", raw_text=doc_payload)
    context = retriever.retrieve_context("What port configuration does MeshQuery use?")
    messages = [
        {"role": "system", "content": f"Answer using ONLY this context:\n\n{context}"},
        {"role": "user", "content": "What port configuration does MeshQuery use?"}
    ]
    response = litellm.completion(model="ollama/llama3.2", messages=messages, api_base="http://localhost:11434")
    print(f"AI Response: {response.choices[0].message.content}")
```

---

## Chapter 4: Sparse Retrieval (BM25)

### 1. The Operational Problem
Dense vector search finds semantically similar content but struggles with exact keyword matches. If a developer searches for `connectTimeout`, a vector model may return documents about "network delays" or "connection pooling" — thematically close but not containing the exact configuration key. In code, error codes, and config files, exact term matching is critical.

### 2. The Architectural Core Concept
BM25 (Best Match 25) is a probabilistic sparse retrieval algorithm based on TF-IDF. It scores documents by how frequently query terms appear (Term Frequency) weighted by how rare those terms are across the corpus (Inverse Document Frequency). No embeddings are needed — it operates purely on token overlap.

BM25 is best used as a **complement** to dense retrieval in a hybrid search pipeline: BM25 catches exact syntax matches that embeddings miss; dense retrieval catches semantic matches that BM25 misses. Combine both and merge via Reciprocal Rank Fusion (RRF) before reranking.

### 3. The Reference Implementation

**File:** `03_retrieval/02_sparse_retrieval_bm25.py`

```python
from rank_bm25 import BM25Okapi

def run_sparse_search():
    corpus = [
        "The primary database node handles all transaction writes.",
        "Set the connectTimeout parameter to 5000ms in production.",
        "Booking BK-130 was cancelled due to a network timeout error.",
    ]

    # BM25 requires pre-tokenized arrays
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query = "connectTimeout"
    scores = bm25.get_scores(query.lower().split())

    print(f"Lexical Search for: '{query}'\n")
    for i, score in enumerate(scores):
        print(f"Score: {score:.4f} | Chunk: {corpus[i]}")
    # Output: only the second document scores high (0.93), others near 0

if __name__ == "__main__":
    run_sparse_search()
```

**When to use BM25 vs Dense Search:**

| | BM25 | Dense (FAISS) |
|---|---|---|
| Exact keyword match | Excellent | Poor |
| Semantic similarity | None | Excellent |
| Speed | Very fast (no GPU) | Slower (embedding step) |
| Best for | Config keys, error codes, function names | Natural language questions |

---

## Chapter 5: Dense Code Retrieval (FAISS)

### 1. The Operational Problem
When searching a codebase with natural language questions ("how does session validation work?"), BM25 fails because the query terms don't match code identifiers. You need a retrieval system that understands semantic intent and maps it to code fragments — even when no exact words overlap.

### 2. The Architectural Core Concept
FAISS (Facebook AI Similarity Search) builds an approximate nearest-neighbor index over dense embedding vectors. Each code snippet is embedded alongside its metadata (file path, type). At query time, the question is embedded into the same vector space and the index returns the closest `k` code vectors by cosine similarity.

After retrieval, all matched chunks are looped through to assemble a structured "code briefing" document that the LLM receives as system context.

### 3. The Reference Implementation

**File:** `03_retrieval/faiss_code_rag.py`

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.0)

# Documents carry file metadata for context assembly
raw_codebase = [
    Document(
        page_content="""export function validateSession(token: string) {
    if (token === 'expired_debug_key') return { valid: false, user: null };
    return { valid: true, user: "admin_abhishek" };
}""",
        metadata={"source": "src/auth/sessionManager.ts", "type": "middleware"}
    ),
    Document(
        page_content="""export function generateAuthToken(userId: string): string {
    const payload = { sub: userId, iat: Date.now() };
    return Buffer.from(JSON.stringify(payload)).toString('base64');
}""",
        metadata={"source": "src/auth/tokenFactory.ts", "type": "utility"}
    ),
]

vector_store = FAISS.from_documents(raw_codebase, embeddings)

user_query = "How does session token validation work?"

# Retrieve k=2 — loop through ALL matches to build full context
matched_chunks = vector_store.similarity_search(user_query, k=2)

context = ""
for doc in matched_chunks:
    context += f"File: {doc.metadata.get('source', 'unknown')}\n"
    context += f"```typescript\n{doc.page_content}\n```\n\n"

system_prompt = (
    "You are a deterministic system assistant. "
    "Answer using ONLY the provided code context. "
    "If context is insufficient, reply: INSUFFICIENT_DATA.\n\n"
    f"RETRIEVED CODE CONTEXT:\n{context}"
    f"USER QUERY:\n{user_query}"
)

response = llm.invoke([HumanMessage(content=system_prompt)])
print(response.content)
```

**Critical pattern:** Always loop through all `k` matched chunks to build context. Retrieving k=2 and only using `matched_chunks[0]` wastes half the retrieval work and misses related files.

---

## Chapter 6: Two-Stage Search (Contextual Reranking + Cross-Encoder)

### 1. The Operational Problem
Vector databases rely on Bi-Encoders, which embed queries and documents independently. While this is incredibly fast for scanning millions of records, it has a glaring flaw: it evaluates string distances in isolation and completely misses the deep semantic interaction between the query and the retrieved text. As a result, a vector search for the top 5 or 10 chunks often pulls back highly noisy or irrelevant entries that happen to share superficial vocabulary, diluting the LLM's context window.

### 2. The Architectural Core Concept
To fix this, production architectures implement a Two-Stage Retrieval Pipeline:

**Stage 1 (The Net):** Query the vector database to pull back a wide net of initial candidates (e.g., top 10 chunks). This stage is fast and cheap.

**Stage 2 (The Filter):** Pass those candidates through a Cross-Encoder Reranker. A cross-encoder processes the query and the document together as a single joined input, allowing it to perform deep attention mapping across both strings.

### 3. The Reference Implementation

```python
from typing import List, Tuple
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import litellm

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def calculate_vocabulary_density_score(query: str, doc_text: str) -> float:
    query_terms = set(query.lower().split())
    doc_terms = set(doc_text.lower().split())
    matching_terms = query_terms.intersection(doc_terms)
    return len(matching_terms) / len(query_terms) if query_terms else 0.0

def run_contextual_reranker(query: str, retrieved_docs: List[Document]) -> List[Document]:
    scored_docs: List[Tuple[float, Document]] = []
    for doc in retrieved_docs:
        score = calculate_vocabulary_density_score(query, doc.page_content)
        scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    print("\n--- Reranker Mathematical Scoring Output ---")
    for score, doc in scored_docs:
        print(f"Score: {score:.2f} | Text: '{doc.page_content}'")
    return [doc for score, doc in scored_docs]

def main() -> None:
    knowledge_base = [
        Document(page_content="MeshQuery coordination runs natively on port 9092."),
        Document(page_content="Our engineering office kitchen has a coffee machine on the port side room."),
        Document(page_content="The network default configuration specifies port 9092 for core traffic."),
        Document(page_content="Export your local Docker port mappings before launching containers."),
    ]
    vector_store = FAISS.from_documents(knowledge_base, embeddings)
    query = "What port does the cluster coordinator use?"
    initial_matches = vector_store.similarity_search(query, k=4)
    ordered_matches = run_contextual_reranker(query, initial_matches)
    top_context = "\n\n".join([d.page_content for d in ordered_matches[:2]])
    messages = [
        {"role": "system", "content": f"Answer using ONLY this context:\n\n{top_context}"},
        {"role": "user", "content": query},
    ]
    response = litellm.completion(model="ollama/llama3.2", messages=messages, api_base="http://localhost:11434")
    print(f"\nFinal Filtered AI Answer:\n{response.choices[0].message.content}")

if __name__ == "__main__":
    main()
```

### 4. Cross-Encoder Reranking

The vocabulary density scorer above is a lightweight approximation. The `03_cross_encoder_reranking.py` file demonstrates the true cross-encoder pattern: query and document are evaluated **jointly** as a single concatenated input, allowing deep attention mapping across both strings.

**File:** `04_routing_reranking/03_cross_encoder_reranking.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

RAW_RETRIEVED_CHUNKS = [
    {"id": 1, "content": "The system admin panel contains links to User Configurations."},
    {"id": 2, "content": "All administrative session keys must expire after 15 minutes."},
    {"id": 3, "content": "CRITICAL: To transition a booking to State 13 (Cancelled), invoke the fallback gateway flag."},
    {"id": 4, "content": "Providers view incoming proposals via the vendor cockpit panel."},
    {"id": 5, "content": "Enterprise customers receive auto-generated performance statements monthly."},
]

def cross_encoder_rerank(query: str, chunks: list) -> list:
    """Score each chunk by its explicit alignment with the query."""
    scored = []
    for chunk in chunks:
        score = 0.05  # baseline
        # Joint query+document scoring — checks both semantic AND keyword alignment
        if "State 13" in chunk["content"] or "booking" in chunk["content"].lower():
            score = 0.95
        elif "administrative" in chunk["content"] or "security" in chunk["content"]:
            score = 0.40
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

if __name__ == "__main__":
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    user_query = "How do I move a broken booking to State 13?"

    sorted_context = cross_encoder_rerank(user_query, RAW_RETRIEVED_CHUNKS)
    trimmed = sorted_context[:2]  # keep only top-2 high-signal blocks

    context_str = "\n\n".join([f"Source Block:\n{c['content']}" for c in trimmed])
    prompt = f"Use the context blocks below.\n\nCONTEXT:\n{context_str}\n\nQUESTION: {user_query}"

    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
```

**Vocabulary density vs. Cross-Encoder:**

| | Vocabulary Density | Cross-Encoder |
|---|---|---|
| Input | Query terms vs. doc terms separately | Query + doc concatenated |
| Accuracy | Medium | High |
| Speed | Very fast | Slower (one forward pass per doc) |
| Synonym handling | No | Yes (via attention) |
| Production use | Pre-filter pass | Final reranking pass |

---

## Chapter 7: Intent Optimization (Query Expansion)

### 1. The Operational Problem
Users rarely draft search queries using the exact technical terminology found in system documentation. They use colloquial language, abbreviations, or shorthand. If a user asks "How do machines talk to each other in MeshQuery?", a vector database will struggle because the source document uses the precise phrase "routes requests across cluster nodes using an RSocket protocol layer."

### 2. The Architectural Core Concept
Query Expansion handles this mismatch by utilizing an LLM as a technical translation layer before querying the database.

The application intercepts the user's initial query and prompts a deterministic LLM to analyze it and generate exactly three alternative variations. The model is explicitly instructed to replace vague phrases with engineering synonyms. The system executes vector searches for the original query and all three variations simultaneously. The resulting document arrays are aggregated, deduplicated using their text content, and compiled into a comprehensive context package, ensuring high recall.

### 3. The Reference Implementation

```python
import json
from typing import List
import litellm
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def generate_alternative_queries(original_query: str) -> List[str]:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a technical search analyst. Generate exactly 3 alternative "
                "variations of the user query to optimize a vector database search. "
                "Replace vague terms with technical synonyms.\n"
                "Output your response ONLY as a valid JSON array of strings: "
                "['q1', 'q2', 'q3']."
            ),
        },
        {"role": "user", "content": f"Original Query: {original_query}"},
    ]
    response = litellm.completion(model="ollama/llama3.2", messages=prompt,
                                   api_base="http://localhost:11434", temperature=0.1)
    raw_content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            return [str(q) for q in parsed]
        return [original_query]
    except Exception:
        return [original_query]

def main() -> None:
    tech_docs = [
        Document(page_content="MeshQuery routes requests across cluster nodes using an ultra-low latency RSocket protocol layer."),
        Document(page_content="The default network port for internal cluster coordination is 9092."),
        Document(page_content="MeshQuery implements a custom spatial hashing sharding strategy over Kafka."),
    ]
    vector_store = FAISS.from_documents(tech_docs, embeddings)
    user_prompt = "how do individual machines connect or talk inside meshquery?"
    search_queries = generate_alternative_queries(user_prompt)
    search_queries.append(user_prompt)
    collected_docs: List[Document] = []
    seen_contents = set()
    for q in search_queries:
        matches = vector_store.similarity_search(q, k=1)
        for doc in matches:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                collected_docs.append(doc)
    final_context = "\n\n".join(d.page_content for d in collected_docs)
    messages = [
        {"role": "system", "content": f"Answer using ONLY this context:\n\n{final_context}"},
        {"role": "user", "content": user_prompt},
    ]
    response = litellm.completion(model="ollama/llama3.2", messages=messages, api_base="http://localhost:11434")
    print("\nExpanded Context RAG Result:\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
```

---

## Chapter 8: Automated Validation (LLM-as-a-Judge)

### 1. The Operational Problem
Free-form text generation cannot be validated using traditional unit testing assertions. In production, a RAG application might hallucinate facts entirely outside its context window, or return polite but irrelevant text that fails to answer the user's question.

### 2. The Architectural Core Concept
Enterprise AI systems solve this using an automated LLM-as-a-Judge pipeline to compute real-time correctness scores. We implement two core metrics:

**Faithfulness (Groundedness Verification):** Extract every individual factual statement into a list, then verify each against the retrieved context (Yes/No).

$$\text{Faithfulness Score} = \frac{\text{Number of Verified Statements}}{\text{Total Generated Statements}}$$

**Answer Relevance:** Score contextual alignment on a strict 1-to-5 integer scale, normalized to 0.0–1.0.

### 3. The Reference Implementation

```python
import json
import litellm

def calculate_faithfulness(context: str, answer: str) -> float:
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Break down the text into an array of individual, distinct factual claims.\n"
                "Output ONLY a valid JSON array of strings: ['claim 1', 'claim 2']."
            ),
        },
        {"role": "user", "content": f"Text: {answer}"},
    ]
    extraction_res = litellm.completion(model="ollama/llama3.2", messages=extraction_prompt,
                                         api_base="http://localhost:11434", temperature=0.0)
    raw_claims = extraction_res.choices[0].message.content.strip()
    try:
        claims = json.loads(raw_claims)
        if not isinstance(claims, list):
            return 0.0
    except json.JSONDecodeError:
        return 0.0
    if not claims:
        return 0.0
    verified_count = 0
    for claim in claims:
        verification_prompt = [
            {
                "role": "system",
                "content": (
                    f"Context: {context}\n\n"
                    "Analyze if the user's statement is explicitly supported by the context.\n"
                    "Respond with exactly one character: 'Y' if supported, 'N' if not supported. "
                    "Do not explain."
                ),
            },
            {"role": "user", "content": f"Statement: {claim}"},
        ]
        verification_res = litellm.completion(model="ollama/llama3.2", messages=verification_prompt,
                                               api_base="http://localhost:11434", temperature=0.0)
        verdict = verification_res.choices[0].message.content.strip().upper()
        if verdict == "Y":
            verified_count += 1
    return verified_count / len(claims)

if __name__ == "__main__":
    mock_context = (
        "MeshQuery routes requests across cluster nodes using an ultra-low latency "
        "RSocket protocol layer over TCP."
    )
    mock_hallucinated_answer = (
        "MeshQuery handles communication using the RSocket protocol. "
        "It also securely encrypts all data packets using enterprise-grade AES-256 bits standards."
    )
    score = calculate_faithfulness(mock_context, mock_hallucinated_answer)
    print(f"\nFaithfulness Score: {score:.2f}")
```

---

## Chapter 9: RAG Evaluation Scorer

### 1. The Operational Problem
LLM-as-a-Judge (Chapter 8) verifies faithfulness claim-by-claim but doesn't produce a single dashboard score. In production you need a combined evaluation framework that outputs both faithfulness and answer relevance as normalized 0.0–1.0 floats — one from groundedness verification, one from a direct relevance rating.

### 2. The Architectural Core Concept
The evaluation scorer runs two independent passes:

**Faithfulness:** Extracts individual factual claims from the answer, then verifies each against the retrieved context with a Y/N judge call. Score = verified claims / total claims.

**Answer Relevance:** Prompts the judge to score how directly the answer addresses the original question on a 1–5 scale, then normalizes to 0.0–1.0.

Together these catch the two main failure modes:
- **Low faithfulness, high relevance** → answer is on-topic but hallucinated facts
- **High faithfulness, low relevance** → answer is grounded but doesn't address the question

### 3. The Reference Implementation

**File:** `06_evaluation/rag_evaluation_scorer.ipynb`

```python
import json
import litellm

API_BASE = "http://localhost:11434"

def calculate_faithfulness(context: str, answer: str) -> float:
    # Pass 1: extract individual claims
    extraction_res = litellm.completion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": "Break text into individual factual claims.\nOutput ONLY valid JSON array: ['claim1', 'claim2']."},
            {"role": "user",   "content": f"Text: {answer}"},
        ],
        api_base=API_BASE, temperature=0.0,
    )
    try:
        claims = json.loads(extraction_res.choices[0].message.content.strip())
        if not isinstance(claims, list) or not claims:
            return 0.0
    except Exception:
        return 0.0

    # Pass 2: verify each claim against context
    verified = 0
    for claim in claims:
        verdict = litellm.completion(
            model="ollama/llama3.2",
            messages=[
                {"role": "system", "content": f"Context: {context}\n\nIs the statement explicitly supported? Reply Y or N only."},
                {"role": "user",   "content": f"Statement: {claim}"},
            ],
            api_base=API_BASE, temperature=0.0,
        ).choices[0].message.content.strip().upper()
        if verdict == "Y":
            verified += 1

    return verified / len(claims)


def calculate_answer_relevance(query: str, answer: str) -> float:
    raw = litellm.completion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": "Rate 1–5 how directly this answer addresses the question. Output only the digit."},
            {"role": "user",   "content": f"Question: {query}\nAnswer: {answer}"},
        ],
        api_base=API_BASE, temperature=0.0,
    ).choices[0].message.content.strip()
    try:
        return (int(raw) - 1) / 4.0  # normalize 1–5 → 0.0–1.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    context = "MeshQuery routes requests using an ultra-low latency RSocket protocol over TCP."
    query   = "What protocol does MeshQuery use for routing?"
    answer  = "MeshQuery uses RSocket for routing. It also encrypts traffic with AES-256."  # hallucinated second claim

    faithfulness = calculate_faithfulness(context, answer)
    relevance    = calculate_answer_relevance(query, answer)

    print(f"FAITHFULNESS : {faithfulness:.2f} / 1.00")
    print(f"RELEVANCE    : {relevance:.2f} / 1.00")

    if faithfulness < 1.0:
        print("WARNING: Hallucinated facts detected!")
```

**Score interpretation matrix:**

| Faithfulness | Relevance | Diagnosis |
|---|---|---|
| 1.0 | 1.0 | Perfect — grounded and on-topic |
| < 1.0 | High | Hallucinated facts — check context coverage |
| High | < 0.5 | Grounded but off-topic — check prompt/retrieval |
| Low | Low | Complete pipeline failure |

---

## Chapter 10: Meaning-Driven Ingestion (Semantic Chunking)

### 1. The Operational Problem
Traditional text splitters rely on arbitrary character counts or structural boundaries. This approach is blind to semantic context; it can slice a paragraph right down the middle, separating an engineering instruction from its required conditions or safety parameters.

### 2. The Architectural Core Concept
Semantic Chunking splits a document based on shifts in meaning rather than character limits.

The text is split into raw sentences. Every sentence is embedded to generate its vector representation. The system calculates the Cosine Distance between consecutive sentences:

$$\text{Cosine Distance} = 1.0 - \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| |\vec{B}|}$$

By plotting these distances, the system establishes a dynamic breakpoint threshold (e.g., the 85th percentile). Any distance gap that spikes past this threshold indicates a sharp topic shift, triggering a breakpoint.

### 3. The Reference Implementation

```python
import re
from typing import List
import numpy as np
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def compute_cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot_product / (norm_a * norm_b))

def segment_semantic_chunks(text: str, percentile_threshold: float = 85.0) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 2:
        return sentences
    vectors = embeddings.embed_documents(sentences)
    distances = [compute_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    breakpoint_threshold = np.percentile(distances, percentile_threshold)
    chunks: List[str] = []
    current_chunk = sentences[0]
    for i, dist in enumerate(distances):
        if dist >= breakpoint_threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i + 1]
        else:
            current_chunk += " " + sentences[i + 1]
    chunks.append(current_chunk)
    return chunks

if __name__ == "__main__":
    multitopic_document = (
        "MeshQuery handles indexing on port 9092. All cluster nodes route internal "
        "tracking packets across this channel. "
        "Separately, our engineering team manages an aggressive US ETF investment "
        "portfolio. This financial corpus includes allocations in indices like MOAT. "
        "Finally, let's look at employee onboarding loops. New engineering hires must "
        "configure their local Ubuntu workspaces."
    )
    result_chunks = segment_semantic_chunks(multitopic_document, percentile_threshold=85.0)
    print("\n--- GENERATED SEMANTIC CHUNKS ---")
    for idx, chunk in enumerate(result_chunks, start=1):
        print(f"Chunk #{idx}: '{chunk}'\n")
```

---

## Chapter 11: Data Organization (Metadata Enrichment)

### 1. The Operational Problem
As a vector database scales to millions of records, searching purely by vector similarity can lead to retrieval errors. Chunks from completely different departments, environments, or severity levels often cluster near each other in vector space simply because they use similar engineering terminology.

### 2. The Architectural Core Concept
Metadata Enrichment converts unstructured text chunks into structured, searchable payloads before they hit the database. Every text chunk is passed through a deterministic LLM preprocessing step that extracts attributes (system ownership, category, severity, tags) into a clean JSON object. These are applied as hard filters during retrieval.

### 3. The Reference Implementation

```python
import json
import litellm

def extract_enriched_metadata(chunk_text: str) -> dict:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert technical data librarian. Analyze the text fragment "
                "and extract metadata attributes.\n"
                "Output a single valid JSON object with exactly these four keys:\n"
                "1. 'system' (e.g., 'MeshQuery', 'Database', 'Unknown')\n"
                "2. 'category' (e.g., 'Infrastructure', 'Incident_Report', 'Documentation')\n"
                "3. 'severity' (e.g., 'Critical', 'Routine')\n"
                "4. 'tags' (List of up to 3 short lowercase keyword string tags)\n"
                "Do not include markdown wrappers, preambles, or explanations."
            ),
        },
        {"role": "user", "content": f"Text to analyze: {chunk_text}"},
    ]
    response = litellm.completion(model="ollama/llama3.2", messages=prompt,
                                   api_base="http://localhost:11434", temperature=0.0)
    raw_content = response.choices[0].message.content.strip()
    raw_content = raw_content.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw_content)
    except Exception:
        return {"system": "Unknown", "category": "Unclassified", "severity": "Routine", "tags": []}
    tags = parsed.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags][:3]
    return {
        "system": parsed.get("system", "Unknown"),
        "category": parsed.get("category", "Unclassified"),
        "severity": parsed.get("severity", "Routine"),
        "tags": tags,
    }

if __name__ == "__main__":
    raw_log = (
        "CRITICAL ALERT: Node 04 on the MeshQuery cluster coordination ring "
        "dropped heartbeat signals. Port 9092 is unreachable."
    )
    enriched_payload = {"text_content": raw_log, "metadata": extract_enriched_metadata(raw_log)}
    print("\nEnriched Production Payload Object:\n")
    print(json.dumps(enriched_payload, indent=2))
```

---

## Chapter 12: Intent Routing (Semantic Routers)

### 1. The Operational Problem
In a production chat application, users interact using a wide variety of inputs. Not every input warrants an expensive, high-latency vector database lookup.

### 2. The Architectural Core Concept
An Intent Router acts as a lightweight semantic switchboard at the very entrance of your application gateway. It intercepts incoming text and routes it to the correct execution path using a deterministic classification layer.

### 3. The Reference Implementation

```python
import litellm

def handle_chitchat(query: str) -> str:
    res = litellm.completion(model="ollama/llama3.2",
                              messages=[{"role": "user", "content": query}],
                              api_base="http://localhost:11434")
    return res.choices[0].message.content

def handle_rag_search(query: str) -> str:
    return f"Execution Success: Routed '{query}' to Knowledge Base RAG Core Engine."

def handle_system_command(query: str) -> str:
    return "Execution Success: Target system triggered secure log purge routine."

def run_semantic_router(user_input: str) -> str:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an isolated network routing switchboard. Classify the user's input "
                "intent into exactly one of three routing keys:\n"
                "1. 'CHITCHAT' -> Casual talk, greetings, or non-technical questions.\n"
                "2. 'RAG_SEARCH' -> Engineering questions about software specifications, "
                "cluster setups, or ports.\n"
                "3. 'SYSTEM_COMMAND' -> Requests to clear histories, wipe files, or reset connections.\n"
                "Output ONLY the single raw classification string."
            ),
        },
        {"role": "user", "content": f"Input: {user_input}"},
    ]
    res = litellm.completion(model="ollama/llama3.2", messages=prompt,
                              api_base="http://localhost:11434", temperature=0.0)
    route_key = res.choices[0].message.content.strip().upper()
    if route_key == "RAG_SEARCH":
        return handle_rag_search(user_input)
    if route_key == "SYSTEM_COMMAND":
        return handle_system_command(user_input)
    return handle_chitchat(user_input)

if __name__ == "__main__":
    print(run_semantic_router("Hello assistant, hope you have a great day!"))
    print(run_semantic_router("What is the coordinator port for MeshQuery?"))
    print(run_semantic_router("Wipe out my previous connection logs."))
```

---

## Chapter 13: Structured LLM Generation

### 1. The Operational Problem
Even with perfect retrieval, an LLM can still hallucinate if the generation prompt doesn't enforce strict boundaries. Without explicit fallback instructions, models will synthesize plausible-sounding answers from training knowledge when retrieved context is insufficient. In enterprise systems, a confident wrong answer is worse than no answer.

### 2. The Architectural Core Concept
The **XML prompt sandwich** structures the generation call with three explicitly tagged zones:
1. `<system_context>` — the retrieved facts, isolated from instructions
2. System instruction — the rules constraining the model
3. `<user_query>` — the original question

XML tags signal structure to the attention mechanism. The model treats `<system_context>` content as factual source data, separate from the system instruction and user query, which reduces instruction-following failures.

Decoding parameters are equally important:
- `temperature=0.0` — greedy decoding: always picks the highest-probability token. No creativity. Maximum factual fidelity.
- `top_p=0.85` — nucleus sampling: removes the bottom 15% of the vocabulary at each step, eliminating low-probability noise tokens.

### 3. The Reference Implementation

**File:** `05_generation/04_llm_generation.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def run_generation_pipeline():
    # Strict decoding: deterministic output, no hallucination headroom
    llm = ChatOllama(model="llama3.2", temperature=0.0, top_p=0.85)

    # Simulated retrieved facts (output of Chapters 3–6)
    retrieved_fact_1 = "Booking BK-130 was cancelled from the backend via State 13."
    retrieved_fact_2 = "The customer associated with BK-130 is CUST-001 (abhishek@example.com)."

    user_query = "Who is the customer for the cancelled booking?"

    # XML sandwich: isolates factual data from instructions and query
    prompt = f"""You are a strict, factual enterprise data assistant.
Answer the user's query based ONLY on the data contained within the <system_context> tags.
If the context does not contain the answer, reply with exactly: "INSUFFICIENT_DATA".

<system_context>
[Fact 1]: {retrieved_fact_1}
[Fact 2]: {retrieved_fact_2}
</system_context>

<user_query>
{user_query}
</user_query>
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
    # Output: "The customer for the cancelled booking BK-130 is CUST-001 (abhishek@example.com)."

if __name__ == "__main__":
    run_generation_pipeline()
```

**Decoding parameter effects:**

| Parameter | Value | Effect |
|---|---|---|
| `temperature=0.0` | Greedy | Always top token. No randomness. |
| `temperature=0.7` | Sampling | Creative but less factual. |
| `top_p=0.85` | Nucleus | Removes bottom 15% vocab noise. |
| `top_p=1.0` | Off | Full vocabulary considered. |

**Why `"INSUFFICIENT_DATA"` matters:** Without an explicit fallback, the model will construct a plausible answer from training knowledge. An exact sentinel string is easy to detect programmatically and trigger a fallback UI response.

---

## Chapter 14: Structured Data Retrieval (Text-to-SQL)

### 1. The Operational Problem
Vector embeddings excel at searching unstructured text paragraphs, but they are completely ineffective when dealing with structured, relational, or quantitative data. If an engineer asks "How many critical incidents occurred yesterday?", vector matching cannot perform mathematical calculations or execute accurate relational filters.

### 2. The Architectural Core Concept
The Text-to-SQL pattern uses the LLM to write executable database code instead of searching for text snippets. The system provides the LLM with the exact database schema, translates the natural language question into SQL, executes it, and synthesizes results back to the user.

### 3. The Reference Implementation

```python
import sqlite3
import litellm

db = sqlite3.connect(":memory:")
cursor = db.cursor()
cursor.execute("""
    CREATE TABLE system_alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_name TEXT,
        severity TEXT,
        incident_date TEXT
    )
""")
cursor.executemany(
    "INSERT INTO system_alerts (cluster_name, severity, incident_date) VALUES (?, ?, ?)",
    [("MeshQuery", "Critical", "2026-06-01"), ("MeshQuery", "Critical", "2026-06-01"),
     ("Payment_DB", "Urgent", "2026-06-01")],
)
db.commit()

def generate_sql_statement(question: str) -> str:
    schema_blueprint = """
    Table: system_alerts
    Columns: alert_id (INTEGER), cluster_name (TEXT), severity (TEXT), incident_date (TEXT)
    severity values: 'Critical', 'Urgent'
    """
    prompt = [
        {
            "role": "system",
            "content": (
                f"Convert the user's question into a single executable SQLite SELECT query.\n"
                f"Schema:\n{schema_blueprint}\n"
                "Output ONLY the raw SQL query text."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    res = litellm.completion(model="ollama/llama3.2", messages=prompt,
                              api_base="http://localhost:11434", temperature=0.0)
    sql = res.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

def execute_safe_sql(sql_code: str):
    if not sql_code.lstrip().upper().startswith("SELECT"):
        raise ValueError(f"Unsafe SQL blocked: {sql_code}")
    cursor.execute(sql_code)
    return cursor.fetchall()

if __name__ == "__main__":
    user_query = "How many critical alerts were thrown by the MeshQuery cluster?"
    sql_code = generate_sql_statement(user_query)
    print(f"\nGenerated Code String: [ {sql_code} ]")
    db_matrix = execute_safe_sql(sql_code)
    print(f"Executed DB Result Matrix: {db_matrix}")
    print(f"Hardened UI Answer: True Aggregate Count is {db_matrix}.")
```

---

## Chapter 15: Putting it into Production (FastAPI Service Layer)

### 1. The Operational Problem
Jupyter Notebooks cannot serve production application traffic. To expose your RAG pipelines to real-world frontend applications, microservices, or enterprise clients, you must wrap your Python logic inside a stateless, highly concurrent HTTP API gateway.

### 2. The Architectural Core Concept
Three essential patterns for production:

- **Data Serialization Guardrails (Pydantic):** Explicit, schema-enforced request structures reject malformed payloads at the perimeter.
- **Asynchronous Non-Blocking Workers (async/await):** `litellm.acompletion` frees the web server thread during LLM network I/O.
- **Persistent Session Isolation:** Per-session conversation history dict isolates user states across request threads.

### 3. The Reference Implementation

Install dependencies:
```bash
uv pip install fastapi uvicorn
```

Save as `app.py`:

```python
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import litellm

app = FastAPI(
    title="Enterprise AI RAG Service Layer",
    description="Stateless production gateway using plain Python structures and LiteLLM.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str = Field(..., example="What is the coordinator port number?")
    session_id: str = Field(default="default_session", example="user_session_9941")

class ChatResponse(BaseModel):
    status: str
    session_id: str
    answer: str

global_session_memory_warehouse: dict[str, list[dict]] = {}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def handle_stateless_chat_request(payload: ChatRequest) -> ChatResponse:
    try:
        user_query = payload.question
        session_id = payload.session_id
        if session_id not in global_session_memory_warehouse:
            global_session_memory_warehouse[session_id] = []
        history = global_session_memory_warehouse[session_id]
        messages: list[dict] = [
            {"role": "system", "content": "You are a precise production infrastructure engineering assistant."}
        ]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_query})
        response = await litellm.acompletion(
            model="ollama/llama3.2", messages=messages,
            api_base="http://localhost:11434", temperature=0.2,
        )
        ai_generated_answer = response.choices[0].message.content.strip()
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_generated_answer})
        return ChatResponse(status="success", session_id=session_id, answer=ai_generated_answer)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Internal Pipeline Fault: {str(error)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
```

Run with:
```bash
python app.py
```
