# LCEL RAG Chain

**File:** `01_foundations/lcel_rag_chain.ipynb` / `lcel_rag_chain.py`

## Core Concept
Build a complete RAG pipeline declaratively using LangChain Expression Language (LCEL). The pipe operator `|` chains embeddings → retrieval → prompt → LLM → parser into a single composable chain.

## What You Learn
- Compose a RAG chain using the `|` pipe operator
- Use `RunnablePassthrough()` to thread the question unchanged alongside retrieved context
- Format retrieved documents into a context string before injecting into the prompt
- Stream token-by-token output with `.stream()`

## Key Constructs
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Blocking
result = rag_chain.invoke("What is the session timeout?")

# Streaming
for chunk in rag_chain.stream("What is the session timeout?"):
    print(chunk, end="", flush=True)
```

## Mental Model
Each step in the chain is a "Runnable" that takes input and produces output. The dict `{"context": ..., "question": ...}` fans the input into two parallel branches, then merges them at the prompt template. `RunnablePassthrough()` passes the original question untouched through the pipeline.

## Pitfalls
- Forgetting `RunnablePassthrough()` loses the question before it reaches the prompt
- `RecursiveCharacterTextSplitter` with large overlap can retrieve redundant chunks
- The system prompt must explicitly restrict the LLM to context only — otherwise it will use training knowledge
