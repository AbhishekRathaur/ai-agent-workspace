# Structured LLM Generation

**File:** `05_generation/04_llm_generation.py`

## Core Concept
The final step of a RAG pipeline: feed retrieved facts to the LLM using a structured XML prompt, strict decoding parameters, and an explicit fallback instruction to prevent hallucination.

## What You Learn
- Set `temperature=0.0` for greedy decoding (maximum factual fidelity)
- Use `top_p` for nucleus sampling to limit the vocabulary to high-probability tokens
- Structure the prompt with XML tags to isolate context from instructions and query
- Force a `"INSUFFICIENT_DATA"` fallback when context doesn't contain the answer

## Key Constructs
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0.0, top_p=0.85)

prompt = f"""You are a strict, factual enterprise data assistant.
Answer ONLY from the data in <system_context>. If unavailable, reply: "INSUFFICIENT_DATA".

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
```

## Decoding Parameters
| Parameter | Value | Effect |
|---|---|---|
| `temperature=0.0` | Greedy | Always picks highest-probability token. No creativity. |
| `top_p=0.85` | Nucleus sampling | Removes bottom 15% of vocabulary at each step. Reduces noise. |

## Why XML Tags?
XML tags signal structure to the attention mechanism — the model treats `<system_context>` content as factual data, separate from system instructions and the user query. This reduces instruction following failures.

## Pitfalls
- Without `"INSUFFICIENT_DATA"` fallback the model will hallucinate an answer
- `temperature=0` is not the same as `temperature=None` — always set it explicitly for factual tasks
- XML tag names are arbitrary but should be consistent across all prompts in your system
