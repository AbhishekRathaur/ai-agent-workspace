# Query Routing

**File:** `04_routing_reranking/query_routing.ipynb`

## Core Concept
Classify incoming queries by intent and route them to different handlers. Avoids running expensive RAG retrieval on casual questions and prevents system commands from hitting the knowledge base.

## What You Learn
- Use an LLM with a strict role-play system prompt to classify query intent
- Map classification output to handler functions (chitchat, RAG search, system command)
- Use substring matching for robust route key extraction (handles model padding/prefixes)
- Default to the safest route on classification failure

## Key Constructs
```python
def semantic_router(query: str) -> str:
    prompt = [
        {"role": "system", "content": (
            "You are a routing switchboard. Classify intent:\n"
            "1. CHITCHAT — casual talk\n"
            "2. RAG_SEARCH — technical questions\n"
            "3. SYSTEM_COMMAND — clear/wipe requests\n"
            "Output ONLY the route key."
        )},
        {"role": "user", "content": query}
    ]
    raw = llm_call(prompt).strip().upper()

    # Substring match — handles "CHITCHAT_MODE", "CHITCHAT:", etc.
    if "CHITCHAT" in raw:      return "CHITCHAT"
    if "RAG_SEARCH" in raw:    return "RAG_SEARCH"
    if "SYSTEM_COMMAND" in raw: return "SYSTEM_COMMAND"
    return "CHITCHAT"  # safe default

def dispatch(query: str):
    route = semantic_router(query)
    handlers = {
        "CHITCHAT":       handle_chitchat,
        "RAG_SEARCH":     handle_rag_search,
        "SYSTEM_COMMAND": handle_system_command,
    }
    return handlers[route](query)
```

## Mental Model
An intelligent receptionist who listens to what a visitor needs before sending them to the right department. Casual visitors don't need the retrieval engine. System commands shouldn't trigger document search. Routing saves cost and prevents misuse.

## Pitfalls
- Exact string comparison fails when model adds punctuation — always use substring matching
- Default fallback must be the **safest** route (chitchat never crashes; system commands can)
- Keep intent categories ≤ 3 — more categories means more classification confusion
- LLM routing adds one extra LLM call per request — consider rule-based routing for high-traffic paths
