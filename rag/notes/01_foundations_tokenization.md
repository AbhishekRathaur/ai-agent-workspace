# Tokenization

**File:** `01_foundations/01_tokenization.py`

## Core Concept
Before building any RAG pipeline, understand how text is split into tokens — the unit LLMs actually process. Token count determines cost, latency, and whether content fits in the context window.

## What You Learn
- Load the `cl100k_base` tokenizer used by modern LLMs
- Encode text into token IDs and decode them back to fragments
- Compare token density between prose and code (code is denser)
- Estimate API cost from token counts

## Key Constructs
```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

prose_tokens = encoder.encode("The platform user request has been closed successfully.")
code_tokens  = encoder.encode("bookingLifecycleState.transitionTo(State.CLOSED_MANUALLY);")

print(len(prose_tokens))   # token count
print([encoder.decode([t]) for t in prose_tokens])  # fragments
```

## Insight
Code is token-dense — `camelCase`, dots, and brackets each often become separate tokens. A code string with similar character length to prose can cost 2–3× more tokens.

## Pitfalls
- Token count ≠ character count — never estimate cost from string length
- `cl100k_base` is for OpenAI/Ollama; different models may use different tokenizers
