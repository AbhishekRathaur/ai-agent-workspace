# RAG Evaluation: Faithfulness & Relevance

**Files:** `06_evaluation/rag_evaluation_scorer.ipynb` · `06_evaluation/llm_as_judge.py`

## Core Concept
Measure RAG quality with two metrics: **Faithfulness** (answer claims are grounded in retrieved context) and **Answer Relevance** (answer actually addresses the question). Both use the LLM as an automated judge.

## What You Learn
- Extract individual factual claims from an answer using an LLM
- Verify each claim against retrieved context with a Y/N LLM call
- Score answer relevance on a 1–5 scale and normalize to 0.0–1.0
- Interpret scores to detect hallucination vs. off-topic answers

## Key Constructs — Faithfulness
```python
def faithfulness(context: str, answer: str) -> float:
    # Pass 1: extract claims
    claims = llm_json_call(f"Break into factual claims: {answer}")  # returns list
    if not claims:
        return 0.0

    # Pass 2: verify each claim against context
    verified = 0
    for claim in claims:
        verdict = llm_call(f"Context: {context}\nIs '{claim}' supported? Answer Y or N.")
        if verdict.strip().upper() == "Y":
            verified += 1

    return verified / len(claims)
```

## Key Constructs — Answer Relevance
```python
def answer_relevance(query: str, answer: str) -> float:
    score = llm_call(
        f"Rate 1–5 how directly this answer addresses the question.\n"
        f"Question: {query}\nAnswer: {answer}\nOutput only the digit."
    )
    try:
        return (int(score.strip()) - 1) / 4.0  # normalize to 0.0–1.0
    except Exception:
        return 0.0
```

## Score Interpretation
| Faithfulness | Relevance | Diagnosis |
|---|---|---|
| High | High | Correct, grounded answer |
| Low | High | Hallucinated but on-topic |
| High | Low | Grounded but misses the question |
| Low | Low | Completely failed |

## Pitfalls
- JSON claim extraction can fail — always wrap in `try/except` with `return 0.0` fallback
- Each evaluation makes 2+ LLM calls — expensive; don't run on every request in production
- "Y" verdict must match exactly, not as substring (`"YES"` ≠ `"Y"`)
- Claim extraction can over-split or under-split compound sentences
- Low scores don't tell you *why* the answer failed — combine with logging the claims list
