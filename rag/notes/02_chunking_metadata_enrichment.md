# Metadata Enrichment

**File:** `02_chunking/metadata_enrichment.ipynb`

## Core Concept
Use an LLM as a data librarian to tag unstructured document chunks with structured metadata (system, category, severity, tags). Turns raw text into queryable records you can filter at retrieval time.

## What You Learn
- Prompt an LLM to extract structured JSON from unstructured text
- Handle model formatting quirks (markdown code block wrappers)
- Build a graceful fallback for JSON parse failures
- Filter retrieved documents by metadata fields (e.g., `severity == "Critical"`)

## Key Constructs
```python
import json

# Strict extraction prompt
prompt = [
    {"role": "system", "content": (
        "Extract exactly: system, category, severity, tags.\n"
        "Output ONLY valid JSON."
    )},
    {"role": "user", "content": f"Text: {raw_text}"}
]

raw = response.choices[0].message.content.strip()

# Strip markdown wrappers (common with local models)
if raw.startswith("```json"):
    raw = raw.replace("```json", "").replace("```", "").strip()

# Graceful fallback
try:
    metadata = json.loads(raw)
except Exception:
    metadata = {"system": "Unknown", "category": "Unclassified",
                "severity": "Routine", "tags": []}

# Filter at retrieval time
if doc["severity"] == "Critical" and doc["system"] == "MeshQuery":
    ...
```

## Mental Model
The LLM acts as a smart librarian tagging each document for a card catalogue. Instead of full-text search, you query the catalogue: "show me all Critical severity entries from MeshQuery". Faster and more precise than scanning raw text.

## Pitfalls
- LLM can hallucinate metadata fields not present in the text — validate required keys exist
- Severity/system strings must match exactly (case-sensitive) in filter logic
- Each enrichment call adds latency — batch where possible
- `temperature=0` reduces variance but doesn't guarantee identical output across model versions
