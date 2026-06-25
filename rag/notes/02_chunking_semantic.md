# Semantic Chunking

**File:** `02_chunking/semantic_chunking.ipynb`

## Core Concept
Split documents at topic boundaries rather than fixed character counts. Measures cosine distance between consecutive sentence embeddings and cuts where the semantic distance spikes above a percentile threshold.

## What You Learn
- Embed sentences individually to capture fine-grained meaning
- Calculate cosine distance between adjacent sentence vectors
- Use a dynamic percentile threshold (not a fixed cutoff) to find topic shifts
- Produce variable-length chunks that respect content boundaries

## Key Constructs
```python
import numpy as np

def cosine_distance(a, b):
    return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Compute distances between consecutive sentences
distances = [cosine_distance(vecs[i], vecs[i+1]) for i in range(len(vecs)-1)]

# Dynamic threshold — adapts to each document
threshold = np.percentile(distances, 85)

# Split where distance spikes above threshold
chunks, current = [], sentences[0]
for i, dist in enumerate(distances):
    if dist >= threshold:
        chunks.append(current)
        current = sentences[i+1]
    else:
        current += " " + sentences[i+1]
chunks.append(current)
```

## Mental Model
Sentences within the same topic sit close together in embedding space (low distance). A topic shift sends the next sentence flying into a different region (high distance spike). The chunker detects these spikes and cuts there, keeping thematically related content together.

## vs. Fixed-Size Chunking
| | Fixed-Size | Semantic |
|---|---|---|
| Chunk size | Constant | Variable |
| Topic coherence | May split mid-topic | Respects boundaries |
| Compute cost | Cheap | Expensive (embed every sentence) |
| Tuning | `chunk_size` param | Percentile threshold |

## Pitfalls
- 85th percentile is a starting point — tune per dataset
- Regex splitting on `.!?` breaks on abbreviations (Dr., Inc.)
- Embedding every sentence is slow for large documents; batch with GPU if possible
