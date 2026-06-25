# FastAPI RAG Gateway

**Files:** `08_serving/fastapi_app.py` · `08_serving/fastapi_gateway.ipynb`

## Core Concept
Wrap a RAG pipeline in a production-grade HTTP API using FastAPI. Provides request validation, async LLM calls, per-session chat history, CORS support, and health checks.

## What You Learn
- Define request/response schemas with Pydantic `BaseModel`
- Use `await litellm.acompletion()` for non-blocking async LLM calls
- Manage per-session message history with an in-memory dict keyed by `session_id`
- Add CORS middleware for frontend access
- Expose a `/health` endpoint for infrastructure monitoring

## Key Constructs
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import litellm, uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

session_store = {}  # { session_id: [{"role":..., "content":...}, ...] }

class ChatRequest(BaseModel):
    question:   str = Field(..., example="What port config is used?")
    session_id: str = Field(default="default", example="user-4452")

class ChatResponse(BaseModel):
    status: str; session_id: str; answer: str

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    history = session_store.setdefault(payload.session_id, [])
    messages = [{"role": "system", "content": "You are a system assistant."}]
    messages += history
    messages.append({"role": "user", "content": payload.question})

    try:
        resp = await litellm.acompletion(model="ollama/llama3.2", messages=messages,
                                          api_base="http://localhost:11434", temperature=0.2)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user",      "content": payload.question})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(status="success", session_id=payload.session_id, answer=answer)

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
```

## Production Checklist
| Concern | Dev setup | Production fix |
|---|---|---|
| Session storage | In-memory dict | Redis with TTL |
| CORS origins | `"*"` | Restrict to known domains |
| Authentication | None | API key / OAuth |
| History growth | Unbounded | Max-turn window or summarization |
| Health check | Static response | Check Ollama + DB connectivity |
| `reload=True` | Fine | Remove for production |

## Pitfalls
- `litellm.completion()` (sync) inside an `async` endpoint blocks the event loop — always use `acompletion()`
- In-memory session history is lost on restart — not suitable for multi-instance deployments
- Never expose raw exception detail in `HTTPException` — it can leak internal paths and secrets
