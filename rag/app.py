import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import litellm

# ==========================================
# 1. APPLICATION INITIALIZATION & CONFIG
# ==========================================
app = FastAPI(
    title="Enterprise AI RAG Service Layer",
    description="Stateless production gateway using plain Python structures and LiteLLM.",
    version="1.0.0"
)

# Enable CORS Middleware so frontend applications can securely consume your endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your specific domain origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. DATA VALIDATION SCHEMAS (PYDANTIC)
# ==========================================
class ChatRequest(BaseModel):
    """ Enforces strict, predictable incoming JSON request payloads """
    question: str = Field(..., example="What port configuration does the indexing cluster use?")
    session_id: str = Field(default="default_session", example="user_session_4452")

class ChatResponse(BaseModel):
    """ Enforces a clean, standard API output model for client applications """
    status: str
    session_id: str
    answer: str

# In-memory production session log simulator (In real production, swap this with Redis)
production_chat_history_store = {}

# ==========================================
# 3. HIGH-PERFORMANCE ASYNCHRONOUS ENDPOINTS
# ==========================================
@app.get("/health")
async def health_check():
    """ Standard Kubernetes / Cloud infrastructure health monitoring endpoint """
    return {"status": "healthy", "engine": "Ollama/LiteLLM"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def execute_chat_gateway(payload: ChatRequest):
    """ Asynchronous stateless pipeline routing incoming user requests to the LLM """
    try:
        user_query = payload.question
        session_id = payload.session_id
        
        # Look up or initialize the target user's historical state array
        if session_id not in production_chat_history_store:
            production_chat_history_store[session_id] = []
        history = production_chat_history_store[session_id]
        
        # Construct our explicit OpenAI-style message dictionary list
        messages = [
            {"role": "system", "content": "You are a precise system infrastructure engineer assistant. Answer accurately."}
        ]
        
        # Inject previous turns from our clean history structure
        for turn in history:
            messages.append(turn)
            
        # Append the active request string
        messages.append({"role": "user", "content": user_query})
        
        # Execute the network call asynchronously using LiteLLM's standard async interface (acompletion)
        # This completely frees the web server thread loop while waiting for model generation tokens
        response = await litellm.acompletion(
            model="ollama/llama3.2",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=0.2
        )
        
        ai_raw_answer = response.choices[0].message.content.strip()
        
        # Save this completed interaction block to our state manager before returning
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_raw_answer})
        
        return ChatResponse(
            status="success",
            session_id=session_id,
            answer=ai_raw_answer
        )
        
    except Exception as error:
        # Catch unexpected pipeline edge cases safely and log them accurately
        raise HTTPException(status_code=500, detail=f"Internal AI Pipeline Error: {str(error)}")

# ==========================================
# 4. SERVER BOOTSTRAP EXECUTOR
# ==========================================
if __name__ == "__main__":
    # Boot the async network server engine local on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)