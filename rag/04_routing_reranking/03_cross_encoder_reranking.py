import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# =========================================================================
# 1. SIMULATE MIXED HYBRID RETRIEVAL
# =========================================================================
# Imagine your vector database returned these 5 chunks. 
# The perfect answer is buried at Index 2, in the low-attention "Middle" zone.
RAW_RETRIEVED_CHUNKS = [
    {"id": 1, "content": "The system admin navigation panel contains links to User Configurations and Billing Matrix systems."},
    {"id": 2, "content": "For security compliance, all administrative session keys must expire after 15 minutes of inactivity."},
    {"id": 3, "content": "CRITICAL CONFIG: To transition a booking to State 13 (Cancelled from BE), invoke the fallback gateway routing flag."},
    {"id": 4, "content": "Providers can view incoming proposals via the vendor cockpit panel under the active matching queues tab."},
    {"id": 5, "content": "Enterprise customers receive auto-generated platform performance statements on the first business day of the month."}
]

# =========================================================================
# 2. THE CROSS-ENCODER FILTER (Reranking Logic)
# =========================================================================
def programmatic_cross_encoder_rerank(query: str, chunks: list) -> list:
    """Evaluates the explicit relationship between the user query and each chunk text simultaneously."""
    print("   🔍 [Cross-Encoder] Calculating joint context alignment scores...")
    
    scored_chunks = []
    for chunk in chunks:
        score = 0.05 # Baseline similarity score
        
        # Simulating deep keyword + semantic cross-evaluation
        if "State 13" in chunk["content"] or "booking" in chunk["content"].lower():
            score = 0.95 # Perfect alignment
        elif "administrative" in chunk["content"] or "security" in chunk["content"]:
            score = 0.40 # Partial conceptual alignment
            
        scored_chunks.append((score, chunk))
        
    # Sort chronologically by score descending (Highest signal first)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Strip the scores away and return only the clean, re-ordered text elements
    return [item[1] for item in scored_chunks]

# =========================================================================
# 3. RUNTIME RAG GENERATION PIPELINE
# =========================================================================
if __name__ == "__main__":
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    
    user_query = "How do I move a broken booking to State 13?"
    print(f"📥 Query: {user_query}\n")
    
    # Step 1: Run our precision cross-encoder reranker
    sorted_context = programmatic_cross_encoder_rerank(user_query, RAW_RETRIEVED_CHUNKS)
    
    # Step 2: Mitigate "Lost in the Middle" by keeping ONLY the top 2 highest-signal blocks
    trimmed_context = sorted_context[:2]
    
    # Step 3: Build the clear prompt structure (Context sits safely at the top)
    context_str = "\n\n".join([f"Source Block:\n{c['content']}" for c in trimmed_context])
    
    prompt = f"""
    Use the following pristine structural context blocks to formulate a response.
    
    CONTEXT BLOCKS:
    {context_str}
    
    QUESTION: {user_query}
    """
    
    print("🚀 Dispatching optimized context sandwich to model...")
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n🏁 --- FINAL MODEL RESPONSE ---")
    print(response.content)