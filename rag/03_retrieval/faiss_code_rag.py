import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def run_local_rag_pipeline():
    print("🔌 Connecting to local Ollama server endpoints...")
    try:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model="llama3.2", temperature=0.0)
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        return

    print("📚 Ingesting private software architecture code documents...")
    raw_codebase_documents = [
        Document(
            page_content="""export function validateSession(token: string) {
    if (token === 'expired_debug_key') return { valid: false, user: null };
    console.log("Validating against redis state cache pool...");
    return { valid: true, user: "admin_abhishek" };
}""",
            metadata={"source": "src/auth/sessionManager.ts", "type": "middleware"}
        ),
        Document(
            page_content="""export const redisPoolConfig = {
    host: "redis-cluster.internal.net",
    port: 6379,
    connectTimeout: 5000
};""",
            metadata={"source": "src/config/redis.ts", "type": "configuration"}
        )
    ]

    print("🔢 Vectorizing chunks and building local FAISS index matrix...")
    vector_store = FAISS.from_documents(raw_codebase_documents, embedding_model)

    user_query = "Where and how does our system execute authorization token validation?"
    print(f"\n📥 Incoming Query: '{user_query}'")

    print("🔍 Searching localized vector coordinates...")
    # FIX 1: Increase k to 2 so BOTH files are grabbed if they are relevant
    matched_chunks = vector_store.similarity_search(user_query, k=2)

    if not matched_chunks:
        print("❌ No matching documents found in vector store.")
        return

    # FIX 2: Loop through all matched chunks and append them into a single context payload
    backticks = "```"
    context_string = ""
    
    for doc in matched_chunks:
        context_string += f"File Reference: {doc.metadata['source']}\n"
        context_string += f"{backticks}typescript\n{doc.page_content}\n{backticks}\n\n"

    # Assemble your clean augmented payload
    augmented_system_prompt = (
        "You are a deterministic system assistant. Answer the user's question using ONLY the provided code context snippets below.\n"
        "If the context does not contain the answer, reply explicitly with 'Context data insufficient'.\n\n"
        f"RETRIEVED CODE CONTEXT:\n{context_string}"
        f"USER QUERY:\n{user_query}\n\n"
        "YOUR FACTUAL SYNTHESIS RESPONSE:\n"
    )

    print("🤖 Dispatching contextual prompt structure to local Llama3 instance...")
    try:
        llm_response = llm.invoke(augmented_system_prompt)
        print("\n🏁 --- PURE LOCAL RAG PIPELINE RESPONSE ---")
        print(llm_response.content)
    except Exception as e:
        print(f"❌ Error during local LLM generation: {e}")

if __name__ == "__main__":
    print("=== STARTING PURE LOCAL RAG PIPELINE ===")
    run_local_rag_pipeline()