from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def run_generation_pipeline():
    print("🔌 Initializing Generation Plane with Strict Decoding Parameters...")
    
    # =========================================================================
    # 1. DECODING ALGORITHMS (Hardware Level Controls)
    # =========================================================================
    # temperature=0.0 -> Forces the model to pick the #1 most probable token (No creativity/hallucination)
    # top_p=0.85 -> Nucleus sampling: chops off the bottom 15% of weird vocabulary options
    llm = ChatOllama(
        model="llama3.2", 
        temperature=0.0, 
        top_p=0.85
    )

    # =========================================================================
    # 2. CONTEXT FRAMING (The XML Sandwich)
    # =========================================================================
    # We pretend these chunks perfectly survived the Lab 3 Reranker
    retrieved_fact_1 = "Booking BK-130 was cancelled from the backend via State 13."
    retrieved_fact_2 = "The customer associated with BK-130 is CUST-001 (abhishek@example.com)."
    
    user_query = "Who is the customer for the cancelled booking?"
    
    # Using explicit XML tags forces the attention matrix to isolate the factual data
    # from the system instructions and the user query.
    xml_structured_prompt = f"""You are a strict, factual enterprise data assistant.
Answer the user's query based ONLY on the data contained within the <system_context> tags.
If the context does not contain the answer, reply with exactly: "INSUFFICIENT_DATA".

<system_context>
[Fact 1]: {retrieved_fact_1}
[Fact 2]: {retrieved_fact_2}
</system_context>

<user_query>
{user_query}
</user_query>
"""

    print(f"\n📥 User Query: {user_query}")
    print("🚀 Firing XML-structured prompt to inference engine...")

    # =========================================================================
    # 3. EXECUTION
    # =========================================================================
    response = llm.invoke([HumanMessage(content=xml_structured_prompt)])
    
    print("\n🏁 --- FINAL SYNTHESIZED OUTPUT ---")
    print(response.content)

if __name__ == "__main__":
    run_generation_pipeline()