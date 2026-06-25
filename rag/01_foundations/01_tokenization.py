import tiktoken

def run_token_analysis():
    # Load the standard cl100k_base tokenizer engine (used by modern LLMs)
    encoder = tiktoken.get_encoding("cl100k_base")
    
    # Payload 1: Clean, standard conversational prose
    prose_text = "The platform user request has been closed successfully."
    
    # Payload 2: Highly compressed code syntax from your workflow diagram
    code_text = "bookingLifecycleState.transitionTo(State.CLOSED_MANUALLY);"
    
    # Calculate character lengths
    print(f"📏 Prose Length: {len(prose_text)} chars | Code Length: {len(code_text)} chars\n")
    
    # Tokenize both payloads
    prose_tokens = encoder.encode(prose_text)
    code_tokens = encoder.encode(code_text)
    
    print("--- 📝 PROSE BREAKDOWN ---")
    print(f"Token Count: {len(prose_tokens)}")
    print(f"Token IDs:   {prose_tokens}")
    print(f"Fragments:   {[encoder.decode([t]) for t in prose_tokens]}\n")
    
    print("--- 💻 CODE BREAKDOWN ---")
    print(f"Token Count: {len(code_tokens)}")
    print(f"Token IDs:   {code_tokens}")
    print(f"Fragments:   {[encoder.decode([t]) for t in code_tokens]}")

if __name__ == "__main__":
    run_token_analysis()