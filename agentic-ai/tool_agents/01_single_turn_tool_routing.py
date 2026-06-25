import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# =========================================================================
# 1. DECLARE ATOMIC AGENTIC TOOLS
# =========================================================================
# Using the standard @tool decorator automatically builds the JSON schema
# that Ollama reads to understand how and when to invoke this tool.

@tool
def check_booking_status(booking_id: str) -> str:
    """Retrieves the live operational status of a platform booking using its ID string."""
    # Simulating a database lookup against your lifecycle states (e.g., state 6 vs state 13)
    mock_db = {
        "BK-601": "State 6: Booking Confirmed (Pending Payment Verification)",
        "BK-130": "State 13: Booking Cancelled from Backend (Refund Complete)",
    }
    print(f"🎯 [TOOL EXECUTION] Querying platform state store for ID: {booking_id}")
    return mock_db.get(booking_id, f"Booking {booking_id} not found in state store.")

# Create our executable lookup map
available_tools = {"check_booking_status": check_booking_status}

# =========================================================================
# 2. BIND SCHEMAS TO LOCAL RUNTIME MODEL
# =========================================================================
print("🔌 Starting Local Agent Routing Engine...")
llm = ChatOllama(model="llama3.2", temperature=0.0)

# Bind the tool array schema directly into the model's generation matrix
llm_with_tools = llm.bind_tools([check_booking_status])

# =========================================================================
# 3. INTERACTIVE AGENT ROUTING RUNTIME LOOP
# =========================================================================
user_prompt = "Can you look into why booking BK-130 has no updates?"
print(f"\n📥 Incoming Prompt: '{user_prompt}'")

# Execute our initial model evaluation step
messages = [HumanMessage(content=user_prompt)]
ai_response = llm_with_tools.invoke(messages)

# Check if the model determined a tool execution parameter is required
if ai_response.tool_calls:
    print("🤖 [LLM DECISION] Structured Tool Call Requested!")
    messages.append(ai_response)
    
    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Pull the physical function pointer from our tracking dictionary
        target_function = available_tools[tool_name]
        
        # Execute the function locally using the arguments parsed by the LLM
        tool_output = target_function.invoke(tool_args)
        
        # Append the structural tool result directly back into the conversational loop
        messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
        
    # Send the augmented chat history array back to the model for final synthesis
    print("🔄 Dispatching function execution metrics back to local model...")
    final_synthesis = llm_with_tools.invoke(messages)
    
    print("\n🏁 --- FINAL AGENT RESPONSE ---")
    print(final_synthesis.content)
else:
    print("🤖 Final Conversation Response:")
    print(ai_response.content)