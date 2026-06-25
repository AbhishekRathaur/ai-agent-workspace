from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# =========================================================================
# 1. DEFINE NATIVE BACKEND TOOLS
# =========================================================================
@tool
def get_booking_details(booking_id: str) -> str:
    """Lookup the status and customer tracking info."""
    print(f"   ⚙️ [Executing Tool] get_booking_details for: {booking_id}")
    return '{"status": "State 13: Cancelled", "customer_id": "CUST-001"}'

@tool
def fetch_customer_email(customer_id: str) -> str:
    """Lookup a customer's registered email address."""
    print(f"   ⚙️ [Executing Tool] fetch_customer_email for: {customer_id}")
    return '{"email": "abhishek@example.com"}'

# Execution mapping registry
tools_map = {
    "get_booking_details": get_booking_details,
    "fetch_customer_email": fetch_customer_email
}

# =========================================================================
# 2. SETUP MODEL ENGINE & BIND SCHEMAS
# =========================================================================
print("🔌 Initializing local tool-bounded engine...")
llm = ChatOllama(model="llama3.2", temperature=0.0)
llm_with_tools = llm.bind_tools([get_booking_details, fetch_customer_email])

# =========================================================================
# 3. RUNTIME STEPPING LOOP
# =========================================================================
messages = [
    HumanMessage(content="Check booking BK-130 and find the customer email.")
]

print(f"📥 User Query: {messages[0].content}\n")

# Run the loop up to a maximum number of steps to prevent infinite hangs
max_steps = 5
step = 0

while step < max_steps:
    step += 1
    print(f"🔄 --- Turn {step} ---")
    
    # Send the historical list of messages to the model
    response = llm_with_tools.invoke(messages)
    
    # Crucial: Save the model's response to the conversation history immediately
    messages.append(response)
    
    # Case A: If the model requests tool calls, we must execute them
    if response.tool_calls:
        print(f"🤖 [Model Decision]: Needs to invoke {len(response.tool_calls)} tool(s).")
        
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            tool_id = tool_call["id"]
            
            # Extract the actual tool function and run it
            target_tool = tools_map[name]
            tool_output = target_tool.invoke(args)
            
            # Append a clean, structured ToolMessage back into our history list
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
            
    # Case B: If the model didn't request any tools, it has computed the final answer
    else:
        print("\n🏁 --- FINAL RESPONSE FROM AGENT ---")
        print(response.content)
        break
else:
    print("⚠️ Guardrail reached: Terminating loop to protect resource footprint.")