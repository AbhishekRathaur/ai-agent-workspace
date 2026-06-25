from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Tools stay clean and simple
@tool
def get_booking_details(booking_id: str) -> str:
    """Lookup the status and customer tracking info for a booking."""
    return '{"status": "State 13: Cancelled", "customer_id": "CUST-001"}'

@tool
def fetch_customer_email(customer_id: str) -> str:
    """Lookup a customer's registered email address."""
    return '{"email": "abhishek@example.com"}'

# 2. Setup the model
llm = ChatOllama(model="llama3.2", temperature=0.0)

# 3. Compile the entire engine in ONE line
# This replaces the entire while loop, state management, and parallel guards automatically
agent_executor = create_react_agent(llm, tools=[get_booking_details, fetch_customer_email])

# 4. Invoke it directly with a state dictionary
print("🚀 Dispatching request to automated prebuilt agent...")
response = agent_executor.invoke({
    "messages": [("user", "Check booking BK-130 and find the customer email.")]
})

print("\n🏁 Final Answer:")
print(response["messages"][-1].content)