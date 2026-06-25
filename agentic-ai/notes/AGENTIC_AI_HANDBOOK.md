## Introduction: The Philosophy of Stateful Agent Engineering

There are two fundamentally different approaches to building AI agents, and confusing them is the most common architectural mistake.

**Tool-calling agents** (simple) give an LLM access to functions and let it call them in sequence until it has enough information to answer. This is fast to build and sufficient for single-turn tasks. The entire loop fits in ~20 lines.

**Stateful graph agents** (LangGraph) model the agent as a directed graph where each node is a function, edges determine flow, and a shared state dict carries context across the entire execution. This is the right choice when you need: conditional routing between agents, persistent memory across sessions, human approval gates before irreversible actions, retry policies for flaky services, or the ability to rewind and re-execute from a past checkpoint.

This handbook is split into two parts reflecting this distinction:

- **Part I: Tool Agents** — understand the raw mechanics first. Build a ReACT loop from scratch before using the framework's prebuilt version.
- **Part II: LangGraph** — add one production capability at a time, in the order you'll actually need them.

Every pattern in Part II builds on the previous. Do not skip ahead.

---

## Table of Contents

### Part I — Tool-Calling Agents
* [[#Chapter 1: Single-Turn Tool Routing]]
* [[#Chapter 2: ReACT Agent Loop (Manual)]]
* [[#Chapter 3: Prebuilt ReACT Agent]]

### Part II — LangGraph Stateful Graphs
* [[#Chapter 4: LangGraph Basics (StateGraph)]]
* [[#Chapter 5: Conditional Routing & Feedback Loops]]
* [[#Chapter 6: Persistent Memory & Checkpointing]]
* [[#Chapter 7: Multi-Agent Hierarchies (Subgraphs)]]
* [[#Chapter 8: Human-in-the-Loop Interrupts]]
* [[#Chapter 9: Production Resilience (Retries + Checkpoints + Interrupts)]]
* [[#Chapter 10: Real-World SQLite Persistence]]
* [[#Chapter 11: Remote & Distributed Agents]]
* [[#Chapter 12: Streaming Execution]]
* [[#Chapter 13: Time-Travel Debugging]]

---

# Part I — Tool-Calling Agents

---

## Chapter 1: Single-Turn Tool Routing

### 1. The Operational Problem
The most basic agentic task: a user asks something that requires looking up live data. The LLM must decide whether to call a tool or answer directly — then synthesize the tool result into a final response. One decision point, one execution, one answer.

### 2. The Architectural Core Concept
Three building blocks:

1. **`@tool` decorator** — turns a Python function into a tool schema the LLM can read. The docstring becomes the description the model uses to decide when to call it.
2. **`bind_tools()`** — injects the tool schemas into the model's generation matrix. The LLM now knows what tools exist and what arguments they need.
3. **`tool_calls` check** — after the first LLM call, inspect `ai_response.tool_calls`. If populated, execute each tool locally, append a `ToolMessage`, and re-invoke the model with the full history for synthesis.

### 3. The Reference Implementation

**File:** `tool_agents/01_single_turn_tool_routing.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def check_booking_status(booking_id: str) -> str:
    """Retrieves the live operational status of a platform booking using its ID."""
    mock_db = {
        "BK-601": "State 6: Booking Confirmed (Pending Payment Verification)",
        "BK-130": "State 13: Booking Cancelled from Backend (Refund Complete)",
    }
    return mock_db.get(booking_id, f"Booking {booking_id} not found in state store.")

available_tools = {"check_booking_status": check_booking_status}

llm = ChatOllama(model="llama3.2", temperature=0.0)
llm_with_tools = llm.bind_tools([check_booking_status])

messages = [HumanMessage(content="Can you look into why booking BK-130 has no updates?")]
ai_response = llm_with_tools.invoke(messages)

if ai_response.tool_calls:
    messages.append(ai_response)  # save LLM decision to history

    for tc in ai_response.tool_calls:
        output = available_tools[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

    # Re-invoke with augmented history for synthesis
    final = llm_with_tools.invoke(messages)
    print(final.content)
else:
    print(ai_response.content)  # model answered directly
```

### 4. Message Flow
```
HumanMessage("Why does BK-130 have no updates?")
  → LLM: calls check_booking_status(booking_id='BK-130')   [AIMessage + tool_calls]
  → Tool runs locally → "State 13: Cancelled"              [ToolMessage]
  → LLM synthesizes final answer                           [AIMessage]
```

### 5. Critical Rules
- Always append `ai_response` (the AIMessage) to history **before** adding ToolMessages — the conversation needs the LLM's decision in the chain
- `ToolMessage` requires the matching `tool_call_id` to link back to the right tool request
- The `@tool` docstring IS the tool description — write it precisely

---

## Chapter 2: ReACT Agent Loop (Manual)

### 1. The Operational Problem
Single-turn routing fails when one tool call isn't enough. A task like "check booking BK-130 and find the customer email" requires two sequential tool calls: first get the booking (to extract `customer_id`), then fetch the email (using that `customer_id`). The LLM must see the first result before deciding to make the second call.

### 2. The Architectural Core Concept
The **ReACT (Reasoning + Acting)** pattern replaces the single if/else with a while loop. Each iteration:

1. **Reason** — the LLM sees the full history and decides what to do next
2. **Act** — if `tool_calls` present, execute them all and append ToolMessages
3. **Observe** — loop back; the LLM now sees the results and reasons again
4. **Exit** — when `tool_calls` is empty, the LLM has computed its final answer

A `max_steps` guard prevents infinite loops.

### 3. The Reference Implementation

**File:** `tool_agents/02_react_agent_loop.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def get_booking_details(booking_id: str) -> str:
    """Lookup the status and customer tracking info."""
    return '{"status": "State 13: Cancelled", "customer_id": "CUST-001"}'

@tool
def fetch_customer_email(customer_id: str) -> str:
    """Lookup a customer registered email address."""
    return '{"email": "abhishek@example.com"}'

tools_map = {"get_booking_details": get_booking_details, "fetch_customer_email": fetch_customer_email}

llm = ChatOllama(model="llama3.2", temperature=0.0)
llm_with_tools = llm.bind_tools([get_booking_details, fetch_customer_email])

messages = [HumanMessage(content="Check booking BK-130 and find the customer email.")]

max_steps = 5
step = 0

while step < max_steps:
    step += 1
    response = llm_with_tools.invoke(messages)
    messages.append(response)              # ALWAYS save LLM response first

    if response.tool_calls:
        for tc in response.tool_calls:
            output = tools_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        # loop continues — model sees results and decides next action
    else:
        print(response.content)            # no tool calls = final answer
        break
else:
    print("Max steps reached — terminating to protect resources.")
```

### 4. Multi-Step Execution Trace
```
Turn 1: → LLM calls get_booking_details("BK-130")
           ← {"status": "State 13", "customer_id": "CUST-001"}
Turn 2: → LLM calls fetch_customer_email("CUST-001")
           ← {"email": "abhishek@example.com"}
Turn 3: → LLM has all info → final answer, no tool_calls → break
```

### 5. Why Full Message History Matters
Each `llm.invoke(messages)` receives the **entire** accumulated history. Without it the LLM has no memory of what it already called — it will repeat tool calls or fail to synthesize across multiple results.

---

## Chapter 3: Prebuilt ReACT Agent

### 1. The Operational Problem
The manual while loop in Chapter 2 is ~20 lines of boilerplate. For standard tool-calling tasks it's identical every time: invoke, check tool_calls, execute, append, repeat. LangGraph's `create_react_agent` factory compiles this loop for you.

### 2. The Architectural Core Concept
`create_react_agent(llm, tools)` compiles a complete `StateGraph` internally — with message accumulation, tool dispatch, and loop-until-done logic — and returns it as a runnable. The entire manual loop is replaced by one line.

### 3. The Reference Implementation

**File:** `tool_agents/03_prebuilt_react_agent.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_booking_details(booking_id: str) -> str:
    """Lookup the status and customer tracking info for a booking."""
    return '{"status": "State 13: Cancelled", "customer_id": "CUST-001"}'

@tool
def fetch_customer_email(customer_id: str) -> str:
    """Lookup a customer registered email address."""
    return '{"email": "abhishek@example.com"}'

llm = ChatOllama(model="llama3.2", temperature=0.0)

# Replaces the entire while loop, message history management, and tool dispatch
agent_executor = create_react_agent(llm, tools=[get_booking_details, fetch_customer_email])

response = agent_executor.invoke({
    "messages": [("user", "Check booking BK-130 and find the customer email.")]
})

print(response["messages"][-1].content)
```

### 4. What the Factory Replaces

| Manual (Ch 2) | Prebuilt (Ch 3) |
|---|---|
| `while step < max_steps:` | Handled internally |
| `messages.append(response)` | Handled internally |
| `if response.tool_calls:` | Handled internally |
| Tool dispatch loop | Handled internally |
| ~20 lines | 1 line |

### 5. The Learning Order Rule
**Never jump straight to `create_react_agent` without understanding Chapters 1 and 2 first.** When it breaks — wrong tool called, infinite loop, missing result — you need to know exactly which part of the manual loop failed. The abstraction hides all of that.

---

# Part II — LangGraph Stateful Graphs

---

## Chapter 4: LangGraph Basics (StateGraph)

### 1. The Operational Problem
Tool-calling agents (Part I) work for sequential tasks but can't express branching, looping, or persistent state across sessions. To build systems that route between specialized agents, remember conversation history, or pause for human review, you need a graph model.

### 2. The Architectural Core Concept
A `StateGraph` is a directed graph where:
- **State** = a shared `TypedDict` passed through every node
- **Nodes** = Python functions that read and write to state
- **Edges** = transitions between nodes (`add_edge`)
- **`Annotated` fields with `operator.add`** = list fields that accumulate across nodes instead of overwriting

### 3. The Reference Implementation

**File:** `langraph/01_langgraph_basics.ipynb`

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

# State schema — the shared data store passed through every node
class MemoryState(TypedDict):
    input_text: str
    processed_steps: Annotated[list, operator.add]  # accumulates across nodes

# Nodes — pure functions that return partial state updates
def uppercase_worker(state: MemoryState):
    return {
        "input_text": state["input_text"].upper(),
        "processed_steps": ["Converted to uppercase"]  # appended, not overwritten
    }

def dramatic_worker(state: MemoryState):
    return {
        "input_text": state["input_text"] + "!!!",
        "processed_steps": ["Added drama"]
    }

# Graph construction
builder = StateGraph(MemoryState)
builder.add_node("upper_node",  uppercase_worker)
builder.add_node("drama_node",  dramatic_worker)
builder.add_edge(START,         "upper_node")
builder.add_edge("upper_node",  "drama_node")
builder.add_edge("drama_node",  END)

app = builder.compile()

final_state = app.invoke({"input_text": "hello cursor mastery"})
print(final_state)
# {'input_text': 'HELLO CURSOR MASTERY!!!', 'processed_steps': ['Converted to uppercase', 'Added drama']}
```

### 4. The `Annotated[list, operator.add]` Pattern
Without this, each node's return value **overwrites** the list field. With it, each node's list is **appended** to the existing list. This is how you build an audit trail or accumulated message history.

### 5. Critical Rules
- Always wire `START → first_node` and `last_node → END` — a missing `START` edge causes silent failure
- Nodes must return a **dict** (partial state update), not the full state
- `builder.compile()` validates the graph structure and returns the executable `app`

---

## Chapter 5: Conditional Routing & Feedback Loops

### 1. The Operational Problem
Linear graphs (Chapter 4) can't express quality control, retries, or improvement cycles. A Writer-Critic pattern — where the Critic evaluates the Writer's output and sends it back for revision if quality is insufficient — requires a conditional edge that routes backward in the graph.

### 2. The Architectural Core Concept
`add_conditional_edges(source, router_fn, mapping)` adds a decision point after a node. The `router_fn` is a pure function of state that returns a string key. The `mapping` dict resolves that key to the next node (or `END`).

The key insight: an edge can point **backward** to a previous node, creating a loop. A `revision_count` field in state acts as the loop exit guard.

### 3. The Reference Implementation

**File:** `langraph/02_conditional_routing_feedback_loops.ipynb`

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    topic: str
    current_draft: str
    feedback: str
    quality_score: int
    revision_count: int

def writer_agent(state: AgentState):
    draft = "Initial draft" if not state["feedback"] else f"Revised: {state['feedback']}"
    return {"current_draft": draft, "revision_count": state["revision_count"] + 1}

def critic_agent(state: AgentState):
    score = min(10, state["revision_count"] * 3)  # improves each revision
    return {"quality_score": score, "feedback": "Needs more detail"}

# Router — pure function of state, returns string key
def gatekeeper_router(state: AgentState) -> str:
    if state["quality_score"] >= 8 or state["revision_count"] >= 3:
        return "approved"
    return "needs_work"

builder = StateGraph(AgentState)
builder.add_node("writer", writer_agent)
builder.add_node("critic", critic_agent)
builder.add_edge(START,    "writer")
builder.add_edge("writer", "critic")

# Conditional edge — routes backward to writer or forward to END
builder.add_conditional_edges(
    "critic",
    gatekeeper_router,
    {"needs_work": "writer", "approved": END}
)

app = builder.compile()
result = app.invoke({"topic": "LangGraph", "feedback": "", "revision_count": 0, "quality_score": 0, "current_draft": ""})
```

### 4. Execution Flow
```
writer → critic → [router] → writer (revision 1)
                           → writer (revision 2)
                           → END   (quality_score >= 8 or revision_count >= 3)
```

### 5. Pitfalls
- Router return strings must **exactly** match keys in the `mapping` dict
- Always add a max-revision guard (`revision_count >= N`) — without it, a low-quality loop never terminates
- Router functions must be **pure** (no side effects) — they may be called multiple times

---

## Chapter 6: Persistent Memory & Checkpointing

### 1. The Operational Problem
By default, every `app.invoke()` starts from a blank state. A multi-turn chatbot that forgets the previous message on every call is useless. Persistent memory requires saving state after each node execution so it can be recovered on the next invocation.

### 2. The Architectural Core Concept
A **checkpointer** is a key-value store that saves a state snapshot after every node. Snapshots are keyed by `(thread_id, checkpoint_id)`.

`thread_id` is the session identifier — same `thread_id` means same conversation. Different `thread_id` values give completely isolated state. The checkpointer is attached at compile time; the graph is otherwise unchanged.

### 3. The Reference Implementation

**File:** `langraph/03_persistent_memory_checkpointing.ipynb`

```python
from langgraph.checkpoint.memory import MemorySaver

memory_storage = MemorySaver()

# Attach checkpointer at compile — graph logic unchanged
persistent_app = builder.compile(checkpointer=memory_storage)

# Session A — first turn
config_a = {"configurable": {"thread_id": "abhishek_session"}}
persistent_app.invoke({"chat_history": ["Hello"]}, config=config_a)

# Session A — second turn (state from first turn is automatically loaded)
persistent_app.invoke({"chat_history": ["Follow-up question"]}, config=config_a)

# Session B — completely isolated, starts blank
config_b = {"configurable": {"thread_id": "john_session"}}
persistent_app.invoke({"chat_history": ["Different user"]}, config=config_b)

# Inspect current state without running the graph
snapshot = persistent_app.get_state(config_a)
print(snapshot.values)
```

### 4. Checkpointer Backends

| Backend | Use case |
|---|---|
| `MemorySaver()` | Development / in-process only |
| `SqliteSaver` | Single-machine persistence (Chapter 10) |
| Postgres / Redis | Production distributed systems |

### 5. Pitfalls
- `MemorySaver` lives in-process only — restart loses all state
- Forgetting `config` argument on `.invoke()` means no checkpointing happens silently

---

## Chapter 7: Multi-Agent Hierarchies (Subgraphs)

### 1. The Operational Problem
Complex tasks require specialized agents working together — a Research agent that searches and scrapes, a Writer agent that drafts from research results, an Executive orchestrator that coordinates both. Each specialist needs its own private state that the others can't corrupt.

### 2. The Architectural Core Concept
Compile a child graph and register it as a **single node** in a parent graph. The child has its own `TypedDict` state schema — completely isolated from the parent's state. The parent sees the child as a black box: input goes in, output comes out, internal state is invisible.

### 3. The Reference Implementation

**File:** `langraph/04_multi_agent_hierarchies.ipynb`

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

# --- Child Graph (Research Agent) ---
class ResearchPrivateState(TypedDict):
    query: str
    raw_urls: list
    extracted_facts: list
    final_summary: str

def search_node(state: ResearchPrivateState):
    return {"raw_urls": [f"url_for_{state['query']}"]}

def summarizer_node(state: ResearchPrivateState):
    return {"final_summary": f"Summary of research on: {state['query']}"}

research_builder = StateGraph(ResearchPrivateState)
research_builder.add_node("search",    search_node)
research_builder.add_node("summarize", summarizer_node)
research_builder.add_edge(START,       "search")
research_builder.add_edge("search",    "summarize")
research_builder.add_edge("summarize", END)

research_subgraph = research_builder.compile()  # compiled child

# --- Parent Graph (Executive Orchestrator) ---
class ExecutiveState(TypedDict):
    project_topic: str
    executive_brief: str

def writer_node(state: ExecutiveState):
    return {"executive_brief": f"Brief on: {state['project_topic']}"}

parent_builder = StateGraph(ExecutiveState)
parent_builder.add_node("research_agent", research_subgraph)  # child as single node
parent_builder.add_node("writer",         writer_node)
parent_builder.add_edge(START,            "research_agent")
parent_builder.add_edge("research_agent", "writer")
parent_builder.add_edge("writer",         END)

orchestrator = parent_builder.compile()
result = orchestrator.invoke({"project_topic": "LangGraph subgraphs"})
```

### 4. State Boundary Crossing
Parent state fields do **not** flow into child state automatically. Values cross boundaries via matching key names — the child's input key must match a key in the parent's state for automatic wiring. For mismatched schemas, add an adapter node before the child to translate parent state → child input.

### 5. When to Use Subgraphs
- The sub-problem has its own multi-node logic (not just a single function)
- You want to reuse the child graph in multiple parent graphs
- The child needs independent memory/checkpointing

---

## Chapter 8: Human-in-the-Loop Interrupts

### 1. The Operational Problem
Agents must not execute irreversible actions (payment processing, sending emails, deleting records) without human review. You need a mechanism to pause the graph mid-execution, expose the pending action to a human, and only continue after explicit approval.

### 2. The Architectural Core Concept
`interrupt_before=["node_name"]` at compile time installs a pause gate before the specified node. When execution reaches that node, the graph freezes and saves its state to the checkpointer. The node itself has **not yet run**.

Resuming is done by calling `app.invoke(None, config=config)` — passing `None` as input signals "no new data, just resume from the checkpoint."

### 3. The Reference Implementation

**File:** `langraph/05_human_in_the_loop.ipynb`

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

app = builder.compile(
    checkpointer=memory,
    interrupt_before=["charger"]   # pause before this node — it has NOT run yet
)

config = {"configurable": {"thread_id": "tx_999"}}

# Phase 1: Run until the interrupt gate
app.invoke({"amount": 4500.00, "account_id": "acc_abhishek", "status": "initiated"}, config=config)

# Inspect the frozen state — "charger" node is pending, not executed
snapshot = app.get_state(config)
print(f"Waiting for approval: {snapshot.next}")  # ('charger',)
print(f"Current status: {snapshot.values['status']}")

# Human reviews and decides to approve...

# Phase 2: Resume — the "charger" node now executes
final_state = app.invoke(None, config=config)
```

### 4. Modifying State Before Resuming
```python
# Human can also edit state before resuming
app.update_state(config, {"approved": True, "reviewer": "admin_abhishek"})
final_state = app.invoke(None, config=config)
```

### 5. Pitfalls
- **Requires a checkpointer** — without one, state can't survive the pause
- `interrupt_before` takes a **list** — gate multiple nodes simultaneously if needed
- The interrupted node has NOT run when you inspect state — it runs on resume
- Passing new input to `invoke()` on resume overrides state — use `None` to continue cleanly

---

## Chapter 9: Production Resilience (Retries + Checkpoints + Interrupts)

### 1. The Operational Problem
Real pipelines fail. Third-party APIs time out, warehouse services drop connections, payment gateways return 503s. Crashing the entire agent on a transient network error and requiring manual restart is not production-grade. You need automatic retry with exponential backoff, combined with checkpointing and human gates.

### 2. The Architectural Core Concept
`RetryPolicy` is attached per-node at `add_node()` time. When that node raises a matching exception type, the framework automatically waits (exponential backoff) and retries up to `max_attempts`. On final failure, the exception propagates normally.

Combined with checkpointing and interrupt gates, this gives the **production trifecta**:

| Layer | Protects Against | Tool |
|---|---|---|
| Checkpointing | Crashes, process restarts | `MemorySaver` / `SqliteSaver` |
| RetryPolicy | Transient failures (network, timeout) | `RetryPolicy` on nodes |
| Interrupts | Wrong / irreversible actions | `interrupt_before` |

### 3. The Reference Implementation

**File:** `langraph/06_checkpoints_retries_interrupts.ipynb`

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

# Define retry behavior per failure type
flaky_network_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,    # wait 0.5s before first retry
    backoff_factor=2.0,      # 0.5s → 1.0s → 2.0s
    retry_on=ConnectionError # ONLY retry this exception type — not logic errors
)

production_memory = MemorySaver()

builder.add_node(
    "inventory",
    verify_inventory_node,
    retry_policy=flaky_network_policy   # per-node retry
)

# Combine all three layers at compile
app = builder.compile(
    checkpointer=production_memory,
    interrupt_before=["billing"]  # human safety brake on the payment node
)
```

### 4. Retry Decision Logic
```
Node raises exception →
  Is exception in retry_on? No  → propagate immediately
  Is exception in retry_on? Yes →
    attempts < max_attempts → wait (backoff) → retry
    attempts == max_attempts → raise final exception
```

### 5. Pitfalls
- **Never retry idempotent-unsafe operations** (e.g., charging a card twice) without deduplication logic
- `retry_on=Exception` is dangerous — it retries logic bugs too. Be specific.
- Exponential backoff adds latency to every failed run — tune `max_attempts` based on your SLA

---

## Chapter 10: Real-World SQLite Persistence

### 1. The Operational Problem
`MemorySaver` lives in process memory — restarting the server loses all state. In a real HTTP API, each request is stateless: the graph must pause, persist to disk, and resume hours later from a different HTTP request with no shared process memory.

### 2. The Architectural Core Concept
`SqliteSaver` is a drop-in replacement for `MemorySaver` that writes checkpoints to a SQLite file. The same `thread_id` recovers the exact frozen graph state across process boundaries, server restarts, or arbitrary time gaps.

The pattern simulates a real HTTP API where separate function calls model separate requests to the same stateless endpoint.

### 3. The Reference Implementation

**File:** `langraph/07_realworld_sqlite_interrupts.ipynb`

```python
from langgraph.checkpoint.sqlite import SqliteSaver

DB_FILE = "production_vault.db"

# --- Simulated HTTP POST /orders (user action) ---
def endpoint_user_buy():
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer, interrupt_before=["disburse"])
        config = {"configurable": {"thread_id": "tx_abhishek_2026"}}
        app.invoke({"payload_id": "REQ-888", "amount": 9500.00, "approved": False}, config=config)
    # Context manager closes — state is frozen in SQLite file

# --- Simulated HTTP POST /orders/{id}/approve (admin, hours later) ---
def endpoint_admin_approve():
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer, interrupt_before=["disburse"])
        config = {"configurable": {"thread_id": "tx_abhishek_2026"}}  # same thread_id
        app.invoke(None, config=config)  # resumes from SQLite

# --- Admin state correction before resuming ---
def endpoint_admin_correct():
    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        app = builder.compile(checkpointer=checkpointer, interrupt_before=["disburse"])
        config = {"configurable": {"thread_id": "tx_abhishek_2026"}}

        app.update_state(config, {"admin_action": "REJECTED"}, as_node="processor")
        app.invoke(None, config=config)
```

### 4. Inspect State via Raw SQL
```python
import sqlite3
conn = sqlite3.connect(DB_FILE)
rows = conn.execute("SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints;").fetchall()
for row in rows:
    print(row)
```

### 5. Production Architecture
```
POST /jobs           → invoke → hits interrupt → saves to SQLite → return job_id
GET  /jobs/{id}      → get_state from SQLite → return current status
POST /jobs/{id}/approve → invoke(None) → resume from SQLite → complete
```

---

## Chapter 11: Remote & Distributed Agents

### 1. The Operational Problem
Not every specialist agent runs in the same process. A research agent, code execution sandbox, or ML inference service may live behind an HTTP endpoint on a different machine. The orchestrating graph needs to call remote agents and fold their results back into local state.

### 2. The Architectural Core Concept
A LangGraph node is just a Python function — it can contain any code, including HTTP requests. The node calls the remote service, waits for the response, and returns the result as a state update. From the graph's perspective, it's just another node.

### 3. The Reference Implementation

**File:** `langraph/08_remote_distributed_agents.ipynb`

```python
def remote_research_agent_node(state: ExecutiveState):
    REMOTE_AGENT_URL = "https://api.research-agent-service.internal/v1/run"

    payload = {"query": state["topic"]}

    # In production: use httpx.AsyncClient() for async or httpx.Client() for sync
    # Here simulated with a mock response
    mock_remote_response = {
        "status": "success",
        "artifacts": {"summary": "Verified architecture spec from remote machine"}
    }

    remote_result = mock_remote_response["artifacts"]["summary"]
    return {"compiled_data": remote_result}

builder.add_node("remote_research", remote_research_agent_node)
```

### 4. Integration Concerns

| Concern | Approach |
|---|---|
| Network failures | Combine with `RetryPolicy` (Chapter 9) |
| Long-running remotes | Interrupt + poll pattern (Chapter 10 style) |
| Authentication | Pass tokens via state or environment variables |
| Serialization | Ensure all state values are JSON-serializable |

### 5. Pitfalls
- Set explicit timeouts on HTTP clients — a hanging remote call blocks the graph indefinitely
- Avoid storing secrets in state — checkpointed state is written to disk
- Remote services should be idempotent if combined with retry policies (retry may call them twice)

---

## Chapter 12: Streaming Execution

### 1. The Operational Problem
`.invoke()` blocks until the entire graph finishes before returning. For multi-step agents with LLM calls, this means waiting 5–30 seconds for any output. Real-time UIs need incremental updates as each node completes.

### 2. The Architectural Core Concept
`.stream()` is a generator that yields after each node completes. Two modes control what's yielded:

- `stream_mode="updates"` — yields only the **delta** (what changed) from the last node
- `stream_mode="values"` — yields the **full accumulated state** after each node

### 3. The Reference Implementation

**File:** `langraph/09_streaming_execution.ipynb`

```python
from typing import TypedDict, Annotated
import operator

class StreamState(TypedDict):
    current_node: str
    counter: int
    log: Annotated[list, operator.add]

initial_payload = {"current_node": "", "counter": 0, "log": ["System Boot"]}

# Mode: updates — delta only (what each node changed)
print("=== UPDATES MODE ===")
for chunk in streaming_app.stream(initial_payload, stream_mode="updates"):
    print(chunk)
    # {'engine': {'current_node': 'engine', 'counter': 1, 'log': ['Engine fired']}}
    # Only shows the node that just ran and what it returned

# Mode: values — full state after each step
print("=== VALUES MODE ===")
for chunk in streaming_app.stream(initial_payload, stream_mode="values"):
    print(chunk)
    # {'current_node': 'engine', 'counter': 1, 'log': ['System Boot', 'Engine fired']}
    # Full accumulated state — history preserved
```

### 4. Mode Comparison

| Mode | Yields | Best for |
|---|---|---|
| `"updates"` | Delta from last node only | Token-by-token LLM streaming, minimal bandwidth |
| `"values"` | Full state after each node | Progress dashboards, debug monitoring |
| `"debug"` | Internal execution events | Graph structure debugging |

### 5. Async Streaming (FastAPI / Async Contexts)
```python
async for chunk in app.astream(input_data, stream_mode="updates"):
    yield f"data: {json.dumps(chunk)}\n\n"  # SSE format for frontend
```

---

## Chapter 13: Time-Travel Debugging

### 1. The Operational Problem
When an agent makes a wrong decision midway through a multi-step run (wrong routing, bad tool output, incorrect state), you need to rewind to the exact checkpoint before the mistake, fix the state, and re-run from there — without starting over from the beginning.

### 2. The Architectural Core Concept
`get_state_history(config)` returns all checkpoints for a thread in **reverse chronological order** (most recent first). Each snapshot contains the full state at that point and a `config` with the checkpoint ID.

`update_state(old_config, values, as_node)` creates a fork: it modifies the state at a historical checkpoint and returns a new config pointing to that fork. `invoke(None, forked_config)` re-runs the graph from that modified fork point.

### 3. The Reference Implementation

**File:** `langraph/10_time_travel_debugging.ipynb`

```python
config = {"configurable": {"thread_id": "debug_session"}}

# Run the graph (it made a mistake somewhere)
app.invoke({"issue_type": "unknown", "status": "open"}, config=config)

# Retrieve full execution history — reverse chronological (index 0 = latest)
history = list(app.get_state_history(config))

for i, snapshot in enumerate(history):
    print(f"[{i}] Next: {snapshot.next} | Values: {snapshot.values}")

# Identify the checkpoint just before the wrong routing decision
target = history[2]  # e.g., state before the "classify" node ran

# Fork: modify the state at that historical point
forked_config = app.update_state(
    target.config,                        # old checkpoint config
    values={"issue_type": "billing"},     # override the wrong value
    as_node="classify"                    # attribute update to this node
)

# Re-execute from the corrected fork
final_forked_result = app.invoke(None, config=forked_config)
print(final_forked_result)
```

### 4. History Snapshot Fields

| Field | Description |
|---|---|
| `snapshot.values` | Full state at this checkpoint |
| `snapshot.next` | Which node(s) run next from here |
| `snapshot.config` | Config with `checkpoint_id` for forking |
| `snapshot.metadata` | Step number, which node created this snapshot |

### 5. Use Cases

| Use case | Operation |
|---|---|
| Bug investigation | `get_state_history` → find where state went wrong |
| What-if analysis | `update_state` → change a field → `invoke(None)` |
| Production replay | Recover prod checkpoint → replay with fixed logic locally |
| Test generation | Save real state snapshots as test fixtures |

### 6. Pitfalls
- **Requires a checkpointer** — no checkpointer = no history to travel through
- `update_state` creates a **fork** — the original thread history is unchanged
- `as_node` determines which edges are followed after the update — set it to the node whose output you're simulating
- History grows unbounded — implement checkpoint pruning for long-running production threads

---

## API Reference

| Call | Purpose | First used |
|---|---|---|
| `@tool` | Declare LLM-callable function with schema | Ch 1 |
| `llm.bind_tools([...])` | Inject tool schemas into model | Ch 1 |
| `response.tool_calls` | Check if LLM requested tool execution | Ch 1 |
| `create_react_agent(llm, tools)` | Compile full ReACT loop as graph | Ch 3 |
| `StateGraph(Schema)` | Create graph with typed state | Ch 4 |
| `add_node(name, fn)` | Register node | Ch 4 |
| `add_edge(from, to)` | Direct transition | Ch 4 |
| `add_conditional_edges(from, router, map)` | Branching / feedback loop | Ch 5 |
| `compile()` | Build executable graph | Ch 4 |
| `compile(checkpointer=...)` | Add persistence | Ch 6 |
| `compile(interrupt_before=[...])` | Add human gate | Ch 8 |
| `add_node(..., retry_policy=...)` | Add fault tolerance to node | Ch 9 |
| `invoke(state, config)` | Execute synchronously | Ch 4 |
| `invoke(None, config)` | Resume from checkpoint | Ch 8 |
| `stream(state, stream_mode=...)` | Stream node-by-node | Ch 12 |
| `get_state(config)` | Inspect current/paused state | Ch 8 |
| `update_state(config, values, as_node)` | Modify checkpoint / fork | Ch 10, 13 |
| `get_state_history(config)` | All checkpoints reverse-chronological | Ch 13 |
| `MemorySaver()` | In-memory checkpointer | Ch 6 |
| `SqliteSaver.from_conn_string(path)` | File-based checkpointer | Ch 10 |
| `RetryPolicy(max_attempts, retry_on)` | Per-node retry policy | Ch 9 |
