# Building AI Agents and Agentic Workflows: Fundamentals of Building AI Agents

This is a compilation of notes from the Coursera Specialization [Building AI Agents and Agentic Workflows (IBM)](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/specializations/building-ai-agents-and-agentic-workflows), which is composed of the following courses:

- [Fundamentals of Building AI Agents](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/fundamentals-of-building-ai-agents?authProvider=deutschetelekom)
- [Agentic AI with LangChain and LangGraph](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langchain-and-langgraph)
- [Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langgraph-crewai-autogen-and-beeai)

This folder contains notes of the second course: **Agentic AI with LangChain and LangGraph**.

Table of contents:

- [Building AI Agents and Agentic Workflows: Fundamentals of Building AI Agents](#building-ai-agents-and-agentic-workflows-fundamentals-of-building-ai-agents)
  - [1. Introduction to LangGraph](#1-introduction-to-langgraph)
    - [Introduction to Agentic AI](#introduction-to-agentic-ai)
      - [Generative AI vs Agentic AI](#generative-ai-vs-agentic-ai)
      - [Agentic AI](#agentic-ai)
    - [LangChain and LangGraph](#langchain-and-langgraph)
      - [Core Components of LangGraph](#core-components-of-langgraph)
      - [Designing Effective LangGraph Workflows](#designing-effective-langgraph-workflows)
      - [When to use LangGraph vs LangChain](#when-to-use-langgraph-vs-langchain)
    - [Build a LangGraph Workflow](#build-a-langgraph-workflow)
      - [LangGraph 101](#langgraph-101)
      - [Exercise: Build a Stateful Workflow with LangGraph](#exercise-build-a-stateful-workflow-with-langgraph)
    - [Summary and Cheat Sheet: Introduction to LangGraph](#summary-and-cheat-sheet-introduction-to-langgraph)
      - [Getting Started With LangGraph](#getting-started-with-langgraph)
      - [Why Graph-Based Agents?](#why-graph-based-agents)
      - [When To Use LangGraph](#when-to-use-langgraph)
      - [Core Concepts](#core-concepts)
      - [Complete Example: Increment Counter](#complete-example-increment-counter)
  - [2. Build Self-Improving Agents with LangGraph](#2-build-self-improving-agents-with-langgraph)
    - [Build Reflection Agents](#build-reflection-agents)
      - [Overview: Type of Agents](#overview-type-of-agents)
      - [Building Reflection Agents with LangGraph](#building-reflection-agents-with-langgraph)
      - [Exercise: Build a Reflection Agent with LangGraph](#exercise-build-a-reflection-agent-with-langgraph)
    - [Advanced Self-Reflexion Agents](#advanced-self-reflexion-agents)
      - [Structuring LLM Tool Calls with Pydantic and JSON Serialization](#structuring-llm-tool-calls-with-pydantic-and-json-serialization)
        - [Why This Matters](#why-this-matters)
        - [Real Example: Addition Tool With Pydantic](#real-example-addition-tool-with-pydantic)
        - [Why Use Pydantic Models for Tool Calls?](#why-use-pydantic-models-for-tool-calls)
        - [Reusable Math Tool Schemas](#reusable-math-tool-schemas)
        - [What Does `Literal` Do?](#what-does-literal-do)
        - [Why JSON-Serializable Pydantic Models Are Powerful](#why-json-serializable-pydantic-models-are-powerful)
        - [Final Thoughts and Alternatives](#final-thoughts-and-alternatives)
      - [Understanding Reflexion Agents: Reflection + Tools + Real-Time Data + Verifiable Outputs](#understanding-reflexion-agents-reflection--tools--real-time-data--verifiable-outputs)
      - [Building Reflexion Agents](#building-reflexion-agents)
      - [Exercise: Building a Reflexion Agent with External Knowledge Integration](#exercise-building-a-reflexion-agent-with-external-knowledge-integration)
    - [ReAct: Reasoning + Action](#react-reasoning--action)
    - [Summary and Cheat Sheet: Build Self-Improving Agents with LangGraph](#summary-and-cheat-sheet-build-self-improving-agents-with-langgraph)
  - [3. Multi-Agent Systems and Agentic RAG with LangGraph](#3-multi-agent-systems-and-agentic-rag-with-langgraph)


## 1. Introduction to LangGraph

### Introduction to Agentic AI

#### Generative AI vs Agentic AI

* Generative AI is reactive: it waits for a prompt and generates content (text, images, code, audio) based on learned patterns; it does not act beyond generation without further input.
* Agentic AI is proactive: it uses a prompt to pursue goals through a loop of perception --> decision --> action --> feedback, with minimal human intervention.
* Both often rely on LLMs:
  * Generative AI uses them for content generation.
  * Agentic AI uses them for reasoning (e.g., chain-of-thought to break tasks into steps).
* Key difference:
  * Generative AI --> produces possibilities; human directs and refines.
  * Agentic AI --> executes multi-step tasks autonomously.
* Use cases:
  * Generative AI: content creation, scripting, media generation, assisted workflows.
  * Agentic AI: task automation (e.g., shopping agents monitoring prices, handling purchases).
* Agent behavior:
  * Breaks complex tasks into steps (planning).
  * Iteratively acts and adapts based on results.
* Future direction:
  * Hybrid systems combining both approaches:
    * generation for exploration
    * agentic execution for action
* Key idea: generative AI "creates", agentic AI "acts"; the most powerful systems will integrate both.

#### Agentic AI

* The evolution of LLMs (e.g., after ChatGPT) moved from simple text generation to tool use, memory, and function calling, enabling the emergence of AI agents.
* AI agents:
  * Single autonomous entities designed for specific tasks.
  * Capabilities: autonomy, task-specificity, and reactivity.
  * Operate with a simple loop: perceive --> reason --> act.
  * Use cases: chatbots, search assistants, email automation.
* Agentic AI:
  * Systems composed of multiple collaborating agents.
  * Capabilities:
    * Task decomposition (break goals into subtasks)
    * Inter-agent communication
    * Shared memory and learning
    * Centralized or distributed orchestration
  * Enables complex, multi-step, and parallel workflows.
* Key differences:
  * AI agent: single, linear, limited scope.
  * Agentic AI: multi-agent, collaborative, adaptive, scalable.
  * Agentic systems support iterative reasoning, planning, and re-planning.
* Architectural advancements in Agentic AI:
  * Multi-agent coordination via messaging/shared memory
  * Advanced reasoning (ReAct, Chain-of-Thought, Tree-of-Thoughts)
    * Chain-of-Thoughts: linear reasoning steps, internal monologue
  * Persistent memory (episodic, semantic, vector-based)
* Applications:
  * AI agents: customer support, internal tools, automation
  * Agentic AI: research assistants, robotics, healthcare systems, enterprise workflows
* Challenges:
  * AI agents: hallucinations, limited reasoning, weak long-horizon planning
  * Agentic AI: coordination failures, error propagation, scalability, explainability
* Emerging solutions:
  * RAG for grounding and shared knowledge
  * Tool/function calling for real-world interaction
  * Advanced memory systems for long-term reasoning
* Future trends:
  * Agents becoming proactive, learning, and more capable
  * Agentic AI evolving into coordinated multi-agent teams with governance
* Tooling ecosystem:
  * LangChain: building blocks for agents (tools, memory, chains)
  * LangGraph: graph-based multi-agent workflows
  * Other frameworks: CrewAI, AutoGen, etc.
* Key idea: AI is evolving from single-task agents to coordinated multi-agent systems (Agentic AI) that can solve complex, real-world problems through collaboration, planning, and memory.

![Agentic AI](./assets/agentic_ai.png)

![Agentic AI Example](./assets/agentic_ai_example.png)

### LangChain and LangGraph

#### Core Components of LangGraph

* LangGraph is a low-level framework (within the LangChain ecosystem) for building **stateful, multi-agent workflows** using graph structures.
* Core primitives:
  * Nodes: computation steps (functions).
  * Edges: define execution flow between nodes.
  * State: shared memory that persists context across the workflow.
* Key capabilities:
  * Looping and branching for dynamic decision-making.
  * State persistence for long-running, context-aware interactions.
  * Human-in-the-loop for manual intervention during execution.
  * Time travel for debugging by reverting to previous states.
* Advantages over traditional control flow (loops/conditionals):
  * Explicit state management across steps.
  * Runtime conditional transitions (dynamic branching).
  * Modular design (independent, reusable nodes).
  * Better observability and debugging of execution paths.
* Use case:
  * Ideal for complex agents requiring memory, adaptability, and multi-step reasoning (e.g., customer support agents that track context and escalate when needed).
* Visualization:
  * Workflows can be represented as graphs (e.g., Mermaid diagrams) to improve understanding and debugging.
* Key idea: LangGraph replaces linear control flow with graph-based orchestration, enabling flexible, stateful, and inspectable AI agent workflows.

![LangGraph Elements](./assets/langgraph_elements.png)

![Graph Visualization](./assets/graph_visualization.png)

#### Designing Effective LangGraph Workflows

* Graph architecture (LangGraph) enables flexible, stateful workflows beyond traditional loops by supporting dynamic branching, clear visualization, and modular reusable components.
* State design:
  * Stores shared context across nodes.
  * Use clear, descriptive names and keep structures flat.
* Node design:
  * Each node should have a single responsibility.
  * Types: processing, validation, integration, decision.
  * Nodes read from state, perform logic, and update state.
* Edges:
  * Control execution flow and enable conditional routing.
* Error handling:
  * Plan explicitly using retries, error states, and fallback/human intervention paths.
* Testing/debugging:
  * Test nodes independently.
  * Ensure predictable state transitions.
  * Build incrementally.
* Performance:
  * Keep state simple.
  * isolate expensive operations.
  * Use caching where needed.
* Integration:
  * Separate external system logic.
  * handle failures and timeouts.
  * design clear human-in-the-loop checkpoints.
* Common pitfalls:
  * oversized nodes
  * complex nested state
  * missing error handling
* Key idea: design workflows as modular, state-driven graphs with clear responsibilities and robust error handling.

```python
# State design: clear, flat, explicit schema
from typing import TypedDict

class SupportAgentState(TypedDict):
    user_input: str
    agent_response: str
    issue_type: str
    retry_count: int


# Node pattern: read state --> process --> update state
def process_request(state: SupportAgentState) -> SupportAgentState:
    # example processing logic
    state["agent_response"] = f"Processing: {state['user_input']}"
    return state


# Decision node: controls branching via edges
def route_decision(state: SupportAgentState) -> str:
    if state["retry_count"] > 2:
        return "human_review"        # route to human node
    elif state["issue_type"] == "resolved":
        return "end_interaction"     # terminate workflow
    else:
        return "continue_processing" # loop or next step


# Error handling pattern
# can be combined with routing logic to escalate
def error_handler(state: SupportAgentState) -> SupportAgentState:
    # increment retry count and log error
    state["retry_count"] += 1
    return state


# Example state for a document processing workflow
from typing import TypedDict

class DocumentProcessingState(TypedDict):
    file_path: str
    text_content: str
    summary: str
    analysis_results: dict

# Example node sequence (conceptual pipeline)
def extract_text(state: DocumentProcessingState) -> DocumentProcessingState:
    # simulate extraction
    state["text_content"] = "extracted text"
    return state

def analyze_text(state: DocumentProcessingState) -> DocumentProcessingState:
    # simulate analysis
    state["analysis_results"] = {"sentiment": "positive"}
    return state

def summarize(state: DocumentProcessingState) -> DocumentProcessingState:
    # simulate summarization
    state["summary"] = "short summary"
    return state
```

#### When to use LangGraph vs LangChain

* Both LangChain and LangGraph are open-source frameworks for building LLM-based applications, but they target different workflow complexities.
* LangChain:
  * Focus: building LLM applications via **sequential chains of operations**.
  * Structure: chain (DAG) --> fixed, forward execution flow.
  * Components: document loaders, text splitters, prompts, LLMs, memory, chains.
  * State: limited; mainly passed forward or handled via memory components.
  * Best for:
    * linear workflows (retrieve --> summarize --> answer)
    * well-defined pipelines with known steps.
* LangGraph:
  * Focus: **stateful, multi-agent systems** and complex workflows.
  * Structure: graph --> nodes (actions), edges (transitions), state (shared memory).
  * Supports loops, branching, and revisiting steps.
  * State: central, persistent, and shared across all nodes.
  * Best for:
    * dynamic, interactive systems
    * long-running, context-aware agents.
* Key differences:
  * Execution:
    * LangChain --> linear, predefined flow.
    * LangGraph --> dynamic, non-linear flow with loops.
  * State:
    * LangChain --> limited/passed along chain.
    * LangGraph --> core shared memory across workflow.
  * Complexity:
    * LangChain --> simpler pipelines.
    * LangGraph --> complex, adaptive multi-agent systems.
* Example contrast:
  * LangChain: retrieve data --> summarize --> answer (fixed sequence).
  * LangGraph: task manager agent --> process input --> branch to add/complete/summarize tasks --> loop back with updated state.
* Key idea:
  * LangChain is for **composing LLM pipelines**,
  * LangGraph is for **orchestrating stateful, adaptive agent systems**.

### Build a LangGraph Workflow

#### LangGraph 101

* LangGraph models workflows as graphs with state, enabling looping, branching, and dynamic execution.
* Core concepts:
    * State: shared data (e.g., variables like n, letter) that evolves across steps.
    * Nodes: functions that read state and return state updates or perform side effects.
    * Edges: define transitions between nodes.
    * `START` and `END`: explicit graph boundaries.
* State is typically defined using `TypedDict` for structured, typed data.
* Nodes usually return **partial state updates** (only changed fields); LangGraph merges those updates into the current state.
* Reducers control how repeated updates are merged:
    * default behavior replaces a state key
    * `Annotated[..., reducer]` can append/merge values (e.g., `operator.add` for lists)
    * chat workflows commonly use `MessagesState` or `add_messages` for message history
* Workflow example (below):
    * Initialize state `(n=1, letter="")`
    * Increment `n` and generate a random letter
    * Print values
    * Loop until `n >= 13`, then stop
* Conditional edges:
    * Control flow based on state (e.g., continue loop or end).
* Building a LangGraph app:
    * Define state schema
    * Create node functions
    * Add nodes and edges
    * Connect `START` to the first node and terminal paths to `END`
    * Compile graph
    * Run with `invoke(initial_state)` or `stream(initial_state)`
* Persistence and long-running workflows:
    * Compile with a checkpointer (e.g., `compile(checkpointer=...)`) to persist state.
    * Use a `thread_id` in the run config to continue the same conversation/workflow later.
    * Checkpoints enable human-in-the-loop interrupts, resume with `Command(resume=...)`, and time travel/debugging.
* Advanced routing:
    * `add_conditional_edges` routes based on state.
    * `Command` can combine a state update with a routing or resume instruction.
* Key idea: execution flows through nodes while state is updated and passed along, enabling dynamic, iterative workflows.

In the following, this graph is created:

```mermaid
flowchart TD
    START([START]) --> ADD["add_node\nincrement n\ngenerate random letter"]
    ADD --> PRINT["print_node\nprint current state"]
    PRINT --> ROUTE{"should_continue(state)"}
    ROUTE -- "n < 13" --> ADD
    ROUTE -- "n >= 13" --> END([END])
```

```python
# 1. Define the graph state.
# Every node receives this state as input. A node can then return updates
# for one or more keys, and LangGraph merges those updates into the state.
from typing import Literal
from typing_extensions import TypedDict

class ChainState(TypedDict):
    # Counter used to decide when the loop should stop.
    n: int
    # Latest generated letter. This is replaced on every "add" node run.
    letter: str

# 2. Define a node that updates state.
# This node does not need to return the full state. It returns only the
# fields that changed: n and letter.
import random
import string

def add_node(state: ChainState) -> dict:
    return {
        "n": state["n"] + 1,
        "letter": random.choice(string.ascii_lowercase)
    }


# 3. Define a node that performs a side effect.
# It reads the current state and prints it. Since it does not change state,
# it returns an empty update.
def print_node(state: ChainState) -> dict:
    print(f"n={state['n']}, letter={state['letter']}")
    return {}  # no state update


# 4. Define conditional routing logic.
# After "print" runs, LangGraph calls this function with the latest state.
# The returned label is mapped to the next graph destination below.
def should_continue(state: ChainState) -> Literal["add", "end"]:
    if state["n"] >= 13:
        return "end"
    return "add"


# 5. Build the graph.
# START is the explicit graph entry boundary, and END is the explicit
# terminal boundary. The graph loops from "print" back to "add" until the
# conditional route returns "end".
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ChainState)
workflow.add_node("add", add_node)
workflow.add_node("print", print_node)
workflow.add_edge(START, "add")  # Entry point
workflow.add_edge("add", "print")

# Conditional edge: after print --> either loop or end
# this mapping translates labels into graph destinations
# Note: should_continue is not a node, but a routing function attached to conditional edges
workflow.add_conditional_edges(
    "print",  # 1. source node
    should_continue,  # 2. routing function
    {  # 3. routing map: outputs from should_continue are mapped to these nodes
        "add": "add",
        "end": END
    }
)

# compile() validates the graph and produces a runnable app.
app = workflow.compile()


# 6. Run the workflow.
# The initial_state starts with n=1 and an empty letter. The first node run
# increments n to 2, generates a letter, prints it, and then the loop repeats.
result = app.invoke({
    "n": 1,
    "letter": ""
})

print("Final state:", result)
# Execution behavior:
# add --> print --> (check condition)
# --> loop until n >= 13 --> END
#
# Example state evolution:
# initial state: {"n": 1, "letter": ""}
# after add_node: {"n": 2, "letter": "q"}  # random letter
# after print_node: {"n": 2, "letter": "q"}  # unchanged
# should_continue returns "add" because n < 13
# after add_node: {"n": 3, "letter": "m"}
# after print_node: {"n": 3, "letter": "m"}
# ...
# after add_node: {"n": 13, "letter": "x"}
# after print_node: {"n": 13, "letter": "x"}
# should_continue returns "end" because n >= 13
# final state: {"n": 13, "letter": "x"}
```

#### Exercise: Build a Stateful Workflow with LangGraph

Notebook: [`lab/01_LangGraph101-v1.ipynb`](./lab/01_LangGraph101-v1.ipynb).

This lab introduces LangGraph by building stateful workflows from small, explicit pieces:

* Setup:
    * installs current `langgraph`, `langchain[openai]`, and `python-dotenv`
    * loads `OPENAI_API_KEY` from `.env`
    * creates an OpenAI chat model with `init_chat_model`
* LangGraph basics:
    * defines typed state with `TypedDict`
    * creates node functions that read state and return partial updates
    * connects nodes with normal edges and conditional edges
    * uses explicit `START` and `END` graph boundaries
* Example 1: Authentication workflow:
    * collects username/password input
    * validates credentials (it looks for exact password)
    * routes to success or failure with a conditional edge
    * loops back to the input node after failed authentication
* Example 2: QA workflow:
    * validates a user question (looking for keywords)
    * adds LangGraph-specific context when the question is relevant
    * calls an OpenAI model to answer from the provided context
    * returns a fallback answer when no context is available
* Final exercise:
    * builds a small looping counter graph
    * increments `n`, generates a random letter, prints state, and loops until `n >= 13`
    * reinforces the modern `START -> node -> ... -> END` style

Example 1: authentication workflow:

```mermaid
flowchart TD
    START([START]) --> INPUT["InputNode\ncollect username/password"]
    INPUT --> VALIDATE["ValidateCredential\ncheck credentials"]
    VALIDATE --> ROUTER{"router(state)"}
    ROUTER -- "authenticated" --> SUCCESS["Success\nwrite success output"]
    ROUTER -- "not authenticated" --> FAILURE["Failure\nwrite failure output"]
    FAILURE --> INPUT
    SUCCESS --> END([END])
```

Code summary:

```python
class AuthState(TypedDict):
    username: Optional[str]
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]

def input_node(state):
    if state.get("username", "") == "":
        username = input("What is your username?")
    else:
        username = state["username"]

    password = input("Enter your password: ")
    return {
        "username": username,
        "password": password,
    }

def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")
    return {
        "is_authenticated": (
            username == "test_user" and password == "secure_password"
        )
    }

def success_node(state):
    return {"output": "Authentication successful! Welcome."}

def failure_node(state):
    return {"output": "Not successful, please try again!"}

def router(state):
    if state["is_authenticated"]:
        return "success_node"
    return "failure_node"

workflow = StateGraph(AuthState)
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)

workflow.add_edge(START, "InputNode")
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_conditional_edges(
    "ValidateCredential",
    router,
    {
        "success_node": "Success",
        "failure_node": "Failure",
    },
)
workflow.add_edge("Success", END)
workflow.add_edge("Failure", "InputNode")

app = workflow.compile()
result = app.invoke({"username": "test_user"})
```

Example 2: QA workflow:

```mermaid
flowchart TD
    START([START]) --> INPUT["InputNode\nvalidate question"]
    INPUT --> CONTEXT["ContextNode\nadd context if relevant"]
    CONTEXT --> QA["QANode\nanswer with OpenAI model\nor fallback if no context"]
    QA --> END([END])
```

Code summary:

```python
class QAState(TypedDict):
    question: Optional[str]
    valid: Optional[bool]
    error: Optional[str]
    context: Optional[str]
    answer: Optional[str]

def input_validation_node(state):
    question = state.get("question", "").strip()
    if not question:
        return {
            "valid": False,
            "error": "Question cannot be empty.",
        }
    return {"valid": True, "error": None}

def context_provider_node(state):
    question = state.get("question", "").lower()
    if "langgraph" in question or "guided project" in question:
        return {
            "context": (
                "This guided project is about using LangGraph, a Python "
                "library to design state-based workflows. LangGraph connects "
                "modular nodes with normal and conditional edges."
            )
        }
    return {"context": None}

def llm_qa_node(state):
    question = state.get("question", "")
    context = state.get("context")

    if not context:
        return {"answer": "I don't have enough context to answer your question."}

    messages = [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    response = qa_llm.invoke(messages)
    return {"answer": response.content.strip()}

qa_workflow = StateGraph(QAState)
qa_workflow.add_node("InputNode", input_validation_node)
qa_workflow.add_node("ContextNode", context_provider_node)
qa_workflow.add_node("QANode", llm_qa_node)

qa_workflow.add_edge(START, "InputNode")
qa_workflow.add_edge("InputNode", "ContextNode")
qa_workflow.add_edge("ContextNode", "QANode")
qa_workflow.add_edge("QANode", END)

qa_app = qa_workflow.compile()
qa_app.invoke({"question": "What is LangGraph?"})
```

### Summary and Cheat Sheet: Introduction to LangGraph

LangGraph is an open-source framework for **stateful, graph-based AI agents**.

* Extends LangChain with explicit control flow.
* Uses shared state across workflow steps.
* Supports branching, loops, persistence, human review, and debugging.
* Works with LangChain models, tools, retrievers, and LangSmith.

#### Getting Started With LangGraph

| Topic | Summary |
| --- | --- |
| Overview | Build graph-shaped AI workflows. |
| LangChain extension | Adds stateful orchestration to LangChain components. |
| State | Shared `TypedDict` or Pydantic object. |
| Flow | Branching, loops, retries, and routing. |
| Agents | Iterative reasoning, tools, review, and coordination. |
| Execution | Durable runs, checkpoints, resume, streaming. |
| Observability | Graph diagrams and LangSmith traces. |

Install:

```bash
pip install langgraph "langchain[openai]" python-dotenv
```

#### Why Graph-Based Agents?

* Linear chains are good for fixed flows: retrieve, call model, parse, answer.
* Agents often need loops: retry a tool, revise a query, or ask for approval.
* LangGraph models workflows as state machines.
* The graph can revisit nodes, branch by state, and stop only when a condition is met.
* Example: poor retrieval -> rewrite query -> retrieve again -> evaluate -> answer.

#### When To Use LangGraph

| Use case | Why LangGraph helps |
| --- | --- |
| Loops | Repeat until a goal is met. |
| Branching | Route with explicit if/else logic. |
| Long runs | Persist and resume with checkpoints. |
| Complex state | Keep workflow data in one shared object. |
| Multi-agent flows | Coordinate agents, tools, or reviewers. |
| Human review | Pause and resume with `Command(resume=...)`. |
| Debugging | Inspect graph, state, streams, and traces. |

#### Core Concepts

| Concept | Explanation |
| --- | --- |
| State | Shared data passed between nodes. |
| `StateGraph` | Blueprint for nodes, edges, and state. |
| Nodes | Functions or `Runnable`s that return state updates. |
| Edges | Fixed or conditional transitions. |
| `START` / `END` | Special graph boundaries. |
| `compile()` | Builds the runnable app. |
| `invoke()` / `stream()` | Run the compiled graph. |
| Checkpointers | Save state for resume and time travel. |
| Reducers | Control how updates merge into state. |

State and graph:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    user_query: str
    summary: str
    step_count: int

graph = StateGraph(WorkflowState)
```

Define a node:

```python
def summarize(state: WorkflowState) -> dict:
    text = state["user_query"]
    summary = llm_summarize(text)

    return {
        "summary": summary,
        "step_count": state["step_count"] + 1,
    }

graph.add_node("summarize", summarize)
```

Edges:

```python
graph.add_edge(START, "summarize")
graph.add_edge("summarize", "finalize")
graph.add_edge("finalize", END)
```

Conditional edges:

```python
from typing import Literal

def decide(state: WorkflowState) -> Literal["repeat", "done"]:
    if state["step_count"] < 2:
        return "repeat"
    return "done"

graph.add_conditional_edges(
    "summarize",
    decide,
    {
        "repeat": "summarize",
        "done": END,
    },
)
```

Run and visualize:

```python
app = graph.compile()
final_state = app.invoke({
    "user_query": "Hello",
    "summary": "",
    "step_count": 0,
})
print(app.get_graph().draw_mermaid())
```

Note: the routing function is **not** a node. It returns a label; the mapping chooses the destination.

#### Complete Example: Increment Counter

This graph increments `count` until it reaches `3`.

```mermaid
flowchart TD
    START([START]) --> INCREMENT["increment\ncount += 1"]
    INCREMENT --> DECIDE{"decide_next(state)"}
    DECIDE -- "again" --> INCREMENT
    DECIDE -- "finish" --> END([END])
```

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    count: int
    message: str

def increment(state: GraphState) -> dict:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "message": f"Count is now {new_count}",
    }

def decide_next(state: GraphState) -> Literal["again", "finish"]:
    if state["count"] < 3:
        return "again"
    return "finish"

graph = StateGraph(GraphState)
graph.add_node("increment", increment)

graph.add_edge(START, "increment")
graph.add_conditional_edges(
    "increment",
    decide_next,
    {
        "again": "increment",
        "finish": END,
    },
)

app = graph.compile()
result = app.invoke({
    "count": 0,
    "message": "",
})

print(result)
# {"count": 3, "message": "Count is now 3"}
```

Key takeaways:

* Use LangGraph for state, branching, loops, retries, or durable execution.
* State is the data; `StateGraph` is the structure.
* Nodes return updates.
* Edges move execution.
* Conditional edges route at runtime.
* `START` and `END` are special boundaries.
* `compile()` builds the app; `invoke()` or `stream()` runs it.
* Mermaid and LangSmith help debug behavior.

## 2. Build Self-Improving Agents with LangGraph

### Build Reflection Agents

#### Overview: Type of Agents

* AI agents are classified by how they make decisions and interact with their environment, ranging from simple rule-based systems to adaptive learning systems.
* Five main types of agents:
    * Simple reflex agents: it *reacts*
        * Use condition–action rules (if --> then).
          * Percept environment with sensors, match conditions, execute actions.
        * No memory or history.
        * Work well in predictable environments.
        * Limitation: fail in dynamic or unseen situations.
    * Model-based reflex agents: it *remembers*
        * Maintain an internal state (memory of the world).
        * Track how the environment evolves and how actions affect it.
        * Still reactive (no planning), but more robust than simple reflex agents.
    * Goal-based agents: it *aims*
        * It builds on the model-based agent by adding a goal or objective to achieve.
        * Make decisions based on achieving a goal.
          * Internal question: which action will lead me closer to the goal?
        * Simulate future outcomes of actions.
        * Choose actions that lead toward the goal.
        * Limitation: any solution meeting the goal is acceptable (no quality ranking).
    * Utility-based agents: it *evaluates*
        * Extend goal-based agents by optimizing for the best outcome.
          * A goal is a binary condition (achieved/not achieved), while utility is a measure of desirability.
        * Use a utility function to evaluate desirability (e.g., speed, cost, safety).
        * Select actions that maximize overall utility.
    * Learning agents: the most adaptable and powerful type. It *improves*.
        * Improve over time through experience and feedback.
        * Components:
            * performance element (acts)
            * critic (evaluates outcome, provides feedback/reward)
            * learning element (updates strategy)
            * problem generator (explores new actions)
        * Most powerful but data-intensive and slower to train.
* Key progression:
    * reflex --> remembers --> plans --> optimizes --> learns
* Multi-agent systems: That's what we want usually to improve performance and capabilities.
    * Combine multiple agents working together in a shared environment.
    * Enable more complex, collaborative problem-solving.
* Key idea:
    * Agent sophistication increases from fixed rules to adaptive learning, but real-world systems often combine multiple agent types and still benefit from human oversight.

![Learning Agent](./assets/learning_agent.png)

#### Building Reflection Agents with LangGraph

* Reflection agents improve outputs through an iterative feedback loop:
    * Generator produces an initial response.
    * Reflector critiques it.
    * Generator refines based on feedback.
    * Loop repeats for several iterations.
* Two roles:
    * Generator: creates content.
    * Reflector: evaluates and suggests improvements.
* State:
    * Maintains full conversation history across iterations.
    * Each iteration adds messages (input, outputs, critiques).
* Implementation:
    * Use LangChain for prompt + LLM chains.
    * Use LangGraph for workflow orchestration.
* LangGraph setup:
    * State = `MessagesState`, which stores and appends conversation messages.
    * Nodes:
        * generate --> produces output
        * reflect --> critiques output
    * Edges:
        * START --> generate
        * reflect --> generate
    * Conditional routing:
        * after generate, either stop or go to reflect.
* Key idea:
    * Reflection agents simulate "self-critique", improving quality over multiple passes.

![Reflection Agent Example](./assets/reflection_agent_example.png)

![Reflection Agent Exercise](./assets/reflection_agent_exercise.png)

![Reflection Agent Exercise Continued](./assets/reflection_agent_exercise_2.png)

Example graph flow:

```mermaid
flowchart TD
    START([START]) --> GENERATE["generate\ncreate or revise content"]
    GENERATE --> ROUTER{"should_continue(state)"}
    ROUTER -- "enough iterations" --> END([END])
    ROUTER -- "continue" --> REFLECT["reflect\ncritique latest draft"]
    REFLECT --> GENERATE
```

```python
# 1. Generator and reflector chains (LangChain)
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")

llm = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-5-nano"),
    model_provider="openai",
)

# Generator prompt
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful content creator. Create a high-quality post."),
    MessagesPlaceholder(variable_name="messages")
])
generate_chain = generation_prompt | llm  # pipe operator builds chain

# Reflector prompt
reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical reviewer. Give feedback."),
    MessagesPlaceholder(variable_name="messages")
])
reflect_chain = reflect_prompt | llm


# 2. LangGraph nodes
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState

def generate_node(state: MessagesState) -> dict:
    # Generate content based on full conversation history.
    response = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=response.content)]}

def reflect_node(state: MessagesState) -> dict:
    # Critique the latest AI draft.
    critique = reflect_chain.invoke({"messages": state["messages"]})
    # Wrap the critique as HumanMessage so the generator treats it as feedback.
    # NOTE: There is no human input in reality, but this is the way to feed the critique back into the generator chain.
    return {"messages": [HumanMessage(content=critique.content)]}


# 3. Build LangGraph workflow
# MessagesState is a built-in state schema with a "messages" key.
# LangGraph appends returned messages instead of replacing the history.
from typing import Literal
from langgraph.graph import StateGraph, START, END

graph = StateGraph(MessagesState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)

# Flow: START --> generate --> reflect --> generate ...
graph.add_edge(START, "generate")
graph.add_edge("reflect", "generate")

# Routing function: stop after several messages, otherwise critique.
def should_continue(state: MessagesState) -> Literal["reflect", END]:
    if len(state["messages"]) > 6:
        return END
    return "reflect"

graph.add_conditional_edges("generate", should_continue)
app = graph.compile()


# 4. Run reflection agent
initial_state = {
    "messages": [
        HumanMessage(
            content="Write a LinkedIn post about getting a dev job under 160 chars"
        )
    ]
}

result = app.invoke(initial_state)

# final refined output
print(result["messages"][-1].content)
# Execution pattern:
# Human --> generate --> reflect --> generate --> reflect --> ... --> END
# Each loop improves the output
```

#### Exercise: Build a Reflection Agent with LangGraph

Notebook: [`lab/02_Tweet-Reflection-Agent-v1.ipynb`](./lab/02_Tweet-Reflection-Agent-v1.ipynb).

This notebook implements the reflection agent as explained in the previous section:

* Setup:
    * installs current `langgraph`, `langchain[openai]`, and `python-dotenv`
    * loads `OPENAI_API_KEY` and optional `OPENAI_MODEL` from `.env`
    * initializes an OpenAI chat model with `init_chat_model`
* Reflection workflow:
    * builds a LinkedIn post generator prompt
    * builds a critique prompt for content strategy feedback
    * chains each prompt to the same OpenAI model
* LangGraph implementation:
    * uses `StateGraph(MessagesState)` instead of the older `MessageGraph`
    * defines `generate` and `reflect` nodes
    * returns partial message updates with `{"messages": [...]}`
    * starts with `START -> generate`
    * loops `reflect -> generate`
    * uses `add_conditional_edges` after `generate` to stop at `END`
* Execution:
    * invokes the graph with an initial `HumanMessage`
    * inspects the first draft, first critique, and final revised post
    * renders the graph as Mermaid with `draw_mermaid()`

Important code (in the notebook the prompts are more detailed):

```python
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")

llm = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-5-nano"),
    model_provider="openai",
)

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional LinkedIn content assistant. Generate the "
        "best LinkedIn post possible. If feedback is provided, revise the "
        "previous draft.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])
generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional LinkedIn content strategist. Critique the "
        "post and provide actionable feedback for the next revision.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])
reflect_chain = reflection_prompt | llm

def generation_node(state: MessagesState) -> dict:
    generated_post = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=generated_post.content)]}

def reflection_node(state: MessagesState) -> dict:
    critique = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=critique.content)]}

def should_continue(state: MessagesState) -> Literal["reflect", END]:
    if len(state["messages"]) > 6:
        return END
    return "reflect"

graph = StateGraph(MessagesState)
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)

graph.add_edge(START, "generate")
graph.add_edge("reflect", "generate")
graph.add_conditional_edges("generate", should_continue)

workflow = graph.compile()

response = workflow.invoke({
    "messages": [
        HumanMessage(
            content="Write a LinkedIn post about getting a software developer job under 160 characters"
        )
    ]
})

print(response["messages"][-1].content)
```

### Advanced Self-Reflexion Agents

#### Structuring LLM Tool Calls with Pydantic and JSON Serialization

LLMs can produce free-form text, but real applications often need structured data that can be passed to APIs, databases, tools, or downstream functions.

* Tool binding lets the model extract arguments for a function-like schema.
* Pydantic models define that schema in Python.
* The model output becomes predictable, typed, validated, and JSON-serializable.
* This is useful for agentic workflows where LLM output becomes another system's input.

##### Why This Matters

Example: a weather API may expect:

* `condition`: weather condition, such as `sunny`, `rainy`, or `cloudy`
* `temperature`: integer value
* `unit`: `celsius` or `fahrenheit`

You can express that contract as a Pydantic model:

```python
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    """Weather information extracted from user text."""

    condition: str = Field(
        description="Weather condition such as sunny, rainy, cloudy"
    )
    temperature: int = Field(description="Temperature value")
    unit: str = Field(
        description="Temperature unit such as fahrenheit or celsius"
    )
```

Bind the schema as a tool:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-5-nano", model_provider="openai")
weather_llm = llm.bind_tools([WeatherSchema])

response = weather_llm.invoke("It's sunny and 75 degrees")

print(response.tool_calls[0]["args"])
# {"condition": "sunny", "temperature": 75, "unit": "fahrenheit"}
```

The tool call arguments are a dictionary of key-value pairs that can be validated, transformed, or passed to a real weather API.

Another conceptual example: spam detection.

```python
class SpamSchema(BaseModel):
    """Classify whether an email is spam."""

    classification: str = Field(description="Email classification: spam or not_spam")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(description="Reason for the classification")

spam_llm = llm.bind_tools([SpamSchema])
response = spam_llm.invoke("I'm a Nigerian prince, you want to be rich")

print(response.tool_calls[0]["args"])
# {
#   "classification": "spam",
#   "confidence": 0.95,
#   "reason": "Nigerian prince scam"
# }
```

These are conceptual examples. The exact output depends on the model and prompt, but the schema gives the model a structured target.

##### Real Example: Addition Tool With Pydantic

In a real workflow, this pattern could extract flight-booking fields such as origin, destination, date, and time before calling an API. A smaller example is addition: the model extracts two numbers, and Python performs the actual calculation.

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

class Add(BaseModel):
    """Add two numbers together."""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")

llm = init_chat_model("gpt-5-nano", model_provider="openai")
initial_chain = llm.bind_tools([Add])

question = "add 1 and 10"
response = initial_chain.invoke([HumanMessage(content=question)])

def extract_and_add(response) -> int:
    tool_call = response.tool_calls[0]
    args = tool_call["args"]
    return args["a"] + args["b"]

result = extract_and_add(response)

print(
    f"LLM extracted: a={response.tool_calls[0]['args']['a']}, "
    f"b={response.tool_calls[0]['args']['b']}"
)
print(f"Result: {result}")
```

Key point:

* The LLM does **extraction and tool selection**.
* Your code performs the deterministic operation.
* The schema makes the handoff reliable.

##### Why Use Pydantic Models for Tool Calls?

In tool-augmented LLM applications, inputs and outputs should be:

* structured
* validated
* easy to serialize as JSON
* easy to deserialize from JSON
* reusable across tools

Pydantic provides runtime validation, type hints, helpful errors, and JSON serialization.

##### Reusable Math Tool Schemas

```python
from typing import Literal
from pydantic import BaseModel

class TwoOperands(BaseModel):
    a: float
    b: float

class AddInput(TwoOperands):
    operation: Literal["add"]

class SubtractInput(TwoOperands):
    operation: Literal["subtract"]

class MathToolRequest(TwoOperands):
    operation: Literal["add", "subtract"]

class MathOutput(BaseModel):
    result: float
```

Tool functions can accept and return Pydantic models:

```python
def add_tool(data: AddInput) -> MathOutput:
    return MathOutput(result=data.a + data.b)

def subtract_tool(data: SubtractInput) -> MathOutput:
    return MathOutput(result=data.a - data.b)
```

Dispatch from JSON input:

```python
incoming_json = '{"a": 7, "b": 3, "operation": "subtract"}'

def dispatch_tool(json_payload: str) -> str:
    # Pydantic v2: parse and validate raw JSON.
    request = MathToolRequest.model_validate_json(json_payload)

    if request.operation == "add":
        output = add_tool(AddInput.model_validate_json(json_payload))
    elif request.operation == "subtract":
        output = subtract_tool(SubtractInput.model_validate_json(json_payload))
    else:
        raise ValueError("Unsupported operation")

    # Pydantic v2: serialize model to JSON.
    return output.model_dump_json()

result_json = dispatch_tool(incoming_json)
print(result_json)
# {"result":4.0}
```

`MathToolRequest` validates the shared fields and allowed operation. The operation-specific model then validates the exact tool payload.

##### What Does `Literal` Do?

`Literal` restricts a field to specific constant values. This prevents unsupported operations from reaching your tool logic.

```python
from typing import Literal
from pydantic import BaseModel, Field

class CalculatorSchema(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

calculator_llm = llm.bind_tools([CalculatorSchema])

response = calculator_llm.invoke("Add 15 and 23")
print(response.tool_calls[0]["args"])
# {"operation": "add", "a": 15.0, "b": 23.0}

response = calculator_llm.invoke("Multiply 7 by 8")
print(response.tool_calls[0]["args"])
# {"operation": "multiply", "a": 7.0, "b": 8.0}
```

##### Why JSON-Serializable Pydantic Models Are Powerful

| Feature | Benefit |
| --- | --- |
| Type validation | Rejects malformed inputs early. |
| Reusability | Share base schemas across tools. |
| JSON serialization | Use `model_dump_json()` for APIs and storage. |
| JSON parsing | Use `model_validate_json()` for incoming payloads. |
| Extensibility | Add tools such as multiply or divide easily. |
| Testability | Validate tool behavior without an LLM. |

##### Final Thoughts and Alternatives

Pydantic schemas make LLM applications:

* more robust
* easier to test
* easier to maintain
* safer to connect to APIs and databases
* easier to orchestrate in LangChain, LangGraph, CrewAI, and similar frameworks

Python `dataclasses` can also define lightweight data containers. However, Pydantic is usually preferred for LLM tool calls because it provides:

* runtime validation
* field descriptions for tool schemas
* JSON parsing and serialization
* custom validators
* strong integration with LangChain, FastAPI, and other Python frameworks

Key idea: Pydantic turns model output from "some text" into a typed contract your software can trust.

#### Understanding Reflexion Agents: Reflection + Tools + Real-Time Data + Verifiable Outputs

* Reflexion agents extend reflection agents by adding tool use, real-time data, and verifiable outputs (citations).
* Core loop:
    * Generator (responder) creates an initial answer.
    * Self-critique identifies weaknesses.
    * Tool (e.g., web search) retrieves external information.
    * Revisor refines the response using critique + tool outputs.
    * Loop repeats for multiple iterations.
* Key capabilities:
    * Continuous self-improvement across iterations.
    * Detection and correction of errors in prior outputs.
    * Integration of up-to-date external data (post-training knowledge).
    * Transparent outputs with citations and references.
* Structured outputs:
    * Responses are not plain text but follow a schema, a kind of a dict/table; example initial query: "I need more minerals in my diet"
      * response: "you can get more minerals by eating rocks"
      * self-critique: "this is not a good answer because humans can't eat rocks"
      * queries: "what are good dietary sources of minerals?"
        * for each query, tool(s) return results (content + URLs); in this case we have a search tool
      * references (created by the revisor by using the tool results)
    * Enables clear separation between reasoning, tool inputs, and final answers.
* Workflow:
    * User query --> responder outputs structured response + search query
    * Tool executes query --> returns results (content, URLs)
    * Revisor updates response using critique + tool data + references
    * Repeat until stopping condition
* State:
    * Maintained as a list (e.g., response_list) containing:
        * original query
        * generated responses
        * critiques
        * tool outputs
        * revised responses
* Key idea:
    * Reflexion agents combine self-critique + external knowledge + structured reasoning to produce more accurate, grounded, and explainable results than standard reflection agents.

#### Building Reflexion Agents

* Reflexion agents are built by combining LLM generation, self-critique, tool use (search), and iterative revision using structured schemas.
* Setup:
    * Configure a search tool (e.g., Tavily) to retrieve external data.
    * Initialize an LLM with `init_chat_model`.
    * Load API keys from `.env` (`OPENAI_API_KEY`, and `TAVILY_API_KEY` if using Tavily).
    * Install provider packages such as `langchain-tavily` when using Tavily.
* Structured outputs:
    * Define schemas (e.g., AnswerQuestion, Reflection) to enforce fields like:
        * answer
        * critique (missing / superfluous)
        * search queries
        * citations (for revised outputs)
    * LLM outputs structured objects instead of plain text.
* Workflow:
    * Responder node:
        * Generates initial answer + critique + search queries.
    * Tool node:
        * Executes search queries and returns results.
    * Revisor node:
        * Improves answer using critique + tool results + citations.
    * Loop:
        * Repeat tools --> revisor until iteration limit.
* State:
    * Maintained with `MessagesState`, which appends:
        * user query
        * AI responses
        * tool outputs
        * revisions
* LangGraph orchestration:
    * Nodes: responder, tool executor, revisor.
    * Edges: `START -> respond -> tools -> revise`.
    * Loop: `revise -> tools` until `END`.
    * Conditional routing controls iteration count.
* Key idea:
    * Reflexion agents produce higher-quality, evidence-backed answers by combining structured reasoning, external knowledge, and iterative self-improvement.

Example graph flow:

```mermaid
flowchart TD
    START([START]) --> RESPOND["respond\ninitial answer + critique\nsearch queries"]
    RESPOND --> TOOLS["tools\nrun Tavily searches"]
    TOOLS --> REVISE["revise\nimprove answer\nadd citations"]
    REVISE --> ROUTER{"should_continue(state)"}
    ROUTER -- "continue" --> TOOLS
    ROUTER -- "iteration limit" --> END([END])
```

```python
# pip install -U langgraph langchain langchain-openai langchain-tavily python-dotenv
# We need to create a Tavily account + TAVILY_API_KEY

# 1. Setup model + search tool
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-5-nano"),
    model_provider="openai",
)

search_tool = TavilySearch(max_results=5, topic="general")


# 2. Define structured output schemas

class Reflection(BaseModel):
    missing: str = Field(description="Important missing information")
    superfluous: str = Field(description="Unnecessary or unsupported information")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="Draft answer to the user's question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: list[str] = Field(
        description="Search queries for external verification"
    )

class ReviseAnswer(AnswerQuestion):
    citations: list[str] = Field(description="URLs supporting the revised answer")


# 3. Bind schemas as model tools
responder_chain = llm.bind_tools([AnswerQuestion])
revisor_chain = llm.bind_tools([ReviseAnswer])


# 4. LangGraph nodes
def respond_node(state: MessagesState) -> dict:
    response = responder_chain.invoke(state["messages"])
    return {"messages": [response]}

def execute_tools(state: MessagesState) -> dict:
    last_ai_msg = state["messages"][-1]
    tool_call = last_ai_msg.tool_calls[0]
    queries = tool_call["args"]["search_queries"]

    tool_results = []
    for query in queries:
        tool_results.append(search_tool.invoke(query))

    return {
        "messages": [
            ToolMessage(
                content=str(tool_results),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        ]
    }

def revise_node(state: MessagesState) -> dict:
    response = revisor_chain.invoke(state["messages"])
    return {"messages": [response]}


# 5. LangGraph workflow
graph = StateGraph(MessagesState)
graph.add_node("respond", respond_node)
graph.add_node("tools", execute_tools)
graph.add_node("revise", revise_node)

graph.add_edge(START, "respond")
graph.add_edge("respond", "tools")
graph.add_edge("tools", "revise")

def should_continue(state: MessagesState) -> Literal["tools", END]:
    if len(state["messages"]) > 6:
        return END
    return "tools"

graph.add_conditional_edges("revise", should_continue)
app = graph.compile()


# 6. Run Reflexion agent
result = app.invoke({
    "messages": [
        HumanMessage(content="I'm pre-diabetic and need to lower blood sugar")
    ]
})

# Final revised answer is stored in the final AIMessage tool call.
final_tool_call = result["messages"][-1].tool_calls[0]
print(final_tool_call["args"]["answer"])
print(final_tool_call["args"]["citations"])

# Execution pattern:
# Human --> respond --> tools --> revise --> tools --> revise --> END
# Each iteration improves answer using critique + external data
```

#### Exercise: Building a Reflexion Agent with External Knowledge Integration



### ReAct: Reasoning + Action



### Summary and Cheat Sheet: Build Self-Improving Agents with LangGraph



## 3. Multi-Agent Systems and Agentic RAG with LangGraph
