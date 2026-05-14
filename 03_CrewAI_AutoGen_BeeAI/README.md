# Building AI Agents and Agentic Workflows: Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI

This is a compilation of notes from the Coursera Specialization [Building AI Agents and Agentic Workflows (IBM)](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/specializations/building-ai-agents-and-agentic-workflows), which is composed of the following courses:

- [Fundamentals of Building AI Agents](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/fundamentals-of-building-ai-agents?authProvider=deutschetelekom)
- [Agentic AI with LangChain and LangGraph](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langchain-and-langgraph)
- [Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langgraph-crewai-autogen-and-beeai)

This folder contains notes of the third course: **Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI**.

Table of Contents:

- [Building AI Agents and Agentic Workflows: Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI](#building-ai-agents-and-agentic-workflows-agentic-ai-with-langgraph-crewai-autogen-and-beeai)
  - [1. Agentic Frameworks and LangGraph Design Patterns for Effective AI Systems](#1-agentic-frameworks-and-langgraph-design-patterns-for-effective-ai-systems)
    - [Introduction to Agentic Frameworks](#introduction-to-agentic-frameworks)
    - [Understanding AI System Design Patterns](#understanding-ai-system-design-patterns)
      - [Essential Design Patterns for Agentic AI Systems: Sequential, Routing, and Parallelization](#essential-design-patterns-for-agentic-ai-systems-sequential-routing-and-parallelization)
      - [Orchestrator Design Pattern](#orchestrator-design-pattern)
      - [Evaluator-Optimizer Design Pattern](#evaluator-optimizer-design-pattern)
      - [Exercise: Prompt Chaining, Routing and Parallelization Patterns in LangGraph](#exercise-prompt-chaining-routing-and-parallelization-patterns-in-langgraph)
      - [Exercise: Orchestration and Evaluation Patterns in LangGraph](#exercise-orchestration-and-evaluation-patterns-in-langgraph)
    - [Summary and Cheat Sheet: Agentic Frameworks and LangGraph Design Patterns](#summary-and-cheat-sheet-agentic-frameworks-and-langgraph-design-patterns)
  - [2. CrewAI Fundamentals and Advanced Applications](#2-crewai-fundamentals-and-advanced-applications)
  - [3. Alternative Agentic Frameworks: BeeAI and AutoGen (AG2)](#3-alternative-agentic-frameworks-beeai-and-autogen-ag2)
  - [4. Extra: Pydantic AI](#4-extra-pydantic-ai)


## 1. Agentic Frameworks and LangGraph Design Patterns for Effective AI Systems

### Introduction to Agentic Frameworks

* Agentic AI systems are autonomous AI systems that make decisions and take actions to achieve goals rather than simply reacting to prompts.
  * Core traits:
    * multi-step reasoning
    * decision-making
    * tool/API usage
    * memory/context retention
    * goal-oriented behavior
* Agentic AI differs from reactive AI because it proactively decomposes problems, gathers information, invokes tools or other agents, and continues until the objective is achieved.
* Frameworks exist because building agentic systems from scratch is complex.
  * Common infrastructure challenges:
    * memory/state management
    * tool integration
    * agent communication
    * coordination logic
    * distributed error handling
    * monitoring/debugging
    * routing/orchestration logic
    * fallback/retry mechanisms
* Multi-agent systems (MAS) use multiple specialized agents instead of one general-purpose agent.
  * Benefits:
    * specialization: each agent focuses on one task
    * parallelism: agents work simultaneously
    * fault tolerance: partial failures do not break the whole system
    * scalability: agents can be added as needed
    * modularity: components can be independently replaced
    * clearer debugging: failures are easier to isolate
* Common collaboration patterns:
  * sequential pipelines:
    * agent A → agent B → agent C
    * example: research → writing → review
  * evaluator-generator loops:
    * generator creates output
    * evaluator critiques/refines
    * loop until accepted
  * conversational collaboration:
    * agents exchange messages in turns
  * router-based workflows:
    * next agent selected dynamically based on state
  * parallel execution + synthesis:
    * multiple agents work independently
    * final agent combines outputs
* Framework comparison:
  * CrewAI
    * role-based agent collaboration ("team of agents")
    * easy onboarding
    * structured outputs
    * explicit concepts:
      * agents with role, goal, backstory, tools
      * tasks with description, expected output, assigned agent
      * `Crew` object orchestrates execution
    * strong for:
      * evaluator/generator workflows
      * role delegation
      * content pipelines
      * reporting
    * drawbacks:
      * less flexible
      * debugging can be harder
  * LangGraph
    * graph-based workflows with nodes, edges, and shared state
    * agents/LLMs represented as explicit workflow nodes
    * routing via conditional router functions
    * fine-grained control over:
      * routing
      * memory/state
      * retries
      * failure handling
      * human-in-the-loop
    * supports reflection/evaluator workflows naturally
    * best for:
      * complex workflows
      * document pipelines
      * decision trees
      * customer service
      * regulated enterprise workflows
    * tradeoff:
      * more verbose / lower-level
  * AutoGen (AG2)
    * conversational agent-to-agent and human-agent collaboration
    * supports:
      * personas via system prompts
      * delegation
      * live code execution
      * group chat orchestration
    * implementation model:
      * agents interact in managed conversations
      * group chat manager controls turns
      * round-robin / scheduled interactions possible
    * best for:
      * collaborative coding
      * education
      * technical support
      * HITL workflows
      * research assistants
  * Pydantic AI
    * schema-first, type-safe structured outputs
    * reduces ambiguity and runtime errors
    * strong validation and deterministic interfaces
    * best for:
      * APIs
      * enterprise integrations
      * production reliability
  * BeeAI
    * enterprise-focused multi-agent orchestration
    * modular workflow design
    * supports:
      * multiple model providers
      * MCP tools
      * memory
      * telemetry
      * persistence
      * structured outputs
      * sequential + parallel execution
    * good for tool-enabled interoperable agent systems
    * best for scalable production agent systems
* Workflow styles by framework:
  * CrewAI:
    * role delegation / team simulation
  * LangGraph:
    * explicit state machines / workflow orchestration
  * AutoGen:
    * conversational collaboration
  * Pydantic AI:
    * structured deterministic API workflows
  * BeeAI:
    * modular enterprise orchestration
* Practical positioning:
  * easiest to start:
    * CrewAI
    * AutoGen
  * most control:
    * LangGraph
    * Pydantic AI
  * strongest enterprise focus:
    * BeeAI
  * fastest prototyping:
    * CrewAI
    * AutoGen
  * best debugging/observability:
    * LangGraph
* Example production uses:
  * CrewAI:
    * reporting
    * content workflows
    * delegated knowledge work
  * LangGraph:
    * structured automation
    * customer workflows
    * multi-step enterprise workflows
  * AutoGen:
    * collaborative technical systems
    * tutoring
    * conversational assistants
  * Pydantic AI:
    * reliable API services
    * structured backend systems
  * BeeAI:
    * enterprise automation
    * scalable multi-agent orchestration


### Understanding AI System Design Patterns

#### Essential Design Patterns for Agentic AI Systems: Sequential, Routing, and Parallelization

* Three core LLM workflow design patterns were introduced: sequential (prompt chaining), routing, and parallelization.
* These patterns can be implemented in LangGraph using the current `StateGraph` API:
  * a typed shared state, usually a `TypedDict`
  * nodes that return partial state updates as dictionaries
  * `START` and `END` sentinels for graph entry/exit
  * `add_edge(...)` for fixed execution flow
  * `add_conditional_edges(...)` for routing and dynamic fan-out
  * reducers such as `Annotated[list[T], operator.add]` when multiple parallel branches update the same state key
* Sequential / prompt chaining
  * One LLM/agent's output becomes the next agent's input.
  * This is useful when a complex task can be decomposed into specialized subtasks.
  * Example:
    * agent 1 summarizes job requirements from a job description
    * agent 2 uses that summary plus the original input to generate a tailored cover letter
  * Benefits:
    * modular decomposition
    * specialization
    * easier reasoning/debugging
  * Best for:
    * multi-step transformations
    * document generation
    * staged reasoning workflows
* Routing pattern
  * A router agent first analyzes the input and decides which downstream workflow to execute.
  * Routing is dynamic and based on runtime classification.
  * Example:
    * router decides whether user input requires summarization or translation
    * workflow routes to the corresponding specialized node
  * Implementation concepts:
    * shared state includes:
      * input
      * task type
      * final output
    * `llm.with_structured_output(...)` can constrain routing decisions to a schema
    * conditional edges route execution
  * Benefits:
    * dynamic task selection
    * modular branching
    * cleaner separation of responsibilities
  * Best for:
    * task classification
    * intent routing
    * multi-capability assistants
* Parallelization pattern
  * Multiple independent tasks run simultaneously rather than sequentially.
  * Results are later merged by an aggregator node.
  * Example:
    * translate one input into French, Spanish, and Japanese in parallel
    * aggregator combines all translations
  * Implementation concepts:
    * multiple nodes receive the same initial state
    * outputs are written to different state fields
    * aggregator merges results
    * for dynamic map-reduce style parallelism, conditional edges can return `Send(...)` objects with branch-specific state
  * Dynamic parallelization with `Send`
    * Use `Send(...)` when the number of parallel tasks is not known until runtime.
    * Instead of hard-coding one node per branch, a routing function returns a list of `Send("node_name", branch_state)` objects.
    * LangGraph schedules one execution of the target node for each `Send`.
    * Each branch can receive a smaller or different state than the main graph state.
    * This is useful for map-reduce workflows, such as processing an arbitrary list of documents, URLs, languages, search queries, or data rows.
    * When many branches write to the same state key, define a reducer such as `Annotated[list[str], operator.add]` so LangGraph knows how to merge the branch outputs.
    * In the example below, each language creates one translation job, and the reducer combines all returned translations into one list.
  * Benefits:
    * lower latency
    * higher throughput
    * efficient decomposition of independent subtasks
  * Best for:
    * concurrent analysis
    * multi-output generation
    * ensemble-style workflows
* Conceptual mapping:
  * sequential:
    * pipeline / assembly line
  * routing:
    * dispatcher / traffic controller
  * parallelization:
    * concurrent workers + aggregator
* LangGraph is particularly suitable because it explicitly models these patterns through graph execution rather than hidden orchestration logic. This also makes routing, fan-out, and fan-in visible in graph diagrams and easier to debug.

![Prompt Chaining](./assets/prompt_chaining.png)

![Routing Pattern](./assets/routing_pattern.png)

![Parallelization Pattern](./assets/parallelization_pattern.png)

| Aspect | Sequential (Prompt Chaining) | Routing | Parallelization |
| --- | --- | --- | --- |
| Core idea | One agent's output becomes the next agent's input | A router agent decides which workflow branch to execute | Independent tasks run simultaneously, then outputs are merged |
| Execution flow | Linear pipeline | Conditional branching | Fan-out → aggregation |
| Example | Job description → resume summary → cover letter | Input classified as **translate** or **summarize**, then routed accordingly | English text translated in parallel into French, Spanish, Japanese, then merged |
| State contents | Intermediate outputs from each stage (`job_description`, `resume_summary`, `cover_letter`) | Input, routing decision, final output (`user_input`, `task_type`, `output`) | Shared input, per-task outputs, merged result (`text`, `french`, `spanish`, `japanese`, `combined_output`) |
| LangGraph mechanism | `START`/node/`END` edges with `add_edge(...)` | `add_conditional_edges(...)` from a router node | Static fan-out with multiple edges, or dynamic fan-out with `Send(...)` + reducer |
| Main benefit | Simplicity, modularity, easier debugging, task specialization | Dynamic task selection and flexible branching | Speed, concurrency, improved throughput |
| Best use cases | Multi-step reasoning, staged content generation, document workflows | Intent classification, assistants with multiple capabilities, decision trees | Concurrent processing, ensemble workflows, multi-output generation |

```python
# Shared setup for the examples below.
# Requires: pip install -U langchain langchain-openai langgraph
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)


# --- 1. Sequential / prompt chaining with LangGraph
# Flow: job_description -> resume_summary -> cover_letter

from typing import TypedDict
from langgraph.graph import START, END, StateGraph

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

def generate_resume_summary(state: ChainState) -> dict:
    prompt = f"""
    You're a resume assistant. Read the following job description and summarize
    the key qualifications and experience the ideal candidate should have.

    Job Description:
    {state["job_description"]}
    """
    response = llm.invoke(prompt)
    return {"resume_summary": response.content.strip()}

def generate_cover_letter(state: ChainState) -> dict:
    prompt = f"""
    You're a professional cover letter writer.

    Use the job description and resume summary to write a tailored cover letter.

    Job Description:
    {state["job_description"]}

    Resume Summary:
    {state["resume_summary"]}
    """
    response = llm.invoke(prompt)
    return {"cover_letter": response.content.strip()}

workflow = StateGraph(ChainState)

workflow.add_node("resume_summary", generate_resume_summary)
workflow.add_node("cover_letter", generate_cover_letter)

workflow.add_edge(START, "resume_summary")
workflow.add_edge("resume_summary", "cover_letter")
workflow.add_edge("cover_letter", END)

app = workflow.compile()

result = app.invoke({
    "job_description": "We need a Python ML engineer with API and cloud experience.",
})

print(result["resume_summary"])
print(result["cover_letter"])


# --- 2. Routing with LangGraph
# Flow: router -> summarize OR translate -> END

# Routing with LangGraph + Pydantic schema
# Flow: router -> summarize OR translate -> END

from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class RouteDecision(BaseModel):
    task_type: Literal["summarize", "translate"] = Field(
        ...,
        description="Decide whether the user wants summarization or translation."
    )

# Use LangChain structured output so the model returns a validated object.
router_llm = llm.with_structured_output(RouteDecision)

def router_node(state: RouterState) -> dict:
    decision = router_llm.invoke(
        f"Classify the task as either summarize or translate: {state['user_input']}"
    )
    return {"task_type": decision.task_type}

def route_task(state: RouterState) -> Literal["summarize", "translate"]:
    return state["task_type"]

def summarize_node(state: RouterState) -> dict:
    response = llm.invoke(f"Summarize this text:\n\n{state['user_input']}")
    return {"output": response.content.strip()}

def translate_node(state: RouterState) -> dict:
    response = llm.invoke(f"Translate this text to Spanish:\n\n{state['user_input']}")
    return {"output": response.content.strip()}

workflow = StateGraph(RouterState)

workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_task,
    {
        "summarize": "summarize",
        "translate": "translate",
    },
)

workflow.add_edge("summarize", END)
workflow.add_edge("translate", END)

app = workflow.compile()


# --- 3. Parallelization with LangGraph
# Flow: START -> french/spanish/japanese in parallel -> aggregator -> END

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

def translate_french(state: ParallelState) -> dict:
    response = llm.invoke(f"Translate to French:\n\n{state['text']}")
    return {"french": response.content.strip()}

def translate_spanish(state: ParallelState) -> dict:
    response = llm.invoke(f"Translate to Spanish:\n\n{state['text']}")
    return {"spanish": response.content.strip()}

def translate_japanese(state: ParallelState) -> dict:
    response = llm.invoke(f"Translate to Japanese:\n\n{state['text']}")
    return {"japanese": response.content.strip()}

def aggregate_translations(state: ParallelState) -> dict:
    return {
        "combined_output": (
            f"French: {state['french']}\n"
            f"Spanish: {state['spanish']}\n"
            f"Japanese: {state['japanese']}"
        )
    }

workflow = StateGraph(ParallelState)

workflow.add_node("french", translate_french)
workflow.add_node("spanish", translate_spanish)
workflow.add_node("japanese", translate_japanese)
workflow.add_node("aggregate", aggregate_translations)

workflow.add_edge(START, "french")
workflow.add_edge(START, "spanish")
workflow.add_edge(START, "japanese")

workflow.add_edge("french", "aggregate")
workflow.add_edge("spanish", "aggregate")
workflow.add_edge("japanese", "aggregate")

workflow.add_edge("aggregate", END)

app = workflow.compile()

result = app.invoke({
    "text": "Hello, how are you?",
})

print(result["combined_output"])


# --- 4. Dynamic parallelization with Send
# Flow: START -> many translate jobs -> END
# Use Send when the number of parallel branches is only known at runtime.
# A conditional edge can return a list of Send("node_name", branch_state) objects.
# LangGraph schedules one execution of the target node for each Send object.
# Each branch can receive branch-specific state instead of the full graph state.
# If many branches write to the same key, use a reducer such as operator.add
# so LangGraph knows how to merge the branch outputs.
# This is map-reduce style fan-out/fan-in: map each language to one translation
# job, then reduce all returned translations into one list.

import operator
from typing import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

class DynamicParallelState(TypedDict):
    text: str
    languages: list[str]
    translations: Annotated[list[str], operator.add]

class TranslationJob(TypedDict):
    text: str
    language: str

def fan_out_translations(state: DynamicParallelState) -> list[Send]:
    return [
        Send("translate_one", {"text": state["text"], "language": language})
        for language in state["languages"]
    ]

def translate_one(state: TranslationJob) -> dict:
    response = llm.invoke(f"Translate to {state['language']}:\n\n{state['text']}")
    return {"translations": [f"{state['language']}: {response.content.strip()}"]}

workflow = StateGraph(DynamicParallelState)
workflow.add_node("translate_one", translate_one)
workflow.add_conditional_edges(START, fan_out_translations)
workflow.add_edge("translate_one", END)

app = workflow.compile()

result = app.invoke({
    "text": "Hello, how are you?",
    "languages": ["French", "Spanish", "Japanese"],
})

print("\n".join(result["translations"]))
```

#### Orchestrator Design Pattern

* The orchestrator pattern is designed for **dynamic workflows** where the number and type of tasks are not known in advance, unlike static sequential/routing/parallel workflows.
* Core idea:
  * a central **orchestrator agent** analyzes the request
  * decomposes it into subtasks (aka. sections in the example)
  * dynamically creates worker assignments
  * workers execute tasks in parallel
  * a **synthesizer** combines all outputs into one final result
* Analogy:
  * like a head chef receiving a complex dinner request
  * the head chef decides how many specialist chefs are needed
  * each chef prepares one dish in parallel
  * the final meal guide combines all dishes
* Core components:
  * **orchestrator node**
    * receives the original request
    * breaks it into structured work packages
    * outputs a list of task objects
  * **assign workers node**
    * reads the orchestrator output
    * dynamically spawns workers using `Send`
    * routes one task object per worker
  * **worker nodes**
    * specialized agents processing individual subtasks
    * operate in parallel
    * produce partial outputs
  * **synthesizer node**
    * aggregates all worker outputs
    * produces the unified final response
* State management:
  * uses two state containers:
    * **shared state**
      * global workflow context
      * original request
      * task list
      * accumulated worker outputs
      * final synthesized result
    * **worker state**
      * branch-specific context for each worker
      * individual assigned task details
      * can be smaller than the full shared state
  * nodes should return partial state updates rather than mutating the state object in place
  * shared aggregation can use reducers such as `Annotated[list[T], operator.add]` to merge worker outputs automatically
  * `Send("worker", branch_state)` lets each worker receive a custom state payload while still writing updates back to the parent graph state
* Workflow execution:
  * user submits a complex request
  * orchestrator uses an LLM with structured output to decompose it into typed subtasks
  * assigner dynamically creates parallel worker executions with `Send`
  * each worker receives its own structured task
  * each worker invokes an LLM to solve its subproblem
  * results are merged into shared output storage through the reducer
  * synthesizer merges all outputs into the final response
* Key LangGraph concepts:
  * `StateGraph` for workflow definition
  * `START` and `END` for graph entry/exit
  * `Send` from `langgraph.types` for dynamic worker spawning
  * `add_conditional_edges(...)` for dynamic fan-out
  * partial state updates returned from node functions
  * reducers for automatic aggregation of worker outputs
* Benefits:
  * handles unknown task complexity
  * dynamically scales worker count
  * parallel execution improves performance
  * modular worker specialization
  * adaptable runtime decomposition
  * suitable for large, unpredictable tasks
* Best use cases:
  * research decomposition
  * document analysis
  * report generation
  * multi-step enterprise workflows
  * planning systems
  * any task requiring dynamic decomposition

![Orchestrator Pattern](./assets/orchestrator.png)

```python
# Orchestrator pattern with LangGraph
# Flow:
# START -> orchestrator -> dynamically spawned workers -> synthesizer -> END

import operator
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

llm = init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)

# Section in the example
class Task(TypedDict):
    name: str
    description: str

class TaskSpec(BaseModel):
    name: str = Field(description="Short task name.")
    description: str = Field(description="Specific work the worker should complete.")

class TaskPlan(BaseModel):
    tasks: list[TaskSpec] = Field(
        ...,
        description="Independent tasks that can be completed in parallel."
    )

class MainState(TypedDict):
    request: str
    tasks: list[Task]
    completed_outputs: Annotated[list[str], operator.add]
    final_result: str

class WorkerState(TypedDict):
    task: Task

planner_llm = llm.with_structured_output(TaskPlan)

def orchestrator(state: MainState) -> dict:
    prompt = f"""
    Break this request into independent tasks:

    {state["request"]}
    """
    plan = planner_llm.invoke(prompt)
    return {"tasks": [task.model_dump() for task in plan.tasks]}

def assign_workers(state: MainState) -> list[Send]:
    # dynamically spawn one worker per task
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

def worker(state: WorkerState) -> dict:
    prompt = f"""
    Solve this task:

    {state["task"]["description"]}
    """
    response = llm.invoke(prompt)

    return {
        "completed_outputs": [response.content.strip()]
    }

def synthesizer(state: MainState) -> dict:
    combined = "\n\n---\n\n".join(state["completed_outputs"])

    prompt = f"""
    Combine these partial outputs into one final response:

    {combined}
    """
    response = llm.invoke(prompt)

    return {
        "final_result": response.content
    }

workflow = StateGraph(MainState)

workflow.add_node("orchestrator", orchestrator)
workflow.add_node("worker", worker)
workflow.add_node("synthesizer", synthesizer)

workflow.add_edge(START, "orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    assign_workers
)

workflow.add_edge("worker", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

result = app.invoke({
    "request": "Create a report about AI trends in healthcare and finance",
})

print(result["final_result"])
```

Comparison with previous patterns:

| Pattern         | Task (work section) count known beforehand? | Dynamic routing? | Parallel workers? |
| --------------- | ---------------------------: | ---------------: | ----------------: |
| Sequential      |                          Yes |               No |                No |
| Routing         |                          Yes |              Yes |                No |
| Parallelization |                          Yes |               No |               Yes |
| Orchestrator    |                           No |              Yes |               Yes |


#### Evaluator-Optimizer Design Pattern

* The Evaluator-Optimizer pattern iteratively improves an output until it meets target criteria.
  * generator LLM creates an initial answer
  * evaluator LLM grades/critiques it
  * if rejected, feedback is passed back to the generator
  * loop continues until accepted or max iterations are reached
* Example: multi-agent investment advisor.
  * investor profile is provided as input
  * grading LLM assigns a target risk grade
  * Kathy Wood persona generates an initial high-growth investment plan
  * Warren Buffett persona evaluates the plan conservatively and returns a risk grade + feedback
  * Ray Dalio persona revises the plan using evaluator feedback
  * workflow stops when current risk grade matches target grade or iteration limit is reached
* State variables track:
  * `investor_profile`: user profile
  * `target_grade`: desired risk level
  * `investment_plan`: generated/revised plan
  * `current_grade`: evaluator-assigned risk level
  * `feedback`: evaluator critique
  * `iterations`: refinement count
* Core nodes:
  * risk grading node: determines target risk grade from investor profile using structured output
  * generator node: creates or revises investment plan
  * evaluator node: grades plan and produces structured feedback
  * router function: decides whether to end or loop back to the generator
* Key LangGraph / LangChain concepts:
  * `StateGraph` models the loop explicitly.
  * `START` and `END` define graph entry and exit.
  * Nodes return partial state updates as dictionaries.
  * `llm.with_structured_output(...)` replaces ad hoc parsing or tool-call extraction for evaluator decisions.
  * `add_conditional_edges(...)` routes either to `END` or back to the generator.
* Key idea:
  * the pattern combines generation, evaluation, feedback, and routing into a reflection loop

![Evaluator-Optimizer Pattern](./assets/evaluator_optimizer_1.png)

![Evaluator-Optimizer Workflow](./assets/evaluator_optimizer_2.png)


```python
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

llm = init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)

RiskGrade = Literal[
    "ultra_conservative",
    "conservative",
    "moderate",
    "growth",
    "high_risk",
]

class InvestmentState(TypedDict):
    investor_profile: str
    target_grade: str
    investment_plan: str
    current_grade: str
    feedback: str
    iterations: int

class RiskProfile(BaseModel):
    grade: RiskGrade = Field(..., description="Target risk grade for the investor.")

class Evaluation(BaseModel):
    grade: RiskGrade = Field(..., description="Risk grade of the investment plan.")
    feedback: str = Field(..., description="Feedback explaining how to improve the plan.")

MAX_ITERATIONS = 3

risk_grader_llm = llm.with_structured_output(RiskProfile)
evaluator_llm = llm.with_structured_output(Evaluation)

def grade_risk_node(state: InvestmentState) -> dict:
    prompt = f"""
    Classify this investor profile into one risk grade:
    ultra_conservative, conservative, moderate, growth, high_risk.

    Investor profile:
    {state["investor_profile"]}
    """
    result = risk_grader_llm.invoke(prompt)
    return {"target_grade": result.grade}

def generate_plan_node(state: InvestmentState) -> dict:
    # First draft uses Kathy Wood persona; revisions use Ray Dalio persona + evaluator feedback.
    if state.get("feedback"):
        prompt = f"""
        You are Ray Dalio. Revise the investment plan using the feedback.

        Investor profile:
        {state["investor_profile"]}

        Target risk grade:
        {state["target_grade"]}

        Previous plan:
        {state["investment_plan"]}

        Evaluator feedback:
        {state["feedback"]}
        """
    else:
        prompt = f"""
        You are Kathy Wood. Create an innovation-driven investment plan.

        Investor profile:
        {state["investor_profile"]}

        Target risk grade:
        {state["target_grade"]}
        """
    response = llm.invoke(prompt)
    return {"investment_plan": response.content.strip()}

def evaluate_plan_node(state: InvestmentState) -> dict:
    prompt = f"""
    You are Warren Buffett. Evaluate this investment plan conservatively.
    Focus on capital preservation, fundamentals, and risk alignment.

    Investor profile:
    {state["investor_profile"]}

    Target risk grade:
    {state["target_grade"]}

    Investment plan:
    {state["investment_plan"]}
    """
    result = evaluator_llm.invoke(prompt)
    return {
        "current_grade": result.grade,
        "feedback": result.feedback,
        "iterations": state["iterations"] + 1,
    }

def route_investment(state: InvestmentState) -> Literal["accepted", "rejected"]:
    # Accept if risk matches target or max iterations reached; otherwise revise.
    if state["current_grade"] == state["target_grade"]:
        return "accepted"
    if state["iterations"] >= MAX_ITERATIONS:
        return "accepted"
    return "rejected"

workflow = StateGraph(InvestmentState)

workflow.add_node("grade_risk", grade_risk_node)
workflow.add_node("generate_plan", generate_plan_node)
workflow.add_node("evaluate_plan", evaluate_plan_node)

workflow.add_edge(START, "grade_risk")
workflow.add_edge("grade_risk", "generate_plan")
workflow.add_edge("generate_plan", "evaluate_plan")

workflow.add_conditional_edges(
    "evaluate_plan",
    route_investment,
    {
        "accepted": END,
        "rejected": "generate_plan",
    },
)

app = workflow.compile()

result = app.invoke({
    "investor_profile": "45-year-old investor, medium income, retirement in 20 years, wants growth but dislikes large losses.",
    "iterations": 0,
})

print(result["investment_plan"])
print(result["current_grade"])
print(result["feedback"])
```


#### Exercise: Prompt Chaining, Routing and Parallelization Patterns in LangGraph

Notebook: [`lab/01_design-patterns-v1.ipynb`](./lab/01_design-patterns-v1.ipynb)

* The notebook implements three core LangGraph workflow patterns with current LangChain/LangGraph APIs: sequential/prompt chaining, routing, and parallelization.
* Setup loads environment variables from `.env` with `load_dotenv()`, then uses OpenAI through `init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)`.
* Prompt chaining builds a job-application assistant where `resume_summary` runs first and `cover_letter` runs second through explicit `START -> node -> node -> END` edges.
* Routing uses `llm.with_structured_output(...)` with a Pydantic schema to classify user input as `summarize` or `translate`, then `add_conditional_edges(...)` dispatches to the correct node.
* Parallelization fans out from `START` to French, Spanish, and Japanese translation nodes, then uses `add_edge([node_a, node_b, node_c], "aggregator")` so the aggregator waits for all branches.
* The completed exercises extend routing into a multi-service assistant with branches for ride hailing, restaurant orders, groceries, and a default handler.
* The exercise solutions define the state/schema, implement all service handler nodes, assemble the graph, and run representative test cases.
* All nodes return partial state updates as dictionaries; the notebook avoids older `set_entry_point(...)`, `set_finish_point(...)`, and `bind_tools(...)` patterns.

Summary code from the notebook:

```python
# %% Cell 1
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

load_dotenv()

llm = init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)


# %% Cell 2
def show_graph(app):
    """Display a graph visualization when notebook rendering support is available."""
    try:
        from IPython.display import Image, display

        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception as exc:
        print(f"Graph visualization skipped: {exc}")


# %% Cell 3
class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str


def generate_resume_summary(state: ChainState) -> dict:
    prompt = f"""
    You're a resume assistant. Read the job description and summarize the key
    qualifications and experience the ideal candidate should have.

    Job Description:
    {state["job_description"]}
    """
    response = llm.invoke(prompt)
    return {"resume_summary": response.content.strip()}


def generate_cover_letter(state: ChainState) -> dict:
    prompt = f"""
    You're a professional cover letter writer. Use the job description and
    resume summary to write a tailored cover letter.

    Job Description:
    {state["job_description"]}

    Resume Summary:
    {state["resume_summary"]}
    """
    response = llm.invoke(prompt)
    return {"cover_letter": response.content.strip()}


# %% Cell 4
prompt_chain = StateGraph(ChainState)
prompt_chain.add_node("resume_summary", generate_resume_summary)
prompt_chain.add_node("cover_letter", generate_cover_letter)

prompt_chain.add_edge(START, "resume_summary")
prompt_chain.add_edge("resume_summary", "cover_letter")
prompt_chain.add_edge("cover_letter", END)

prompt_chain_app = prompt_chain.compile()
show_graph(prompt_chain_app)


# %% Cell 5
chain_result = prompt_chain_app.invoke({
    "job_description": (
        "We are looking for a data scientist with experience in machine learning, "
        "NLP, Python, large datasets, and production model deployment."
    )
})

print(chain_result["resume_summary"])
print("\n--- COVER LETTER ---\n")
print(chain_result["cover_letter"])


# %% Cell 6
class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str


class RouteDecision(BaseModel):
    task_type: Literal["summarize", "translate"] = Field(
        description="Route to summarize for summaries, or translate for French translation."
    )


router_llm = llm.with_structured_output(RouteDecision)


def router_node(state: RouterState) -> dict:
    decision = router_llm.invoke(
        f"Classify the user request as summarize or translate: {state['user_input']}"
    )
    return {"task_type": decision.task_type}


def route_task(state: RouterState) -> Literal["summarize", "translate"]:
    return state["task_type"]


def summarize_node(state: RouterState) -> dict:
    response = llm.invoke(f"Summarize this text:\n\n{state['user_input']}")
    return {"output": response.content.strip()}


def translate_node(state: RouterState) -> dict:
    response = llm.invoke(f"Translate this text to French:\n\n{state['user_input']}")
    return {"output": response.content.strip()}


# %% Cell 7
routing_graph = StateGraph(RouterState)
routing_graph.add_node("router", router_node)
routing_graph.add_node("summarize", summarize_node)
routing_graph.add_node("translate", translate_node)

routing_graph.add_edge(START, "router")
routing_graph.add_conditional_edges(
    "router",
    route_task,
    {
        "summarize": "summarize",
        "translate": "translate",
    },
)
routing_graph.add_edge("summarize", END)
routing_graph.add_edge("translate", END)

routing_app = routing_graph.compile()
show_graph(routing_app)


# %% Cell 8
for user_input in [
    "Can you translate this sentence: I love programming?",
    "Can you summarize this sentence: I love programming so much that it is all I want to do?",
]:
    result = routing_app.invoke({"user_input": user_input})
    print(f"Input: {user_input}")
    print(f"Route: {result['task_type']}")
    print(f"Output: {result['output']}\n")


# %% Cell 9
class TranslationState(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str


def translate_french(state: TranslationState) -> dict:
    response = llm.invoke(f"Translate to French:\n\n{state['text']}")
    return {"french": response.content.strip()}


def translate_spanish(state: TranslationState) -> dict:
    response = llm.invoke(f"Translate to Spanish:\n\n{state['text']}")
    return {"spanish": response.content.strip()}


def translate_japanese(state: TranslationState) -> dict:
    response = llm.invoke(f"Translate to Japanese:\n\n{state['text']}")
    return {"japanese": response.content.strip()}


def aggregate_translations(state: TranslationState) -> dict:
    combined = (
        f"Original Text: {state['text']}\n\n"
        f"French: {state['french']}\n\n"
        f"Spanish: {state['spanish']}\n\n"
        f"Japanese: {state['japanese']}"
    )
    return {"combined_output": combined}


# %% Cell 10
parallel_graph = StateGraph(TranslationState)
parallel_graph.add_node("translate_french", translate_french)
parallel_graph.add_node("translate_spanish", translate_spanish)
parallel_graph.add_node("translate_japanese", translate_japanese)
parallel_graph.add_node("aggregator", aggregate_translations)

parallel_graph.add_edge(START, "translate_french")
parallel_graph.add_edge(START, "translate_spanish")
parallel_graph.add_edge(START, "translate_japanese")
parallel_graph.add_edge(
    ["translate_french", "translate_spanish", "translate_japanese"],
    "aggregator",
)
parallel_graph.add_edge("aggregator", END)

parallel_app = parallel_graph.compile()
show_graph(parallel_app)


# %% Cell 11
parallel_result = parallel_app.invoke({
    "text": "Good morning! I hope you have a wonderful day."
})

print(parallel_result["combined_output"])


# %% Cell 12
class ServiceRouterState(TypedDict):
    user_input: str
    task_type: str
    output: str


ServiceRoute = Literal[
    "ride_hailing_call",
    "restaurant_order",
    "groceries",
    "default_handler",
]


class ServiceRouteDecision(BaseModel):
    task_type: ServiceRoute = Field(
        description="Classify the request into the best matching service branch."
    )


service_router_llm = llm.with_structured_output(ServiceRouteDecision)


def service_router_node(state: ServiceRouterState) -> dict:
    decision = service_router_llm.invoke(
        "Classify this request as ride_hailing_call, restaurant_order, "
        f"groceries, or default_handler:\n\n{state['user_input']}"
    )
    return {"task_type": decision.task_type}


def route_service(state: ServiceRouterState) -> ServiceRoute:
    return state["task_type"]


# %% Cell 13
def ride_hailing_node(state: ServiceRouterState) -> dict:
    prompt = f"""
    You are a ride hailing assistant. Extract pickup, destination, timing,
    ride preferences, and special requirements from this request:

    {state["user_input"]}
    """
    response = llm.invoke(prompt)
    return {"output": response.content.strip()}


def restaurant_order_node(state: ServiceRouterState) -> dict:
    prompt = f"""
    You are a restaurant ordering assistant. Extract food items, quantities,
    delivery or pickup details, and missing information from this request:

    {state["user_input"]}
    """
    response = llm.invoke(prompt)
    return {"output": response.content.strip()}


def groceries_node(state: ServiceRouterState) -> dict:
    prompt = f"""
    You are a grocery assistant. Organize this request into a clear shopping list
    and note any missing quantities or preferences:

    {state["user_input"]}
    """
    response = llm.invoke(prompt)
    return {"output": response.content.strip()}


def default_handler_node(state: ServiceRouterState) -> dict:
    response = llm.invoke(
        "Politely explain that this assistant handles rides, restaurant orders, "
        f"and groceries. User request:\n\n{state['user_input']}"
    )
    return {"output": response.content.strip()}


# %% Cell 14
service_graph = StateGraph(ServiceRouterState)
service_graph.add_node("router", service_router_node)
service_graph.add_node("ride_hailing_call", ride_hailing_node)
service_graph.add_node("restaurant_order", restaurant_order_node)
service_graph.add_node("groceries", groceries_node)
service_graph.add_node("default_handler", default_handler_node)

service_graph.add_edge(START, "router")
service_graph.add_conditional_edges(
    "router",
    route_service,
    {
        "ride_hailing_call": "ride_hailing_call",
        "restaurant_order": "restaurant_order",
        "groceries": "groceries",
        "default_handler": "default_handler",
    },
)
service_graph.add_edge("ride_hailing_call", END)
service_graph.add_edge("restaurant_order", END)
service_graph.add_edge("groceries", END)
service_graph.add_edge("default_handler", END)

service_app = service_graph.compile()
show_graph(service_app)


# %% Cell 15
test_cases = [
    "I need a ride from downtown to the airport at 3pm.",
    "I want to order 2 large pepperoni pizzas for delivery.",
    "I need milk, bread, eggs, and vegetables for the week.",
    "What's the weather like today?",
]

for user_input in test_cases:
    result = service_app.invoke({"user_input": user_input})
    print(f"Question: {user_input}")
    print(f"Route: {result['task_type']}")
    print(f"Output: {result['output']}")
    print("-" * 80)
```

#### Exercise: Orchestration and Evaluation Patterns in LangGraph

Notebook: [`lab/02_Agentic_Design_Patterns_in_LangGraph-v1.ipynb`](./lab/02_Agentic_Design_Patterns_in_LangGraph-v1.ipynb)

* The notebook implements two completed agentic design patterns with current LangChain/LangGraph APIs: orchestration-worker and evaluator-optimizer.
* Setup loads environment variables from `.env` with `load_dotenv()`, then initializes OpenAI with `init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)`.
* The orchestrator-worker exercise builds a meal planning workflow where an orchestrator produces structured dish tasks with `llm.with_structured_output(...)`.
* Dynamic worker fan-out uses `Send` from `langgraph.types`; each chef worker receives branch-specific state for one dish.
* Worker outputs are merged with `Annotated[list[str], operator.add]`, then a synthesizer combines the parallel results into `final_meal_guide`.
* The evaluator-optimizer exercise builds an investment-plan reflection loop with a target-risk classifier, generator, evaluator, and conditional router.
* Structured Pydantic outputs are used for both risk-profile classification and investment-plan evaluation.
* The loop uses `add_conditional_edges(...)` to either end at `END` or route back to the generator until the grade matches or the iteration limit is reached.
* All graph nodes return partial state updates as dictionaries; the notebook avoids older pinned install cells, direct `ChatOpenAI` setup, `set_entry_point(...)`, `set_finish_point(...)`, and tool-call parsing patterns.

Summary code from the notebook:

```python
# %% Cell 1
import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model(model="gpt-5.4", model_provider="openai", temperature=0)


# %% Cell 2
def show_graph(app):
    """Display a Mermaid graph when notebook rendering support is available."""
    try:
        from IPython.display import Image, display

        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception as exc:
        print(f"Graph visualization skipped: {exc}")


# %% Cell 3
class DishTask(BaseModel):
    name: str = Field(description="Name of the dish to prepare.")
    cuisine: str = Field(description="Cuisine or cultural origin of the dish.")
    ingredients: list[str] = Field(description="Ingredients needed for the dish.")


class MealPlan(BaseModel):
    dishes: list[DishTask] = Field(
        description="Independent dish tasks that can be assigned to chef workers."
    )


class MealState(TypedDict):
    request: str
    dishes: list[dict]
    completed_menu: Annotated[list[str], operator.add]
    final_meal_guide: str


class ChefWorkerState(TypedDict):
    dish: dict


meal_planner_llm = llm.with_structured_output(MealPlan)


# %% Cell 4
def orchestrator(state: MealState) -> dict:
    plan = meal_planner_llm.invoke(
        "Break this meal request into independent dishes. For each dish, "
        "include a cuisine and ingredient list.\n\n"
        f"Meal request: {state['request']}"
    )
    return {"dishes": [dish.model_dump() for dish in plan.dishes]}


def assign_workers(state: MealState) -> list[Send]:
    return [Send("chef_worker", {"dish": dish}) for dish in state["dishes"]]


def chef_worker(state: ChefWorkerState) -> dict:
    dish = state["dish"]
    response = llm.invoke(
        f"""
        You are a world-class chef specializing in {dish['cuisine']} cuisine.
        Create a practical cooking guide for this dish.

        Dish: {dish['name']}
        Ingredients: {', '.join(dish['ingredients'])}

        Include preparation steps, cooking guidance, and serving notes.
        """
    )
    return {"completed_menu": [response.content.strip()]}


def synthesize_menu(state: MealState) -> dict:
    guide = "\n\n---\n\n".join(state["completed_menu"])
    return {"final_meal_guide": guide}


# %% Cell 5
meal_graph = StateGraph(MealState)
meal_graph.add_node("orchestrator", orchestrator)
meal_graph.add_node("chef_worker", chef_worker)
meal_graph.add_node("synthesizer", synthesize_menu)

meal_graph.add_edge(START, "orchestrator")
meal_graph.add_conditional_edges("orchestrator", assign_workers)
meal_graph.add_edge("chef_worker", "synthesizer")
meal_graph.add_edge("synthesizer", END)

meal_app = meal_graph.compile()
show_graph(meal_app)


# %% Cell 6
meal_result = meal_app.invoke({
    "request": "Plan a dinner with spaghetti bolognese, chicken stir fry, and carrot cake."
})

print(meal_result["final_meal_guide"][:2000])


# %% Cell 7
RiskGrade = Literal[
    "ultra_conservative",
    "conservative",
    "moderate",
    "growth",
    "high_risk",
]


class InvestmentState(TypedDict):
    investor_profile: str
    target_grade: RiskGrade
    investment_plan: str
    current_grade: RiskGrade
    feedback: str
    iterations: int


class RiskProfile(BaseModel):
    grade: RiskGrade = Field(description="Target risk grade for the investor profile.")


class InvestmentEvaluation(BaseModel):
    grade: RiskGrade = Field(description="Risk grade assigned to the investment plan.")
    feedback: str = Field(description="Specific feedback for improving risk alignment.")


MAX_ITERATIONS = 3
risk_profile_llm = llm.with_structured_output(RiskProfile)
investment_evaluator_llm = llm.with_structured_output(InvestmentEvaluation)


# %% Cell 8
def determine_target_grade(state: InvestmentState) -> dict:
    result = risk_profile_llm.invoke(
        "Classify this investor profile into one risk grade: "
        "ultra_conservative, conservative, moderate, growth, or high_risk.\n\n"
        f"Investor profile: {state['investor_profile']}"
    )
    return {"target_grade": result.grade}


def generate_investment_plan(state: InvestmentState) -> dict:
    if state.get("feedback"):
        prompt = f"""
        You are a diversified, risk-aware investment strategist. Revise the plan
        using the evaluator feedback while targeting this risk grade: {state['target_grade']}.

        Investor profile:
        {state['investor_profile']}

        Previous plan:
        {state['investment_plan']}

        Evaluator feedback:
        {state['feedback']}
        """
    else:
        prompt = f"""
        You are a growth-oriented investment strategist. Create an initial plan
        for the investor below while respecting the target risk grade: {state['target_grade']}.

        Investor profile:
        {state['investor_profile']}
        """

    response = llm.invoke(prompt)
    return {"investment_plan": response.content.strip()}


def evaluate_investment_plan(state: InvestmentState) -> dict:
    result = investment_evaluator_llm.invoke(
        f"""
        Evaluate this investment plan against the investor profile and target risk grade.
        Return a risk grade and concrete feedback.

        Investor profile:
        {state['investor_profile']}

        Target risk grade:
        {state['target_grade']}

        Investment plan:
        {state['investment_plan']}
        """
    )
    return {
        "current_grade": result.grade,
        "feedback": result.feedback,
        "iterations": state.get("iterations", 0) + 1,
    }


def route_investment(state: InvestmentState) -> Literal["accepted", "revise"]:
    if state["current_grade"] == state["target_grade"]:
        return "accepted"
    if state["iterations"] >= MAX_ITERATIONS:
        return "accepted"
    return "revise"


# %% Cell 9
optimizer_graph = StateGraph(InvestmentState)
optimizer_graph.add_node("determine_target_grade", determine_target_grade)
optimizer_graph.add_node("generate_plan", generate_investment_plan)
optimizer_graph.add_node("evaluate_plan", evaluate_investment_plan)

optimizer_graph.add_edge(START, "determine_target_grade")
optimizer_graph.add_edge("determine_target_grade", "generate_plan")
optimizer_graph.add_edge("generate_plan", "evaluate_plan")
optimizer_graph.add_conditional_edges(
    "evaluate_plan",
    route_investment,
    {
        "accepted": END,
        "revise": "generate_plan",
    },
)

optimizer_app = optimizer_graph.compile()
show_graph(optimizer_app)


# %% Cell 10
investment_result = optimizer_app.invoke({
    "investor_profile": (
        "Age: 29\n"
        "Salary: $110,000\n"
        "Assets: $40,000\n"
        "Goal: Achieve financial independence by age 45\n"
        "Risk tolerance: High"
    ),
    "iterations": 0,
})

print("Target grade:", investment_result["target_grade"])
print("Final grade:", investment_result["current_grade"])
print("Iterations:", investment_result["iterations"])
print("\nFeedback:\n", investment_result["feedback"])
print("\nFinal investment plan:\n", investment_result["investment_plan"])
```

### Summary and Cheat Sheet: Agentic Frameworks and LangGraph Design Patterns

## 2. CrewAI Fundamentals and Advanced Applications



## 3. Alternative Agentic Frameworks: BeeAI and AutoGen (AG2)


## 4. Extra: Pydantic AI

