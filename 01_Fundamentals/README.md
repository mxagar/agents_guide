# Building AI Agents and Agentic Workflows: Fundamentals of Building AI Agents

This is a compilation of notes from the Coursera Specialization [Building AI Agents and Agentic Workflows (IBM)](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/specializations/building-ai-agents-and-agentic-workflows), which is composed of the following courses:

- [Fundamentals of Building AI Agents](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/fundamentals-of-building-ai-agents?authProvider=deutschetelekom)
- [Agentic AI with LangChain and LangGraph](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langchain-and-langgraph)
- [Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langgraph-crewai-autogen-and-beeai)

This folder contains notes of the first course: **Fundamentals of Building AI Agents**.

Table of contents:

- [Building AI Agents and Agentic Workflows: Fundamentals of Building AI Agents](#building-ai-agents-and-agentic-workflows-fundamentals-of-building-ai-agents)
  - [1. Foundations of Tool Calling and Chaining](#1-foundations-of-tool-calling-and-chaining)
    - [Welcome to the Course](#welcome-to-the-course)
    - [Introduction to AI Agents](#introduction-to-ai-agents)
      - [What Are AI Agents?](#what-are-ai-agents)
      - [Comparing AI System Designs](#comparing-ai-system-designs)
      - [When Should We Use Agents?](#when-should-we-use-agents)
        - [AI System Spectrum](#ai-system-spectrum)
        - [When to Use AI Agents: 4-Criteria Framework](#when-to-use-ai-agents-4-criteria-framework)
        - [Common Challenges of Agents](#common-challenges-of-agents)
        - [When NOT to Use Agents](#when-not-to-use-agents)
        - [Risk Management Strategies](#risk-management-strategies)
        - [Key Elements of Agent Architecture](#key-elements-of-agent-architecture)
        - [Deployment Best Practices](#deployment-best-practices)
        - [Key Takeaways](#key-takeaways)
    - [Getting Started with Tool Calling](#getting-started-with-tool-calling)
      - [Tool Calling for LLMs](#tool-calling-for-llms)
      - [Why LLMs Need Tools](#why-llms-need-tools)
      - [Tools, Agents, and Function Calling in LangChain](#tools-agents-and-function-calling-in-langchain)
    - [Building and Orchestrating Tools](#building-and-orchestrating-tools)
      - [Build Effective AI Tools for Advanced LLMs](#build-effective-ai-tools-for-advanced-llms)
      - [Build Intelligent Agents for Dynamic LLM Tool Use](#build-intelligent-agents-for-dynamic-llm-tool-use)
  - [2. LCEL and Manual Tool Calling in LangChain](#2-lcel-and-manual-tool-calling-in-langchain)
  - [3. Using Built-In Agents in LangChain](#3-using-built-in-agents-in-langchain)

## 1. Foundations of Tool Calling and Chaining

### Welcome to the Course

Course Outline:

- Module 1: Foundations of Tool Calling and Chaining
  - Lesson 0: Welcome
  - Lesson 1: Introduction to AI Agents
  - Lesson 2: Getting Started with Tool Calling 
  - Lesson 3: Building and Orchestrating Tools
  - Lesson 4: Module Summary and Evaluation  

- Module 2:  LCEL and Manual Tool Calling in LangChain  
  - Lesson 1: Introduction to Chaining and LCEL Basics
  - Lesson 2: Manual Tool Calling Basics
  - Lesson 3: Parsing and Validating Tool Calls  
  - Lesson 4: Module Summary and Evaluation  

- Module 3: Using Built-in Agents in LangChain 
  - Lesson 1: Natural Language Data Visualization 
  - Lesson 2: Conversational Database Access
  - Lesson 3: Module Summary and Evaluation
  - Lesson 4: Course Wrap-Up

Tools/Software:

- LangChain: To design and implement structured AI workflows, orchestrate large language models (LLMs) with external tools, and build intelligent agents. 

- LangChain Expression Language (LCEL): For building custom chains and flexible, production-ready AI workflows. 

- Large Language Models (LLMs): To experiment with different AI models, understand their capabilities, and integrate them into agent applications. 

- LangGraph: To serve as an extension for building advanced agents with LangChain.

- Python: For coding AI applications, defining custom tools, integrating APIs, and implementing LangChain's functionalities effectively. 

- LangChain's Built-in Agents (e.g., DataFrame Agent and SQL Agent): For natural language data analysis, visualization, and conversational database access.

### Introduction to AI Agents

#### What Are AI Agents?

![What are agents](./assets/what_are_agents.png)

* Generative AI is shifting from standalone models to compound AI systems.
* Standalone models are limited by training data and lack access to real-time or private information.
* Compound systems combine models with tools, databases, and programmatic components.
* This modular approach is more flexible and easier to adapt than retraining models.
* Retrieval augmented generation (RAG) is a common example.
* Traditional systems rely on fixed, human-defined control logic, which can fail outside predefined paths.
* AI agents move control logic to the model itself.
* This is enabled by improved reasoning in large language models.
* Agents follow a "think slow" approach: plan, act, evaluate, and iterate.
* Core components of AI agents:
  * Reasoning: The model plans and breaks down tasks.
  * Acting: The model uses external tools such as APIs, search, or code.
  * Memory: The system stores past interactions and intermediate steps.
* ReAct (Reason + Act) is a common agent pattern.
* The agent plans, uses tools, observes results, and iterates to a final answer.
* Agents handle complex, multi-step problems across multiple data sources.
* Systems exist on a spectrum from low autonomy (fixed logic) to high autonomy (agentic).
* Fixed systems are efficient for narrow tasks, while agents are better for complex tasks.
* There is a trade-off between efficiency and flexibility.
* The field is evolving toward more agentic systems, with human oversight still important.

#### Comparing AI System Designs

| AI System Type | Process | Use Case | Pros | Cons |
| --- | --- | --- | --- | --- |
| Single LLM | Input --> LLM --> Output | Summarization, classification | Simple, fast, low cost | Not adaptable, lacks context |
| Workflow | Parallel LLMs --> Aggregation --> Output | Structured multi-step tasks | Predictable, easy to audit | Rigid, not dynamic |
| Agent | Plan --> Act --> Observe --> (repeat agent loop) | Complex, adaptive automation | Flexible, learns from feedback | Unpredictable, complex, pricier |

#### When Should We Use Agents?

##### AI System Spectrum

| Type                   | Description                                              | Best Use Cases                 |
| ---------------------- | -------------------------------------------------------- | ------------------------------ |
| Simple AI Features     | Single-task models (e.g., classification, summarization) | Fast, repeatable tasks         |
| Orchestrated Workflows | Predefined multi-step pipelines                          | Structured processes           |
| Autonomous Agents      | Adaptive, decision-making systems                        | Complex reasoning and strategy |

##### When to Use AI Agents: 4-Criteria Framework

| Criterion      | Use Agents When...                  | Use Workflows When...          |
| -------------- | ----------------------------------- | ------------------------------ |
| Task Nature    | Ambiguous, exploratory, creative    | Predictable, rule-based        |
| Value vs Cost  | High-value tasks justify cost       | Low-value or high-volume tasks |
| Capabilities   | Agent passes key skill tests        | Agent fails reliability checks |
| Risk of Errors | Errors are manageable or reversible | Errors are costly or critical  |


##### Common Challenges of Agents

| Challenge               | Why It Matters                              |
| ----------------------- | ------------------------------------------- |
| Reasoning inconsistency | Unreliable performance across similar tasks |
| Unpredictable costs     | Token usage can vary widely                 |
| Tool integration issues | Requires stable APIs and tooling            |

##### When NOT to Use Agents

* High-volume, low-margin tasks
* Real-time systems (e.g., fraud detection)
* Zero-error domains (e.g., medical, security)
* Highly regulated environments

##### Risk Management Strategies

| Risk Level                  | Strategy                            |
| --------------------------- | ----------------------------------- |
| High-stakes, hard to detect | Human review + multiple validations |
| High-stakes, visible        | Automated checks + oversight        |
| Low-stakes                  | Monitoring + lightweight validation |

##### Key Elements of Agent Architecture

* Environment: Where the agent operates
* Tools: External systems/APIs it uses
* System prompts: Rules and goals guiding behavior

##### Deployment Best Practices

* Start simple and increase complexity gradually
* Begin with read-only tool access
* Add human approval for critical steps
* Use staged deployment (PoC --> Pilot --> Production)
* Enable logging and monitoring

##### Key Takeaways

* Agents are best for complex, ambiguous, high-value tasks
* Workflows are better for predictable and repeatable tasks
* Agents are powerful but costly and less reliable
* Human oversight and risk control are essential
* Start simple and scale cautiously as reliability improves

### Getting Started with Tool Calling

#### Tool Calling for LLMs

* Tool calling connects an LLM to external tools like APIs, databases, or code to access real-time data.
* A client sends user messages and tool definitions, and the LLM decides which tool to use.
* The client executes the tool and returns the result, and the LLM produces the final answer or another tool call.
* Tool definitions include name, description, and input parameters, and can represent APIs, databases, or code.
* Traditional tool calling can fail due to hallucinations or incorrect tool calls.
* Embedded tool calling uses a library between the client and LLM to manage tools and execution.
* The library sends messages with tools, executes tool calls, retries if needed, and returns final answers.
* This reduces errors and simplifies the system by centralizing tool handling. 

#### Why LLMs Need Tools

* LLMs are strong at text generation but cannot access real-time data, perform reliable calculations, or interact with external systems, so they often "guess."
* Tools extend LLM capabilities by enabling actions like math computation, API calls, data retrieval, and interaction with software.
* Without tools, LLMs rely only on training data patterns, which leads to hallucinations and errors, especially in tasks like math or logic.
* Tools improve accuracy and reliability by allowing the model to execute precise operations instead of guessing.
* Tools enable retrieval-augmented generation (RAG), letting LLMs access external data such as company documents or databases.
* They also support multimodal processing, including images, audio, and other non-text inputs.
* Tools help overcome limitations like lack of memory across sessions and restricted context window size.
* They allow LLMs to interact with APIs, databases, and digital services to perform real-world tasks.
* Examples include calculators for exact math, web tools for real-time information, code execution tools, and SQL queries.
* With tools, LLMs evolve into agents that follow a loop: understand the request, choose a tool, execute it, and return a result.
* Tools transform LLMs from passive text generators into active systems capable of solving real-world problems.

#### Tools, Agents, and Function Calling in LangChain

* Tools are functions (APIs, code, databases) that extend LLM capabilities.
* Tool calling means the LLM generates a structured request (not execution).
* An external system executes the tool and returns results for the final answer.
* Function calling and tool calling are the same concept (different naming).

Tool structure:

| Component   | Purpose         |
| ----------- | --------------- |
| Name        | Identifier      |
| Description | When/how to use |
| Parameters  | Inputs          |

Workflow:

* User query --> LLM selects tool.
* LLM outputs structured call (e.g., JSON).
* System executes tool.
* Result --> LLM --> final answer.

Tools in LangChain:

* Built-in (Wikipedia, search, math).
* Custom (`@tool`, `Tool`).
* Toolkits = grouped tools.
* Can be bound to OpenAI function calling.

Agents vs tools:

| Tools             | Agents                   |
| ----------------- | ------------------------ |
| Execute functions | Decide + orchestrate     |
| No reasoning      | Use LLM + tools + memory |

Agent architecture:

* LLM: decides actions.
* Tools: external capabilities.
* Memory: context (RAM, SQL, vector DB).
* Actions: structured tool calls.
* External world: APIs, OS, etc.

Agent flow: 

* Query --> decide tool --> call tool --> get result --> respond.

![Architecture of an AI agent in LangChain](./assets/Architecture-of-an-AI-agent-in-LangChain.png)

### Building and Orchestrating Tools

#### Build Effective AI Tools for Advanced LLMs

![Agents vs LLMs](./assets/agents_vs_llms.png)

* An agent extends an LLM by using tools to act, access data, and perform multi-step reasoning; tools are the mechanism that enables real-world interaction.
* Tool calling workflow: the LLM selects a tool, extracts parameters from the user query, calls the tool with structured inputs, and returns the result.
* A tool is a Python function with a clear purpose, well-defined inputs (string or JSON), a descriptive name, a docstring (critical for tool selection), and a consistent output format (usually a dictionary).
* Simple tools use unstructured string inputs and basic parsing; they are fragile and limited.
* Structured tools define typed inputs (e.g., lists, booleans), support multiple parameters, and integrate better with function-calling LLMs.
* Inputs must be JSON-serializable; outputs should be simple and predictable because some LLMs struggle with complex formats.
* Tools can return flexible outputs using typing (e.g., Union), but this increases parsing complexity.
* Not all LLMs support multi-argument tools reliably; testing and version control are important due to LangChain instability.

```python
# Simple tool (fragile)
def add_numbers(inputs: str) -> dict:
    """
    Adds all integer numbers found in a string.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces or text.

    Returns:
    - dict: A dictionary with key 'result' containing the sum of extracted integers.

    Example:
    >>> add_numbers("10 20 30")
    {'result': 60}
    """
    digits = [int(x) for x in inputs.split() if x.isdigit()]
    return {"result": sum(digits)}  # We return a dictionary

# LangChain Tool wrapper
from langchain.tools import Tool

add_tool = Tool(
    name="add_numbers",
    func=add_numbers,
    description="Adds numbers from a string input"  # complements docstring
)

result = add_tool.invoke("10 20 30")  # {'result': 60}

# Accessing metadata
print(add_tool.name)  # "add_numbers"
print(add_tool.description)  # "Adds numbers from a string input"
print(getattr(add_tool, "args", None))  # Depends on version


# Tool decorator: same as Tool, but cleaner syntax
from langchain.tools import tool
import re

@tool
def add_numbers(inputs: str) -> dict:
    """
    Extracts and sums all integers from a string using regex.

    Parameters:
    - inputs (str): A string possibly containing numbers.

    Returns:
    - dict: A dictionary with key 'result' containing the sum.

    Example:
    >>> add_numbers("The numbers are 10 and 20")
    {'result': 30}
    """
    numbers = [int(x) for x in re.findall(r"\d+", inputs)]
    return {"result": sum(numbers)}


# Structured tool with multiple typed inputs and detailed docstring
# Better!
from typing import List
from langchain.tools import tool

@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Sums a list of numbers, optionally using absolute values.

    Parameters:
    - numbers (List[float]): List of numbers to sum.
    - absolute (bool): If True, sums absolute values of the numbers.

    Returns:
    - float: The computed sum.

    Examples:
    >>> add_numbers_with_options([1.0, -2.0], absolute=False)
    -1.0
    >>> add_numbers_with_options([1.0, -2.0], absolute=True)
    3.0
    """
    if absolute:
        numbers = [abs(x) for x in numbers]
    return sum(numbers)

# Invocation: We pass a dictionary matching the parameter names
result = add_numbers_with_options.invoke({
    "numbers": [-1.2, -5.0],
    "absolute": True
})  # 6.2

# Accessing metadata (structured)
print(add_numbers_with_options.name)  # "add_numbers_with_options"
print(add_numbers_with_options.description)  # extracted from docstring
# args is a JSON-like schema
print(add_numbers_with_options.args)
# Full tool schema (often available)
print(getattr(add_numbers_with_options, "args_schema", None))  # Pydantic model if present


# Tool with flexible output
from typing import Dict, Union

def safe_add(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Attempts to sum integers in a string; returns an error message if none are found.

    Parameters:
    - inputs (str): Input string containing numbers.

    Returns:
    - Dict[str, Union[float, str]]:
        - 'result' (float): Sum if numbers are found.
        - 'result' (str): Error message if no numbers are found.

    Example:
    >>> safe_add("no numbers here")
    {'result': 'No numbers found'}
    """
    numbers = [int(x) for x in inputs.split() if x.isdigit()]
    if not numbers:
        return {"result": "No numbers found"}
    return {"result": sum(numbers)}

```

#### Build Intelligent Agents for Dynamic LLM Tool Use

* An agent combines an LLM with tools to reason, decide, and act; it follows a loop: receive query, reason, call tools, observe results, iterate, and produce an answer.
* Key design factors:
    * LLM choice determines tool-use and reasoning capabilities.
    * Tools must use JSON-serializable inputs/outputs; structured tools are preferred.
    * Agent strategy matters: simple agents vs ReAct (multi-step reasoning + tool use).
* ReAct pattern: think --> act (call tool) --> observe --> plan next step --> repeat or answer; zero-shot ReAct solves tasks without examples using step-by-step reasoning.
* Agent initialization in LangChain uses `initialize_agent` to bind LLM + tools + strategy.
* Agent types depend on tool format; `agent` parameter specifies the reasoning strategy and tool handling:
    1. `zero-shot-react-description` --> expects string inputs/outputs.
    2. `structured-chat-zero-shot-react-description` --> supports typed inputs (`StructuredTools`).
    3. `openai-functions` --> supports structured outputs (JSON/dicts).
* `invoke` is preferred over `run` for debugging and structured I/O; `verbose=True` exposes reasoning; `handle_parsing_errors=True` improves robustness.
* Different LLM-agent-tool combinations behave differently; structured outputs may break weaker combinations and require compatible agents.

![How Agents Work](./assets/how_agents_work.png)

![ReAct Agent](./assets/react_agent.png)

![Agent Input/Output](./assets/agent_input_output.png)

```python
# Initialize an LLM (example with IBM Granite via langchain_ibm)
from langchain_ibm import ChatWatsonx

llm = ChatWatsonx(
    model_id="ibm/granite-3-2-8b-instruct",
    url="https://xxx.ml.cloud.ibm.com",  # placeholder
    project_id="your_project_id",
    apikey="YOUR_API_KEY"
)

# Basic usage
response = llm.invoke("What is 2 + 2?")

# 1. Zero-shot ReAct agent (string tools)
# First, we define simple tool + tool wrapper (string-based, for zero-shot-react-description)
from langchain.tools import Tool

def add_numbers(inputs: str) -> str:
    """
    Adds numbers from a string input.
    Parameters:
    - inputs (str): Numbers in text form.
    Returns:
    - str: Sum as string (important for this agent type).
    """
    digits = [int(x) for x in inputs.split() if x.isdigit()]
    return str(sum(digits))  # must return string for this agent

add_tool = Tool(
    name="add_numbers",
    func=add_numbers,
    description="Adds numbers from a string"
)


# Zero-shot ReAct agent (string tools)
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=[add_tool],  # We pass a list of tools
    llm=llm,  # The LLM that powers the agent's reasoning
    agent="zero-shot-react-description",
    verbose=True,  # prints reasoning steps
    handle_parsing_errors=True  # recovers from malformed outputs
)

# Run query (internally: think -> act -> observe loop)
result = agent.invoke(
    "What is the sum of 27.72, 2.14, and 1.79?"
)

print(result)


# 2. Structured tool (typed inputs) for structured-chat agent
from langchain.tools import tool
from typing import List

@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Sums a list of numbers, optionally using absolute values.
    Parameters:
    - numbers (List[float]): List of numbers.
    - absolute (bool): Whether to use absolute values.
    Returns:
    - float: Sum result.
    """
    if absolute:
        numbers = [abs(x) for x in numbers]
    return sum(numbers)

# Structured ReAct agent (supports typed inputs)
structured_agent = initialize_agent(
    tools=[add_numbers_with_options],
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True
)

# Structured invocation (input/output wrapped in dict)
result = structured_agent.invoke({
    "input": "Sum -1.2 and -5.0 using absolute values"
})

print(result["output"])


# 3. Agent supporting structured outputs (e.g., OpenAI functions)
from langchain_openai import ChatOpenAI

# OpenAI function calling agent (supports structured outputs, i.e., dict/JSON)
llm_openai = ChatOpenAI(model="gpt-4.1-nano")

agent_functions = initialize_agent(
    tools=[add_numbers_with_options],
    llm=llm_openai,
    agent="openai-functions",  # supports JSON/dict outputs
    verbose=True
)

result = agent_functions.invoke({
    "input": "Sum -1.2 and -5.0 using absolute values"
})

print(result)

# Notes:
# - zero-shot-react-description: string I/O tools only
# - structured-chat-zero-shot-react-description: supports typed inputs
# - openai-functions: supports structured outputs (dict/JSON)
# - invoke(): preferred for structured debugging
# - verbose=True: shows internal reasoning loop (ReAct trace)
```

## 2. LCEL and Manual Tool Calling in LangChain



## 3. Using Built-In Agents in LangChain


