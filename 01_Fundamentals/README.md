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
      - [Exercises: Tool Calling](#exercises-tool-calling)
        - [Note: `create_agent` replaces `create_react_agent`](#note-create_agent-replaces-create_react_agent)
      - [Popular Built-in Tools in LangChain](#popular-built-in-tools-in-langchain)
        - [Search tools](#search-tools)
        - [Code interpretation and data analysis](#code-interpretation-and-data-analysis)
        - [Web browsing and interaction](#web-browsing-and-interaction)
        - [Productivity and collaboration](#productivity-and-collaboration)
        - [File and document processing](#file-and-document-processing)
        - [Financial and business tools](#financial-and-business-tools)
        - [AI and machine learning integration](#ai-and-machine-learning-integration)
    - [Summary and Cheat Sheet: Tool Calling and Chaining](#summary-and-cheat-sheet-tool-calling-and-chaining)
      - [1. Tool calling fundamentals](#1-tool-calling-fundamentals)
      - [2. Tool calling workflow](#2-tool-calling-workflow)
      - [3. Tool creation methods (LangChain)](#3-tool-creation-methods-langchain)
      - [4. Inspecting and using tools](#4-inspecting-and-using-tools)
      - [5. Built-in tools (by use case)](#5-built-in-tools-by-use-case)
      - [6. Agents (LangChain)](#6-agents-langchain)
      - [7. LCEL (LangChain Expression Language)](#7-lcel-langchain-expression-language)
      - [Key takeaways](#key-takeaways-1)
  - [2. LCEL and Manual Tool Calling in LangChain](#2-lcel-and-manual-tool-calling-in-langchain)
    - [Introduction Chaining and LCEL Basics (LangChain Expression Language)](#introduction-chaining-and-lcel-basics-langchain-expression-language)
      - [LangChain Expression Language (LCEL) and Chaining](#langchain-expression-language-lcel-and-chaining)
      - [Exercise: AI-Powered Data Analysis with LCEL](#exercise-ai-powered-data-analysis-with-lcel)
      - [LCEL Cheat Sheet](#lcel-cheat-sheet)
        - [What changed in v1](#what-changed-in-v1)
        - [Why LCEL is useful](#why-lcel-is-useful)
        - [Core Runnable types](#core-runnable-types)
        - [Common LCEL operations](#common-lcel-operations)
        - [Common LCEL patterns](#common-lcel-patterns)
        - [Minimal example](#minimal-example)
        - [When to use what](#when-to-use-what)
    - [Manual Tooling Calling Basics](#manual-tooling-calling-basics)
      - [When to Call Tools Manually](#when-to-call-tools-manually)
      - [Structured Outputs for Tool Calling](#structured-outputs-for-tool-calling)
    - [Parsing and Validating Tool Calls](#parsing-and-validating-tool-calls)
      - [LLM Agents with Tools](#llm-agents-with-tools)
      - [Interactive LLM Agents](#interactive-llm-agents)
      - [Exercise: Build Interactive Agents with Tools](#exercise-build-interactive-agents-with-tools)
      - [Exercise: Build a Tool-Calling Agent](#exercise-build-a-tool-calling-agent)
    - [Summary and Cheat Sheet: Manual Tool Calling in LangChain](#summary-and-cheat-sheet-manual-tool-calling-in-langchain)
      - [1. What is manual tool calling?](#1-what-is-manual-tool-calling)
      - [2. Key concepts](#2-key-concepts)
      - [3. How to manually call a tool](#3-how-to-manually-call-a-tool)
  - [3. Using Built-In Agents in LangChain](#3-using-built-in-agents-in-langchain)
    - [Natural Language Data Visualization](#natural-language-data-visualization)
      - [From Natural Language to Data Visualization](#from-natural-language-to-data-visualization)
      - [Exercise: Build Your Own Data Visualization Agent](#exercise-build-your-own-data-visualization-agent)
    - [Conversational Database Access](#conversational-database-access)
      - [Introduction to SQL Agents](#introduction-to-sql-agents)
      - [Natural Language Interfaces for Data Systems](#natural-language-interfaces-for-data-systems)
      - [Lab: Implementing LangChain's SQL Agent](#lab-implementing-langchains-sql-agent)
    - [Summary and Cheat Sheet: Built-In Agents in LangChain](#summary-and-cheat-sheet-built-in-agents-in-langchain)
      - [1. Understanding built-in agents](#1-understanding-built-in-agents)
      - [2. LangChain v1 default: `create_agent`](#2-langchain-v1-default-create_agent)
      - [3. Legacy agent types and v1 equivalents](#3-legacy-agent-types-and-v1-equivalents)
      - [4. Model compatibility](#4-model-compatibility)
      - [5. Prebuilt task-specific agents](#5-prebuilt-task-specific-agents)
      - [6. LangGraph agents](#6-langgraph-agents)
      - [Key takeaways](#key-takeaways-2)

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

# Alternative LLM:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4.1-nano")

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

#### Exercises: Tool Calling

Notebooks and contents:

- [`01_tools.ipynb`](./lab/01_tools.ipynb): automatically generated notebook which covers the basics of tool calling in LangChain.
  - Introduces the notebook as a runnable companion to the README section on building and orchestrating tools.
  - Loads `dotenv`, LangChain tool utilities, and basic Python typing/helpers.
  - Demonstrates a simple string-based addition function wrapped as a LangChain tool.
  - Shows the `@tool` decorator approach with a regex-based addition tool.
  - Builds a structured tool with typed inputs and an `absolute` flag.
  - Inspects tool metadata such as `name`, `description`, `args`, and `args_schema`.
  - Includes a `safe_add` example with flexible output for success and error cases.
  - Uses OpenAI models only for the agent section.
  - Creates agents with the current `create_agent(...)` API for string tools and structured tools.
  - Adds a structured final-output example using a Pydantic schema.
  - Ends with takeaways comparing fragile string tools, structured tools, and modern agent orchestration.
- [`02_AI-Math-Assistant-Tool-Calling.ipynb`](./lab/02_AI-Math-Assistant-Tool-Calling.ipynb): lab exercise from the course, where tool calling is covered.
  - Starts as a guided lab on building an AI math assistant with LangChain tool calling.
  - Covers setup, required libraries, and environment preparation.
  - Introduces IBM `ChatWatsonx` as the main model in the original lab, with notes for local OpenAI and IBM configuration.
  - Explains the difference between plain functions and LangChain tools.
  - Builds and tests a basic `add_numbers` function.
  - Wraps functions with both the `Tool` class and the `@tool` decorator.
  - Demonstrates structured tools with multiple inputs using `add_numbers_with_options`.
  - Explores typed and flexible tool return values, including complex dictionary outputs.
  - Introduces classic agent setup with `initialize_agent(...)`.
  - Demonstrates multiple agent styles, including `zero-shot-react-description`, `structured-chat-zero-shot-react-description`, and `openai-functions`.
  - Introduces `create_react_agent` from LangGraph as a newer alternative.
  - Builds a multi-tool math toolkit with addition, subtraction, multiplication, and division.
  - Tests the math agent, diagnoses a subtraction mismatch, and fixes the tool behavior.
  - Rebuilds the agent and runs automated test cases across multiple prompts.
  - Discusses stronger validation and error handling for the math tools.
  - Adds a built-in/community Wikipedia tool and combines factual lookup with math reasoning.
  - Ends with an exercise to build, wrap, and test a power/exponentiation tool.

In the second notebook [`02_AI-Math-Assistant-Tool-Calling.ipynb`](./lab/02_AI-Math-Assistant-Tool-Calling.ipynb)

- `create_react_agent(...)` is used to build a graph-based loop where the model can reason, pick a tool, observe the result, and keep going until it has enough information. That makes it especially useful for multi-step tasks, multi-tool orchestration, and inspecting the full message/tool-call sequence. It is also closer to the LangGraph-style direction the ecosystem is moving toward, so it is better suited for more agentic workflows than a simple one-shot tool demo.
- `WikipediaAPIWrapper` is used, which allows the agent to fetch information from Wikipedia. This means the agent is no longer limited to computation from user-provided numbers. It can fetch outside factual context first, then use your math tools on top of that result. That is the big practical advantage: the agent can combine retrieval plus computation in one flow, which is much closer to real-world agent behavior.

##### Note: `create_agent` replaces `create_react_agent`

Example:

![Calculator ReAct Agent](./assets/calculator_agent.png)

```python
from typing import List
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# 1. Define math tools
@tool
def add(numbers: List[float]) -> float:
    """Add a list of numbers."""
    return sum(numbers)

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def multiply(numbers: List[float]) -> float:
    """Multiply a list of numbers."""
    result = 1
    for n in numbers:
        result *= n
    return result

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# 2. Define external information tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool_base = WikipediaQueryRun(api_wrapper=wiki_api)

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia and return a short factual summary.

    Args:
        query: Search query for Wikipedia.

    Returns:
        A short Wikipedia summary.
    """
    return wiki_tool_base.run(query)


# 3. Create tool list
tools = [
    add,
    subtract,
    multiply,
    divide,
    search_wikipedia,
]


# 4. Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# 5. Create LangChain v1 agent
system_prompt = (
    "You are a helpful mathematical assistant. "
    "Use math tools for calculations and Wikipedia for factual lookup. "
    "When a query requires both retrieval and math, retrieve first, then calculate."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)


# 6. Invoke agent with pure math query
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Multiply 2, 3, and 4."}
    ]
})
print(response["messages"][-1].content)


# 7. Invoke agent with retrieval + math query
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the population of Canada? Multiply it by 0.75."}
    ]
})
print(response["messages"][-1].content)
```


#### Popular Built-in Tools in LangChain

##### Search tools

| Tool/Toolkit  | Function              | Purpose                                                                |
| ------------- | --------------------- | ---------------------------------------------------------------------- |
| SerpAPI       | Web search            | Performs web searches and returns answers                              |
| Google Search | Web search            | Executes Google searches and returns URLs, snippets, and titles        |
| Tavily Search | AI-optimized search   | Designed for AI agents; returns URLs, content, titles, images, answers |
| Wikipedia     | Knowledge base search | Searches Wikipedia and returns relevant summaries                      |


##### Code interpretation and data analysis

| Tool/Toolkit         | Function                 | Purpose                                                     |
| -------------------- | ------------------------ | ----------------------------------------------------------- |
| Python REPL          | Code execution           | Executes Python code for calculations, analysis, automation |
| Pandas DataFrame     | Data manipulation        | Enables interaction with tabular data                       |
| SQL Database Toolkit | Database querying        | Queries and manipulates SQL databases via natural language  |
| LLMMathChain         | Mathematical computation | Solves math problems via Python execution                   |
| JSON Toolkit         | JSON manipulation        | Handles large JSON/dictionary objects efficiently           |

##### Web browsing and interaction

| Tool/Toolkit       | Function                | Purpose                                      |
| ------------------ | ----------------------- | -------------------------------------------- |
| Requests Toolkit   | HTTP requests           | Sends HTTP requests and fetches web content  |
| PlayWright Browser | Browser automation      | Automates browser navigation and interaction |
| MultiOn Toolkit    | Web app interaction     | Enables interaction with web applications    |
| ArXiv              | Scientific paper search | Retrieves scientific papers from arXiv       |

##### Productivity and collaboration

| Tool/Toolkit      | Function            | Purpose                                |
| ----------------- | ------------------- | -------------------------------------- |
| Gmail Toolkit     | Email management    | Reads, sends, and manages Gmail emails |
| Office365 Toolkit | Office integration  | Interacts with Microsoft 365 apps      |
| Slack Toolkit     | Team communication  | Sends and reads Slack messages         |
| Github Toolkit    | Repo management     | Manages repositories, issues, PRs      |
| Google Calendar   | Calendar management | Creates and manages calendar events    |

##### File and document processing

| Tool/Toolkit     | Function              | Purpose                                      |
| ---------------- | --------------------- | -------------------------------------------- |
| File System      | Local file operations | Reads, writes, and manages local files       |
| Google Drive     | Cloud storage         | Accesses and manages files in Google Drive   |
| VectorStoreQA    | Document querying     | Queries documents stored in vector databases |
| Document Loaders | Content extraction    | Extracts content from formats like PDF, DOCX |

##### Financial and business tools

| Tool/Toolkit  | Function               | Purpose                                       |
| ------------- | ---------------------- | --------------------------------------------- |
| Yahoo Finance | Financial news         | Retrieves financial news and market data      |
| GOAT          | Financial transactions | Handles payments, purchases, investments      |
| Polygon IO    | Market data            | Provides real-time and historical market data |
| Stripe        | Payment processing     | Manages payments and subscriptions            |

##### AI and machine learning integration

| Tool/Toolkit           | Function         | Purpose                                 |
| ---------------------- | ---------------- | --------------------------------------- |
| DALL·E Image Generator | Image creation   | Generates images from text              |
| HuggingFace Hub Tools  | Model access     | Connects to ML models on HuggingFace    |
| Google Imagen          | Image generation | Uses Google Vertex AI image generation  |
| Nuclia Understanding   | Data indexing    | Indexes unstructured data for retrieval |

### Summary and Cheat Sheet: Tool Calling and Chaining

#### 1. Tool calling fundamentals

* Tool calling lets an LLM decide which tool to use and generate arguments, but the application executes the tool.
* The model does not execute code; it only proposes structured tool calls.

#### 2. Tool calling workflow

* Define tools + ask question.
* LLM selects tool and generates arguments.
* Application executes tool.
* Tool returns structured output (dict/JSON).
* Result is passed back to LLM.
* LLM produces final natural language answer.

```python
# Example tool
def get_weather(location: str) -> dict:
    return {"temperature": 14}

# Simulated flow
query = "What's the weather in Paris?"
tool_call = {"tool": "get_weather", "args": {"location": "paris"}}

result = get_weather(**tool_call["args"])  # {"temperature": 14}

final_answer = f"It's currently {result['temperature']}°C in Paris."
```

#### 3. Tool creation methods (LangChain)

* BaseTool (subclassing)

  * Maximum control, supports sync/async, custom logic.
  * More boilerplate.

* `Tool` class

  * Wraps a function with metadata.
  * Mostly single string input.
  * Legacy compatibility.

* `@tool` decorator (recommended)

  * Infers name, description, args from signature + docstring.
  * Creates StructuredTool automatically.

* StructuredTool

  * Supports multiple typed inputs and complex schemas.
  * Best for modern function-calling LLMs.

```python
# @tool (recommended)
from langchain.tools import tool
from typing import List

@tool
def add_numbers(numbers: List[float]) -> float:
    """Sum a list of numbers."""
    return sum(numbers)
```

#### 4. Inspecting and using tools

* Inspect schema (name, description, args)

```python
print(add_numbers.name)
print(add_numbers.description)
print(add_numbers.args)
```

* Direct invocation (useful for testing)

```python
add_numbers.invoke({"numbers": [1, 2, 3]})  # 6
```

* Bind tools to model

```python
llm_with_tools = llm.bind_tools([add_numbers])
```

* Model generates tool call (app executes it)

```python
response = llm_with_tools.invoke("Sum 1, 2, 3")
# response contains tool call info (not execution)
```

#### 5. Built-in tools (by use case)

* Search: SerpAPI, Wikipedia, Tavily --> web/knowledge search
* Math & Code: LLMMathChain, Python REPL, Pandas --> computation, analysis
* Web/API: Requests Toolkit, PlayWright --> HTTP, scraping
* Productivity: Gmail, Calendar, Slack, GitHub --> communication, scheduling
* Files/Docs: FileSystem, Google Drive, VectorStoreQA --> file/document access
* Finance: Stripe, Yahoo Finance, Polygon --> payments, market data
* ML: DALL·E, HuggingFace --> model/image generation


#### 6. Agents (LangChain)

* Agent = LLM + Tools + Memory + Execution loop

* Iteratively:

  * reason --> act (tool call) --> observe --> repeat --> answer

* Components:

  * LLM: reasoning
  * Tools: actions
  * Memory: context
  * Executor: loop controller

* Common agent types:

  * zero-shot-react-description
  * chat-zero-shot-react-description
  * create_openai_functions_agent
  * LangGraph agents

#### 7. LCEL (LangChain Expression Language)

* Used to build chains (pipelines) using `|`
* Based on Runnables (standard interface)
* Enables composable, readable workflows

```python
# LCEL chain example
from langchain_core.runnables import RunnableLambda

chain = (
    RunnableLambda(lambda x: x + 1)
    | RunnableLambda(lambda x: x * 2)
)

chain.invoke(3)  # (3 + 1) * 2 = 8
```

#### Key takeaways

* LLM decides tool usage; application executes it.
* Structured tools are preferred for reliability and flexibility.
* Agents implement iterative reasoning loops with tools.
* LCEL enables clean chaining of components.
* Always validate tool schemas and compatibility with chosen LLM/agent.

## 2. LCEL and Manual Tool Calling in LangChain

### Introduction Chaining and LCEL Basics (LangChain Expression Language)

#### LangChain Expression Language (LCEL) and Chaining

* [LangChain Expression Language (LCEL)](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/01-basic/07-lcel-interface) is the modern LangChain pattern for composing workflows by chaining components with the **pipe** operator (`|`), improving readability, composability, and data flow clarity over legacy `LLMChain`.
* It is built on runnables (standard interfaces for prompts, LLMs, tools, etc.), enabling consistent chaining and execution.
  * Runnables can be chained in a pipe: `chain = Runnable1 | Runnable2 | Runnable3`
* Basic workflow:
  * Define a prompt template with variables.
  * Instantiate the template.
  * Connect components with the pipe operator.
  * Invoke with input data.
* Execution patterns:
  * Sequential: output flows step-by-step (pipe `|` replaces `RunnableSequence`).
  * Parallel: multiple components run on the same input (`dict` --> `RunnableParallel`).
* Automatic type coercion, i.e., regular code functions can be used as runnables without manual wrapping, making it easy to integrate custom logic.
  * Functions become `RunnableLambda`.
  * Dictionaries become `RunnableParallel`.
  * No manual wrapping required.
* Data flow example: input --> prompt formatting --> LLM --> output parser.
* Parallel use case: same input processed into multiple outputs (e.g., summary, translation, sentiment).
* LCEL supports async execution, streaming, tracing, and reusable pipelines.
* Best suited for simple to medium workflows; use LangGraph for complex orchestration, embedding LCEL inside nodes.

![LCEL Runnables](./assets/lcel_runnables.png)

```python
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-4")

## Sequential LCEL chain
def format_prompt(inputs):
    return f"Tell me a {inputs['adjective']} joke about {inputs['content']}"

chain = (
    RunnableLambda(format_prompt)  # function auto-converted
    | llm
    | StrOutputParser()
)

chain.invoke({"adjective": "funny", "content": "AI"})

## Prompt template + pipe
prompt = ChatPromptTemplate.from_template(
    "Write a {adjective} story about {topic}"
)
chain = prompt | llm

chain.invoke({"adjective": "short", "topic": "robots"})

## Parallel execution (dict --> RunnableParallel) 
parallel_chain = {
  "summary": ChatPromptTemplate.from_template("Summarize: {text}") | llm,
  "translation": ChatPromptTemplate.from_template("Translate to French: {text}") | llm,
  "sentiment": ChatPromptTemplate.from_template("Analyze sentiment: {text}") | llm
}

chain = RunnableParallel(parallel_chain)

result = chain.invoke({"text": "LangChain is powerful"})
# returns dict with all outputs
```

#### Exercise: AI-Powered Data Analysis with LCEL

Notebook: [`03_LLM-Powered Data Science-v1.ipynb`](./lab/03_LLM-Powered%20Data%20Science-v1.ipynb)

This exercise builds a small data-analysis assistant that can inspect CSV files, summarize their structure, and choose the right evaluation path for classification vs. regression. The notebook starts with plain LangChain tools and then wires them into a LangChain v1 agent.

Key ideas:

- Use tools to expose concrete dataset operations such as file discovery, caching, summarization, dataframe inspection, and ML evaluation.
- Cache loaded dataframes in memory so the agent can work across multiple tool calls without reloading files every time.
- Let the agent decide which tool to call next based on the dataset structure and the user's question.
- Use the newer LangChain v1 interface: `create_agent(...)`, `system_prompt=...`, and `agent.invoke({"messages": [...]})`.
- Keep notebook usage and CLI usage separate: a notebook helper function for interactive cells, plus a terminal-style `while` loop example for scripts.

Core dataset tools:

```python
from langchain_core.tools import tool

@tool
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the local directory."""
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]

DATAFRAME_CACHE = {}

@tool
def preload_datasets(paths: List[str]) -> str:
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    return f"Loaded datasets: {loaded}\nAlready cached: {cached}"
```

The notebook then adds two especially useful analysis tools:

- `get_dataset_summaries(...)` returns column names and dtypes for each CSV.
- `call_dataframe_method(...)` lets the agent call simple dataframe methods such as `head()` or `describe()` on cached datasets.

It also includes lightweight ML evaluation tools so the agent can move from inspection to actual scoring:

```python
@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred)}

@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "r2_score": r2_score(y_test, y_pred),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
    }
```

The agent wiring is the part that was updated to align with the newer LangChain v1 interface:

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=False)

system_prompt = (
    "You are a data science assistant. Use the available tools to analyze CSV files. "
    "Your job is to determine whether each dataset is for classification or regression, based on its structure."
)

tools = [
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
]

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
```

Instead of older `AgentExecutor`-style code, the notebook now uses the v1 message format directly:

```python
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Can you summarize the dataset?"}
    ]
})
```

For notebook usage, the cleanest entry point is a helper like this:

```python
def ask_datawizard(user_input: str):
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })
    answer = get_final_text(result)
    print(f"my Agent: {answer}")
    return result
```

And for terminal usage, the notebook keeps a CLI-style loop example at the end:

```python
# Run in a Python script or terminal, not inside the notebook UI.
while True:
    user_input = input(" You:")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("see ya later")
        break
    ask_datawizard(user_input)
```

#### LCEL Cheat Sheet

LangChain Expression Language (LCEL) is LangChain's compositional layer for building deterministic chains from reusable `Runnable` components. In LangChain v1, LCEL is still the right tool for prompt-model-parser pipelines, retrieval pipelines, and lightweight orchestration. For higher-level agent loops, the newer v1 interface centers on `create_agent(...)`, which is built on LangGraph.

##### What changed in v1

- LCEL and `Runnable` concepts are still current and widely used.
- Agent-building examples have shifted toward `create_agent(...)` and `{"messages": [...]}` inputs.
- `AgentExecutor`-style examples are older patterns; for many common cases, v1 agents manage the tool loop for you.
- A good rule of thumb is: use LCEL for predictable dataflows, use `create_agent(...)` for tool-calling agents, and use LangGraph when you need explicit state, branching, loops, or multi-agent workflows.

##### Why LCEL is useful

- It gives you a concise way to connect components with the `|` pipe operator.
- It supports synchronous, async, batching, and streaming workflows through a shared interface.
- It makes it easy to compose prompts, models, retrievers, parsers, and custom Python functions.
- It works well for RAG pipelines and other structured transformations where the flow is mostly linear.

##### Core Runnable types

- `ChatModel`: calls an LLM or chat model.
- `PromptTemplate` or `ChatPromptTemplate`: formats structured prompts from variables.
- `OutputParser`: converts model output into plain text or structured data.
- `RunnableLambda`: wraps custom Python logic as a runnable step.
- `RunnableParallel`: runs multiple branches concurrently on the same input.
- `RunnablePassthrough`: forwards input unchanged or augments dictionary-shaped state.

##### Common LCEL operations

- `invoke()` / `ainvoke()`: run one input through a chain.
- `batch()` / `abatch()`: process many inputs efficiently.
- `stream()` / `astream()`: stream incremental output.
- `|` or `.pipe()`: compose steps into a sequence.
- `.bind()`: preset model or runnable arguments.
- `.with_retry()`: retry transient failures automatically.
- `.with_fallbacks()`: try backup runnables if the primary path fails.
- `.with_config()`: attach reusable runtime configuration.
- `astream_events()`: inspect detailed execution events.

##### Common LCEL patterns

- Simple QA: `prompt | model | StrOutputParser()`
- RAG: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()`
- Structured output: `prompt | model | parser`
- Parallel fan-out: `RunnableParallel(summary=chain_a, keywords=chain_b)`

##### Minimal example

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

response = chain.invoke({"input": "Summarize LCEL in one sentence."})
print(response)
```

##### When to use what

- Use a direct model call when you just need one prompt and one response.
- Use LCEL when you have a mostly linear pipeline of prompts, retrieval, parsing, and small transformations.
- Use `create_agent(...)` in LangChain v1 when the model needs to decide which tools to call.
- Use LangGraph when you need durable state, explicit branching, loops, interrupts, or multi-agent coordination.

### Manual Tooling Calling Basics

#### When to Call Tools Manually

* Automated agents follow: user prompt --> LLM selects tool + parameters --> agent executes --> result returned, with no manual validation.
* This automation is efficient but risky, especially in sensitive domains (e.g., finance), where incorrect actions can cause serious consequences.
* Manual tool invocation provides control by allowing developers to review tool selection, validate parameters, and verify outputs before execution.
* Key advantages of manual invocation:
  * Safety: prevents unintended or harmful actions.
  * Cost control: avoids unnecessary or excessive API/tool calls.
  * Accuracy: ensures correct tool usage and parameter selection.
* Manual control enables input/output validation, alignment with intent, and selective execution of only safe and necessary operations.
* Trade-off: automation increases speed and convenience, while manual invocation increases reliability, precision, and oversight.
* Best practice: choose between automation and manual control depending on risk level, required accuracy, and system constraints.

#### Structured Outputs for Tool Calling

* Structured outputs enforce LLM responses to follow a predefined schema (instead of free text), enabling reliable use in databases, APIs, and programmatic workflows.
* Benefits:
  * Consistent data formats
  * Easy programmatic processing
  * Guaranteed presence of required fields
  * Schema validation
* Two-step process:
  * Define schema (expected structure).
  * Generate output conforming to that schema.
* Schema definition options:
  * JSON-like (dict/list): simple, lightweight.
  * Pydantic models: preferred for type validation, field descriptions, and integration with LangChain.
* Two methods to generate structured outputs:
  * Tool calling:
    * Bind schema as a tool.
    * LLM returns arguments matching schema.
    * Extract as dict and optionally parse to Pydantic.
  * JSON mode:
    * Supported by some models.
    * Forces valid JSON output directly.
    * Returns ready-to-use dictionary.
* LangChain helper:
  * with_structured_output():
    * Binds schema as a tool.
    * Forces model to use it.
    * Parses output automatically into schema.
* Use cases:
  * Database storage
  * API integration
  * UI formatting
  * Multi-step workflows
  * Data extraction from text
* Key idea: structured outputs turn LLMs from text generators into reliable data producers with enforceable formats.

Practical examples:

1. Extract entities from free text into a validated schema

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

class SupportTicket(BaseModel):
    customer_name: str = Field(description="Customer full name")
    issue_type: str = Field(description="Short category such as billing, login, or bug")
    priority: str = Field(description="low, medium, or high")
    summary: str = Field(description="One-sentence summary of the issue")

structured_model = model.with_structured_output(SupportTicket)

ticket = structured_model.invoke(
    "Maria Gomez cannot log in after resetting her password. She needs access today."
)

print(ticket)
```

This is useful when you want to move directly from raw user language to a typed object you can store in a database or send to another service.

2. Use a schema as a tool and inspect the generated arguments

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

class CreateCalendarEvent(BaseModel):
    title: str = Field(description="Short title of the event")
    date: str = Field(description="Date in YYYY-MM-DD format")
    start_time: str = Field(description="Start time in HH:MM format")
    attendees: list[str] = Field(description="List of attendee email addresses")

llm_with_tools = model.bind_tools([CreateCalendarEvent])

ai_msg = llm_with_tools.invoke(
    "Set up a planning meeting called Q3 roadmap on 2026-05-03 at 14:00 with ana@acme.com and lee@acme.com"
)

print(ai_msg.tool_calls)
# Example shape:
# [{'name': 'CreateCalendarEvent',
#   'args': {'title': 'Q3 roadmap', 'date': '2026-05-03', 'start_time': '14:00',
#            'attendees': ['ana@acme.com', 'lee@acme.com']}}]
```

This is the manual tool-calling flavor: the model does not create the calendar event itself. It returns validated arguments, and your application decides whether and how to execute the action.

3. Produce API-ready JSON-style output for a downstream workflow

```python
from typing import Literal
from pydantic import BaseModel
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

class ReviewLabel(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    needs_followup: bool
    short_summary: str

structured_model = model.with_structured_output(ReviewLabel)

result = structured_model.invoke(
    "The delivery was late and the box was damaged, but support fixed it quickly."
)

payload = result.model_dump()
print(payload)
```

This pattern is handy when the next step is programmatic, such as:

- inserting a row into a database
- sending a JSON payload to an API
- routing a case in a workflow engine
- rendering a predictable UI card


### Parsing and Validating Tool Calls

#### LLM Agents with Tools

* Manual tool calling enables an LLM to act as an agent by selecting tools, extracting parameters, invoking functions, and incorporating results into responses.
* Workflow: user query --> LLM selects tool + extracts parameters --> tool executes --> result returned --> LLM generates final answer.
* Setup:
  * Initialize a chat model (e.g., GPT-4 mini) as the central interface (`llm.invoke`).
  * Define tools using the `@tool` decorator; docstrings guide tool selection.
  * Bind tools to the model so it can recognize and use them.
* Extend capability by adding multiple tools (e.g., add, subtract, multiply).
* Use a mapping dictionary to dynamically call tools by name, enabling flexible function execution based on LLM output.
* Inputs are passed as dictionaries matching function parameters; `invoke()` maps keys to arguments automatically.
* Binding tools wraps the LLM into a tool-aware model capable of handling queries requiring computation.
* Chat history can be maintained to improve context and response quality.

![Calculator Agent](./assets/calculator_agent.png)

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Initialize chat model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# llm.invoke(...) uses this instance

# Define tools with @tool (docstring guides LLM)
# The docstring is used  by the agent to understand when to use the tool!
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# Bind tools to LLM
tools = [add, subtract, multiply]
llm_with_tools = llm.bind_tools(tools)
# Model can now select and call these tools

# Dynamic tool invocation via mapping dictionary
tool_map = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply
}

inputs = {"a": 1, "b": 2}
result = tool_map["add"].invoke(inputs)  # 3
# invoke() maps dict keys to function parameters automatically

# Use LLM looping tool calls and results manually
# This is an ALTERNATIVE to create_agent.
# Here, we loop manually instead of letting the agent do it for us, which gives us more control and visibility into the process.
# create_agent(...) abstracts this entire loop (it does it on its own or asks LLM to choose a tool).
# So it is preferable to use create_agent, unless we want to have full control.
messages = [("human", "Add 1 and 2, then multiply 3 and 4.")]
response = llm_with_tools.invoke(messages)

# Call all tools, collect results, and tool descriptions
tool_messages = []
for tool_call in response.tool_calls:
    result = tool_map[tool_call["name"]].invoke(tool_call["args"])
    tool_messages.append(
        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    )
# Collect all messages (original + tool results) and invoke the model again to get the final answer.
final_response = llm_with_tools.invoke([*messages, response, *tool_messages])
```

#### Interactive LLM Agents

Here, the example above is extended.

* Interactive agents extend manual tool calling by managing full conversation state (chat history), extracting tool calls from LLM outputs, executing them, and feeding results back for final responses.
* Workflow:
  * Convert user query into a `HumanMessage` and store in `chat_history`.
  * Invoke LLM with tools using full chat history.
  * LLM returns an `AIMessage` containing `tool_calls` (not final text).
  * Extract tool name, arguments, and call ID from `tool_calls`.
  * Manually execute the tool using these parameters.
  * Wrap result in a `ToolMessage` and append to `chat_history`.
  * Re-invoke LLM with updated history to generate final natural language answer.
* Tool call structure includes:
  * name: tool to call.
  * args: JSON parameters.
  * id: links tool response to request (important for multiple calls).
  * type: indicates tool call.
* Chat history maintains full context: user input --> model tool request --> tool result --> final response.
* Mapping dictionaries are used to dynamically resolve tool names to functions for execution.
* This loop enables precise control while supporting multi-step reasoning and multiple tool calls.
* Encapsulating this logic in an agent class (e.g., ToolCallingAgent) automates:
  * tool binding
  * chat history management
  * tool extraction and execution
  * response generation
* Result: transforms LLM into a context-aware, multi-step agent capable of interpreting intent, selecting tools, and producing coherent final answers even from imperfect input.

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

tools = [add, subtract, multiply]
tool_map = {tool.name: tool for tool in tools}

# This tool agent class encapsulates the manual tool-calling loop,
# managing chat history and tool execution.
# Note that we can reset the history for each new query, or keep it for multi-turn conversations.
# For our case, it makes sense to reset the history for each new query.
class ToolCallingAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map
        self.chat_history = []

    def run(self, query: str, reset_history: bool = True) -> str:
        if reset_history:
            self.chat_history = []

        self.chat_history.append(HumanMessage(content=query))
        response = self.llm_with_tools.invoke(self.chat_history)
        if not response.tool_calls:
            self.chat_history.append(response)
            return response.content

        while response.tool_calls:
            self.chat_history.append(response)

            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                tool_result = self.tool_map[tool_name].invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                )

            self.chat_history.extend(tool_messages)
            response = self.llm_with_tools.invoke(self.chat_history)

        self.chat_history.append(response)
        return response.content


my_agent = ToolCallingAgent(llm)

my_agent.run("one plus 2")  # fresh conversation
my_agent.run("one - 2")  # fresh conversation

# Multi-turn usage:
my_agent.run("one plus 2", reset_history=True)
my_agent.run("now multiply that by 3", reset_history=False)
```

#### Exercise: Build Interactive Agents with Tools

Notebook: [`04_Interactive Tool-Calling Agent-v1.ipynb`](./lab/04_Interactive%20Tool-Calling%20Agent-v1.ipynb)

- This notebook is approximately the same as the `Interactive LLM Agents` section above: it defines arithmetic tools, binds them to the model, inspects `tool_calls`, executes the selected tool manually, wraps the result in a `ToolMessage`, and sends that result back to the LLM for a final answer.
- It first walks through the tool-calling flow step by step with `add`, `subtract`, and `multiply`, then wraps the same logic in a small `ToolCallingAgent` class.
- It also includes a second example with a `calculate_tip` tool and a simple `TipAgent`, reinforcing the same manual tool-calling pattern with a different task.
- The notebook version is a bit simpler and older in style than the code shown above: it mainly demonstrates a single-turn flow and a single tool-call round trip.
- The updated reference implementation is in the `Interactive LLM Agents` section above, where the code has already been modernized and extended with the `reset_history` option for fresh vs. multi-turn conversations.

#### Exercise: Build a Tool-Calling Agent

**Very important notebook**.

Notebook: [`05_Tool-Calling Agent-v1.ipynb`](./lab/05_Tool-Calling-v1.ipynb)

- This notebook builds a YouTube-focused tool-calling agent that can search videos, extract video IDs, fetch transcripts, pull metadata, retrieve thumbnails, and rank search results by recency, views, or likes.
- It starts by defining each capability as a LangChain tool with `@tool`, then adds those tools to a shared `tools` list and binds them to the chat model with `llm.bind_tools(tools)`.
- It demonstrates manual tool calling first, so you can see the underlying mechanics: the LLM proposes a tool call, the application executes it, wraps the result in a `ToolMessage`, and sends that result back to the LLM.
- It then builds a fixed-sequence summarization chain for a common workflow such as "summarize this YouTube video", where the model typically needs to extract the video ID first and fetch the transcript second.
- After that, it generalizes the approach into a recursive universal chain that keeps processing tool calls until the model stops requesting them.
- This makes the notebook more flexible than a single hardcoded demo: the same chain can summarize one video, fetch metadata for ranked search results, or combine multiple tools in sequence depending on the query.
- Compared with the earlier trending-page version, the notebook now uses `get_ranked_videos(...)` instead of scraping YouTube Trending, because the old Trending page is no longer reliable.

Most important code parts:

The notebook starts by defining reusable YouTube tools. These are the core building blocks that the LLM can call:

```python
@tool
def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"

@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=[language])
    return " ".join([snippet.text for snippet in transcript.snippets])

@tool
def get_full_metadata(url: str) -> dict:
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }
```

It also includes a more general search-and-rank tool that replaces the broken trending-page approach:

```python
@tool
def get_ranked_videos(query: str, sort_by: str = "views", max_results: int = 10) -> List[Dict]:
    valid_sort_keys = {"latest", "views", "likes"}
    if sort_by not in valid_sort_keys:
        return [{"error": f"sort_by must be one of: {sorted(valid_sort_keys)}"}]

    search_pool = max(max_results * 3, 15)
    with yt_dlp.YoutubeDL({
        'quiet': True,
        'no_warnings': True,
        'logger': yt_dpl_logger,
        'playlistend': search_pool,
    }) as ydl:
        info = ydl.extract_info(f"ytsearch{search_pool}:{query}", download=False)

    entries = info.get('entries', [])
    videos = []
    for entry in entries:
        if entry.get('_type') not in (None, 'video'):
            continue
        videos.append({
            'title': entry.get('title'),
            'video_id': entry.get('id'),
            'url': entry.get('webpage_url') or f"https://youtu.be/{entry.get('id')}",
            'channel': entry.get('channel') or entry.get('uploader', 'N/A'),
            'duration': entry.get('duration', 0) or 0,
            'view_count': entry.get('view_count', 0) or 0,
            'like_count': entry.get('like_count', 0) or 0,
            'upload_date': entry.get('upload_date', ''),
        })

    sort_key_map = {
        'latest': lambda item: item['upload_date'],
        'views': lambda item: item['view_count'],
        'likes': lambda item: item['like_count'],
    }
    videos.sort(key=sort_key_map[sort_by], reverse=True)
    return videos[:max_results]
```

All tools are then registered with the model and made available through a mapping dictionary:

```python
llm_with_tools = llm.bind_tools(tools)

tool_mapping = {
    "get_thumbnails": get_thumbnails,
    "get_ranked_videos": get_ranked_videos,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata,
}
```

The helper below is the key bridge between the model and the actual Python tools:

```python
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call["id"]
        )
```

For common video-summary tasks, the notebook builds a fixed two-step chain:

```python
summarization_chain = (
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc) for tc in x["ai_response"].tool_calls
        ]
    )
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    | RunnablePassthrough.assign(
        ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
    )
    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    | RunnablePassthrough.assign(
        summary=lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    | RunnableLambda(lambda x: x["summary"])
)
```

And finally, it generalizes that into a recursive tool-calling loop that can keep going until the model no longer asks for tools:

```python
def process_tool_calls(messages):
    last_message = messages[-1]
    tool_messages = [
        execute_tool(tc)
        for tc in getattr(last_message, 'tool_calls', [])
    ]
    updated_messages = messages + tool_messages
    next_ai_response = llm_with_tools.invoke(updated_messages)
    return updated_messages + [next_ai_response]

def should_continue(messages):
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))

def _recursive_chain(messages):
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages

recursive_chain = RunnableLambda(_recursive_chain)

universal_chain = (
    RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
    | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
    | recursive_chain
)
```

### Summary and Cheat Sheet: Manual Tool Calling in LangChain

#### 1. What is manual tool calling?

Manual tool calling in LangChain gives you precise control over how external tools are used. Instead of relying on the LLM to autonomously invoke tools, developers parse the LLM's output to extract tool calls, validate inputs, and execute functions manually.

This approach is particularly beneficial in production environments where reliability, security, and auditability are paramount.

#### 2. Key concepts

| Term | Definition |
| --- | --- |
| Tool | A Python function paired with a schema that defines its name, description, and expected arguments. Tools can be created using the `@tool` decorator or by defining a class inherited from `BaseTool`. |
| Tool Schema | A structured definition, often using Pydantic models, that spells out exactly what input a tool expects. It helps ensure the information is correct and easy to work with. |
| Tool Call | An instruction generated by the LLM indicating which tool to invoke and with what arguments. Typically represented in a structured format like JSON. |
| Automatic Tool Calling | The model autonomously decides to invoke tools based on the input and handles execution without developer intervention. |
| Manual Tool Calling | The developer or user intercepts the model's tool call suggestions, validates inputs, and executes the tools, providing greater control over the process. |
| `AIMessage` | A message type that represents the model's response, which may include tool call instructions in the `.tool_calls` attribute. |
| `ToolMessage` | A message type used to convey the result of a tool execution back to the model, maintaining context and enabling informed subsequent responses. It contains the tool output and associated `tool_call_id`. |
| `tool_call_id` | A unique identifier for each tool call, allowing the system to match the tool's output (`ToolMessage`) with the corresponding request (`AIMessage`). This is especially useful when handling multiple tool calls concurrently. |

#### 3. How to manually call a tool

Here is a step-by-step look at how you can take control and manually call a tool.

| Step | Details |
| --- | --- |
| Define your tools | First, define a simple tool using the `@tool` shortcut. |
| Bind tools to the model | Attach the tools to a chat model that supports tool calling. |
| Parse tool calls | After invoking the model, parse its output to extract tool calls. |
| Validate tool arguments | Use a Pydantic model or perform a manual check to validate inputs. |
| Understand tool execution | Decide whether to execute the tool directly and whether to manually create a `ToolMessage`. |
| Purpose of `ToolMessage` | Feed tool results back to the LLM so it can continue the conversation and produce a final answer. |
| Controlled tool calling | Add validation, filtering, business rules, and custom error handling before executing tools. |

Define your tools:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

Bind tools to the model:

```python
model_with_tools = model.bind_tools([multiply])
```

Parse tool calls:

```python
response = model_with_tools.invoke("What is 2 multiplied by 3?")
tool_calls = response.tool_calls  # Contains tool name and arguments
```

This is what `tool_calls` looks like:

```python
[{
    'name': 'multiply',
    'args': {'a': 2, 'b': 3},
    'id': 'chatcmpl-tool-94d27a8e35b44212bfe6c8d26553c149',
    'type': 'tool_call'
}]
```

Validate tool arguments:

```python
from pydantic import BaseModel

class MultiplyInput(BaseModel):
    a: int
    b: int

validated_input = MultiplyInput(**tool_calls[0]['args'])
```

This is what `validated_input` looks like:

```python
MultiplyInput(a=2, b=3)
```

Understand tool execution:

- `tool.invoke(args_dict)` returns a raw result such as `6`.
- `tool.invoke(tool_call_object)` can return a `ToolMessage` automatically.
- The "manual" part of manual tool calling is about controlling whether, which, and when tools are executed.

When you invoke a LangChain tool with a `ToolCall` object, you often get back a `ToolMessage`, so you do not always need to create one manually.

```python
result = multiply.invoke(validated_input.model_dump())
```

Purpose of `ToolMessage`:

`ToolMessage` is used to maintain context and state throughout the conversation between the user and the model. `ToolMessage` objects are essential for feeding tool results back to the LLM so it can continue the conversation. The purpose is not just getting the tool result, but feeding that result back to the LLM so it can:

- see what the tool returned
- continue the conversation with that context
- give a final answer to the user

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Complete conversation flow
messages = [
    HumanMessage("What is 2 multiplied by 3?"),
    AIMessage("I'll use the multiply tool", tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_123'}]),
    ToolMessage(content="6", tool_call_id="call_123")  # This tells LLM the result!
]

final_response = model.invoke(messages)
# LLM: "The result of 2 multiplied by 3 is 6."
```

When you run the statement below, it is not truly manual tool calling. It is more like semi-automatic tool calling because the tool is always executed:

```python
result = multiply.invoke(validated_input.model_dump())
```

Controlled tool calling:

Here is how you can implement controlled tool calling:

```python
from langchain_core.messages import ToolMessage

# Manual decision making and validation for multiply tool
for tool_call in response.tool_calls:
    if tool_call['name'] == 'multiply':
        # Check if we should execute this specific call
        a, b = tool_call['args']['a'], tool_call['args']['b']

        # Example: Only allow positive number multiplication
        if a > 0 and b > 0:
            # Validate and execute
            validated_input = MultiplyInput(**tool_call['args'])
            tool_msg = multiply.invoke(tool_call)
            messages.append(tool_msg)
        else:
            # Reject negative numbers
            error_msg = ToolMessage(
                content="Multiplication with negative numbers not allowed",
                tool_call_id=tool_call['id']
            )
            messages.append(error_msg)
    else:
        # Skip unknown tools
        skip_msg = ToolMessage(
            content=f"Tool '{tool_call['name']}' execution skipped",
            tool_call_id=tool_call['id']
        )
        messages.append(skip_msg)
```

What makes the above code manual?

1. Conditional execution: you decide whether to run the tool rather than blindly executing every tool the LLM asks for.

```python
# MANUAL DECISION: Should we execute this tool call?
if a > 0 and b > 0:
    # YES - execute the tool
    tool_msg = multiply.invoke(tool_call)
else:
    # NO - reject the tool call
    error_msg = ToolMessage(content="Multiplication with negative numbers not allowed")
```

2. Custom business logic: you can override the LLM's preference with your own rules.

```python
# CUSTOM RULE: Only allow positive number multiplication
if a > 0 and b > 0:
```

3. Tool filtering: you can skip tools you do not recognize or want to allow.

```python
if tool_call['name'] == 'multiply':
    # Handle multiply tool
else:
    # DECISION: Skip unknown tools
    skip_msg = ToolMessage(content=f"Tool '{tool_call['name']}' execution skipped")
```

4. Custom error handling: instead of letting a tool fail, you can proactively return a more meaningful message.

## 3. Using Built-In Agents in LangChain

### Natural Language Data Visualization

#### From Natural Language to Data Visualization

* The LangChain Pandas agent enables natural language interaction with a Pandas DataFrame, automatically generating Python code to analyze data and produce results or visualizations.
* It is preconfigured with tools and prompts, operates directly on a provided DataFrame, and returns answers such as counts, summaries, or plots.
* Setup:
    * Load data into a Pandas DataFrame.
    * Initialize a LangChain v1 chat model with `ChatOpenAI` from `langchain-openai`.
    * Create the agent with `create_pandas_dataframe_agent(llm, df)` from `langchain-experimental`.
* Usage:
    * Query with natural language via `agent.invoke({"input": "..."})`.
    * Agent translates query -> Python code (e.g., filtering, aggregation) -> executes -> returns result.
    * Intermediate steps expose generated code for transparency/debugging.
* Capabilities:
    * Data queries (e.g., counts, filters).
    * Statistical summaries.
    * Visualization generation (e.g., bar charts).
    * Semantic understanding (e.g., mapping "gender" -> the `sex` column).
* Key difference vs other agents:
    * Specialized for DataFrame operations by giving the agent Python execution tools over the provided DataFrame.
* Best practices:
    * Use a sandboxed environment because the agent can execute arbitrary Python code.
    * Keep `allow_dangerous_code=True` only for trusted local or isolated environments.
    * Store the OpenAI API key in `OPENAI_API_KEY`; do not hard-code credentials.
    * Write clear prompts to reduce ambiguity.
    * Validate results with human oversight.
    * Iterate prompts for better accuracy.
* Key idea: the agent turns natural language into executable data analysis code, enabling rapid exploration but requiring careful control for safety and correctness.

![Create Pandas Agent](./assets/create_pandas_agent.png)

```python
# 1. Load a dataset into a Pandas DataFrame
import pandas as pd

df = pd.read_csv("student-mat.csv")

# Inspect columns and first rows
print(df.head())
print(df.columns)

# 2. Initialize a LangChain v1 chat model with OpenAI
# Requires: pip install -U langchain langchain-openai langchain-experimental pandas matplotlib tabulate
# Requires the OPENAI_API_KEY environment variable.
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=512,
)

# 3. Create the LangChain Pandas DataFrame agent.
# The pandas agent lives in langchain-experimental because it executes Python code.
from langchain_experimental.agents import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,  # Opt in only inside a sandbox or trusted environment.
)

# 4. Ask a natural language question
response = agent.invoke({"input": "How many rows are in this file?"})
print(response["output"])
# Example: "There are 395 rows."

# 5. Inspect the generated Python code
for step in response["intermediate_steps"]:
    print(step)
# You may see code similar to:
# len(df)

# 6. Ask a filtered data question
response = agent.invoke({"input": "How many students are 18 years old?"})
print(response["output"])
# Generated logic is conceptually similar to:
count_18 = len(df[df["age"] == 18])
print(count_18)

# 7. Ask for a visualization
response = agent.invoke({"input": "Plot the gender count with bars."})
print(response["output"])
# Generated code may be conceptually similar to:
df["sex"].value_counts().plot(kind="bar")

# 8. Safer manual equivalent for production-like workflows
# Instead of letting the agent execute arbitrary code, explicitly write and review logic.
gender_counts = df["sex"].value_counts()
print(gender_counts)
ax = gender_counts.plot(kind="bar", title="Student count by gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")

# 9. Recommended safety pattern
# Use the agent only in a sandboxed environment.
# Validate outputs manually before using them for decisions.
question = "How many students are 18 years old?"
response = agent.invoke({"input": question})
print("Agent answer:", response["output"])
# Human-verifiable check
manual_check = len(df[df["age"] == 18])
print("Manual check:", manual_check)
```

#### Exercise: Build Your Own Data Visualization Agent

Notebook: [`06-Chat-with-your-dataframe-using-langchain-v1.ipynb`](./lab/06-Chat-with-your-dataframe-using-langchain-v1.ipynb)

* Goal: build a conversational data visualization agent that answers natural-language questions over a CSV file and can generate charts from those questions.
* Dataset: uses the UCI Student Alcohol Consumption mathematics dataset (`student-mat.csv`) loaded into a Pandas DataFrame.
* Environment: loads `OPENAI_API_KEY` from the repository-level `.env` file using `python-dotenv`.
* Model: uses `ChatOpenAI` through LangChain v1 instead of the original Watsonx model setup.
* Agent: creates a Pandas DataFrame agent with `create_pandas_dataframe_agent`, enabling natural-language analysis over `df`.
* Safety: opts in to `allow_dangerous_code=True` because the pandas agent executes Python code; this should be used only in a trusted local or sandboxed environment.
* Interactions: asks row-count and filtering questions, then verifies simple answers with direct Pandas code.
* Visualization tasks: prompts the agent to create bar charts, pie charts, box plots, and scatter plots for gender counts, alcohol consumption, free time, absences, and grades.
* Debugging: inspects `response["intermediate_steps"]` to see the Python code generated by the LLM.
* Exercises: asks you to generate plots for parental education vs grades, internet access vs grades, and absences vs final grades.

```python
from pathlib import Path
import os
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

# Optional: suppress noisy warnings in the notebook.
warnings.filterwarnings("ignore")

# The notebook is in 01_Fundamentals/lab, so ../.. points to the repo root.
load_dotenv(dotenv_path=Path("../..") / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY was not found. Add it to ../../.env.")

# Load the CSV into a DataFrame.
df = pd.read_csv("student-mat.csv")

# Initialize the OpenAI chat model through LangChain v1.
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=256,
)

# Create the pandas DataFrame agent.
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
)

# Ask questions about the data.
response = agent.invoke({"input": "how many rows of data are in this file?"})
print(response["output"])

# Verify simple answers with Pandas.
print(len(df))

# Inspect generated Python code.
print(response["intermediate_steps"][-1][0].tool_input.replace("; ", "\n"))

# Ask for filtered data.
response = agent.invoke({
    "input": "Give me all the data where student's age is over 18 years old."
})
print(response["output"])

# Ask for visualizations with natural language.
response = agent.invoke({"input": "Generate a bar chart to plot the gender count."})
print(response["intermediate_steps"][-1][0].tool_input.replace("; ", "\n"))

response = agent.invoke({
    "input": "Create box plots to analyze the relationship between freetime and G3."
})
print(response["output"])

response = agent.invoke({
    "input": (
        "Generate scatter plots to examine the correlation between Dalc and G3, "
        "and between Walc and G3."
    )
})
print(response["output"])
```

### Conversational Database Access

SQL guide: [`mxagar/sql_guide`](https://github.com/mxagar/sql_guide)

#### Introduction to SQL Agents

* AI-powered SQL agents translate natural language into SQL, making data accessible without requiring SQL expertise.
* Benefits:
    * Lower barrier to data access for non-technical users.
    * Faster querying and interpretation of databases.
* Capabilities:
    * Understand database schemas and select relevant tables.
    * Generate SQL queries from natural language.
    * Support multi-step queries for complex questions.
    * Handle errors by analyzing failures and retrying with corrected queries.
* Limitations:
    * Possible misinterpretation of user intent.
    * Complex queries may still require manual refinement.
    * Require continuous testing and validation for reliability.
* End-to-end workflow:
    * User query (natural language)
    * LLM generates SQL query
    * Query executed via database connector
    * Database returns raw data
    * LLM processes and formats results
    * Final answer returned in natural language
* Key idea: SQL agents act as an interface between human language and databases, automating query generation and result interpretation while requiring oversight for accuracy.

#### Natural Language Interfaces for Data Systems

* Natural Language Interfaces (NLIs) allow users to query and analyze data using everyday language, removing the need for SQL or technical expertise and democratizing access to insights.
* Evolution of interfaces:
    * Command-line --> graphical tools --> dashboards --> natural language interfaces
    * Shift from humans adapting to systems --> systems understanding human language.
* End-to-end workflow:
    * User query (natural language, often ambiguous)
    * AI interprets intent and maps to schema (entities, metrics, operations)
    * Generates structured query (SQL/API)
    * Executes query and retrieves data
    * Cleans and analyzes data (aggregations, patterns)
    * Synthesizes insights and explanations
    * Presents results (visualizations + natural language summaries)
* Types of NLIs:
    * One-shot:
        * Single query, no context
        * Simpler, faster, limited exploration
    * Conversational:
        * Multi-turn, context-aware
        * Supports refinement and exploration
        * More complex but more powerful
* Key technologies:
    * LLMs: intent understanding, language generation
    * Semantic parsing / NER: extract entities and map to schema
    * SQL generation: build correct and optimized queries
    * Dialogue management: maintain context and control interaction flow
* Design approaches:
    * Rule-based: precise, domain-aware, but brittle
    * ML/deep learning: robust to language variation, needs data
    * Hybrid: combines both for better accuracy and adaptability
* Applications:
    * Business intelligence (KPIs, trends, segmentation)
    * Data science (EDA, hypothesis testing)
    * Enterprise systems (cross-domain data access)
* Challenges:
    * Ambiguity in natural language
    * Mapping to complex schemas
    * Handling advanced queries (joins, nested logic)
    * Data security and governance constraints
* Benchmarks:
    * WikiSQL, Spider, SParC, CoSQL --> evaluate text-to-SQL systems
* Future trends:
    * Multimodal interaction (voice, visual)
    * Autonomous insight generation
    * Explainable AI for transparency and trust
* Key idea: NLIs transform data access into a conversational process, enabling broader adoption while requiring robust handling of ambiguity, complexity, and security.

#### Lab: Implementing LangChain's SQL Agent

See [`lab/07_sql_agent_project/README.md`](./lab/07_sql_agent_project/README.md).

* Project goal: build a natural-language SQL agent over the Chinook MySQL database.
* Dataset: uses the Chinook media-store schema, including artists, albums, tracks, playlists, customers, employees, invoices, and invoice lines.
* Database setup: load the already-downloaded [`chinook_mysql.sql`](./lab/07_sql_agent_project/chinook_mysql.sql) file into a local MySQL server and verify that `Album` contains 347 rows.
* Environment: use the existing `agents` conda environment and repository-level [`requirements.in`](../requirements.in), not a separate project virtual environment.
* Configuration: load `OPENAI_API_KEY`, optional OpenAI model settings, and MySQL connection variables from `.env` in the SQL project folder or repository root.
* Model: [`llm_agent.py`](./lab/07_sql_agent_project/llm_agent.py) creates a `ChatOpenAI` model and provides a quick model smoke test.
* Agent: [`sql_agent.py`](./lab/07_sql_agent_project/sql_agent.py) builds a `SQLDatabase` connection, creates a LangChain SQL agent, and accepts prompts from the command line with `--prompt`.
* Notebook: [`07-test-sql-agent-connection.ipynb`](./lab/07_sql_agent_project/07-test-sql-agent-connection.ipynb) walks through `.env` loading, OpenAI model access, MySQL connection, direct SQL checks, and a final agent invocation.
* Example questions: count albums or employees, describe `PlaylistTrack`, join `Artist` and `Album`, and find which country spent the most by invoice.
* Safety: the agent executes model-generated SQL, so use a local training database or a read-only database user and verify important answers with direct SQL.

```python
from pathlib import Path
from urllib.parse import quote_plus
import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

# Load .env from the current SQL project folder first, then the repo root.
project_dir = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parents[3]

for env_path in (Path.cwd() / ".env", project_dir / ".env", project_root / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "512")),
)

mysql_username = os.getenv("MYSQL_USERNAME", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", "")
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_port = os.getenv("MYSQL_PORT", "3306")
database_name = os.getenv("MYSQL_DATABASE", "Chinook")

mysql_uri = (
    f"mysql+mysqlconnector://{quote_plus(mysql_username)}:"
    f"{quote_plus(mysql_password)}@{mysql_host}:{mysql_port}/{database_name}"
)
db = SQLDatabase.from_uri(mysql_uri)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    handle_parsing_errors=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

result = agent_executor.invoke({
    "input": "Which country's customers spent the most by invoice?"
})
print(result["output"])
```

### Summary and Cheat Sheet: Built-In Agents in LangChain

#### 1. Understanding built-in agents

LangChain agents let an LLM decide when to use tools, execute those tools, observe results, and continue until it can answer. In LangChain v1, the main built-in agent interface is `create_agent`, which returns a LangGraph-backed agent.

There are two common ways to use built-in agent functionality:

| Approach | Use when | Examples |
| --- | --- | --- |
| General-purpose agent | You have a model plus one or more tools and want the agent to choose actions dynamically. | `create_agent(...)` |
| Task-specific utility | You want a shortcut for a common data workflow. | SQL toolkit tools, `create_sql_agent`, `create_pandas_dataframe_agent` |

#### 2. LangChain v1 default: `create_agent`

The old course material used `initialize_agent(..., agent=AgentType...)`. In LangChain v1, prefer `create_agent` from `langchain.agents`. You pass the model, tools, and optional `system_prompt`; LangChain handles the agent loop using LangGraph under the hood.

```python
# Requires: pip install -U langchain "langchain[openai]"
from langchain.agents import create_agent


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[multiply],
    system_prompt="You are a careful math assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 12 times 9?"}]}
)

print(result["messages"][-1].content)
```

Use this pattern for most new agents: define clear tools with type hints and docstrings, pass them to `create_agent`, and invoke the agent with a chat-style `messages` input.

#### 3. Legacy agent types and v1 equivalents

The course cheat sheet lists older `AgentType` values. They are useful historically, but they are no longer the first choice for new LangChain v1 code.

| Legacy agent type | Original purpose | LangChain v1 recommendation |
| --- | --- | --- |
| `ZERO_SHOT_REACT_DESCRIPTION` | ReAct-style reasoning using tool descriptions. | Use `create_agent(model, tools, ...)`. |
| `REACT_DOCSTORE` | ReAct agent with document-store lookup. | Use `create_agent` with retriever/search tools, or build a RAG workflow. |
| `SELF_ASK_WITH_SEARCH` | Break complex questions into subquestions and search. | Use `create_agent` with a search tool and an appropriate system prompt. |
| `CONVERSATIONAL_REACT_DESCRIPTION` | ReAct agent with chat history. | Use `create_agent`; provide conversation history in `messages`, or add memory/checkpointing when needed. |
| `CHAT_ZERO_SHOT_REACT_DESCRIPTION` | Chat-model zero-shot ReAct. | Use `create_agent`; v1 assumes chat-model style interactions. |
| `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` | Chat-model conversational ReAct. | Use `create_agent` with prior messages or persistent state. |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` | ReAct with structured multi-input tools. | Use `create_agent` with typed Python functions or `BaseTool` objects. |
| `OPENAI_FUNCTIONS` | OpenAI function-calling agent. | Use `create_agent` with an OpenAI model; tool calling is handled by the model integration. |
| `OPENAI_MULTI_FUNCTIONS` | OpenAI multi-function tool calling. | Use `create_agent` with multiple tools. |

Legacy example from the course, updated to v1:

```python
from langchain.agents import create_agent


def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_word_length],
    system_prompt="Use tools when they help answer exactly.",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "How many letters are in LangChain?"}]}
)

print(response["messages"][-1].content)
```

#### 4. Model compatibility

Tool-calling agents work best with chat models that support structured tool calls. If a model has weak tool-calling support, you may see parsing, validation, or missing-argument errors, especially with multi-argument tools or structured outputs.

Practical tips:

* Prefer current tool-calling chat models.
* Give every tool precise type hints and a clear docstring.
* Keep tool outputs simple and JSON-serializable.
* Test the same agent with representative prompts before trusting it in a workflow.
* Add human review, read-only credentials, or sandboxing for tools that can modify data or execute code.

#### 5. Prebuilt task-specific agents

Task-specific helpers still exist, but in v1 the general direction is to use `create_agent` plus the right toolkit/tools where possible.

| Utility | Purpose | Current note |
| --- | --- | --- |
| `create_pandas_dataframe_agent()` | Natural language DataFrame analysis and visualization. | Lives in `langchain-experimental`; executes Python, so it requires `allow_dangerous_code=True`. |
| `create_csv_agent()` | Natural language CSV querying. | Also experimental/community depending on version; consider loading CSV into Pandas and using explicit analysis code for safer workflows. |
| `create_sql_agent()` | Natural language SQL querying. | Still available, but v1 docs also show building SQL agents with `SQLDatabaseToolkit` + `create_agent`. |
| `create_openai_functions_agent()` | OpenAI function-calling tools. | Older constructor; prefer `create_agent` for new work. |
| `create_tool_calling_agent()` | Generic structured tool-calling agent. | Older/lower-level constructor; prefer `create_agent` unless you need its specific runnable composition. |

Pandas DataFrame agent example:

```python
# Requires:
# pip install -U langchain langchain-openai langchain-experimental pandas tabulate
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

df = pd.read_csv("student-mat.csv")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,  # Use only in a trusted sandbox.
)

response = agent.invoke(
    {"input": "Generate a bar chart showing the count of students by sex."}
)

print(response["output"])
print(response["intermediate_steps"])
```

SQL agent in the v1 style with `SQLDatabaseToolkit` and `create_agent`:

```python
# Requires:
# pip install -U langchain "langchain[openai]" langchain-community sqlalchemy
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
model = init_chat_model("openai:gpt-4.1-mini", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query,
execute it, inspect the result, and return a concise answer.

Unless the user specifies otherwise, limit queries to at most 5 rows.
Never make DML statements such as INSERT, UPDATE, DELETE, or DROP.
Always inspect the available tables before querying table contents.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Which country's customers spent the most by invoice?",
            }
        ]
    }
)

print(result["messages"][-1].content)
```

Classic SQL helper example:

```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    handle_parsing_errors=True,
)

result = agent_executor.invoke(
    {"input": "How many artists are in the database?"}
)

print(result["output"])
```

#### 6. LangGraph agents

LangChain v1 agents are built on LangGraph, so `create_agent` is already the recommended high-level way to get a graph-backed ReAct-style agent. Use LangGraph directly when you need lower-level control over graph nodes, conditional edges, persistence, human approval, or custom loops.

Older examples may show `create_react_agent` from LangGraph. For new LangChain v1 code, start with `create_agent`; reach for LangGraph primitives only when the high-level agent interface is too constrained.

```python
from langchain.agents import create_agent


def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for a short answer."""
    examples = {
        "refund policy": "Refunds are available within 30 days of purchase.",
        "support hours": "Support is available Monday through Friday, 9:00-17:00.",
    }
    return examples.get(query.lower(), "No matching article found.")


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a support agent. Use the knowledge-base tool before answering "
        "policy or support-hour questions."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the refund policy?"}]}
)

print(result["messages"][-1].content)
```

#### Key takeaways

* Yes: `create_agent` was missing from the old cheat sheet, and it is the main v1 agent constructor.
* Treat `initialize_agent` and `AgentType` as legacy course-era concepts.
* Use `create_agent` for general tool-using agents.
* Use task-specific helpers only when they save real setup effort, and check whether they are experimental.
* Be especially careful with Pandas and SQL agents because they execute generated code or queries.
