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

### Summary and Cheat Sheet: Agentic Frameworks and LangGraph Design Patterns

## 2. CrewAI Fundamentals and Advanced Applications



## 3. Alternative Agentic Frameworks: BeeAI and AutoGen (AG2)


## 4. Extra: Pydantic AI

