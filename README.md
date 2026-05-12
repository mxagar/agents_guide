# Agentic AI: My Notes

This is a compilation of notes on Agentic AI.

The main source has been the Coursera Specialization [Building AI Agents and Agentic Workflows (IBM)](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/specializations/building-ai-agents-and-agentic-workflows), which is composed of the following courses:

- [Fundamentals of Building AI Agents](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/fundamentals-of-building-ai-agents?authProvider=deutschetelekom)
- [Agentic AI with LangChain and LangGraph](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langchain-and-langgraph)
- [Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI](https://www.coursera.org/programs/deutsche-telekom-learning-program-ddjuh/learn/agentic-ai-with-langgraph-crewai-autogen-and-beeai)

<!--
Additionally, I have also taken notes from other courses/tutorials:

- [Complete N8N and AI Automation Masterclass](https://www.udemy.com/course/complete-n8n/)
- [AI Engineer Agentic Track: The Complete Agent & MCP Course](https://www.udemy.com/course/the-complete-agentic-ai-engineering-course/)
- [AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)
- [AI Engineer Production Track: Deploy LLMs & Agents at Scale](https://www.udemy.com/course/generative-and-agentic-ai-in-production/)
- [LLM Observability and Cost Management: Langfuse, Monitoring](https://www.udemy.com/course/llm-observability-cost)
- [Vector Databases for RAG: An Introduction](https://www.coursera.org/learn/vector-databases-for-rag-an-introduction)
- [Advanced RAG with Vector Databases and Retrievers](https://www.coursera.org/learn/advanced-rag-with-vector-databases-and-retrievers)
- [Build Multimodal Generative AI Applications (Coursera, IBM)](https://www.coursera.org/learn/build-multimodal-generative-ai-applications)
- [Build AI Agents using MCP](https://www.coursera.org/learn/build-ai-agents-using-mcp)

-->

Each course/module has its own subdirectory in this repository, where I have organized the notes and code examples.

## Setup

Each subdirectory may contain its own setup instructions. If you need a generic Python environment, you can use the following recipe to based on [conda](https://docs.conda.io/en/latest/) and [pip-tools](https://github.com/jazzband/pip-tools):

```bash
# Create the necessary Python environment
conda env create -f conda.yaml
conda activate agents

# Compile and install all dependencies
pip-compile requirements.in
pip-sync requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# And then:
pip-compile requirements.in
pip-sync requirements.txt
```

The environment variables are stored in the `.env` file, which is ignored by git. You can create a `.env` file with the necessary environment variables for your setup.

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Git Submodules

Some exercises include upstream projects as Git submodules. Clone this repository with submodules when you want the lab code available immediately:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned the repository, initialize or refresh the submodules with:

```bash
git submodule update --init --recursive
```

When a submodule should be moved to the commit recorded by this repository, run the same update command. When you intentionally update a submodule to a newer upstream commit, `cd` into the submodule, pull or checkout the desired commit, then commit the changed submodule pointer in the parent repository.

## Authorship

Mikel Sagardia, 2026.  
No guarantees.  
