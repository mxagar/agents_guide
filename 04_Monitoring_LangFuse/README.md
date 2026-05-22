# Monitoring LLMs with LangFuse

There are several libraries and tools available for monitoring LLMs, for instance:

- [LangSmith](https://smith.langchain.com/)
- [LangFuse](https://langfuse.com/)

I have already a brief [LangSmith Guide](https://github.com/mxagar/generative_ai_udacity/blob/main/06_RAGs_DeepDive/01_RAG_from_Scratch/README.md#extra-langsmith).

This module is focused on LangFuse, which is a self-hosted open-source alternative to LangSmith. It provides similar features for monitoring and analyzing LLM interactions, but with the flexibility of hosting it on your own infrastructure.

The main resources for this module are:

- [LLM Observability and Cost Management: Langfuse, Monitoring (Udemy Course, Paulo Dichone)](https://www.udemy.com/course/llm-observability-cost). The repository related to this course is available in [`lab/udemy-langfuse/`](./lab/udemy-langfuse/), cloned as a submodule from the original [pdichone/llm-observability-course](https://github.com/pdichone/llm-observability-course).
- [LLM Observability with Self-Hosted Langfuse and vLLM (PyImageSearch, Vikram Singh)](https://pyimagesearch.com/2026/05/18/llm-observability-with-self-hosted-langfuse-and-vllm/). The code related to this tutorial is available in [`lab/pyimagesearch/`](./lab/pyimagesearch/).

Table of Contents:

- [Monitoring LLMs with LangFuse](#monitoring-llms-with-langfuse)
  - [1. Introduction to LangFuse](#1-introduction-to-langfuse)
    - [Local Setup and Quick Start](#local-setup-and-quick-start)
      - [Where Is the Data Stored?](#where-is-the-data-stored)
      - [CLI 101](#cli-101)
    - [Why LLM Observability?](#why-llm-observability)
    - [Traditional Monitoring vs LLM Observability](#traditional-monitoring-vs-llm-observability)
    - [The Three Pillars of LLM Observability](#the-three-pillars-of-llm-observability)
    - [ROI Calculation](#roi-calculation)
  - [2. Understanding LLM Costs](#2-understanding-llm-costs)
    - [Input and Output Tokens](#input-and-output-tokens)
    - [Where Costs Hide: RAG and Agentic Pipelines](#where-costs-hide-rag-and-agentic-pipelines)
    - [The Hidden Cost Multipliers](#the-hidden-cost-multipliers)
  - [3. LangFuse as Observability Platform](#3-langfuse-as-observability-platform)
    - [LLM Monitoring Platforms](#llm-monitoring-platforms)
    - [Setting Up LangFuse on the Cloud](#setting-up-langfuse-on-the-cloud)
    - [Creating First Trace with `@observe()`](#creating-first-trace-with-observe)
      - [LangFuse Data Model](#langfuse-data-model)
    - [First LLM Trace with LangFuse with OpenAI Wrapper](#first-llm-trace-with-langfuse-with-openai-wrapper)
    - [LangFuse API Levels: Decorator, Context Manager, Low-Level, Drop-in](#langfuse-api-levels-decorator-context-manager-low-level-drop-in)
      - [Decorator API Example](#decorator-api-example)
      - [Context Manager API Example](#context-manager-api-example)
      - [Low-Level API Example](#low-level-api-example)
  - [4. Instrumenting LLM Applications with LangFuse](#4-instrumenting-llm-applications-with-langfuse)
    - [LLM App for Production](#llm-app-for-production)
    - [RAG Pipeline](#rag-pipeline)
    - [LangChain Integration](#langchain-integration)
  - [5. Cost Optimization Strategies](#5-cost-optimization-strategies)
    - [Overview](#overview)
    - [Prompt Optimization](#prompt-optimization)
    - [Semantic Caching](#semantic-caching)
    - [Smart Model Routing](#smart-model-routing)
  - [6. Monitoring, Alerting, and Debugging](#6-monitoring-alerting-and-debugging)
  - [7. Production Patterns and Security](#7-production-patterns-and-security)

## 1. Introduction to LangFuse

### Local Setup and Quick Start

The repository related to this course is available in [`lab/udemy-langfuse/`](./lab/udemy-langfuse/), cloned as a submodule from the original [pdichone/llm-observability-course](https://github.com/pdichone/llm-observability-course).

To locally install and use LangFuse with docker-compose, follow this link: [LangFuse Docker Compose Deployment](https://langfuse.com/self-hosting/deployment/docker-compose).

```bash
# Install the necessary dependencies
# See ../README.md

# Clone the LangFuse repository
git clone https://github.com/langfuse/langfuse.git

# Navigate to the LangFuse directory and start the services
cd langfuse
docker compose up
# Access the LangFuse dashboard at http://localhost:3000
```

Sometimes, some of the ports are taken, and we need to change them. For instance, in case the Postgres port is taken, we can change them in the `docker-compose.yml` file:

```yaml
services:
  postgres:
    ports:
      - "5433:5432"  # Change the host port from 5432 to 5433
```

To start using LangFuse in your Python code...

After the Docker Compose stack is running, open the Langfuse UI at
[`http://localhost:3000`](http://localhost:3000), create an account, create a
project, and create API keys from the project settings. A local Python client
usually needs these environment variables:

```bash
# Each project has its own keys!
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

We can also add the Cursor/Claude/Copilot skill with this prompt:

```text
Install the Langfuse AI skill from github.com/langfuse/skills and use it to add tracing to this application with Langfuse following best practices.
```

Some older examples use `LANGFUSE_HOST` instead of `LANGFUSE_BASE_URL`. For
course notebooks and mixed SDK versions, setting both to `http://localhost:3000`
is harmless and can avoid confusion.

A **project** is the main workspace boundary inside Langfuse. API keys belong to
one project, and traces sent with those keys appear in that project's tracing,
metrics, prompts, datasets, and evaluation views. For this course, the two
LLM examples and the smoke test below should use the same project so the simple
Python trace, the LangChain/OpenAI trace, and the direct OpenAI trace are easy to
compare side by side. In real systems, use
separate projects when you want isolation, for example `dev` vs. `prod`, separate
applications, separate teams, or separate customers.

The Langfuse Python SDK reads these values automatically, so they can be stored
in a local `.env` file and loaded with `python-dotenv`. The minimal workflow is:

1. Start the Langfuse services with Docker Compose.
2. Create a project and copy the public/secret API keys.
3. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.
4. Add tracing to Python functions with `@observe()`.
5. Call `get_client().flush()` in notebooks and short-lived scripts so buffered
   traces are sent before the process exits.

A minimal notebook is available at
[`lab/01_langfuse_test.ipynb`](./lab/01_langfuse_test.ipynb). It traces a small
Python function without calling an LLM, which makes it useful as a first
connection test:

```python
from langfuse import observe, get_client
 
 
@observe(name="local-langfuse-smoke-test")
def greet(name: str) -> str:
    message = f"Hello, {name}!"
    get_client().update_current_span(
        input={"name": name},
        output={"message": message},
        metadata={
            "module": "04_Monitoring_LangFuse",
            "example": "01_langfuse_test",
            "user_id": "course-student",
            "session_id": "notebook-01",
            "tags": ["local", "notebook", "smoke-test"],
        },
    )
    get_client().set_current_trace_io(input={"name": name}, output={"message": message})
    return message


message = greet("Langfuse")
get_client().flush()
 
print(message)
```

![LangFuse Quickstart Example 1](./assets/langfuse_quickstart_example_1.png)

The same notebook also contains a minimal LangChain/OpenAI example. It uses the
same Langfuse project keys plus `OPENAI_API_KEY`, then attaches the Langfuse
callback handler to a small `ChatOpenAI` chain:

```bash
OPENAI_API_KEY=sk-...
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

prompt = ChatPromptTemplate.from_template(
    "Answer in one short paragraph: {question}"
)
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model

response = chain.invoke(
    {"question": "Why is observability useful for LLM applications?"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {"example": "langchain-openai-minimal"},
        "tags": ["local", "notebook", "langchain"],
    },
)

get_client().flush()
print(response.content)
```

![LangFuse Quickstart Example 2](./assets/langfuse_quickstart_example_2.png)

The notebook also shows the same idea without LangChain, using the OpenAI SDK
directly through Langfuse's OpenAI wrapper. This is useful when the application
does not need LangChain chains, callbacks, tools, or prompt templates:

```python
from langfuse import get_client
from langfuse.openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    name="openai-sdk-minimal",
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You explain technical ideas clearly and briefly.",
        },
        {
            "role": "user",
            "content": "In one paragraph, explain why LLM tracing is useful.",
        },
    ],
    metadata={"example": "openai-sdk-minimal"},
)

get_client().flush()
print(completion.choices[0].message.content)
```

![LangFuse Quickstart Example 3](./assets/langfuse_quickstart_example_3.png)

The difference is mostly where instrumentation is attached:

- With `@observe()`, Langfuse traces regular Python function calls and your own
  metadata.
- With LangChain, `CallbackHandler()` observes a LangChain run tree: prompt,
  model call, chain steps, tags, and metadata.
- With the direct OpenAI wrapper, Langfuse observes the OpenAI API request
  without adding LangChain as an abstraction layer.

Then open the Langfuse dashboard and check the project's **Tracing** view. You
should see the `local-langfuse-smoke-test` trace, the LangChain/OpenAI trace,
and the direct OpenAI SDK trace in the same project.

#### Where Is the Data Stored?

If you run:

```bash
cd langfuse
docker compose up
```

the Langfuse data is **not stored in this course repository** and it is **not
stored in the cloned `langfuse/` source folder**. By default, the data lives in
Docker-managed storage attached to the containers. In practice, this means it is
inside Docker's own volume area, which on Docker Desktop is usually inside
Docker's Linux VM rather than in a normal folder you browse from the project.

The default Langfuse Compose file uses named Docker volumes such as:

```text
langfuse_postgres_data
langfuse_clickhouse_data
langfuse_clickhouse_logs
langfuse_minio_data
langfuse_redis_data
```

You can inspect them with:

```bash
docker volume ls
docker volume inspect langfuse_postgres_data
```

If you keep the default Docker Compose setup and want to shut it down while
keeping the data, stop the containers without deleting volumes:

```bash
# From the cloned langfuse/ directory
docker compose down
```

or, if you only want to stop them without removing the containers:

```bash
docker compose stop
```

Both commands keep the named Docker volumes. Later, restart Langfuse from the
same `langfuse/` directory with:

```bash
docker compose up
```

Docker Compose will reattach the existing named volumes, so the Langfuse account,
projects, API keys, traces, and stored data should still be there.

The Langfuse application is not just one container. In the current self-hosted
Docker Compose setup, the background services include:

- **Langfuse web/API**: the dashboard and ingestion API at port `3000`.
- **Langfuse worker**: background jobs and async processing.
- **Postgres**: transactional state such as users, organizations, projects,
  datasets, settings, and encrypted API keys.
- **ClickHouse**: analytical event storage for traces, observations, scores, and
  high-volume observability data.
- **Redis/Valkey**: cache and queue-related infrastructure, including API key
  caching.
- **Blob storage**: S3-compatible storage, often MinIO in local Compose setups,
  for media and larger payloads.

If you want the data to live in an explicit folder, create a
`docker-compose.override.yml` file next to Langfuse's `docker-compose.yml` and
replace the named volumes with bind mounts:

```yaml
services:
  postgres:
    volumes:
      - ./data/postgres:/var/lib/postgresql/data

  clickhouse:
    volumes:
      - ./data/clickhouse:/var/lib/clickhouse
      - ./data/clickhouse-logs:/var/log/clickhouse-server

  minio:
    volumes:
      - ./data/minio:/data

  redis:
    volumes:
      - ./data/redis:/data
```

Then start Langfuse normally:

```bash
docker compose up
```

With this override, the persistent data is stored under:

```text
langfuse/data/
```

Without this override, assume the data remains in Docker-managed container
storage and not in a project folder. Also be careful with:

```bash
docker compose down -v
```

The `-v` flag removes Docker volumes, which deletes the local Langfuse data.
Use plain `docker compose down` for normal shutdowns if you want to keep the
data.
If you already started Langfuse once with named volumes and then switch to the
folder-based bind mounts above, Langfuse will look like a fresh installation
unless you migrate the old volume data first.

For production deployments, the managed equivalents of Postgres, ClickHouse,
Redis/Valkey, and S3-compatible object storage are usually preferred over the
single-machine Docker Compose setup.


#### CLI 101

Langfuse also provides a CLI called `langfuse-cli`. It wraps the Langfuse public
API, so it can query and manage the same project-level resources exposed by the
API: traces, observations, prompts, datasets, scores, sessions, metrics, and
more.

The quickest way to use it is with `npx`, which downloads and runs the CLI
without adding it to the project:

```bash
npx langfuse-cli api <resource> <action>
```

If you prefer a reusable global command, install it with npm:

```bash
npm install -g langfuse-cli
langfuse-cli api <resource> <action>
```

The CLI uses the same project API keys as the Python SDK. For a local self-hosted
Langfuse instance, configure:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="http://localhost:3000"
```

On PowerShell:

```powershell
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_BASE_URL = "http://localhost:3000"
```

There is no separate login step. The key pair belongs to one Langfuse project,
so CLI commands operate on that project. To work with another project, switch to
that project's public/secret key pair.

Because the CLI is generated from the Langfuse OpenAPI specification, the most
important first commands are help/discovery:

```bash
npx langfuse-cli --help
npx langfuse-cli api --help
```

Then inspect the resource/action help for the thing you want to do, for example:

```bash
npx langfuse-cli api traces --help
npx langfuse-cli api prompts --help
npx langfuse-cli api datasets --help
```

Typical uses are quick lookups, scripting, and CI jobs, for example listing or
fetching traces, checking prompt versions, creating datasets, adding scores, or
exporting data for evaluation workflows. The exact resource and action names can
change as the API evolves, so use `--help` as the source of truth for the
installed CLI version.

### Why LLM Observability?

- LLM observability is essential because LLM applications can become expensive, slow, unreliable, or risky without clear visibility into what is happening.
- Teams often discover problems too late:
  - runaway prompts or retry loops can create large unexpected bills;
  - token usage can spike without an obvious cause;
  - users can experience slow or low-quality responses while engineers lack
    enough context to debug them;
  - RAG systems can retrieve poor context and cause confident hallucinations.
- Observability is not just monitoring for its own sake. It protects:
  - budget, by exposing token usage, retries, and expensive workflows;
  - product quality, by revealing poor prompts, weak retrieval, and bad outputs;
  - reputation, by catching failures before users or customers escalate them.
- The hidden costs of LLM systems include:
  - API/token costs;
  - compute costs for embeddings, inference, and retrieval;
  - engineering time spent debugging;
  - incident costs from downtime, bad answers, compliance issues, or customer
    impact.
- Without observability, teams cannot manage what they cannot measure. Costs and
  failures tend to grow silently until they show up in bills, support tickets, or
  incidents.
- Key risks without observability:
  - **Token spikes**: high impact and high probability.
  - **Silent failures**: medium impact but high probability.
  - **Performance degradation**: medium impact and high probability.
  - **Compliance violations**: critical impact, even if probability is lower.

### Traditional Monitoring vs LLM Observability

- Traditional monitoring focuses on generic application health:
  - request latency;
  - error rates;
  - uptime;
  - infrastructure metrics.
- Traditional monitoring often treats the application as a black box: a request
  enters, internal work happens, and a response leaves.
- LLM observability looks inside the LLM workflow and captures details that
  traditional monitoring misses:
  - prompts and responses;
  - token input/output flows;
  - model name and model behavior;
  - retrieval quality in RAG systems;
  - tool calls and agent steps;
  - cost attribution by request, user, feature, or model.
- LLM-specific questions include:
  - Did the prompt produce a good answer?
  - Did the RAG pipeline retrieve relevant chunks?
  - Did the model hallucinate, refuse, or become overly verbose?
  - Which feature or user segment is driving cost?
  - Which step in the chain is slow or failing?
- In production, LLM observability helps teams:
  - debug issues in minutes instead of hours;
  - optimize token and model costs;
  - catch quality or latency issues before users report them;
  - explain to stakeholders where LLM budget is going.

### The Three Pillars of LLM Observability

- **Traces**
  - Show the end-to-end flow of a request.
  - Capture each step in the workflow: prompt construction, retrieval, tool
    calls, model calls, parsing, evaluation, and final response.
  - Make it easier to identify which step is slow, expensive, incorrect, or
    failing.
- **Metrics**
  - Aggregate operational and cost signals across many requests.
  - Useful metrics include token usage, latency, cost per request, cost per
    feature, cost per user, error rates, and model usage.
  - Metrics help teams spot trends, spikes, regressions, and budget issues.
- **Evaluations**
  - Measure output quality, not just system health.
  - Can track relevance, hallucination risk, correctness, safety, tone,
    groundedness, or task-specific quality.
  - Evaluation scores make it possible to compare prompts, models, retrieval
    strategies, and application versions.
- Common measurements:
  - input/output token ratio;
  - latency by workflow step;
  - cost by request, feature, model, user, or customer;
  - retrieval relevance;
  - hallucination rate;
  - error rate by type, model, prompt, or workflow.

### ROI Calculation

- Observability has a direct business case because it reduces preventable LLM
  costs and engineering effort.
- Example monthly baseline without observability:
  - LLM spend: `$20,000`;
  - estimated waste: `30%`, or about `$6,000`;
  - debugging effort: `10 hours/week * $100/hour`, or about `$4,000/month`;
  - amortized incident cost: `$5,000/month`;
  - total preventable cost: about `$15,000/month`.
- Example investment:
  - observability platform: about `$500-$2,000/month`;
  - setup time: about `8 hours` one time.
- Potential benefits:
  - lower token spend through prompt/model optimization;
  - less wasted retry traffic;
  - faster debugging with traces;
  - fewer surprise bills through cost alerts;
  - faster incident response.
- Example outcomes mentioned in the course:
  - token cost reduction from prompt optimization;
  - up to `80%` less debugging time with proper tracing;
  - fewer runaway-cost surprises with alerts.
- Simple ROI formula:

```text
savings = token_waste + (debug_time * hourly_rate) + (incidents_prevented * incident_cost)
```

- The practical question is not only "Can we afford observability?", but "How
  much are we already losing by not observing the system?"

## 2. Understanding LLM Costs

### Input and Output Tokens

- LLM pricing is based on **tokens**, not characters or words.
- A rough rule of thumb is that one token is about four English characters, but the ratio depends on the text.
- Tokenization varies by content type:
  - `"Hello, world!"` has 13 characters but only 4 tokens.
  - A normal sentence such as `"The quick brown fox jumps over the lazy dog."` has 44 characters and about 10 tokens.
  - Code often tokenizes less efficiently because syntax creates many small tokens.
  - Long words can also split into many tokens.
- Output tokens usually cost **2-5x more** than input tokens.
- Example per-1M-token prices from the course material:
  - GPT-4o: about `$2.50` input and `$10.00` output.
  - GPT-4o mini: about `$0.15` input and `$0.60` output.
  - Claude 3.5 Sonnet: about `$3.00` input and `$15.00` output.
  - Claude 3.5 Haiku: about `$0.25` input and `$1.25` output.
  - Gemini 1.5 Pro: about `$1.25` input and `$5.00` output.
- The exact prices change over time; the important pattern is the input/output ratio and the large gap between model tiers.
- Verbose responses are expensive because they increase output tokens.
- Long system prompts are usually cheaper than long responses because they are input tokens.
- Asking the model to be concise is a direct cost optimization.
- Cost estimation requires:
  - Counting input and output tokens.
  - Looking up the selected model's input and output prices per million tokens.
  - Calculating input cost and output cost separately, then adding them.
- Tokenizers can differ between model providers, so token counts for non-OpenAI models may only be approximate when using a fallback tokenizer.
- A small single-request cost can become significant at production volume.
- In the course example, the same prompt/response shape costs about `$415` per 1M monthly requests with GPT-4o, about `$25` with GPT-4o mini, and about `$50` with Claude 3.5 Haiku.
- Key optimization levers:
  - Choose the cheapest model that performs the task well.
  - Shorten prompts and remove repetitive instructions.
  - Limit output length with concise instructions and `max_tokens` where appropriate.

Examples in [`lab/udemy-langfuse/`](./lab/udemy-langfuse/):

```python
##### -- tokens-demo-1.py

import tiktoken

# Initialize the tokenizer for GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

# Let's count some tokens
examples = [
    "Hello, world!",  # Simple
    "The quick brown fox jumps over the lazy dog.",  # Standard sentence
    "def calculate_total(items): return sum(item.price for item in items)",  # Code
    "supercalifragilisticexpialidocious",  # Long word
]

for text in examples:
    tokens = enc.encode(text)
    print(f"'{text}'")
    print(f"  Characters: {len(text)}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Tokens: {tokens}")
    print()


##### -- token-calculator-2.py

import tiktoken
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelPricing:
    name: str
    input_cost_per_million: float
    output_cost_per_million: float

# Current pricing (January 2026)
MODELS = {
    "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60),
    "claude-3.5-sonnet": ModelPricing("claude-3.5-sonnet", 3.00, 15.00),
    "claude-3.5-haiku": ModelPricing("claude-3.5-haiku", 0.25, 1.25),
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a given text."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def calculate_cost(
    input_text: str,
    output_text: str,
    model: str = "gpt-4o"
) -> Dict[str, float]:
    """Calculate the cost of an LLM interaction."""

    pricing = MODELS.get(model)
    if not pricing:
        raise ValueError(f"Unknown model: {model}")

    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)

    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
    }
```

### Where Costs Hide: RAG and Agentic Pipelines

![RAG and Agentic Pipeline Costs](./assets/rag_agent_costs.png)

- RAG and agentic pipelines hide cost across multiple steps:
  - User query embedding is usually very cheap.
  - Vector search is usually minimal.
  - Context assembly increases the amount of text passed to the model.
  - The first LLM call processes the assembled prompt and retrieved context.
  - Agent decisions can trigger tool calls, API lookups, and additional LLM calls.
  - The final response adds another output-token cost.
- A single query can look inexpensive, but production volume changes the economics quickly.
- In the course example, a sub-cent query becomes about `$700/day`, `$21,000/month`, or `$252,000/year` at `100,000` queries per day.
- That estimate assumes the happy path: no retries, no errors, and no overly verbose responses.
- LLM calls often make up more than 95% of total pipeline cost, even as model inference prices fall.
- Top cost drivers:
  - Bloated system prompts sent with every request.
  - Excessive retrieved context; the goal is the **right** context, not the most context.
  - Agent reasoning loops that repeatedly call the model.
  - Growing chat history, which increases context size linearly.
  - Wrong model selection, which can create a very large price difference for the same workflow.

### The Hidden Cost Multipliers

![Task-Model Matrix](./assets/task_model_matrix.png)

- Hidden multipliers are costs that repeat or compound quietly:
  - System prompts are sent with every request.
  - RAG context can add thousands of tokens.
  - Retries mean paying for the same work twice or more.
  - Chat history grows linearly as conversations get longer.
- The same prompt can have a dramatically different cost depending on the model, with the course material emphasizing up to a `200x` gap.
- Use a task-model matrix to match model quality to task difficulty:
  - Classification: use smaller models such as Haiku or mini models.
  - Simple extraction: use smaller models when they perform well enough.
  - Simple Q&A and summarization: prefer cheaper models unless quality requires an upgrade.
  - Complex reasoning: use stronger models such as Sonnet or GPT-4-class models.
  - Code generation: test both cheaper and stronger models; paying more can be justified when quality matters.
- Intelligent model routing selects the most cost-effective model for each task instead of sending every request to the same expensive model.
- A router can use task type and complexity to choose among cheap, mid-tier, and strong models.
- For code generation, a useful pattern is to start cheaper and upgrade only when needed.
- For complex reasoning, analysis, and creative tasks, route to stronger models that reliably produce better results.
- Model selection is one of the highest-leverage LLM cost optimizations.

## 3. LangFuse as Observability Platform

### LLM Monitoring Platforms

![LLM Monitoring Platforms](./assets/llm_monitoring_platforms.png)

- There are many LLM observability tools, but teams usually only need to compare the main production-ready options.
- Langfuse is recommended in this course because it is open source, self-hostable, vendor neutral, framework friendly, and has a clean UI, strong tracing, metrics, evaluations, and a generous free tier.
- LangSmith is a strong choice for teams heavily invested in the LangChain ecosystem, but it is closed source.
- Arize Phoenix is useful for ML and evaluation-focused teams, with both open-source and commercial options.
- Helicone is especially useful when cost tracking is the primary concern.
- Portkey is worth considering for systems that route across multiple LLM providers.
- Platform choice depends on the team's needs:
  - Heavy LangChain usage: LangSmith.
  - Open source, self-hosting, and control: Langfuse.
  - Cost tracking: Helicone.
  - Multiple LLM providers: Portkey.
- The course uses Langfuse, but the observability concepts should transfer to other platforms.

### Setting Up LangFuse on the Cloud

To see how set up your own managed LangFuse, check [Local Setup and Quick Start](#local-setup-and-quick-start) above.

- Langfuse can be used through the managed cloud service or self-hosted with Docker.
- The cloud option is the fastest path for the course because it requires no hosting, no infrastructure management, and no credit card for the free tier (50k traces/month).
- Basic cloud setup:
  - Sign up with email, Google, GitHub, or Azure AD.
  - Create an organization.
  - Create a project.
  - Copy the project hostname/base URL.
  - Create API keys.
  - Add the Langfuse public key, secret key, and base URL to the project's `.env` file.
- For local Python usage, install the Langfuse SDK and load environment variables with `python-dotenv`.
- Instantiate a Langfuse client with the public key, secret key, and base URL from the environment.
- The Langfuse project settings include API keys, LLM connections, models, score configurations, members, integrations, exports, batch actions, audit logs, notifications, and billing.
- After setup, the next step is instrumenting a real application so traces and LLM activity appear in the dashboard.

```bash
LANGFUSE_SECRET_KEY="sk-lf-xxx"
LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
LANGFUSE_BASE_URL="http://localhost:3000"
```

![LangFuse Project Settings](./assets/langfuse_project_settings.png)

### Creating First Trace with `@observe()`

- The new SDK pattern uses `@observe` and `get_client()` instead of manually instantiating a `LangFuse` object.
- Environment variables still come from `.env`.
- Running the script creates a first trace, flushes it to Langfuse, and lets you verify it in the dashboard under **Observability > Tracing**.
- This same pattern can later be attached to real LLM calls for tracing, debugging, and cost monitoring.

![LangFuse Observe Decorator](./assets/langfuse_observe_decorator.png)

File: [`lab/udemy-langfuse/langf_obs.py`](./lab/udemy-langfuse/langf_obs.py)

```python
from langfuse import observe, get_client
from dotenv import load_dotenv

# Load LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL
# from the local .env file.
load_dotenv()

@observe
def verify_connection():
    # Nested observed functions appear as child observations inside the trace.
    test_generation()

    # Short-lived scripts and notebooks should flush before exiting so buffered
    # traces are sent to Langfuse immediately.
    client = get_client()
    client.flush()

    print("Connection to Langfuse is successful!")
    print("check your dashboard at http://localhost:3000")

@observe
def test_generation():
    """A simple test generation function to verify Langfuse connection."""

    # The @observe decorator automatically logs this function call, including
    # timing and return value, to Langfuse.
    return "Hello, Langfuse!"

if __name__ == "__main__":
    # Run the observed function, then check the Langfuse dashboard.
    verify_connection()
    test_generation()
```

#### LangFuse Data Model

- Langfuse organizes observability data hierarchically so complex LLM workflows can be inspected end to end.
- Typical LLM workflows include retrieval, processing, generation, tool calls, and response handling.
- Main hierarchy:
  - **Session**: a group of related traces, such as one chat conversation.
  - **Trace**: one request or operation inside a session, such as one user message to an LLM.
  - **Observation**: one step inside a trace, such as retrieval, processing, generation, a tool call, or an event.

![LangFuse Data Model Hierarchy](./assets/langfuse_data_model_hierarchy.png)

- A trace is the container for everything that happens during a single request.
- Example trace flow:
  - User asks, "What's the weather like today?"
  - The system receives the question.
  - Retrieval or other context-building steps may run.
  - The LLM call is made.
  - The response is returned.
- Important trace properties:
  - `name`: identifies the operation.
  - `input`: what went into the request.
  - `output`: what came out.
  - `user_id`: identifies who made the request.
  - `session_id`: links the trace to its parent conversation/session.
  - `tags`: labels used for filtering, grouping, and cost breakdowns.

![LangFuse Data Model Traces](./assets/langfuse_data_model_traces.png)

- Observation types:
  - **Generation**: an LLM API call, including completions, token counts, model information, and calculated cost.
  - **Span**: an operation with duration, such as database queries, retrieval steps, or processing logic.
  - **Event**: a point-in-time occurrence, such as a cache hit or an error.
- Observations can be nested hierarchically:
  - A generation can have child spans.
  - A span can have child events.

![LangFuse Data Model Observations](./assets/langfuse_data_model_observations.png)

- Sessions group related traces together.
- Example session:
  - One chat conversation with ID `ABC123`.
  - A 10-message conversation can be represented as one session with 10 traces, one per message.
- Sessions are useful because they show:
  - Full conversation flow.
  - User behavior patterns.
  - How the LLM application behaves across multiple turns.
  - Conversation-level metrics such as resolution rate and turns to completion.
  - A/B test comparisons across different approaches.

![LangFuse Data Model Sessions](./assets/langfuse_data_model_sessions.png)

- Example end-to-end data flow:
  - A user sends a message.
  - Langfuse creates a trace with `session_id` and `user_id`.
  - A retrieval span records its duration, for example `45 ms`.
  - A generation records the LLM call, for example `150` input tokens, `50` output tokens, and the resulting cost.
  - The trace is updated with final output and total latency, for example `850 ms`.

![LangFuse Data Model: How It All Connects](./assets/langfuse_data_model.png)

- Cost tracking happens at multiple levels:
  - **Generation level**: cost per individual LLM call, based on token counts and model pricing.
  - **Trace level**: total cost per request, aggregating all generations inside the trace.
  - **Session level**: cost per conversation, useful for chat-based products.
  - **Tag level**: cost by feature, user tier, experiment variant, or other labels.
  - **User/group level**: identify the most expensive users, cohorts, tiers, or usage patterns.
- This data model makes observability useful for debugging, performance analysis, user behavior analysis, A/B testing, and cost management.

![LangFuse Data Model: Cost Management](./assets/langfuse_data_model_costs.png)


### First LLM Trace with LangFuse with OpenAI Wrapper

- This example creates a real OpenAI chat completion through the Langfuse OpenAI wrapper.
- The wrapper keeps the familiar `chat.completions.create(...)` shape while automatically logging the LLM call as a Langfuse generation.
- In the Langfuse UI, the trace appears with its timestamp, name, input, output, and total duration.
- Opening the generation shows detailed observability data:
  - Latency, for example around `1.xs` in the demo.
  - Model used, here `gpt-4o-mini`.
  - Token usage, including input tokens, output tokens, and total usage.
  - Cost breakdown for input, cached input if applicable, and output.
  - Captured system prompt, user message, assistant response, and custom metadata.
- The metadata value `project: first_trace_llm_project` confirms that application-specific fields can be attached to traces for filtering and analysis.

File: [`lab/udemy-langfuse/first_trace_llm.py`](./lab/udemy-langfuse/first_trace_llm.py).

```python
# Import the OpenAI-compatible client from Langfuse.
# The API call looks like a normal OpenAI call, but Langfuse records the trace.
from langfuse.openai import openai
from dotenv import load_dotenv


# Load OPENAI_API_KEY plus LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# and LANGFUSE_BASE_URL from .env.
load_dotenv()

completion = openai.chat.completions.create(
    # This name appears in the Langfuse trace list.
    name="first_trace_llm",
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            # System prompt captured in the generation details.
            "content": "You are a very accurate calculator. You output only the result of the calculation.",
        },
        # User message captured as the generation input.
        {"role": "user", "content": "123 + 456 * 2 = "},
    ],
    # Custom metadata is stored with the trace and can be used for filtering.
    metadata={"project": "first_trace_llm_project"},
)

# Print the assistant response locally; the same output is visible in Langfuse.
print(completion.choices[0].message.content)

```

![LLM Generation Trace](./assets/llm_generation_trace.png)

### LangFuse API Levels: Decorator, Context Manager, Low-Level, Drop-in

![API Levels](./assets/api_levels.png)

- Langfuse offers multiple API levels that can capture similar observability data with different amounts of control.
- Main API levels:
  - **Decorator-based**: use `@observe()` on functions; modern, simple, and usually the recommended default.
  - **Context manager**: use `with` blocks to create traces, spans, and generations explicitly.
  - **Low-level SDK**: manually create and update traces, spans, generations, scores, events, and other objects.
  - **Drop-in wrappers**: use provider wrappers such as the Langfuse OpenAI wrapper so normal LLM calls are traced automatically.
- The previous OpenAI wrapper example used the **drop-in** level: the code still calls `chat.completions.create(...)`, while Langfuse records traces in the background.
- The first code block below demonstrates the **decorator-based** level:
  - `@observe()` creates traces/spans automatically.
  - Nested observed functions become nested spans.
  - `update_current_span(...)` adds metadata and tags for filtering and grouping.
  - `flush()` sends buffered observations before the script exits.
- Context-manager tracing (2nd code block) is more explicit: use `with` blocks to name and structure spans/generations manually.
- Low-level tracing provides (3rd code block) the most control but requires more code, including manually starting, updating, ending, and linking spans or generations.
- Choose the API level based on how much automation vs. control the application needs.

This section uses three scripts:

- [`decorator_trace_llm.py`](./lab/udemy-langfuse/decorator_trace_llm.py) demonstrates the decorator-based API.
- [`context_manager_trace_llm.py`](./lab/udemy-langfuse/context_manager_trace_llm.py) demonstrates the context manager API.
- [`low_level_trace_llm.py`](./lab/udemy-langfuse/low_level_trace_llm.py) demonstrates the low-level API.

Additionally, the notebook [`lab/02_langfuse_api_levels.ipynb`](./lab/02_langfuse_api_levels.ipynb) walks through all three API levels interactively, using the same code as in the scripts.

#### Decorator API Example

File: [`lab/udemy-langfuse/decorator_trace_llm.py`](./lab/udemy-langfuse/decorator_trace_llm.py)

```python
# Decorator-level API: use normal Python functions and annotate them with
# @observe() so Langfuse creates traces/spans automatically.
from langfuse import observe, Langfuse
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI and Langfuse credentials from .env.
load_dotenv()

# Use the regular OpenAI client for the LLM call.
client = OpenAI()

# Langfuse client is used here to enrich the current span and flush data.
langfuse = Langfuse()

@observe()  # Creates a trace/span automatically.
def calculator(expression: str) -> str:
    """Single calculation - becomes a span when called from another @observe function."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a very accurate calculator. You output only the result of the calculation.",
            },
            {"role": "user", "content": expression},
        ],
    )

    # Add metadata and tags to the active span so it can be filtered in Langfuse.
    # Other SDK helpers can update traces, spans, generations, prompts, scores,
    # datasets, and events depending on what needs to be recorded.
    langfuse.update_current_span(
        metadata={"project": "decorator_example", "tags": ["calculator", "math"]},
    )

    return completion.choices[0].message.content


@observe()  # Nested observed function = parent span with child spans.
def process_calculations(expressions: list[str]) -> list[str]:
    """Process multiple calculations - each calculator() call becomes a child span."""
    results = []
    for expr in expressions:
        # Each calculator() call is captured as a child span under this trace.
        result = calculator(expr)
        results.append(f"{expr} = {result}")
    return results


if __name__ == "__main__":
    # Running this creates one trace for the batch and child spans for each calculation.
    expressions = ["123 + 456", "789 * 2", "100 / 4"]
    results = process_calculations(expressions)

    print("Results:")
    for r in results:
        print(f"  {r}")

    # Flush is important for short-lived scripts so buffered observations are sent.
    langfuse.flush()

```

![Decorator Trace](./assets/decorator_trace.png)

#### Context Manager API Example

File: [`lab/udemy-langfuse/context_manager_trace_llm.py`](./lab/udemy-langfuse/context_manager_trace_llm.py).

```python
# Context-manager API: explicitly open trace/span/generation scopes with
# `with` blocks, while Langfuse manages the active current object.
from langfuse import Langfuse
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI and Langfuse credentials from .env.
load_dotenv()

# Regular OpenAI client performs the model call.
client = OpenAI()

# Langfuse client creates and updates the trace hierarchy.
langfuse = Langfuse()

expression = "123 + 456 * 2"

# Create the root span. Because it is the outermost active span, it becomes
# the root of a new trace in Langfuse.
with langfuse.start_as_current_observation(
    name="calculator_context_manager",
    as_type="span",
) as trace:

    # Add a child span for preprocessing/validation work.
    with langfuse.start_as_current_observation(
        name="input_validation",
        as_type="span",
    ) as validation_span:
        # Update the currently active span with input and output data.
        langfuse.update_current_span(
            input={"expression": expression},
            output={"status": "valid"},
        )

    # Add a generation for the actual LLM call. Generations are the right
    # Langfuse object for model requests because they track model, tokens, and cost.
    with langfuse.start_as_current_observation(
        name="llm_calculation",
        as_type="generation",
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are a very accurate calculator. You output only the result of the calculation.",
            },
            {"role": "user", "content": expression},
        ],
    ) as generation:
        # The OpenAI call itself still uses the regular OpenAI SDK.
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a very accurate calculator. You output only the result of the calculation.",
                },
                {"role": "user", "content": expression},
            ],
        )
        result = completion.choices[0].message.content

        # Attach the LLM output, token usage, and metadata to the active generation.
        langfuse.update_current_generation(
            output=result,
            usage_details={
                "input": completion.usage.prompt_tokens,
                "output": completion.usage.completion_tokens,
            },
            metadata={"project": "context_manager_example"},
        )

    # Attach final request-level output and tags to the root span.
    trace.update(
        output={"result": result},
        tags=["calculator", "context_manager"],
    )

print(f"{expression} = {result}")

# Flush is important for short-lived scripts so buffered observations are sent.
langfuse.flush()
```

#### Low-Level API Example

File: [`lab/udemy-langfuse/low_level_trace_llm.py`](./lab/udemy-langfuse/low_level_trace_llm.py).

```python
# Low-level API: manually create, update, end, and link Langfuse objects.
# This gives maximum control, but requires more lifecycle code.
from langfuse import Langfuse
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load OpenAI and Langfuse credentials from .env.
load_dotenv()

# Regular OpenAI client performs the model call.
client = OpenAI()

# Langfuse client is used directly to create spans, generations, scores, and events.
langfuse = Langfuse()

expression = "123 + 456 * 2"

# Create the root span. Starting a root span creates a new trace automatically.
root_span = langfuse.start_observation(
    name="calculator_low_level",
    as_type="span",
    input={"expression": expression},
    metadata={"project": "low_level_example"},
)

# Keep the trace ID so later objects, such as scores, can be attached to this trace.
trace_id = root_span.trace_id

# Create and manually finish a child span for preprocessing/validation.
preprocessing_span = root_span.start_observation(
    name="preprocessing",
    as_type="span",
    input={"raw_expression": expression},
)

# Update the child span with the validation result.
preprocessing_span.update(
    output={"validated_expression": expression, "status": "valid"},
)

# Low-level spans must be ended explicitly.
preprocessing_span.end()

# Track custom timing around the model call.
start_time = time.time()

# The LLM request itself is still a normal OpenAI SDK call.
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a very accurate calculator. You output only the result of the calculation.",
        },
        {"role": "user", "content": expression},
    ],
)

end_time = time.time()
result = completion.choices[0].message.content

# Create the generation manually as another child observation so we can control
# its name, model, parameters, input, output, usage, status, and lifecycle.
generation = root_span.start_observation(
    name="llm_calculation",
    as_type="generation",
    model="gpt-4o-mini",
    model_parameters={"temperature": 1.0},
    input=[
        {
            "role": "system",
            "content": "You are a very accurate calculator. You output only the result of the calculation.",
        },
        {"role": "user", "content": expression},
    ],
)

# Update the generation with output, token usage, severity level, and status.
generation.update(
    output=result,
    usage_details={
        "input": completion.usage.prompt_tokens,
        "output": completion.usage.completion_tokens,
        "total": completion.usage.total_tokens,
    },
    level="DEFAULT",  # Options: "DEBUG", "DEFAULT", "WARNING", "ERROR"
    status_message="Calculation completed successfully",
)

# Low-level generations must also be ended explicitly.
generation.end()

# Update and end the root span with final request-level output and custom timing.
root_span.update(
    output={"result": result},
    metadata={"duration_ms": (end_time - start_time) * 1000},
)
root_span.end()

# Attach an evaluation score to the trace.
langfuse.create_score(
    trace_id=trace_id,
    name="accuracy",
    value=1.0,
    comment="Correct calculation verified",
)

# Create a point-in-time event for debugging or audit-style logging.
event = langfuse.create_event(
    name="calculation_complete",
    trace_context={"trace_id": trace_id},
    input={"expression": expression},
    output={"result": result},
    metadata={"duration_ms": (end_time - start_time) * 1000},
)

print(f"{expression} = {result}")
print(f"Duration: {(end_time - start_time) * 1000:.2f}ms")
print(f"Tokens used: {completion.usage.total_tokens}")
print(f"Trace ID: {trace_id}")

# Flush is important for short-lived scripts so buffered observations are sent.
langfuse.flush()

```

## 4. Instrumenting LLM Applications with LangFuse

### LLM App for Production

Files:

- [`lab/udemy-langfuse/instrumented_llm.py`](./lab/udemy-langfuse/instrumented_llm.py)
- [`lab/03_langfuse_llm_production.ipynb`](./lab/03_langfuse_llm_production.ipynb)

- The production wrapper normalizes responses from multiple providers into one `LLMResponse` dataclass.
- Each response stores `content`, `input_tokens`, `output_tokens`, `model`, `duration_ms`, and estimated `cost`.
- Pricing is kept in a model-to-price dictionary using per-1M-token input/output rates; update these values when provider pricing changes.
- Provider calls are decorated as Langfuse **generations**, the right observation type for LLM calls because it can hold model, token, cost, latency, input, and output data.
- `call_claude(...)` and `call_openai(...)` follow the same production pattern:
  - Build the provider-specific request.
  - Measure latency.
  - Read token usage from the provider response.
  - Estimate cost.
  - Enrich the current Langfuse generation with model, input/output, usage, cost, and metadata.
  - Return a normalized `LLMResponse`.
- `compare_models(...)` wraps both provider calls in one observed span so the two generations appear as child observations in the same trace.
- `propagate_attributes(...)` sets trace-level fields such as `user_id`, `session_id`, tags, metadata, and trace name so runs are easy to filter later.
- In the Langfuse UI, this setup exposes latency per provider, total trace latency, token usage, estimated cost, inputs/outputs, structured metadata, trace IDs, user IDs, tags, logs, comments, and errors.
- `langfuse.flush()` is required in notebooks and short-lived scripts so buffered observations are sent before the process exits.

```python
from dataclasses import asdict, dataclass
import time
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from openai import OpenAI

load_dotenv()

# Provider clients read their API keys from environment variables.
anthropic_client = Anthropic()
openai_client = OpenAI()

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()


@dataclass
class LLMResponse:
    """Normalized response shape used across providers."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    duration_ms: float
    cost: float


# Example standard prices per 1M tokens, last checked 2026-05-22.
# This intentionally covers both the current course defaults and a few common
# alternatives. It does not include prompt caching, batch, flex, priority, long
# context, regional, or tool-specific pricing modifiers.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude API model IDs.
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI standard short-context prices.
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost from token usage."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


@observe(name="call_claude", as_type="generation")
def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    system: str | None = None,
    max_tokens: int = 1024,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Call Claude and enrich the active Langfuse generation."""
    start = time.perf_counter()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = anthropic_client.messages.create(**kwargs)

    duration_ms = (time.perf_counter() - start) * 1000
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    content = response.content[0].text

    # Because this function is observed as a generation, model-specific fields
    # belong on the current generation rather than generic span metadata.
    langfuse.update_current_generation(
        model=model,
        input=[{"role": "user", "content": prompt}],
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        cost_details={"total": cost},
        metadata={
            **(metadata or {}),
            "provider": "anthropic",
            "duration_ms": duration_ms,
            "stop_reason": response.stop_reason,
        },
    )

    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        duration_ms=duration_ms,
        cost=cost,
    )


@observe(name="call_openai", as_type="generation")
def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str | None = None,
    max_tokens: int = 1024,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Call OpenAI and enrich the active Langfuse generation."""
    start = time.perf_counter()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    duration_ms = (time.perf_counter() - start) * 1000
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    content = response.choices[0].message.content or ""

    langfuse.update_current_generation(
        model=model,
        input=messages,
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        cost_details={"total": cost},
        metadata={
            **(metadata or {}),
            "provider": "openai",
            "duration_ms": duration_ms,
        },
    )

    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        duration_ms=duration_ms,
        cost=cost,
    )


@observe(name="compare_models", as_type="span")
def compare_models(
    prompt: str,
    user_id: str = "demo-user",
    session_id: str | None = "demo-session",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Compare Claude and OpenAI for the same prompt in one trace."""
    trace_tags = tags or ["production-example", "model-comparison"]

    # These attributes are applied to this trace and propagated to child
    # generations so the run can be filtered by user, session, tags, and metadata.
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=trace_tags,
        metadata={"comparison": True},
        trace_name="llm-model-comparison",
    ):
        claude_response = call_claude(
            prompt,
            metadata={"comparison_role": "candidate_a"},
        )
        openai_response = call_openai(
            prompt,
            metadata={"comparison_role": "candidate_b"},
        )

    total_cost = claude_response.cost + openai_response.cost
    total_duration_ms = claude_response.duration_ms + openai_response.duration_ms

    langfuse.update_current_span(
        output={
            "claude": asdict(claude_response),
            "openai": asdict(openai_response),
            "total_cost": total_cost,
            "total_duration_ms": total_duration_ms,
        },
        metadata={
            "comparison": True,
            "total_cost": total_cost,
            "total_duration_ms": total_duration_ms,
        },
    )

    return {
        "claude": claude_response,
        "openai": openai_response,
        "total_cost": total_cost,
        "total_duration_ms": total_duration_ms,
    }


if __name__ == "__main__":
    result = compare_models("Explain the theory of relativity in simple terms.")

    print("Claude:", result["claude"].content)
    print("OpenAI:", result["openai"].content)
    print(f"Total cost: ${result['total_cost']:.6f}")
    print(f"Total duration: {result['total_duration_ms']:.0f}ms")

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
```

![LLM Production Example](./assets/llm_production_example.png)

### RAG Pipeline

Files:

- [`lab/udemy-langfuse/rag_pipeline_obs.py`](./lab/udemy-langfuse/rag_pipeline_obs.py)
- [`lab/04_langfuse_rag.ipynb`](./lab/04_langfuse_rag.ipynb)

- Real applications are usually multi-step pipelines, not single LLM calls; RAG is a good example because it includes loading, chunking, embedding, retrieval, context assembly, and generation.
- Each step is traced as its own Langfuse observation so latency, inputs, outputs, metadata, and errors can be inspected independently.
- The indexing side loads Markdown files, splits them into chunks, embeds each chunk with `all-MiniLM-L6-v2`, and stores the vectors in a persistent ChromaDB collection.
- The query side embeds the user query, retrieves the top matching chunks, builds a context string with source labels, and sends the query plus context to Claude.
- Current Langfuse observation types make the trace easier to read:
  - `span` for orchestration, loading, chunking, and context assembly.
  - `embedding` for chunk and query embedding work.
  - `retriever` for ChromaDB similarity search.
  - `generation` for the final Claude call.
- Retrieval metadata includes `top_k`, number of chunks retrieved, average distance, source paths, distance scores, and content previews.
- Generation metadata includes the model, prompt/context length, input/output token counts, total tokens, and the generated answer.
- In the Langfuse UI, the trace reveals where time is spent; generation is often the slowest and most expensive step.
- Failed runs are useful too: Langfuse shows exceptions, missing outputs, and bad SDK arguments, making pipeline debugging much faster.
- This observability makes it possible to tune retrieval quality, context size, latency, and cost instead of guessing.

![RAG Example](./assets/rag_example.png)

```python
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import get_client, observe, propagate_attributes
from sentence_transformers import SentenceTransformer

load_dotenv()

# Directories need to be changed, depending on noteebook/script location
BASE_DIR = Path(".").resolve().parent
DOCS_DIR = BASE_DIR / "lab" / "udemy-langfuse" / "docs"
CHROMA_DIR = BASE_DIR / "lab" / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL = "claude-sonnet-4-20250514"

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()

# Chroma keeps the vector index on disk so indexing can be reused across runs.
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Load the embedding model once. all-MiniLM-L6-v2 is small and returns 384 dims.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
anthropic_client = anthropic.Anthropic()


@observe(name="load_and_index_documents", as_type="span")
def load_and_index_documents(docs_dir: str | Path = DOCS_DIR) -> int:
    """Load, chunk, embed, and index Markdown documents."""
    documents = load_markdown_docs(docs_dir)
    if not documents:
        langfuse.update_current_span(
            output={"chunks_indexed": 0},
            metadata={"reason": "no_documents"},
        )
        return 0

    chunks = chunk_documents(documents)
    chunks_indexed = index_chunks(chunks)

    langfuse.update_current_span(
        output={"chunks_indexed": chunks_indexed},
        metadata={"docs_loaded": len(documents), "chunks_created": len(chunks)},
    )
    return chunks_indexed


@observe(name="load_markdown_docs", as_type="span")
def load_markdown_docs(docs_dir: str | Path) -> list[Document]:
    """Load all Markdown files from a directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    langfuse.update_current_span(
        output={"documents_loaded": len(documents)},
        metadata={"docs_dir": str(docs_path), "glob": "**/*.md"},
    )
    return documents


@observe(name="chunk_documents", as_type="span")
def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into retrieval-sized chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    langfuse.update_current_span(
        output={"chunks": len(chunks)},
        metadata={
            "documents": len(documents),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return chunks


@observe(name="index_chunks", as_type="embedding")
def index_chunks(chunks: list[Document]) -> int:
    """Create embeddings and upsert chunks into ChromaDB."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for index, chunk in enumerate(chunks):
        chunk_id = f"chunk_{index}"
        embedding = embedding_model.encode(chunk.page_content).tolist()

        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append(
            {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": index,
            }
        )
        embeddings.append(embedding)

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    langfuse.update_current_span(
        output={"chunks_indexed": len(ids)},
        metadata={
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "collection": "documents",
        },
    )
    return len(ids)


@observe(name="rag_pipeline", as_type="span")
def rag_pipeline(
    query: str,
    top_k: int = 5,
    user_id: str = "demo-user",
    session_id: str | None = "rag-demo-session",
) -> str:
    """Run the full RAG query pipeline in one trace."""
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["rag", "retrieval", "generation"],
        metadata={"top_k": top_k, "embedding_model": EMBEDDING_MODEL_NAME},
        trace_name="rag-query",
    ):
        query_embedding = embed_query(query)
        chunks = retrieve_chunks(query_embedding, top_k=top_k)
        context = build_context(chunks)
        response = generate_response(query, context)

    langfuse.update_current_span(
        output={"response": response},
        metadata={
            "query": query,
            "chunks_retrieved": len(chunks),
            "context_length": len(context),
        },
    )
    return response


@observe(name="embed_query", as_type="embedding")
def embed_query(query: str) -> list[float]:
    """Embed the user query before vector search."""
    embedding = embedding_model.encode(query).tolist()

    langfuse.update_current_span(
        output={"embedding_dim": len(embedding)},
        metadata={"query_length": len(query), "embedding_model": EMBEDDING_MODEL_NAME},
    )
    return embedding


@observe(name="retrieve_chunks", as_type="retriever")
def retrieve_chunks(embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant document chunks from ChromaDB."""
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for index, document in enumerate(documents[0]):
        chunks.append(
            {
                "content": document,
                "metadata": metadatas[0][index],
                "distance": distances[0][index],
            }
        )

    avg_distance = (
        sum(chunk["distance"] for chunk in chunks) / len(chunks) if chunks else 0.0
    )

    langfuse.update_current_span(
        output={
            "chunks": [
                {
                    "source": chunk["metadata"].get("source", "unknown"),
                    "distance": chunk["distance"],
                    "content_preview": chunk["content"][:240],
                }
                for chunk in chunks
            ]
        },
        metadata={
            "chunks_retrieved": len(chunks),
            "top_k": top_k,
            "avg_distance": avg_distance,
        },
    )
    return chunks


@observe(name="build_context", as_type="span")
def build_context(chunks: list[dict[str, Any]]) -> str:
    """Assemble retrieved chunks into the context sent to the LLM."""
    if not chunks:
        context = "No relevant context found."
    else:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {index} - {source}]: {chunk['content']}")
        context = "\n\n".join(context_parts)

    langfuse.update_current_span(
        output={"context_preview": context[:500]},
        metadata={"context_length": len(context), "num_chunks_used": len(chunks)},
    )
    return context


@observe(name="generate_response", as_type="generation")
def generate_response(query: str, context: str) -> str:
    """Generate an answer from the retrieved context using Claude."""
    prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = anthropic_client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text

    langfuse.update_current_generation(
        model=GENERATION_MODEL,
        input=[{"role": "user", "content": prompt}],
        output=answer,
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        },
        metadata={"prompt_length": len(prompt), "context_length": len(context)},
    )
    return answer


if __name__ == "__main__":
    print("Indexing documents from ./docs folder...")
    try:
        num_indexed = load_and_index_documents(DOCS_DIR)
        print(f"Indexed {num_indexed} chunks")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Create a 'docs' folder with .md files first.")
        raise SystemExit(1) from exc

    print("\nQuerying RAG pipeline...")
    result = rag_pipeline("What is LlamaIndex and how do I set it up?", top_k=5)
    print(f"\nResponse:\n{result}")

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
```

### LangChain Integration

Files:

- [`lab/udemy-langfuse/instrumentation_langchain.py`](./lab/udemy-langfuse/instrumentation_langchain.py)
- [`lab/05_langfuse_langchain.ipynb`](./lab/05_langfuse_langchain.ipynb)

- Langfuse can trace LangChain applications through a drop-in `CallbackHandler`.
- The handler can be attached to any LangChain runnable through the `config={"callbacks": [...]}` argument at invocation time.
- This is easier than manually instrumenting every LangChain component: chains, prompt steps, model calls, token usage, latency, metadata, tags, tools, and retrievers can be captured automatically.
- The example uses LangChain Expression Language (LCEL): `prompt | llm`.
- `ChatPromptTemplate` creates a reusable prompt with a dynamic `{topic}` variable.
- `ChatAnthropic` calls Claude through LangChain; similar wrappers exist for OpenAI and other providers.
- The code wraps the chain in `@observe(..., as_type="chain")` so the LangChain run sits inside a named parent observation.
- `propagate_attributes(...)` adds `user_id`, `session_id`, tags, metadata, and trace name for filtering and analysis in Langfuse.
- The helper `create_langfuse_handler(...)` supports both the current documented `langfuse_client=` constructor and installed SDK versions that only accept `trace_context=`.
- In the Langfuse UI, the trace shows the parent chain, prompt formatting, Anthropic generation, latency, token usage, input prompt, model response, and cost dashboards by model/use case/user where available.
- `langfuse.flush()` is required in notebooks and short-lived scripts so buffered observations are sent before the process exits.

![LangChain Example](./assets/langchain_example.png)

```python
from inspect import signature
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

load_dotenv()

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()


def create_langfuse_handler(trace_seed: str) -> CallbackHandler:
    """Create a Langfuse LangChain callback handler.

    Newer Langfuse SDK versions accept `langfuse_client=...`; some installed
    versions accept only `trace_context=...`. This keeps the example compatible
    while still using the current callback-based integration pattern.
    """
    trace_context = {"trace_id": Langfuse.create_trace_id(seed=trace_seed)}
    handler_params = signature(CallbackHandler).parameters

    if "langfuse_client" in handler_params:
        return CallbackHandler(
            langfuse_client=langfuse,
            trace_context=trace_context,
        )

    return CallbackHandler(trace_context=trace_context)


@observe(name="run_langchain_example", as_type="chain")
def run_langchain_example(
    topic: str = "quantum computing",
    user_id: str = "demo-user",
    session_id: str | None = "langchain-demo-session",
) -> str:
    """Run a LangChain LCEL chain and trace it with Langfuse."""
    handler = create_langfuse_handler(trace_seed=f"langchain-{topic}")

    # Trace-level attributes are propagated to the observed wrapper and child
    # LangChain callback observations where supported by the SDK.
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["langchain", "callback-handler", "demo"],
        metadata={"topic": topic, "framework": "langchain"},
        trace_name="langchain-demo",
    ):
        # ChatAnthropic is the LangChain chat-model wrapper for Anthropic.
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")

        # The prompt variable name must match the dict passed to chain.invoke().
        prompt = ChatPromptTemplate.from_template(
            "Explain {topic} in simple terms."
        )
        chain = prompt | llm

        # Attach the Langfuse callback at invocation time. The callback captures
        # the prompt step, model generation, token usage, latency, metadata, and tags.
        response = chain.invoke(
            {"topic": topic},
            config={
                "callbacks": [handler],
                "metadata": {"use_case": "langchain_example"},
                "tags": ["course", "langchain"],
            },
        )

    langfuse.update_current_span(
        output={"content": response.content},
        metadata={"topic": topic},
    )
    return response.content


if __name__ == "__main__":
    answer = run_langchain_example("quantum computing")
    print(answer)

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
```

## 5. Cost Optimization Strategies

### Overview

### Prompt Optimization

### Semantic Caching

### Smart Model Routing

## 6. Monitoring, Alerting, and Debugging

## 7. Production Patterns and Security

