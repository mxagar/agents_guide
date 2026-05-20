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

