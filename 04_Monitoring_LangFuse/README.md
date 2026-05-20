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

Now, let's talk about LLM observability and cost management.

This is a very hot topic and most importantly, it's a very crucial and important topic for

anyone or any organization building or using large-language models or AI in general.

The truth of the matter is you're spending more money on large-language models, probably

more than you think.

Now, here's what most teams discover too late.

Number one, they discover that a single runaway prompt can cost 10,000 in an afternoon. That's scary.

Or they discover that token user spikes 300% and no one knows why.

Or users complain about low responses, but you can't identify the bottleneck.

Or maybe your RAG pipeline retrieves garbage and the LLM hallucinates confidently, which is horrible.

That's really bad for business.

And I've seen teams burn through their entire quarterly budget in two weeks because one

developer left a retry loop running on an accident.

Now this isn't about monitoring for the sake of monitoring.

This is about protecting your budget and of course, your reputation.

Now let's look at real numbers real quick.

So here is a real cost comparison table here.

So monthly LLM spend category without OBS will be $35,000 or more.

Now remember, this is a huge scale operation.

And with OBS will be about $15,000.

So you can see that is a huge reduced cost.

For wasted retries, $5,000 plus without OBS.

With OBS, you have about zero.

And debug time per week, about 20 hours or more without OBS and about two hours if you

are using OBS, if you are managing everything using observability.

Incident response will take about days without OBS and pretty much same day with OBS.

Now the question isn't can we afford observability?

Let me show you a real example of what happens when things go wrong.

So we're going to talk about the hidden costs of LLM applications.

What you need to understand is that what you don't measure, you can't manage.

So this is a saying that is applicable to any industry, essentially.

So we must measure things.

That way, it's easier for us to actually manage them.

The real cost breakdown is as follows.

So we have API costs.

These are tokens.

We have compute costs.

So talking about embeddings inference, we have debugging time.

These are engineering hours.

We also have incident costs, so downtime, reputation, and so on.

Now, costs tend to increase without visibility.

For instance, you find teams saying that token costs grew 400%.

We only found out about that at the end of the month.

So they actually had no idea that the cost had grown 400%.

So this is one thing that I hear from enterprise teams all the time.

Now, let's look at the risk matrix here.

So without observability.

So we have risk column.

We have impact and probability token spike.

The impact without observability is really high, which means usually the cost

is up to 10K plus, depending on the kind of AI application or LM based

applications we're building, and the probability is also, of course, very high.

Silent failures.

The impact is quite medium, not as high,

but the probability for that to happen without observability is high.

And we have performance degradation.

So the impact is fairly medium.

Probability still high.

We have compliance violations.

So the impact is extremely critical, but the probability is medium.

So looking at this matrix here, you can see the correlation between the risk,

the impact, and the probability without observability.

If you're running your LM workflows without observability.

### Traditional Monitoring vs LLM Observability

Keep in mind that observability for LLMs isn't the same as traditional application monitoring.

So you're not just tracking request response times.

What you're doing, you're tracking token flows, you're tracking prompt effectiveness, retrieval

quality, model behavior, and cost attribution.

So these are pieces that you need to keep in mind.

Token flows, for instance, how many tokens in, how many out, what's the ratio, you're

tracking prompt effectiveness, is your prompt getting good results?

That's a good question, right, for prompt effectiveness.

Retrieval quality, is your RAG returning relevant chunks?

Model behavior, is the LLM hallucinating?

Is it being verbose?

Is it refusing to answer?

As well as cost attribution.

So this is where we talk about which feature is eating your budget.

So traditional monitoring looks like this.

So essentially, sees your app as a black box.

So essentially, you have a request, something happens, and we get a response.

And in this black box, we're looking at latency and errors that may happen, but all is contained.

But if we look at LLM observability, we have a full traceability of what's going on.

We can look at the token flows, because these are being tracked.

We can look at prompt effectiveness, as we talked about, as well as the cost attribution and so on.

Now, this visibility isn't optional in production, because this is how we're able to debug issues

in minutes instead of hours, we're able to, say, optimize costs by 50 to 80%.

We're also able to start catching problems before users report them to us, right.

And also, it proves to leadership exactly where the budget goes.

That way, you don't have the friction between the leadership and the managers and the operation

layer of your business or your company.

### The Three Pillars of LLM Observability

LLM observability requires us to look at the three pillars for LLMs.

The first one is traces.

This is end-to-end request flows through every step.

So essentially, going deep and look at what's going on with our LLMs.

And we have the metric side of things.

So this is the part where we look at token usage, latency, cost per request, and evaluation.

So evaluating the performance of our large knowledge models or large knowledge model

systems is crucial because this is where we get the quality scores and output assessment.

So we know that things are working as intended.

Now what you'll measure in this case.

So the first thing you'll measure, of course, is token usage.

So input-output ratio.

You also measure the latency per step breakdown, right, how fast things are actually working or slow.

And the cost is very important, but the cost, you can subdivide it per request, per feature, or per user.

So you have all those breakdowns, as well as the quality, because it's important to

see that things are coming in or coming out with relevancy, and then what is the hallucination

rate of the large knowledge model.

And we also have the errors.

Very importantly, we wanted to also separate them by type, by model, by prompt, and many

other categories we may need.

### ROI Calculation

So now I'm going to show you this ROI calculator or calculations to make the

business case. Okay, so in this case here, let's say current state without

observability, your monthly LLM spend is $20,000.

Your estimated waste about 30%, so that's 60 or 6,000 I should say. Your debug

time, let's say 10 hours per week times 100, that's 4,000 a month. And your

incident cost amortized, of course, is $5,000. So that means total preventable

cost is $15,000. So if you add the incident cost, debug time, estimated waste,

that's $15,000. If you were to go and make some investment on an

observability, LLM observability platform, cost $500 to $2,000 a month, and the

setup time would be about eight hours, and this is just one time, that is

awesome because then you have 7 to 30% return on your ROI. This is only in the

first month, so this potentially results in 47% token cost reduction. All of that

from prompt optimization as an example. And 80% less debugging time with proper

tracing, of course, because you have access to that. And then $0 in

runway surprises with cost alerts. This isn't theoretical. So here's the ROI

formula that I would suggest you look into or use in your own organizations. So

savings is equal to token waste plus debug time times rate plus the incidents

prevented times cost. So this will give you the savings that you would get from

implementing a good observability system.
