# SQL Agent

This project builds a LangChain SQL agent for the [Chinook Database](https://docs.yugabyte.com/stable/sample-data/chinook/). The agent receives natural-language questions, generates SQL, executes the query against MySQL, and returns a human-readable answer.

![Chinook ERD](./assets/chinook_erd.png)

The original lab instructions are in [`Instructions.pdf`](./Instructions.pdf). This implementation keeps the same project goal but updates the stack:

* Uses OpenAI chat models with `langchain-openai` instead of IBM/Watson models.
* Loads secrets and connection settings from the repository-level `.env` file.
* Uses the repository-level [`requirements.in`](../../../requirements.in) instead of creating a separate project virtual environment.
* Uses the already-downloaded [`chinook_mysql.sql`](./chinook_mysql.sql) file.
* Adds [`07-test-sql-agent-connection.ipynb`](./07-test-sql-agent-connection.ipynb) to test the connection step by step.

## Files

| File | Purpose |
| --- | --- |
| [`chinook_mysql.sql`](./chinook_mysql.sql) | Creates and populates the MySQL `Chinook` database. |
| [`llm_agent.py`](./llm_agent.py) | Smoke test for the OpenAI chat model. |
| [`sql_agent.py`](./sql_agent.py) | CLI app that creates the LangChain SQL agent and runs natural-language prompts. |
| [`07-test-sql-agent-connection.ipynb`](./07-test-sql-agent-connection.ipynb) | Notebook for testing `.env`, OpenAI, MySQL, direct SQL, and the SQL agent. |
| [`Instructions.pdf`](./Instructions.pdf) | Original project instructions. |

## Requirements

Use the existing environment for this repository. The conda environment is expected to have the packages from [`requirements.in`](../../../requirements.in).

If you need to refresh the environment, run this from the repository root:

```bash
conda activate agents
python -m pip install -r requirements.in
```

The important packages for this project are:

* `langchain`
* `langchain-classic`
* `langchain-community`
* `langchain-openai`
* `mysql-connector-python`
* `python-dotenv`
* `sqlalchemy`
* `jupyter`

## Environment Variables

Create or update the repository-level `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=512

MYSQL_USERNAME=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=Chinook
```

Only `OPENAI_API_KEY` is strictly required by the OpenAI model. The MySQL values default to `root`, empty password, `localhost`, `3306`, and `Chinook` if omitted.

For safer real-world use, create a read-only MySQL user for the agent instead of connecting as `root`.

## MySQL Setup

### macOS

Install MySQL with Homebrew:

```bash
brew install mysql
brew services start mysql
mysql_secure_installation
```

Open the MySQL CLI:

```bash
mysql -u root -p
```

Useful UI options on macOS:

* MySQL Workbench: `brew install --cask mysqlworkbench`
* DBeaver Community: `brew install --cask dbeaver-community`
* TablePlus or Sequel Ace if you prefer a lighter local UI.

### Windows

Install MySQL with the official MySQL Installer:

1. Download MySQL Installer from <https://dev.mysql.com/downloads/installer/>.
2. Choose MySQL Server and MySQL Workbench during installation.
3. Set a root password and keep the server running as a Windows service.
4. Open "MySQL Command Line Client" from the Start menu, or use PowerShell if MySQL is on your `PATH`.

Open the MySQL CLI from PowerShell:

```powershell
mysql -u root -p
```

Useful UI options on Windows:

* MySQL Workbench from the official installer.
* DBeaver Community from <https://dbeaver.io/download/>.

## Load the Chinook Database

From the MySQL CLI, source the SQL file:

```sql
SOURCE /absolute/path/to/agents_guide/01_Fundamentals/lab/07_sql_agent_project/chinook_mysql.sql;
SHOW DATABASES;
USE Chinook;
SELECT COUNT(*) FROM Album;
```

The album count should be `347`.

If your terminal is already in `01_Fundamentals/lab/07_sql_agent_project` when you open `mysql`, this shorter command also works:

```sql
SOURCE chinook_mysql.sql;
```

## Test the OpenAI Model

Run the model smoke test:

```bash
conda activate agents
cd 01_Fundamentals/lab/07_sql_agent_project
python llm_agent.py
```

Expected behavior: the script returns a short answer to a simple question. If it fails, check that `.env` exists at the repository root and contains `OPENAI_API_KEY`.

## Test the MySQL Connection

Use the notebook for a guided check:

```bash
conda activate agents
jupyter lab 01_Fundamentals/lab/07_sql_agent_project/07-test-sql-agent-connection.ipynb
```

The notebook checks:

* `.env` loading.
* OpenAI model access.
* MySQL connection with `SQLDatabase.from_uri`.
* Direct SQL queries against `Album` and `Artist`.
* A final SQL agent question: "How many albums are in the database?"

## Run the SQL Agent

Default prompt:

```bash
conda activate agents
cd 01_Fundamentals/lab/07_sql_agent_project
python sql_agent.py
```

Custom prompt:

```bash
python sql_agent.py --prompt "How many employees are there?"
python sql_agent.py --prompt "Describe the PlaylistTrack table"
python sql_agent.py --prompt "Can you left join table Artist and table Album by ArtistId? Please show me 5 Name and AlbumId in the joined table."
python sql_agent.py --prompt "Which country's customers spent the most by invoice?"
```

Disable verbose LangChain tracing:

```bash
python sql_agent.py --quiet --prompt "How many albums are in the database?"
```

## How It Works

1. `llm_agent.py` loads the repo-level `.env` file and creates a `ChatOpenAI` model.
2. `sql_agent.py` builds a MySQL SQLAlchemy URI from `MYSQL_*` environment variables.
3. `SQLDatabase.from_uri(...)` wraps the Chinook database for LangChain.
4. `create_sql_agent(...)` gives the model SQL tools for schema inspection and query execution.
5. `agent_executor.invoke({"input": prompt})` lets the agent translate natural language into SQL, execute it, recover from query errors when possible, and format the answer.

## Safety Notes

SQL agents execute model-generated SQL. Keep permissions narrow:

* Prefer a read-only database user.
* Use a local training database, not production data.
* Review verbose traces to understand generated SQL.
* Validate important answers with direct SQL queries.
