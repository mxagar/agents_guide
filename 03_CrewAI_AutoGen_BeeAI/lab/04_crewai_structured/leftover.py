from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="openai/gpt-4o", temperature=0.2)
search_tool = TavilySearchTool()


@CrewBase
class LeftoversCrew:
    """YAML-backed crew components for leftover management."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def leftover_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["leftover_manager"],  # type: ignore[index]
            tools=[search_tool],
            llm=llm,
            verbose=False,
        )

    @task
    def leftover_task(self) -> Task:
        return Task(
            config=self.tasks_config["leftover_task"],  # type: ignore[index]
            agent=self.leftover_manager(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
