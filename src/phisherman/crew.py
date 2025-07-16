from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.gmail_comprehensive_tool import GmailAnalysisTool
from tools.checkphish_tool import CheckPhishTool
from tools.gemini_tool import GeminiAnalysisTool
from tools.report_tool import ReportGeneratorTool
import os
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Phisherman():
    """Phisherman crew"""

    # agents: List[BaseAgent]
    # tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.gmail_creds = os.getenv("GMAIL_API_KEY")
        self.checkphish_api = os.getenv("CHECKPHISH_API_KEY")
        self.gemini_api = os.getenv("GEMINI_API_KEY")

    @agent
    def email_fetch_parse_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['email_fetch_parse_agent'], # type: ignore[index]
            verbose=True,
            tools=[
                GmailAnalysisTool(
                    creds_path = self.gmail_creds,
                    checkphish_api_key = self.checkphish_api
                )
            ]
        )

    @agent
    def llm_analyser_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['llm_analyser_agent'], # type: ignore[index]
            verbose=True,
            tools=[
                GeminiAnalysisTool(api_key=self.gemini_api)
            ]
        )
    
    @agent
    def security_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['security_advisor_agent'],
            verbose=True,
            tools=[
                ReportGeneratorTool()
            ]
        )

    # Tasks START :)

    @task
    def email_fetch_and_parse_task(self) -> Task:
        return Task(
            config=self.tasks_config['email_fetch_and_parse_task'],
             agent=self.email_fetch_parse_agent() # type: ignore[index]
        )

    @task
    def content_analysis_via_llms_task(self) -> Task:
        return Task(
            config=self.tasks_config['content_analysis_via_llms_task'], # type: ignore[index]
            agent=self.llm_analyser_agent(),
            context=[self.email_fetch_and_parse_task()]
        )
    
    @task
    def security_expert_recommendation_task(self) -> Task:
        return Task(
            config=self.tasks_config['security_expert_recommendation_task'], # type: ignore[index]
            agent=self.security_advisor_agent(),
            context=[self.email_fetch_and_parse_task(), self.content_analysis_via_llms_task()]

        )

    @crew
    def crew(self) -> Crew:
        """Creates the Phisherman crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/embedding-001",
                    "api_key": self.gemini_api
                }
            }
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
