from crewai import Task
from textwrap import dedent

"""
Creating Tasks Cheat Sheet:
- Begin with the end in mind. Identify the specific outcome your tasks are aiming to achieve.
- Break down the outcome into actionable tasks, assigning each task to the appropriate agent.
- Ensure tasks are descriptive, providing clear instructions and expected deliverables.

Goal:
- Develop a detailed itinerary, including city selection, attractions, and practical travel advice.

Key Steps for Task Creation:
1. Identify the Desired Outcome: Define what success looks like for your project.
    - A detailed 7 day travel itenerary.

2. Task Breakdown: Divide the goal into smaller, manageable tasks that agents can execute.
    - Itenerary Planning: develop a detailed plan for each day of the trip.
    - City Selection: Analayze and pick the best cities to visit.
    - Local Tour Guide: Find a local expert to provide insights and recommendations.

3. Assign Tasks to Agents: Match tasks with agents based on their roles and expertise.

4. Task Description Template:
  - Use this template as a guide to define each task in your CrewAI application. 
  - This template helps ensure that each task is clearly defined, actionable, and aligned with the specific goals of your project.

  Template:
  ---------
  def [task_name](self, agent, [parameters]):
      return Task(description=dedent(f'''
      **Task**: [Provide a concise name or summary of the task.]
      **Description**: [Detailed description of what the agent is expected to do, including actionable steps and expected outcomes. This should be clear and direct, outlining the specific actions required to complete the task.]

      **Parameters**: 
      - [Parameter 1]: [Description]
      - [Parameter 2]: [Description]
      ... [Add more parameters as needed.]

      **Note**: [Optional section for incentives or encouragement for high-quality work. This can include tips, additional context, or motivations to encourage agents to deliver their best work.]

      '''), agent=agent)

"""


class ResearchTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def gather_historical_financial_information(self, agent, symbol):
        return Task(
            description=dedent(
                f"""
            **Task**: Gather historical financial information on the stock with the symbol {symbol}.
            **Description**: The agent should gather historical financial information from the most 
            recent quarterly earnings reports, financial news and expert opinions in the last 6 quarters. 
            The result should be a written report of maximum 150 words.

            **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )

    def gather_current_financial_information(self, agent, symbol):
        return Task(
            description=dedent(
                f"""
                    **Task**:  Gather the latest information on the stock with the symbol {symbol}.
                    **Description**: Analyze the last available quarterly earnings reports, financial 
                    news and expert opinions of the given stock listed company and provide a detailed analysis 
                    about potential deviations from the upcoming estimated earnings per share number. 
                    The result should be a written report of maximum 150 words.

                    **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )

    def gather_economic_data(self, agent, symbol):
        return Task(
            description=dedent(
                f"""
                    **Task**:  Gather economic data related to the stock with the symbol {symbol}.
                    **Description**: Analyze the stock listed company and provide a detailed analysis 
                    about how different economic indicators and KPIs could influence the upcoming 
                    estimated earnings per share number. 
                    The result should be a written report of maximum 150 words.

                    **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
        )

    def write_newsletter(self, agent, context: list):
        return Task(
            description=dedent(
                f"""
                    **Task**:  Write the newsletter by summarizing all reports.
                    **Description**: Gather the reports from the Senior Financial Analyst, 
                    Senior Equity Research Analyst, Senior Economist and create a 250 word article for a newsletter.

                    **Note**: {self.__tip_section()}
        """
            ),
            agent=agent,
            context=context
        )