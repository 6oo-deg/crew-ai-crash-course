from crewai import Crew, Process
from textwrap import dedent
from agents import FinancialAgents
from tasks import ResearchTasks

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

today = datetime.today().strftime('%Y-%m-%d')

class FinancialCrew:
    def __init__(self, date, symbol):
        self.date = date
        self.symbol = symbol

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = FinancialAgents()
        tasks = ResearchTasks()

        # Define your custom agents and tasks here
        head_equity_research = agents.head_equity_research()
        financial_analyst = agents.financial_analyst(today)
        equity_research_analyst = agents.equity_research_analyst(today)
        economist = agents.economist(today)
        writer = agents.financial_writer()

        # Custom tasks include agent name and variables as input
        gather_info_hist = tasks.gather_historical_financial_information(
            financial_analyst,
            self.symbol
        )

        gather_info_curr = tasks.gather_current_financial_information(
            equity_research_analyst,
            self.symbol
        )

        gather_info_econ = tasks.gather_economic_data(
            economist,
            self.symbol
        )

        write_nl = tasks.write_newsletter(
            writer,
            [gather_info_hist, gather_info_curr, gather_info_econ]
        )


        # Define your custom crew here
        crew = Crew(
            agents=[head_equity_research,
                    financial_analyst,
                    equity_research_analyst,
                    economist,
                    writer
                    ],
            tasks=[
                gather_info_hist,
                gather_info_curr,
                gather_info_econ,
                write_nl
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to the Financial Crew AI Newsletter")
    print('-------------------------------')
    symbol = input(
        dedent("""
      Which company would you like to write about?
    """))

    fin_crew = FinancialCrew(today, symbol)
    result = fin_crew.run()
    print("\n\n########################")
    print("## Here is your Newsletter")
    print("########################\n")
    print(result)
