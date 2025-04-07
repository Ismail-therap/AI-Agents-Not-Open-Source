from phi.agent import Agent
import phi.api

from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")




## This is my web search agent
web_search_agent = Agent(name = "Web Search Agent",
                         role = "Search the Web for the information",
                         model = Groq(id = "deepseek-r1-distill-llama-70b"),
                          tools = [DuckDuckGo()],
                          instructions = ["Always include sources."],
                          show_tools_calls = True,
                          markdown = True)


## Financial agent
financial_agent = Agent(name = "Financial AI Agent",
                         role = "Analyze financial data",
                         model = Groq(id = "deepseek-r1-distill-llama-70b"),
                         tools = [YFinanceTools(stock_price = True,
                                                analyst_recommendations = True,
                                                stock_fundamentals = True,
                                                company_news = True)],
                         instructions = ["Use tables to display the data."],
                         show_tools_calls = True,
                         markdown = True)


app = Playground(agents = [financial_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload = True)  # Run the app with reloading enabled