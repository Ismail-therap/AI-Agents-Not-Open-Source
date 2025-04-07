from phi.agent import Agent
import phi.api

from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")




# ## This is my web search agent
# web_search_agent = Agent(name = "Web Search Agent",
#                          role = "Search the Web for the information.",
#                          description = "This agent can search the web for information.",
#                          model = Groq(id = "Llama-3.3-70b-Versatile"),
#                           tools = [DuckDuckGo()],
#                           instructions = ["Always include sources."],
#                           show_tools_calls = True,
#                           markdown = True,
#                           debug_mode=True)

Google_search_agent = Agent(
    tools=[GoogleSearch()],
    description="An intelligent news aggregator that retrieves the latest news articles related to stocks, enhancing decision-making by providing recent developments and trends.",
    instructions=[
        "1. Receive a stock name or topic from the user and perform a web search to retrieve the 10 latest news articles related to the topic or stock.",
        "2. Extract and present the headline, summary, source, and publication date for each article.",
        "3. If multiple stock names are mentioned, gather news for each stock and organize them clearly.",
        "4. Provide the website link for each article.",
        "5. Pass relevant stock names to the Financial Agent for further analysis.",
        "6. Do sentiment analysis of the news (Positive, Negetive or Neutral)."
    ],
    model=Groq(id="Llama-3.3-70b-Versatile"),
    show_tool_calls=True,
    debug_mode=True,
)



## Financial agent
financial_agent = Agent(
    name="Financial AI Agent",
    role="Financial Analyst & Data Scientist",
    description="An intelligent financial agent that performs stock analysis using financial data, statistical modeling, and AI-driven insights.",
    model=Groq(id="Llama-3.3-70b-Versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=[
        "1. Receive stock names from the Google Search Agent or user input and retrieve relevant financial data.",
        "2. Display stock price trends, analyst recommendations, stock fundamentals, and recent company news using tables for clarity.",
        "3. If multiple stocks are mentioned, perform Correlation Heatmap Analysis to identify relationships between stocks.",
        "4. Conduct Volatility Analysis to assess short-term and long-term investment potential.",
        "5. Apply ARIMA Time Series Forecasting for predicting future stock prices when requested.",
        "6. Provide AI-driven insights with clear explanations and visualizations to support decision-making.",
        "7. Reference the website links provided by the Google Search Agent for credibility.",
        "8. Ensure results are displayed in a structured format with markdown for better readability."
    ],
    show_tools_calls=True,
    markdown=True,
    debug_mode=True
)



app = Playground(agents = [financial_agent,Google_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload = True)  # Run the app with reloading enabled