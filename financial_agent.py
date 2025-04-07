from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch

# Set the OpenAI API key
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print("Loaded GROQ_API_KEY:", groq_api_key)

if not groq_api_key:
    print("Error: GROQ_API_KEY is not set or loaded.")
    exit()

# Web search agent
# web_search_agent = Agent(
#     name="Web Search Agent",
#     role="Search the Web for the information",
#     model=Groq(id="deepseek-r1-distill-llama-70b"),  # Ensure this model ID is valid
#     tools=[DuckDuckGo()],
#     instructions=["Always include sources."],
#     show_tools_calls=True,
#     markdown=True,
# )


Google_search_agent = Agent(
    tools=[GoogleSearch()],
    description="You are a news agent that helps users find the latest news about the stock.",
    instructions=[
        "Given a topic by the user, respond with 10 latest news items about that topic.",
        "Search for 10 news items and select the top 10 unique items.",
        "Search in English.",
        "Give me the sentiment of the each news in 5 star scale, worst get 1 star.",
        "Pass the stock name to the financial agent.",
        "Use tables to display the data."
    ],
     model=Groq(id="Llama-3.3-70b-Versatile"),
    show_tool_calls=True,
    debug_mode=True,
)

# Financial agent
financial_agent = Agent(
    name="Financial AI Agent",
    role="Analyze financial data",
    model=Groq(id="deepseek-r1-distill-llama-70b"),  # Ensure this model ID is valid
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data."],
    show_tools_calls=True,
    markdown=True,
)

# Multi-agent setup
multi_ai_agent = Agent(
    team=[Google_search_agent, financial_agent],
    model=Groq(id="Llama-3.3-70b-Versatile"),  # Ensure this model ID is valid
    instructions=["Always include sources.", "Use table to display data."],
    show_tools_calls=True,
    markdown=True,
)

# Run the multi-agent and handle errors
try:
    multi_ai_agent.print_response(
        "Which stock better MSFT or Walmart?.",
        stream=True,
    )
    print("\nSuccess!")
except Exception as e:
    print("Error occurred:", e)