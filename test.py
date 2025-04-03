print("Start")
from phi.agent import Agent
print("1")
from phi.model.groq import Groq
print("2")
from phi.tools.yfinance import YFinanceTools
print("3")
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

print("Loading environment variables...")


# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print("Loaded GROQ_API_KEY:", groq_api_key)

if not groq_api_key:
    print("Error: GROQ_API_KEY is not set or loaded.")
    exit()