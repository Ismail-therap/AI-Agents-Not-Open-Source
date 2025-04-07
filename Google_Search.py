
from phi.model.groq import Groq
from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch


# Set the OpenAI API key
import os
from dotenv import load_dotenv
load_dotenv()

Google_search_agent = Agent(
    tools=[GoogleSearch()],
    description="You are a news agent that helps users find the latest news.",
    instructions=[
        "Given a topic by the user, explore 10 latest news items about that topic.",
        "Follow the instructions given by the user.",
        "Use tables to display the data.",
         "Put the latest news you found at the bottom about the topic also put the weblink."
    ],
     model=Groq(id="Llama-3.3-70b-Versatile"),
    show_tool_calls=True,
    debug_mode=True,
)
Google_search_agent.print_response("future of MSFT", markdown=True)  # Example topic to search for news about
