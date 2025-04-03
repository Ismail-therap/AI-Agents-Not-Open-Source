from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print("Loaded GROQ_API_KEY:", groq_api_key)

if not groq_api_key:
    print("Error: GROQ_API_KEY is not set or loaded.")
    exit()

# Initialize Groq client
client = Groq()

# Attempt to create a completion
try:
    completion = client.chat.completions.create(
        model="Llama-3.3-70b-Versatile",
        messages=[{"role": "user", "content": "Summarize analyst recommendation and share the latest news for NVDA."}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

except Exception as e:
    print("Error occurred:", e)



