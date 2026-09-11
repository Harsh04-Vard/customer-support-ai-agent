import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

# Create Groq client
client = Groq(api_key=api_key)

# Make a simple request
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "In one sentence, explain what a customer support AI agent does."
        }
    ],
)

print("\nGROQ RESPONSE:")
print(response.choices[0].message.content)