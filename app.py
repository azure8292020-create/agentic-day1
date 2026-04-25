from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- PART 1: Naïve Invocation (Context Break) ---
# This demonstrates how independent calls lose context because the model doesn't "remember" previous interactions.
print("Running Part 1: Naïve Invocation...")
resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
def print_content(content):
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                print(item['text'])
            else:
                print(item)
    else:
        print(content)

print_content(resp1.content)

resp2 = llm.invoke("What are the main risks in this system?")
# Note: This second call might give a generic answer or ask for clarification 
# because it doesn't know "this system" refers to the medical insurance system from the previous call.
print(f"Response 2: {resp2.content}")
print("-" * 30)

# --- PART 2: Messages API (Context Fix) ---
# This uses the Messages API to provide the full conversation history, ensuring context is maintained.
print("Running Part 2: Messages API...")
messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

print("Connecting to Gemini and processing your request...")
response = llm.invoke(messages)
print("\n--- ARCHITECT REVIEW ---")
print_content(response.content)


"""
Reflection:

1. Why did string-based invocation fail?
   - String-based invocation treats each call as a completely new session. The LLM is stateless by nature, so without passing the previous conversation back to it, it has no memory of what was discussed in the first call.

2. Why does message-based invocation work?
   - By passing a list of messages (SystemMessage, HumanMessage, AIMessage), we are essentially providing the LLM with its "memory" for that specific interaction. The model sees the entire history and can understand references like "this system".

3. What would break in a production AI system if we ignore message history?
   - Any multi-turn interaction would fail (e.g., a chatbot wouldn't remember your name or the problem you're trying to solve).
   - Context-dependent tasks (like "summarize the above" or "fix the bug in that code") would be impossible because the model wouldn't have access to the "above" or "that code".
   - Personalization and complex reasoning across steps would be lost.
"""



