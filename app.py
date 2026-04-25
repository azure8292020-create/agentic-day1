from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize the NVIDIA model
# You can change the model to any supported NVIDIA NIM model
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY")
)

# response1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
# print(f"Response 1: {response1.content}")

# response2 = llm.invoke("What are the main risks in this system?")
# print(f"Response 2: {response2.content}")

messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

print("Connecting to NVIDIA Inference Service and processing your request... (this may take a minute)")
response = llm.invoke(messages)
print("\n--- ARCHITECT REVIEW ---")
print(response.content, flush=True)



"""
Reflection:

1. Why did string-based invocation fail?
   - String-based invocation doesn't "fail" technically (it is treated as a single HumanMessage), but it lacks context management. If you call it multiple times separately, each call is a fresh start, so the model won't remember previous messages in the conversation.

2. Why does message-based invocation work?
   - Message-based invocation allows you to pass a list of historical messages (System, Human, AI). This gives the model the full context of the conversation and instructions (SystemMessage), allowing for more coherent and context-aware responses.

3. What would break in a production AI system if we ignore message history?
   - Chatbot functionality would break as the system would have "amnesia" (forgetting user names, previous questions, or current tasks).
   - Multi-step reasoning would be impossible.
   - Contextual disambiguation (e.g., "Tell me more about that") would fail because "that" would have no reference point.
"""


