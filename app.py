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

llm.invoke(messages)


"""
Reflection:

1. Why did string-based invocation fail?
2. Why does message-based invocation work?
3. What would break in a production AI system if we ignore message history?
"""

