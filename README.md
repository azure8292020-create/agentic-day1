# Day 1 Homework: Agentic AI Enterprise Mastery Bootcamp

This repository contains the Day 1 Homework for the Agentic AI Enterprise Mastery Bootcamp. The objective is to demonstrate the difference between naïve LLM invocations (which lose context) and message-based invocations using LangChain's Messages API.

## Project Structure
- `app.py`: Main implementation demonstrating the context break and fix.
- `.env`: Environment variables (API keys) - **Not committed for security**.
- `.gitignore`: Ensures sensitive files are not pushed to GitHub.
- `requirements.txt` / `pyproject.toml`: Project dependencies.

## Setup Instructions
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or using `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file and add your `NVIDIA_API_KEY`:
   ```env
   NVIDIA_API_KEY=your_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Implementation Details
The `app.py` script is divided into three parts:
1. **Part 1: Naïve Invocation**: Shows how independent calls to the LLM fail to maintain context.
2. **Part 2: Messages API Fix**: Uses `SystemMessage` and `HumanMessage` to maintain a conversation history.
3. **Reflection**: Answers key questions about why context management is critical for production AI systems.
