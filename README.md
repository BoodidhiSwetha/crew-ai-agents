# CrewAI + Groq SEC Sentiment Project

This project uses **CrewAI + Groq (via LiteLLM)** to:
- Fetch SEC insider trading activity from the last 48 hours
- Summarize activity using CrewAI agents with guardrails
- Perform sentiment analysis on X (Twitter) creators
- Generate a daily report

## Setup

1. Clone repo or unzip folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your **Groq API key** to a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the project:
   ```bash
   python main.py
   ```
    