from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

agent = Agent(
    model = Groq(id="qwen-2.5-32b"),
    description="you are an AI  assistant to me please respond to me based on my queries",
    tools=[DuckDuckGoTools()],
    markdown=True
)
agent.print_response("What are the new ai agents released today? and what are more towards AGI?")