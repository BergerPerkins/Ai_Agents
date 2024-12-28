from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


#web search Agent
web_search_aent = Agent(
    name = 'Web Search Agent',
    role = "search the web for information",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools = [DuckDuckGo(),],
    instructions = ["Always include source"],
    show_tool_calls= True,
    markdown = True,
)


## financial Agent
financial_agent = Agent(
    name = 'Financial AI Agent',
    role = "financial analyst",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                         company_news=True,historical_prices=True)],

    description="You are an investment analyst that researches stock prices, analyst recommendations, company_news, currency_exchange and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)

#Multi Ai_Agent
multi_ai_agent=Agent(
    team=[web_search_aent,financial_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("summarize analyst recomentations and share the latest news for Bitcoin",stream=True)