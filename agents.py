from crewai import Agent
from config import llm
from tools import search_news_db, get_news, search_tool


def create_news_search_agent():
    """Create and return the news search agent."""
    return Agent(
        role='News Searcher',
        goal='Generate key points for each news article from the latest news',
        backstory='Expert in analysing and generating key points from news content for quick updates.',
        tools=[search_news_db.news],
        allow_delegation=True,
        verbose=True,
        llm=llm
    )


def create_writer_agent():
    """Create and return the writer agent."""
    return Agent(
        role='Writer',
        goal='Identify all the topics received. Use the Get News Tool to verify each topic to search. Use the Search tool for detailed exploration of each topic. Summarise the retrieved information in depth for every topic.',
        backstory='Expert in crafting engaging narratives from complex information.',
        tools=[get_news.news, search_tool],
        allow_delegation=True,
        verbose=True,
        llm=llm
    )


# Initialize agents
news_search_agent = create_news_search_agent()
writer_agent = create_writer_agent()