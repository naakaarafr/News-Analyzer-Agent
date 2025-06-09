from crewai import Task
from agents import news_search_agent, writer_agent
from tools import search_news_db, get_news, search_tool


def create_news_search_task(topic=None):
    """Create and return the news search task with dynamic topic."""
    if not topic:
        topic = "AI 2024"  # Default fallback
    
    return Task(
        description=f'Search for {topic} and create key points for each news.',
        agent=news_search_agent,
        tools=[search_news_db.news],
        expected_output=f"A comprehensive list of key points from recent news articles about {topic}."
    )


def create_writer_task(news_search_task, topic=None):
    """Create and return the writer task with dynamic topic."""
    if not topic:
        topic = "the specified topic"
    
    return Task(
        description=f"""
        Go step by step.
        Step 1: Identify all the topics received related to {topic}.
        Step 2: Use the Get News Tool to verify each topic by going through one by one.
        Step 3: Use the Search tool to search for information on each topic one by one. 
        Step 4: Go through every topic and write an in-depth summary of the information retrieved.
        Don't skip any topic.
        Focus specifically on: {topic}
        """,
        agent=writer_agent,
        context=[news_search_task],
        tools=[get_news.news, search_tool],
        expected_output=f"An in-depth summary report covering all identified topics related to {topic} with detailed information retrieved from both the news database and web search."
    )


def initialize_tasks(topic):
    """Initialize tasks with the provided topic."""
    news_search_task = create_news_search_task(topic)
    writer_task = create_writer_task(news_search_task, topic)
    return news_search_task, writer_task