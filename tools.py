import requests
import os
import time
import logging
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import GoogleSerperRun
from config import embedding_function, NEWSAPI_KEY, NEWS_API_BASE_URL, NEWS_API_PARAMS, CHROMA_DB_PATH, SERPER_API_KEY

# Configure logging
logger = logging.getLogger(__name__)


class RateLimitedTool:
    """Base class for tools that need rate limiting."""
    
    def __init__(self, requests_per_minute=3):  # Even more conservative
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def _wait_for_rate_limit(self):
        """Wait if we're approaching rate limits."""
        current_time = datetime.now()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - timedelta(minutes=1)
        self.request_times = [req_time for req_time in self.request_times if req_time > cutoff_time]
        
        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = min(self.request_times)
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - current_time).total_seconds()
            
            if wait_seconds > 0:
                logger.info(f"🕒 Tool rate limiting: waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds + 3)  # Increased buffer
        
        # Always add a small delay between tool calls
        time.sleep(2)
        
        # Record this request
        self.request_times.append(datetime.now())


class SearchNewsDB(RateLimitedTool):
    def __init__(self):
        super().__init__(requests_per_minute=2)  # Ultra conservative - only 2 per minute
    
    @tool("News DB Tool")
    def news(query: str):
        """Fetch news articles and process their contents with rate limiting."""
        
        # Create instance to access rate limiting
        instance = SearchNewsDB()
        instance._wait_for_rate_limit()
        
        params = {
            **NEWS_API_PARAMS,
            'q': query,
            'apiKey': NEWSAPI_KEY,
        }
        
        try:
            logger.info(f"📰 Fetching news for query: {query}")
            response = requests.get(NEWS_API_BASE_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"News API returned status code: {response.status_code}")
                return f"Failed to retrieve news. Status code: {response.status_code}"
            
            articles = response.json().get('articles', [])
            
            if not articles:
                return "No articles found for the given query."
            
            all_splits = []
            processed_count = 0
            
            # Process only 1 article to minimize API calls
            for i, article in enumerate(articles[:1]):
                try:
                    # Check if URL is valid
                    if not article.get('url'):
                        continue
                        
                    logger.info(f"📄 Processing article: {article['title'][:50]}...")
                    
                    # Add longer delay between operations
                    if processed_count > 0:
                        logger.info("⏳ Waiting 10 seconds between articles...")
                        time.sleep(10)
                    
                    # Load content from article URL
                    loader = WebBaseLoader(article['url'])
                    docs = loader.load()

                    # Split the documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600,  # Even smaller chunks
                        chunk_overlap=50
                    )
                    splits = text_splitter.split_documents(docs)
                    all_splits.extend(splits)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing article {article.get('url', 'Unknown URL')}: {str(e)}")
                    continue

            # Index the accumulated content splits if there are any
            if all_splits:
                try:
                    logger.info("📚 Creating vector store...")
                    # Add longer delay before creating vector store (uses embedding API)
                    time.sleep(5)
                    
                    # Create vector store - the embedding function now handles its own rate limiting
                    vectorstore = Chroma.from_documents(
                        all_splits[:5],  # Limit to 5 chunks to reduce embedding calls
                        embedding=embedding_function, 
                        persist_directory=CHROMA_DB_PATH
                    )
                    
                    # Add delay before similarity search
                    time.sleep(3)
                    retriever = vectorstore.similarity_search(query, k=2)  # Reduced from 3 to 2
                    
                    # Format the results for better readability
                    formatted_results = []
                    for doc in retriever:
                        formatted_results.append({
                            'content': doc.page_content[:300],  # Further reduced content length
                            'metadata': doc.metadata
                        })
                    
                    result = f"Successfully processed {processed_count} article(s). Found {len(retriever)} relevant chunks:\n\n" + \
                           "\n---\n".join([f"Content: {r['content']}\nSource: {r['metadata'].get('source', 'Unknown')}" 
                                         for r in formatted_results])
                    
                    logger.info(f"✅ News DB search completed successfully")
                    return result
                    
                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return f"Error creating vector store: {str(e)}"
            else:
                return f"Processed {processed_count} article(s) but no content was available for indexing."
                
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return f"Error fetching news: {str(e)}"


class GetNews(RateLimitedTool):
    def __init__(self):
        super().__init__(requests_per_minute=6)  # Slightly higher for local DB queries
    
    @tool("Get News Tool")
    def news(query: str) -> str:
        """Search Chroma DB for relevant news information based on a query with rate limiting."""
        
        # Create instance to access rate limiting
        instance = GetNews()
        instance._wait_for_rate_limit()
        
        try:
            # Check if the database exists
            if not os.path.exists(CHROMA_DB_PATH):
                return "No news database found. Please run the News DB Tool first to populate the database."
            
            logger.info(f"Searching news database for: {query}")
            
            # Add delay for embedding API calls
            time.sleep(1)
            
            # Create vector store - embedding function handles its own rate limiting
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH, 
                embedding_function=embedding_function
            )
            
            # Add delay before similarity search
            time.sleep(1)
            retriever = vectorstore.similarity_search(query, k=3)  # Reduced from 5 to 3
            
            if not retriever:
                return f"No relevant news found for query: {query}"
            
            # Format the results
            formatted_results = []
            for doc in retriever:
                formatted_results.append({
                    'content': doc.page_content[:400],  # Reduced content length
                    'metadata': doc.metadata
                })
            
            result = f"Found {len(retriever)} relevant articles:\n\n" + \
                   "\n---\n".join([f"Content: {r['content']}\nSource: {r['metadata'].get('source', 'Unknown')}" 
                                 for r in formatted_results])
            
            logger.info("News database search completed successfully")
            return result
                   
        except Exception as e:
            logger.error(f"Error retrieving news from database: {str(e)}")
            return f"Error retrieving news from database: {str(e)}"


# Enhanced Serper search tool with rate limiting
class RateLimitedSerperTool(RateLimitedTool):
    def __init__(self):
        super().__init__(requests_per_minute=5)  # Conservative limit for Serper API
        
        if SERPER_API_KEY:
            serper_key = str(SERPER_API_KEY).strip()
            os.environ["SERPER_API_KEY"] = serper_key
            
            try:
                self.serper_wrapper = GoogleSerperAPIWrapper()
                self.search_enabled = True
                logger.info("Serper search tool initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Serper tool: {e}")
                self.search_enabled = False
        else:
            logger.warning("SERPER_API_KEY not found, search tool will not be available")
            self.search_enabled = False
    
    def search(self, query: str) -> str:
        """Perform web search with rate limiting."""
        if not self.search_enabled:
            return "Serper API key not configured. Please set SERPER_API_KEY environment variable."
        
        try:
            logger.info(f"Performing web search for: {query}")
            self._wait_for_rate_limit()
            
            # Add delay before API call
            time.sleep(2)
            
            result = self.serper_wrapper.run(query)
            logger.info("Web search completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in Serper search: {str(e)}")
            return f"Search error: {str(e)}"


# Initialize rate-limited Serper tool
serper_tool_instance = RateLimitedSerperTool()

@tool("Search Tool")
def search_tool(query: str) -> str:
    """Enhanced search tool with automatic rate limiting."""
    return serper_tool_instance.search(query)

# Initialize tool instances
search_news_db = SearchNewsDB()
get_news = GetNews()