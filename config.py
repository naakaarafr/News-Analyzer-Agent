import os
import time
import logging
import re
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys and Configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Ensure Google API key is a string and properly formatted
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Convert to string and strip any whitespace
GOOGLE_API_KEY = str(GOOGLE_API_KEY).strip()

# Set environment variable explicitly to avoid SecretStr issues
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class QuotaAwareLLM(ChatGoogleGenerativeAI):
    """
    Custom LLM class that extends ChatGoogleGenerativeAI with intelligent quota handling.
    This bypasses LangChain's retry mechanism and implements our own.
    """
    
    def __init__(self, **kwargs):
        # Set max_retries to 0 to disable LangChain's built-in retry
        kwargs['max_retries'] = 0
        kwargs['request_timeout'] = 120
        
        super().__init__(**kwargs)
        
        # Initialize custom attributes after super().__init__()
        object.__setattr__(self, 'requests_per_minute', 6)  # Very conservative
        object.__setattr__(self, 'request_times', [])
        object.__setattr__(self, 'max_quota_retries', 5)
        object.__setattr__(self, 'system_instruction', "You are a helpful AI assistant that can analyze news content and generate comprehensive reports.")
    
    def _wait_for_rate_limit(self):
        """Proactive rate limiting based on request history."""
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
                logger.warning(f"🕒 Proactive rate limiting: waiting {wait_seconds:.1f} seconds to avoid quota...")
                time.sleep(wait_seconds + 3)
        
        # Record this request
        self.request_times.append(datetime.now())
    
    def _extract_retry_delay(self, error_message: str) -> Optional[int]:
        """Extract retry delay from Google API error message."""
        try:
            # Look for retry_delay { seconds: X } pattern
            retry_pattern = r'retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}'
            match = re.search(retry_pattern, error_message)
            
            if match:
                return int(match.group(1))
            
            # Alternative patterns
            seconds_patterns = [
                r'seconds:\s*(\d+)',
                r'retry.*?(\d+).*?second',
                r'wait.*?(\d+).*?second'
            ]
            
            for pattern in seconds_patterns:
                match = re.search(pattern, error_message, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                    
        except Exception as e:
            logger.warning(f"Failed to extract retry delay: {e}")
        
        return None
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is quota/rate limit related."""
        error_str = str(error).lower()
        quota_indicators = [
            '429',
            'quota',
            'rate limit',
            'resourceexhausted',
            'exceeded your current quota',
            'rate-limits',
            'too many requests'
        ]
        
        return any(indicator in error_str for indicator in quota_indicators)
    
    def _handle_quota_error(self, error: Exception) -> int:
        """Handle quota error and return wait time."""
        error_str = str(error)
        retry_delay = self._extract_retry_delay(error_str)
        
        if retry_delay:
            wait_time = retry_delay + 10  # API delay + buffer
            logger.warning(f"🚫 API quota exceeded! Waiting {wait_time}s (API specified: {retry_delay}s + 10s buffer)")
        else:
            wait_time = 70  # Default wait time
            logger.warning(f"🚫 API quota exceeded! Waiting {wait_time}s (default + buffer)")
        
        return wait_time
    
    def _countdown_wait(self, wait_seconds: int):
        """Display countdown for long waits."""
        logger.info(f"⏳ Starting {wait_seconds}s countdown...")
        
        remaining = wait_seconds
        last_log = remaining
        
        while remaining > 0:
            # Log every 10 seconds for long waits, or every second for short waits
            if remaining <= 10 or remaining % 10 == 0 or (last_log - remaining) >= 10:
                logger.info(f"⏳ {remaining}s remaining...")
                last_log = remaining
            
            time.sleep(1)
            remaining -= 1
        
        logger.info("✅ Wait complete! Resuming...")
    
    def invoke(self, input_messages, config=None, **kwargs):
        """Override invoke with quota-aware retry logic."""
        
        for attempt in range(self.max_quota_retries):
            try:
                # Proactive rate limiting
                self._wait_for_rate_limit()
                
                logger.info(f"🤖 Making LLM request (attempt {attempt + 1}/{self.max_quota_retries})")
                
                # Call parent's invoke method
                result = super().invoke(input_messages, config, **kwargs)
                
                logger.info("✅ LLM request successful!")
                return result
                
            except Exception as e:
                logger.error(f"❌ LLM request failed: {str(e)}")
                
                if self._is_quota_error(e):
                    if attempt < self.max_quota_retries - 1:
                        wait_time = self._handle_quota_error(e)
                        
                        # Clear request history to reset rate limiting
                        object.__setattr__(self, 'request_times', [])
                        
                        # Wait with countdown
                        self._countdown_wait(wait_time)
                        
                        logger.info(f"🔄 Retrying LLM request...")
                        continue
                    else:
                        logger.error(f"💀 Max quota retries ({self.max_quota_retries}) exceeded!")
                        raise Exception(
                            f"Google API quota exceeded after {self.max_quota_retries} attempts. "
                            f"Please check your quota at https://ai.google.dev/gemini-api/docs/rate-limits or try again later."
                        )
                else:
                    # Non-quota error, re-raise immediately
                    raise
        
        raise Exception("Failed to invoke LLM after maximum attempts")
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override _generate method for compatibility."""
        
        for attempt in range(self.max_quota_retries):
            try:
                # Proactive rate limiting
                self._wait_for_rate_limit()
                
                logger.info(f"🤖 Making LLM generation request (attempt {attempt + 1}/{self.max_quota_retries})")
                
                # Call parent's _generate method
                result = super()._generate(messages, stop, run_manager, **kwargs)
                
                logger.info("✅ LLM generation successful!")
                return result
                
            except Exception as e:
                logger.error(f"❌ LLM generation failed: {str(e)}")
                
                if self._is_quota_error(e):
                    if attempt < self.max_quota_retries - 1:
                        wait_time = self._handle_quota_error(e)
                        
                        # Clear request history
                        object.__setattr__(self, 'request_times', [])
                        
                        # Wait with countdown
                        self._countdown_wait(wait_time)
                        
                        logger.info(f"🔄 Retrying LLM generation...")
                        continue
                    else:
                        logger.error(f"💀 Max quota retries ({self.max_quota_retries}) exceeded!")
                        raise Exception(
                            f"Google API quota exceeded after {self.max_quota_retries} attempts. "
                            f"Please check your quota at https://ai.google.dev/gemini-api/docs/rate-limits or try again later."
                        )
                else:
                    # Non-quota error, re-raise immediately
                    raise
        
        raise Exception("Failed to generate LLM response after maximum attempts")


# Create the quota-aware LLM instance
llm = QuotaAwareLLM(
    model="gemini-2.0-flash-exp",
    temperature=0.7
)


# Global rate limiting for embeddings - using a simple class approach
class EmbeddingRateLimiter:
    """Simple rate limiter for embeddings that doesn't interfere with Pydantic."""
    
    def __init__(self):
        self.request_times = []
        self.requests_per_minute = 8
    
    def wait_for_rate_limit(self):
        """Proactive rate limiting for embeddings."""
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
                logger.warning(f"🕒 Embedding rate limit: waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds + 2)
        
        # Record this request
        self.request_times.append(datetime.now())


# Global instance for embedding rate limiting
_embedding_rate_limiter = EmbeddingRateLimiter()


class QuotaAwareEmbeddings(GoogleGenerativeAIEmbeddings):
    """Quota-aware embeddings that wait when limits are hit using external rate limiter."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def embed_documents(self, texts):
        """Override with rate limiting using global rate limiter."""
        global _embedding_rate_limiter
        
        _embedding_rate_limiter.wait_for_rate_limit()
        logger.info(f"🔤 Creating embeddings for {len(texts)} documents...")
        
        try:
            time.sleep(1)  # Small delay before embedding
            result = super().embed_documents(texts)
            logger.info("✅ Embeddings created successfully!")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                logger.warning("🚫 Embedding quota exceeded, waiting 60 seconds...")
                time.sleep(60)
                # Try once more after waiting
                try:
                    result = super().embed_documents(texts)
                    logger.info("✅ Embeddings created successfully after retry!")
                    return result
                except Exception as retry_error:
                    logger.error(f"❌ Embedding failed after retry: {retry_error}")
                    raise
            raise
    
    def embed_query(self, text):
        """Override with rate limiting using global rate limiter."""
        global _embedding_rate_limiter
        
        _embedding_rate_limiter.wait_for_rate_limit()
        logger.info("🔍 Creating query embedding...")
        
        try:
            time.sleep(0.5)  # Small delay before embedding
            result = super().embed_query(text)
            logger.info("✅ Query embedding created successfully!")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                logger.warning("🚫 Query embedding quota exceeded, waiting 60 seconds...")
                time.sleep(60)
                # Try once more after waiting
                try:
                    result = super().embed_query(text)
                    logger.info("✅ Query embedding created successfully after retry!")
                    return result
                except Exception as retry_error:
                    logger.error(f"❌ Query embedding failed after retry: {retry_error}")
                    raise
            raise


embedding_function = QuotaAwareEmbeddings(
    model="models/embedding-001"
)

# Database Configuration
CHROMA_DB_PATH = "./chroma_db"

# News API Configuration
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"
NEWS_API_PARAMS = {
    'sortBy': 'publishedAt',
    'language': 'en',
    'pageSize': 2,  # Very conservative - only 2 articles
}