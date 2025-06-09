import time
import logging
import re
from crewai import Crew, Process
from dotenv import load_dotenv
from config import llm
from agents import news_search_agent, writer_agent
from tasks import initialize_tasks

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedNewsCrew:
    """Enhanced news crew with advanced rate limiting and error handling."""
    
    def __init__(self, topic=None):
        self.topic = topic
        self.crew = None
        self.max_retries = 5
        self.base_retry_delay = 30  # Base delay in seconds
        self.max_retry_delay = 300  # Maximum delay (5 minutes)
    
    def get_topic_from_user(self):
        """Get topic from user input if not provided."""
        if not self.topic:
            print("\n" + "="*60)
            print("NEWS ANALYSIS CREW - TOPIC SELECTION")
            print("="*60)
            print("Welcome! Please specify what topic you'd like to get news about.")
            print("Examples: 'AI developments', 'climate change', 'cryptocurrency', 'space exploration', etc.")
            print("-"*60)
            
            while True:
                try:
                    topic = input("Enter the topic for news analysis: ").strip()
                    
                    if not topic:
                        print("❌ Please enter a valid topic. Cannot be empty.")
                        continue
                    
                    # Confirmation
                    print(f"\n📋 You've selected: '{topic}'")
                    confirm = input("Is this correct? (y/n): ").strip().lower()
                    
                    if confirm in ['y', 'yes']:
                        self.topic = topic
                        print(f"✅ Topic confirmed: {topic}")
                        break
                    elif confirm in ['n', 'no']:
                        print("Let's try again...")
                        continue
                    else:
                        print("❌ Please enter 'y' for yes or 'n' for no.")
                        
                except KeyboardInterrupt:
                    print("\n\n❌ Operation cancelled by user.")
                    return None
                except Exception as e:
                    print(f"❌ Error getting input: {e}")
                    continue
            
            print(f"\n🚀 Proceeding with topic: '{self.topic}'")
            print("="*60)
        
        return self.topic
    
    def create_news_crew(self):
        """Create and return the news crew with user-specified topic."""
        try:
            # Get topic from user if not provided
            if not self.get_topic_from_user():
                return None
            
            # Initialize tasks with the topic
            news_search_task, writer_task = initialize_tasks(self.topic)
            
            self.crew = Crew(
                agents=[news_search_agent, writer_agent],
                tasks=[news_search_task, writer_task],
                process=Process.sequential,
                manager_llm=llm,
                verbose=True
            )
            logger.info(f"News crew created successfully for topic: {self.topic}")
            return self.crew
        except Exception as e:
            logger.error(f"Error creating news crew: {e}")
            raise
    
    def _extract_retry_delay(self, error_message: str) -> int:
        """
        Extract retry delay from error message.
        
        Args:
            error_message: The error message containing retry information
            
        Returns:
            Retry delay in seconds
        """
        try:
            # Look for retry_delay { seconds: X } pattern
            retry_pattern = r'retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}'
            match = re.search(retry_pattern, error_message)
            
            if match:
                return int(match.group(1))
            
            # Alternative pattern: look for "seconds: X" anywhere in the message
            seconds_pattern = r'seconds:\s*(\d+)'
            match = re.search(seconds_pattern, error_message)
            
            if match:
                return int(match.group(1))
                
        except Exception as e:
            logger.warning(f"Failed to extract retry delay: {e}")
        
        return None
    
    def _is_quota_error(self, error_message: str) -> bool:
        """Check if the error is related to quota/rate limiting."""
        error_lower = error_message.lower()
        quota_indicators = [
            '429',
            'quota',
            'rate limit',
            'resourceexhausted',
            'exceeded your current quota',
            'rate-limits'
        ]
        
        return any(indicator in error_lower for indicator in quota_indicators)
    
    def _calculate_wait_time(self, attempt: int, error_message: str) -> int:
        """
        Calculate how long to wait based on the attempt number and error message.
        
        Args:
            attempt: Current attempt number (0-based)
            error_message: Error message from the API
            
        Returns:
            Wait time in seconds
        """
        # Try to extract API-specified retry delay
        api_retry_delay = self._extract_retry_delay(error_message)
        
        if api_retry_delay:
            # Use API-specified delay with a buffer
            wait_time = api_retry_delay + 10
            logger.info(f"Using API-specified retry delay: {api_retry_delay}s + 10s buffer = {wait_time}s")
        else:
            # Use exponential backoff with jitter
            wait_time = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
            logger.info(f"Using exponential backoff: {wait_time}s")
        
        return wait_time
    
    def execute_with_retry(self):
        """Execute the crew with intelligent retry handling for quota errors."""
        if not self.crew:
            if not self.create_news_crew():
                return None
        
        # Add initial delay to space out requests
        logger.info(f"🚀 Starting crew execution for topic '{self.topic}' with initial 10-second delay...")
        time.sleep(10)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📋 Starting crew execution (attempt {attempt + 1}/{self.max_retries})")
                
                # Add progressive delay before each retry attempt
                if attempt > 0:
                    delay = 15 + (attempt * 10)  # 15s, 25s, 35s, etc.
                    logger.info(f"⏳ Adding {delay}-second delay before retry...")
                    time.sleep(delay)
                
                # Execute the crew
                logger.info(f"🔄 Kicking off crew tasks for topic: {self.topic}...")
                result = self.crew.kickoff()
                
                logger.info("🎉 Crew execution completed successfully!")
                return result
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"💥 Attempt {attempt + 1} failed: {error_str}")
                
                # Check if it's a quota/rate limit error
                if self._is_quota_error(error_str):
                    if attempt < self.max_retries - 1:
                        wait_time = self._calculate_wait_time(attempt, error_str)
                        
                        logger.warning(f"🚫 Quota/rate limit error detected!")
                        logger.info(f"📊 Progress: {attempt + 1}/{self.max_retries} attempts completed")
                        logger.warning(f"⏱️  Waiting {wait_time} seconds before retry...")
                        
                        # Always show countdown for quota errors
                        self._countdown_wait(wait_time)
                        
                        continue
                    else:
                        logger.error(f"💀 Max retries ({self.max_retries}) reached!")
                        raise Exception(
                            f"Google API quota exceeded after {self.max_retries} attempts. "
                            f"Please check your API quota and billing at https://ai.google.dev/gemini-api/docs/rate-limits. "
                            f"Consider upgrading your plan or trying again later."
                        )
                else:
                    # For non-quota errors, raise immediately
                    logger.error(f"❌ Non-quota error encountered: {error_str}")
                    raise
        
        raise Exception("Failed to execute crew after maximum retry attempts")
    
    def _countdown_wait(self, wait_seconds: int):
        """Display a countdown while waiting."""
        logger.info(f"⏳ Starting countdown: {wait_seconds} seconds...")
        
        # Show progress more frequently for better user experience
        remaining = wait_seconds
        while remaining > 0:
            if remaining % 5 == 0 or remaining <= 10:
                logger.info(f"⌛ {remaining} seconds remaining...")
            time.sleep(1)
            remaining -= 1
        
        logger.info("✅ Wait complete! Resuming execution...")


def create_news_crew(topic=None):
    """Create and return the news crew with optional topic (backward compatibility)."""
    enhanced_crew = EnhancedNewsCrew(topic)
    return enhanced_crew.create_news_crew()


def main(topic=None):
    """Main function to execute the crew with enhanced error handling and user input."""
    print("="*60)
    print("ENHANCED NEWS ANALYSIS CREW")
    print("="*60)
    
    try:
        # Create enhanced crew instance with optional topic
        enhanced_crew = EnhancedNewsCrew(topic)
        
        # Execute with automatic retry handling
        result = enhanced_crew.execute_with_retry()
        
        if result:
            print("\n" + "="*60)
            print("CREW EXECUTION COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Topic analyzed: {enhanced_crew.topic}")
            print("-"*60)
            print(result)
        else:
            print("\n" + "="*60)
            print("EXECUTION CANCELLED")
            print("="*60)
        
        return result
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("CREW EXECUTION FAILED")
        print("="*60)
        print(f"Final error: {str(e)}")
        
        # Log additional debugging information
        logger.error(f"Crew execution failed with error: {e}", exc_info=True)
        
        print("\nTroubleshooting tips:")
        print("1. Check your Google API key and quota limits at https://ai.google.dev/gemini-api/docs/rate-limits")
        print("2. Consider upgrading your API plan for higher quotas")
        print("3. Verify your internet connection")
        print("4. Try running with fewer articles (reduce pageSize in config)")
        print("5. Check the logs for more detailed error information")
        print("6. Wait a few minutes before retrying if quota is exhausted")
        
        return None


# Function to run with specific topic (for programmatic use)
def run_news_analysis(topic):
    """Run news analysis for a specific topic programmatically."""
    return main(topic)


if __name__ == "__main__":
    # Allow topic to be passed as command line argument or ask user
    import sys
    
    topic = None
    if len(sys.argv) > 1:
        # Topic provided as command line argument
        topic = " ".join(sys.argv[1:])
        print(f"Using topic from command line: {topic}")
    
    result = main(topic)