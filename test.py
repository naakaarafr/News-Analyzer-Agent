#!/usr/bin/env python3
"""
Test script to verify API keys and basic functionality before running the main crew.
Run this script first to identify any configuration issues.
"""

import os
from dotenv import load_dotenv

def test_environment_setup():
    """Test if environment variables are properly set."""
    print("🔍 Testing Environment Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['GOOGLE_API_KEY', 'NEWSAPI_KEY', 'SERPER_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Convert to string and strip whitespace
            value = str(value).strip()
            # Set the environment variable to ensure it's properly formatted
            os.environ[var] = value
            print(f"✅ {var}: {'*' * 10}{value[-4:] if len(value) > 4 else '****'}")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All environment variables are set!")
    return True

def test_google_api():
    """Test Google Gemini API connection."""
    print("\n🔍 Testing Google Gemini API...")
    
    try:
        # Ensure environment variable is set properly
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = str(google_api_key).strip()
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Don't pass the API key explicitly, let it use the environment variable
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1
        )
        
        # Test with a simple prompt
        response = llm.invoke("Say 'Hello, I am working!' in exactly those words.")
        print(f"✅ Google Gemini API: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Google Gemini API Error: {str(e)}")
        # Print more details for debugging
        if "SecretStr" in str(e):
            print("💡 Hint: This is a SecretStr issue. Make sure to use environment variables properly.")
        return False

def test_embedding_api():
    """Test Google Embedding API connection."""
    print("\n🔍 Testing Google Embedding API...")
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Don't pass the API key explicitly
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Google Embedding API: Generated embedding with {len(embedding)} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ Google Embedding API Error: {str(e)}")
        return False

def test_serper_api():
    """Test Serper API connection."""
    print("\n🔍 Testing Serper API...")
    
    try:
        serper_key = os.getenv('SERPER_API_KEY')
        if serper_key:
            os.environ["SERPER_API_KEY"] = str(serper_key).strip()
        
        from langchain_community.utilities import GoogleSerperAPIWrapper
        from langchain_community.tools import GoogleSerperRun
        
        serper_wrapper = GoogleSerperAPIWrapper()
        search_tool = GoogleSerperRun(api_wrapper=serper_wrapper)
        
        # Test search
        result = search_tool.run("AI news 2024")
        print(f"✅ Serper API: Search returned {len(result)} characters of data")
        return True
        
    except Exception as e:
        print(f"❌ Serper API Error: {str(e)}")
        return False

def test_news_api():
    """Test News API connection."""
    print("\n🔍 Testing News API...")
    
    try:
        import requests
        
        api_key = os.getenv('NEWSAPI_KEY')
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': 'AI',
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'language': 'en',
            'pageSize': 1,
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles_count = len(data.get('articles', []))
            print(f"✅ News API: Retrieved {articles_count} articles")
            if articles_count > 0:
                print(f"   Sample title: {data['articles'][0]['title'][:50]}...")
            return True
        else:
            print(f"❌ News API Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data}")
            except:
                print(f"   Response text: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ News API Error: {str(e)}")
        return False

def test_vector_database():
    """Test if we can create a simple vector database."""
    print("\n🔍 Testing Vector Database Creation...")
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.schema import Document
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create test documents
        test_docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test1"}),
            Document(page_content="Another document about machine learning.", metadata={"source": "test2"})
        ]
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            test_docs,
            embedding=embeddings,
            persist_directory="./test_chroma_db"
        )
        
        # Test search
        results = vectorstore.similarity_search("AI", k=1)
        print(f"✅ Vector Database: Created and searched successfully, found {len(results)} results")
        
        # Clean up
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector Database Error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting API Configuration Tests...\n")
    
    tests = [
        test_environment_setup,
        test_google_api,
        test_embedding_api,
        test_serper_api,
        test_news_api,
        test_vector_database
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! You can now run the main crew application.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the main application.")
        print("💡 Check your .env file and API key configurations.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)