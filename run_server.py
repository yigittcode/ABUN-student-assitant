#!/usr/bin/env python3
"""
🚀 IntelliDocs Semantic RAG Server
Enhanced semantic document intelligence with advanced retrieval
"""

import os
import sys
import uvicorn
import asyncio
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    return True

def check_environment():
    """Check if environment is properly configured"""
    required_vars = ['OPENAI_API_KEY', 'MONGODB_URL', 'PERSISTENT_COLLECTION_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Create .env file with required variables")
        return False
    
    return True

async def test_dependencies():
    """Test core dependencies"""
    try:
        # Test core imports
        import weaviate
        import openai
        import sentence_transformers
        import pymongo
        print("✅ Core dependencies available")
        
        # Test Weaviate connection
        from config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
        try:
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
            collections = client.collections.list_all()
            client.close()
            print(f"✅ Weaviate connection successful ({len(collections)} collections)")
        except Exception as e:
            print(f"❌ Weaviate connection failed: {e}")
            return False
        
        # Test MongoDB connection
        from config import MONGODB_URL
        try:
            import pymongo
            client = pymongo.MongoClient(MONGODB_URL)
            client.admin.command('ping')
            client.close()
            print("✅ MongoDB connection successful")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Run: pip install -r requirements.txt")
        return False

def show_semantic_features():
    """Show new semantic features"""
    print("\n🧠 NEW SEMANTIC FEATURES:")
    print("  • Content-aware semantic chunking")
    print("  • Document structure detection (automatic)")
    print("  • Multi-pattern content recognition") 
    print("  • Adaptive embedding enhancement")
    print("  • Improved parallel processing")
    print("  • Enhanced retrieval quality")
    print("  • Better token allocation")
    print("")

def main():
    """Main server startup"""
    print("🚀 IntelliDocs Semantic RAG Server v2.1")
    print("=" * 55)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    if not check_environment():
        return False
    
    # Test dependencies and connections
    print("🔍 Testing system dependencies...")
    deps_ok = asyncio.run(test_dependencies())
    
    if not deps_ok:
        print("\n❌ System check failed!")
        print("💡 Fix the issues above and try again")
        return False
    
    # Show new features
    show_semantic_features()
    
    print("🌟 All systems ready!")
    print("🔗 Starting enhanced semantic RAG server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 Docs available at: http://localhost:8000/docs")
    print("")
    
    try:
        # Start the server
        uvicorn.run(
            "api_backend:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 