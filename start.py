#!/usr/bin/env python3
"""
IntelliDocs v2.0 Startup Script
Enhanced startup with migration options and system checks
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import pymongo
        import weaviate
        import openai
        import sentence_transformers
        import fastapi
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install dependencies: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment configuration"""
    print("🔧 Checking environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["OPENAI_API_KEY", "MONGODB_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("📝 Please check your .env file")
        return False
    
    print("✅ Environment configuration is valid")
    return True

def test_mongodb_connection():
    """Test MongoDB connection"""
    print("🔗 Testing MongoDB connection...")
    
    try:
        from config import MONGODB_URL, MONGODB_DB_NAME
        from mongodb_manager import MongoDBManager
        
        mongodb_manager = MongoDBManager(MONGODB_URL, MONGODB_DB_NAME)
        stats = mongodb_manager.get_database_stats()
        mongodb_manager.close()
        
        print(f"✅ MongoDB connection successful")
        print(f"📊 Database stats: {stats['documents_count']} docs, {stats['chunks_count']} chunks")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_weaviate_connection():
    """Test Weaviate connection"""
    print("🗄️ Testing Weaviate connection...")
    
    try:
        import weaviate
        from config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
        
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        
        # Test connection
        collections = client.collections.list_all()
        client.close()
        
        print(f"✅ Weaviate connection successful")
        print(f"📚 Found {len(collections)} collections")
        return True
        
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        print("💡 Make sure Weaviate is running on localhost:8009")
        return False

def offer_migration():
    """Offer to run migration from old system"""
    print("\n📋 Migration Options:")
    print("1. Run migration from existing documents")
    print("2. Skip migration (start fresh)")
    print("3. Exit")
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == "1":
        print("🔄 Starting migration process...")
        try:
            subprocess.run([sys.executable, "migrate_to_mongodb.py"], check=True)
            print("✅ Migration completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Migration failed: {e}")
            return False
    elif choice == "2":
        print("⏭️ Skipping migration - starting with fresh system")
        return True
    else:
        print("👋 Exiting...")
        return False

def start_api_server():
    """Start the API server"""
    print("\n🚀 Starting IntelliDocs v2.0 API Server...")
    print("🔗 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: Open frontend/index.html in your browser")
    print("\n⏹️ Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api_backend:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

def show_usage_instructions():
    """Show usage instructions for the new system"""
    print("\n" + "="*60)
    print("🎉 INTELLIDOCS V2.0 - USAGE GUIDE")
    print("="*60)
    print()
    print("📚 KEY FEATURES:")
    print("  • Persistent document storage (no more sessions!)")
    print("  • Dynamic document management")
    print("  • MongoDB Atlas integration")
    print("  • Continuous chatbot entity")
    print()
    print("🔗 ENDPOINTS:")
    print("  • POST /api/upload - Upload documents")
    print("  • POST /api/chat - Chat with documents")
    print("  • GET /api/documents - List all documents")
    print("  • DELETE /api/documents/{id} - Remove specific document")
    print("  • DELETE /api/documents - Clear all documents")
    print("  • GET /api/chat/history - Get chat history")
    print("  • GET /api/stats - System statistics")
    print()
    print("🌐 FRONTEND:")
    print("  • Open frontend/index_v2.html in your browser")
    print("  • Upload documents to persistent storage")
    print("  • Chat with all documents simultaneously")
    print("  • Manage document library dynamically")
    print()
    print("🔧 MANAGEMENT:")
    print("  • Documents persist between sessions")
    print("  • Add/remove documents anytime")
    print("  • View system statistics and health")
    print("  • Access chat history")
    print()

def main():
    """Main startup function"""
    print("🚀 IntelliDocs v2.0 - Persistent Document Intelligence")
    print("=" * 55)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check environment
    if not check_environment():
        return False
    
    # Test connections
    mongodb_ok = test_mongodb_connection()
    weaviate_ok = test_weaviate_connection()
    
    if not mongodb_ok:
        print("❌ Cannot start without MongoDB connection")
        return False
    
    if not weaviate_ok:
        print("❌ Cannot start without Weaviate connection")
        print("💡 Start Weaviate with: docker-compose up weaviate")
        return False
    
    # Check for existing data and offer migration
    uploads_exist = os.path.exists("./uploads") and os.listdir("./uploads")
    pdfs_exist = os.path.exists("./pdfs") and any(f.endswith('.pdf') for f in os.listdir("./pdfs"))
    
    if uploads_exist or pdfs_exist:
        print(f"\n📄 Found existing documents:")
        if uploads_exist:
            print(f"  • Session uploads in ./uploads/")
        if pdfs_exist:
            print(f"  • PDF files in ./pdfs/")
        
        if not offer_migration():
            return False
    
    # Show usage guide
    show_usage_instructions()
    
    # Start server
    input("\nPress Enter to start the server...")
    start_api_server()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Startup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)