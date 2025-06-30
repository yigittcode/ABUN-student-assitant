#!/usr/bin/env python3
"""
Migration Script: Transfer Existing Documents to MongoDB
Migrates documents from the current session-based storage to persistent MongoDB storage
"""

import os
import logging
import weaviate
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Import our modules
from config import (
    MONGODB_URL, MONGODB_DB_NAME, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT,
    EMBEDDING_MODEL, DOCUMENTS_DIRECTORY
)
from mongodb_manager import MongoDBManager
from document_processor import load_and_process_documents
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_existing_documents():
    """Main migration function"""
    print("🚀 Starting migration from session-based storage to MongoDB...")
    
    try:
        # Initialize MongoDB Manager
        print("📡 Connecting to MongoDB Atlas...")
        mongodb_manager = MongoDBManager(MONGODB_URL, MONGODB_DB_NAME)
        
        # Initialize embedding model
        print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Weaviate client
        print("🗄️ Connecting to Weaviate...")
        weaviate_client = weaviate.connect_to_local(
            host=WEAVIATE_HOST, 
            port=WEAVIATE_PORT, 
            grpc_port=WEAVIATE_GRPC_PORT
        )
        
        # Get migration stats
        print("\n📊 Analyzing current data...")
        migration_stats = analyze_migration_data(mongodb_manager)
        print(f"📄 Found {migration_stats['pdf_files']} PDF files")
        print(f"📁 Found {migration_stats['session_dirs']} session directories")
        print(f"💾 MongoDB has {migration_stats['existing_docs']} existing documents")
        
        # Choose migration source
        choice = input("\nChoose migration source:\n1. PDF files from ./pdfs directory\n2. Session uploads from ./uploads directory\n3. Both sources\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            migrate_pdf_directory(mongodb_manager, embedding_model)
        elif choice == "2":
            migrate_session_uploads(mongodb_manager, embedding_model)
        elif choice == "3":
            migrate_pdf_directory(mongodb_manager, embedding_model)
            migrate_session_uploads(mongodb_manager, embedding_model)
        else:
            print("❌ Invalid choice. Exiting.")
            return
        
        # Final stats
        print("\n📈 Migration completed! Final statistics:")
        final_stats = mongodb_manager.get_database_stats()
        print(f"📄 Total documents in MongoDB: {final_stats['documents_count']}")
        print(f"📑 Total chunks in MongoDB: {final_stats['chunks_count']}")
        print(f"💬 Total chat messages: {final_stats['chat_messages_count']}")
        
        # Cleanup
        mongodb_manager.close()
        weaviate_client.close()
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

def analyze_migration_data(mongodb_manager: MongoDBManager) -> dict:
    """Analyze existing data for migration planning"""
    stats = {
        "pdf_files": 0,
        "session_dirs": 0,
        "existing_docs": 0
    }
    
    # Count PDF files in main directory
    if os.path.exists(DOCUMENTS_DIRECTORY):
        pdf_files = [f for f in os.listdir(DOCUMENTS_DIRECTORY) if f.endswith('.pdf')]
        stats["pdf_files"] = len(pdf_files)
    
    # Count session directories
    uploads_dir = "./uploads"
    if os.path.exists(uploads_dir):
        session_dirs = [d for d in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, d))]
        stats["session_dirs"] = len(session_dirs)
    
    # Count existing MongoDB documents
    existing_docs = mongodb_manager.get_all_documents()
    stats["existing_docs"] = len(existing_docs)
    
    return stats

def migrate_pdf_directory(mongodb_manager: MongoDBManager, embedding_model: SentenceTransformer):
    """Migrate PDF files from the main documents directory"""
    print(f"\n📚 Migrating PDF files from {DOCUMENTS_DIRECTORY}...")
    
    if not os.path.exists(DOCUMENTS_DIRECTORY):
        print(f"⚠️ Directory {DOCUMENTS_DIRECTORY} does not exist. Skipping PDF migration.")
        return
    
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIRECTORY) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("📄 No PDF files found in documents directory.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to migrate")
    
    for filename in tqdm(pdf_files, desc="Migrating PDF files"):
        try:
            # Check if document already exists
            existing_doc = mongodb_manager.get_document_by_filename(filename)
            if existing_doc:
                print(f"⏭️ Document {filename} already exists, skipping...")
                continue
            
            # Read file content
            file_path = os.path.join(DOCUMENTS_DIRECTORY, filename)
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Store document in MongoDB
            doc_id = mongodb_manager.store_document(
                file_name=filename,
                file_content=file_content,
                metadata={
                    "source": "pdf_directory_migration",
                    "migrated_at": datetime.utcnow().isoformat()
                }
            )
            
            # Process and store chunks
            documents = load_and_process_documents(DOCUMENTS_DIRECTORY, specific_files=[filename])
            
            if documents:
                chunk_count = mongodb_manager.store_document_chunks(doc_id, documents)
                print(f"✅ Migrated {filename}: {chunk_count} chunks")
            else:
                print(f"⚠️ No chunks extracted from {filename}")
                
        except Exception as e:
            logger.error(f"❌ Error migrating {filename}: {e}")
            continue

def migrate_session_uploads(mongodb_manager: MongoDBManager, embedding_model: SentenceTransformer):
    """Migrate documents from session upload directories"""
    print(f"\n📁 Migrating documents from session uploads...")
    
    uploads_dir = "./uploads"
    if not os.path.exists(uploads_dir):
        print(f"⚠️ Directory {uploads_dir} does not exist. Skipping session migration.")
        return
    
    session_dirs = [d for d in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, d))]
    
    if not session_dirs:
        print("📁 No session directories found.")
        return
    
    print(f"📁 Found {len(session_dirs)} session directories to migrate")
    
    migrated_files = set()  # Track already migrated files to avoid duplicates
    
    for session_dir in tqdm(session_dirs, desc="Processing sessions"):
        session_path = os.path.join(uploads_dir, session_dir)
        
        try:
            pdf_files = [f for f in os.listdir(session_path) if f.endswith('.pdf')]
            
            for filename in pdf_files:
                # Skip if we've already migrated this file
                if filename in migrated_files:
                    continue
                
                # Check if document already exists in MongoDB
                existing_doc = mongodb_manager.get_document_by_filename(filename)
                if existing_doc:
                    migrated_files.add(filename)
                    continue
                
                # Read file content
                file_path = os.path.join(session_path, filename)
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Store document in MongoDB
                doc_id = mongodb_manager.store_document(
                    file_name=filename,
                    file_content=file_content,
                    metadata={
                        "source": "session_upload_migration",
                        "original_session": session_dir,
                        "migrated_at": datetime.utcnow().isoformat()
                    }
                )
                
                # Process and store chunks
                documents = load_and_process_documents(session_path, specific_files=[filename])
                
                if documents:
                    chunk_count = mongodb_manager.store_document_chunks(doc_id, documents)
                    print(f"✅ Migrated {filename} from session {session_dir}: {chunk_count} chunks")
                    migrated_files.add(filename)
                else:
                    print(f"⚠️ No chunks extracted from {filename}")
                    
        except Exception as e:
            logger.error(f"❌ Error processing session {session_dir}: {e}")
            continue
    
    print(f"📄 Total unique files migrated from sessions: {len(migrated_files)}")

def cleanup_old_data():
    """Optional cleanup of old session-based data"""
    print("\n🧹 Cleanup Options:")
    print("⚠️ WARNING: This will permanently delete old session data!")
    
    choice = input("Do you want to clean up old session directories? (y/N): ").strip().lower()
    
    if choice == 'y':
        uploads_dir = "./uploads"
        if os.path.exists(uploads_dir):
            import shutil
            shutil.rmtree(uploads_dir)
            print("🗑️ Removed old session uploads directory")
        else:
            print("📁 No uploads directory to clean")
    else:
        print("⏭️ Skipping cleanup - old data preserved")

if __name__ == "__main__":
    try:
        migrate_existing_documents()
        
        # Ask about cleanup
        cleanup_choice = input("\nWould you like to clean up old session data? (y/N): ").strip().lower()
        if cleanup_choice == 'y':
            cleanup_old_data()
        
        print("\n🎉 Migration process completed successfully!")
        print("🔄 You can now use the new API backend (api_backend_v2.py)")
        
    except KeyboardInterrupt:
        print("\n⏸️ Migration interrupted by user")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        exit(1)