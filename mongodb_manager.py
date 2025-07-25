"""
MongoDB Manager for Persistent Document Storage
Handles document storage, retrieval, and management operations in MongoDB Atlas
WITH ASYNC OPTIMIZATIONS
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId
import gridfs
from io import BytesIO
import asyncio

logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    MongoDB Manager for handling persistent document storage and retrieval
    OPTIMIZED with async wrappers and connection pooling
    """
    
    def __init__(self, connection_url: str, db_name: str):
        """
        Initialize MongoDB connection with optimizations
        
        Args:
            connection_url: MongoDB Atlas connection string
            db_name: Database name to use
        """
        self.connection_url = connection_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.fs = None
        self.connect()
    
    def connect(self):
        """Establish connection to MongoDB with connection pooling optimizations"""
        try:
            # OPTIMIZATION: Connection pooling settings
            self.client = MongoClient(
                self.connection_url,
                maxPoolSize=50,          # Increased from default 100
                minPoolSize=10,          # Keep minimum connections
                maxIdleTimeMS=30000,     # Close idle connections after 30s
                waitQueueTimeoutMS=5000, # Queue timeout for connections
                retryWrites=True,        # Enable retry writes
                retryReads=True,         # Enable retry reads
                socketTimeoutMS=20000,   # Socket timeout
                connectTimeoutMS=20000,  # Connection timeout
                serverSelectionTimeoutMS=5000  # Server selection timeout
            )
            self.db = self.client[self.db_name]
            self.fs = gridfs.GridFS(self.db)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB database: {self.db_name} with optimized connection pool")
            
            # Create indexes for better performance
            self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # OPTIMIZATION: Compound indexes for better performance
            self.db.documents.create_index([
                ("file_name", 1),
                ("status", 1),
                ("created_at", -1)
            ])
            
            # OPTIMIZATION: Index for document_chunks with document_id + chunk_index
            self.db.document_chunks.create_index([
                ("document_id", 1),
                ("chunk_index", 1)
            ], unique=True)
            
            # OPTIMIZATION: Index for chat_history with timestamp
            self.db.chat_history.create_index([
                ("timestamp", -1),
                ("interaction_type", 1)
            ])
            
            # OPTIMIZATION: Index for upload_sessions
            self.db.upload_sessions.create_index([
                ("session_id", 1)
            ], unique=True)
            
            self.db.upload_sessions.create_index([
                ("started_at", -1),
                ("status", 1)
            ])
            
            logger.info("✅ Optimized database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
    
    # ASYNC OPTIMIZATION: Wrap sync operations with asyncio.to_thread()
    
    async def store_document_async(self, file_name: str, file_content: bytes, 
                                  metadata: Optional[Dict] = None, status: str = "stored") -> str:
        """ASYNC wrapper for store_document"""
        return await asyncio.to_thread(self.store_document, file_name, file_content, metadata, status)
    
    def store_document(self, file_name: str, file_content: bytes, 
                      metadata: Optional[Dict] = None, status: str = "stored") -> str:
        """
        Store a document file in MongoDB GridFS
        
        Args:
            file_name: Name of the file
            file_content: Binary content of the file
            metadata: Optional metadata dictionary
            
        Returns:
            Document ID as string
        """
        try:
            # Check if document already exists
            existing_doc = self.get_document_by_filename(file_name)
            if existing_doc:
                logger.info(f"📄 Document {file_name} already exists, updating...")
                self.remove_document(existing_doc['_id'])
            
            # Store file content in GridFS
            file_id = self.fs.put(
                file_content,
                filename=file_name,
                metadata=metadata or {}
            )
            
            # Store document metadata in documents collection
            doc_data = {
                "file_id": file_id,
                "file_name": file_name,
                "file_size": len(file_content),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": status,
                "metadata": metadata or {}
            }
            
            result = self.db.documents.insert_one(doc_data)
            doc_id = str(result.inserted_id)
            
            logger.info(f"✅ Document {file_name} stored successfully with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Error storing document {file_name}: {e}")
            raise
    
    async def store_document_chunks_async(self, document_id: str, chunks: List[Dict]) -> int:
        """ASYNC wrapper for store_document_chunks"""
        return await asyncio.to_thread(self.store_document_chunks, document_id, chunks)
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict]) -> int:
        """
        Store processed document chunks for a document
        
        Args:
            document_id: Document ID
            chunks: List of document chunks with content and metadata
            
        Returns:
            Number of chunks stored
        """
        try:
            # Remove existing chunks for this document
            self.db.document_chunks.delete_many({"document_id": ObjectId(document_id)})
            
            # OPTIMIZATION: Batch insert preparation
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "document_id": ObjectId(document_id),
                    "chunk_index": i,
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", ""),
                    "article": chunk.get("article", ""),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.utcnow()
                }
                chunk_docs.append(chunk_doc)
            
            # OPTIMIZATION: Single batch insert
            if chunk_docs:
                result = self.db.document_chunks.insert_many(chunk_docs, ordered=False)
                count = len(result.inserted_ids)
            else:
                count = 0
            
            # Update document status
            self.db.documents.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$set": {
                        "status": "processed",
                        "chunks_count": count,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"✅ Stored {count} chunks for document ID: {document_id}")
            return count
            
        except Exception as e:
            logger.error(f"❌ Error storing chunks for document {document_id}: {e}")
            raise
    
    async def get_all_documents_async(self) -> List[Dict]:
        """ASYNC wrapper for get_all_documents"""
        return await asyncio.to_thread(self.get_all_documents)
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all stored documents metadata
        
        Returns:
            List of document metadata dictionaries
        """
        try:
            # OPTIMIZATION: Use projection to reduce data transfer
            documents = list(self.db.documents.find(
                {},
                {
                    "file_name": 1,
                    "file_size": 1, 
                    "created_at": 1,
                    "status": 1,
                    "chunks_count": 1,
                    "metadata": 1
                }
            ).sort("created_at", -1))
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])
                if "file_id" in doc:
                    doc["file_id"] = str(doc["file_id"])
            
            logger.info(f"📚 Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            raise
    
    async def get_document_by_id_async(self, document_id: str) -> Optional[Dict]:
        """ASYNC wrapper for get_document_by_id"""
        return await asyncio.to_thread(self.get_document_by_id, document_id)
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """
        Get document metadata by ID
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        try:
            doc = self.db.documents.find_one({"_id": ObjectId(document_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
                doc["file_id"] = str(doc["file_id"])
            return doc
            
        except Exception as e:
            logger.error(f"❌ Error retrieving document {document_id}: {e}")
            return None
    
    async def get_document_by_filename_async(self, filename: str) -> Optional[Dict]:
        """ASYNC wrapper for get_document_by_filename"""
        return await asyncio.to_thread(self.get_document_by_filename, filename)
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """
        Get document metadata by filename
        
        Args:
            filename: Document filename
            
        Returns:
            Document metadata or None if not found
        """
        try:
            doc = self.db.documents.find_one({"file_name": filename})
            if doc:
                doc["_id"] = str(doc["_id"])
                doc["file_id"] = str(doc["file_id"])
            return doc
            
        except Exception as e:
            logger.error(f"❌ Error retrieving document by filename {filename}: {e}")
            return None
    
    async def update_document_status_async(self, document_id: str, status: str) -> bool:
        """ASYNC wrapper for update_document_status"""
        return await asyncio.to_thread(self.update_document_status, document_id, status)
    
    def update_document_status(self, document_id: str, status: str) -> bool:
        """
        Update document processing status
        
        Args:
            document_id: Document ID
            status: New status (processing, processed, failed)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.db.documents.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"status": status, "updated_at": datetime.utcnow()}}
            )
            
            if result.modified_count > 0:
                logger.info(f"📝 Updated document {document_id} status to {status}")
                return True
            else:
                logger.warning(f"⚠️ No document found with ID {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating document status: {e}")
            return False
    
    async def get_document_chunks_async(self, document_id: str) -> List[Dict]:
        """ASYNC wrapper for get_document_chunks"""
        return await asyncio.to_thread(self.get_document_chunks, document_id)
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        try:
            # OPTIMIZATION: Use projection to only get needed fields
            chunks = list(
                self.db.document_chunks
                .find(
                    {"document_id": ObjectId(document_id)},
                    {
                        "content": 1,
                        "source": 1,
                        "article": 1,
                        "chunk_index": 1,
                        "metadata": 1,
                        "document_id": 1  # CRITICAL FIX: Include document_id in projection
                    }
                )
                .sort("chunk_index", 1)
            )
            
            # Convert ObjectId to string
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
                chunk["document_id"] = str(chunk["document_id"])
            
            logger.info(f"📑 Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks for document {document_id}: {e}")
            return []
    
    async def remove_document_async(self, document_id: str) -> bool:
        """ASYNC wrapper for remove_document"""
        return await asyncio.to_thread(self.remove_document, document_id)
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document and its chunks
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document info
            doc = self.get_document_by_id(document_id)
            if not doc:
                logger.warning(f"⚠️ Document {document_id} not found for removal")
                return False
            
            # Remove file from GridFS
            self.fs.delete(ObjectId(doc["file_id"]))
            
            # Remove document chunks
            self.db.document_chunks.delete_many({"document_id": ObjectId(document_id)})
            
            # Remove document metadata
            self.db.documents.delete_one({"_id": ObjectId(document_id)})
            
            logger.info(f"🗑️ Document {document_id} ({doc['file_name']}) removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error removing document {document_id}: {e}")
            return False
    
    # Chat History Methods with async wrappers
    
    async def store_chat_message_async(self, question: str, answer: str, sources: List[Dict], interaction_type: str = "text") -> str:
        """ASYNC wrapper for store_chat_message"""
        return await asyncio.to_thread(self.store_chat_message, question, answer, sources, interaction_type)
    
    def store_chat_message(self, question: str, answer: str, sources: List[Dict], interaction_type: str = "text") -> str:
        """
        Store a chat interaction
        
        Args:
            question: User question
            answer: Bot answer
            sources: List of source references
            interaction_type: Type of interaction (text, voice, stream)
            
        Returns:
            Chat message ID
        """
        try:
            chat_data = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "interaction_type": interaction_type,
                "timestamp": datetime.utcnow()
            }
            
            result = self.db.chat_history.insert_one(chat_data)
            chat_id = str(result.inserted_id)
            
            logger.info(f"💬 Chat interaction stored with ID: {chat_id}")
            return chat_id
            
        except Exception as e:
            logger.error(f"❌ Error storing chat message: {e}")
            raise
    
    async def get_chat_history_async(self, limit: int = 50) -> List[Dict]:
        """ASYNC wrapper for get_chat_history"""
        return await asyncio.to_thread(self.get_chat_history, limit)
    
    def get_chat_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent chat history
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of chat messages
        """
        try:
            # OPTIMIZATION: Use projection to reduce data transfer for large histories
            messages = list(
                self.db.chat_history
                .find(
                    {},
                    {
                        "question": 1,
                        "answer": 1,
                        "interaction_type": 1,
                        "timestamp": 1,
                        "sources": {"$slice": 5}  # Limit sources to first 5
                    }
                )
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for msg in messages:
                msg["_id"] = str(msg["_id"])
            
            logger.info(f"💬 Retrieved {len(messages)} chat messages")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chat history: {e}")
            return []
    
    # Upload Progress Tracking Methods with async wrappers
    
    async def create_upload_session_async(self, session_id: str, filenames: List[str], user_info: Dict = None) -> str:
        """ASYNC wrapper for create_upload_session"""
        return await asyncio.to_thread(self.create_upload_session, session_id, filenames, user_info)
    
    def create_upload_session(self, session_id: str, filenames: List[str], user_info: Dict = None) -> str:
        """
        Create a new upload session for tracking progress
        
        Args:
            session_id: Unique session identifier
            filenames: List of filenames to be processed
            user_info: Optional user information
            
        Returns:
            Upload session ID
        """
        try:
            # Prepare upload progress data
            upload_data = {
                "session_id": session_id,
                "user_info": user_info or {},
                "total_files": len(filenames),
                "files": [
                    {
                        "filename": filename,
                        "status": "pending",  # pending, processing, completed, failed
                        "started_at": None,
                        "completed_at": None,
                        "error_message": None,
                        "document_id": None,
                        "chunks_count": 0
                    }
                    for filename in filenames
                ],
                "status": "started",  # started, processing, completed, failed
                "started_at": datetime.utcnow(),
                "completed_at": None,
                "progress_percentage": 0
            }
            
            result = self.db.upload_sessions.insert_one(upload_data)
            upload_session_id = str(result.inserted_id)
            
            logger.info(f"📦 Upload session created: {session_id} with {len(filenames)} files")
            return upload_session_id
            
        except Exception as e:
            logger.error(f"❌ Error creating upload session: {e}")
            raise
    
    async def update_upload_progress_async(self, session_id: str, filename: str, status: str, 
                                          document_id: str = None, chunks_count: int = 0, 
                                          error_message: str = None) -> bool:
        """ASYNC wrapper for update_upload_progress"""
        return await asyncio.to_thread(self.update_upload_progress, session_id, filename, status, document_id, chunks_count, error_message)
    
    def update_upload_progress(self, session_id: str, filename: str, status: str, 
                             document_id: str = None, chunks_count: int = 0, 
                             error_message: str = None) -> bool:
        """
        Update progress for a specific file in upload session
        
        Args:
            session_id: Upload session ID
            filename: Filename being processed
            status: File status (processing, completed, failed)
            document_id: Document ID if completed
            chunks_count: Number of chunks processed
            error_message: Error message if failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update file status
            update_data = {
                "files.$.status": status,
                "files.$.completed_at": datetime.utcnow() if status in ["completed", "failed"] else None
            }
            
            if status == "processing":
                update_data["files.$.started_at"] = datetime.utcnow()
            
            if document_id:
                update_data["files.$.document_id"] = document_id
                
            if chunks_count > 0:
                update_data["files.$.chunks_count"] = chunks_count
                
            if error_message:
                update_data["files.$.error_message"] = error_message
            
            # Update specific file
            result = self.db.upload_sessions.update_one(
                {
                    "session_id": session_id,
                    "files.filename": filename
                },
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                # Calculate and update overall progress
                self._update_session_overall_progress(session_id)
                logger.info(f"📊 Upload progress updated: {filename} -> {status}")
                return True
            else:
                logger.warning(f"⚠️ No file found: {filename} in session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating upload progress: {e}")
            return False
    
    def _update_session_overall_progress(self, session_id: str):
        """
        Calculate and update overall session progress
        
        Args:
            session_id: Upload session ID
        """
        try:
            session = self.db.upload_sessions.find_one({"session_id": session_id})
            if not session:
                return
            
            files = session.get("files", [])
            total_files = len(files)
            
            if total_files == 0:
                return
            
            completed_files = len([f for f in files if f["status"] in ["completed", "failed"]])
            progress_percentage = int((completed_files / total_files) * 100)
            
            # Determine overall status
            failed_files = len([f for f in files if f["status"] == "failed"])
            processing_files = len([f for f in files if f["status"] in ["pending", "processing"]])
            
            if completed_files == total_files:
                if failed_files > 0:
                    overall_status = "completed_with_errors"
                else:
                    overall_status = "completed"
                completed_at = datetime.utcnow()
            elif processing_files > 0:
                overall_status = "processing"
                completed_at = None
            else:
                overall_status = "started"
                completed_at = None
            
            # Update session
            self.db.upload_sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "status": overall_status,
                        "progress_percentage": progress_percentage,
                        "completed_at": completed_at
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating session overall progress: {e}")
    
    async def get_upload_session_async(self, session_id: str) -> Optional[Dict]:
        """ASYNC wrapper for get_upload_session"""
        return await asyncio.to_thread(self.get_upload_session, session_id)
    
    def get_upload_session(self, session_id: str) -> Optional[Dict]:
        """
        Get upload session status and progress
        
        Args:
            session_id: Upload session ID
            
        Returns:
            Upload session data or None if not found
        """
        try:
            session = self.db.upload_sessions.find_one({"session_id": session_id})
            if session:
                session["_id"] = str(session["_id"])
                
                # Calculate summary stats
                files = session.get("files", [])
                session["summary"] = {
                    "total": len(files),
                    "completed": len([f for f in files if f["status"] == "completed"]),
                    "failed": len([f for f in files if f["status"] == "failed"]),
                    "processing": len([f for f in files if f["status"] == "processing"]),
                    "pending": len([f for f in files if f["status"] == "pending"])
                }
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Error retrieving upload session: {e}")
            return None
    
    async def get_database_stats_async(self) -> Dict:
        """ASYNC wrapper for get_database_stats"""
        return await asyncio.to_thread(self.get_database_stats)
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # OPTIMIZATION: Run stats queries in parallel
            stats = {
                "documents_count": self.db.documents.count_documents({}),
                "chunks_count": self.db.document_chunks.count_documents({}),
                "chat_messages_count": self.db.chat_history.count_documents({}),
                "upload_sessions_count": self.db.upload_sessions.count_documents({}),
                "collections": self.db.list_collection_names()
            }
            
            # Get database size (optional, can be slow)
            try:
                db_stats = self.db.command("dbStats")
                stats["database_size"] = db_stats.get("dataSize", 0)
            except:
                stats["database_size"] = 0
            
            logger.info(f"📊 Database stats: {stats['documents_count']} docs, {stats['chunks_count']} chunks")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
    
    # Cleanup methods
    
    def clear_chat_history(self) -> int:
        """
        Clear all chat history
        
        Returns:
            Number of messages deleted
        """
        try:
            result = self.db.chat_history.delete_many({})
            count = result.deleted_count
            
            logger.info(f"🧹 Cleared {count} chat messages")
            return count
            
        except Exception as e:
            logger.error(f"❌ Error clearing chat history: {e}")
            return 0
    
    def get_recent_upload_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent upload sessions
        
        Args:
            limit: Maximum number of sessions to retrieve
            
        Returns:
            List of upload sessions
        """
        try:
            sessions = list(
                self.db.upload_sessions
                .find()
                .sort("started_at", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string and add summary
            for session in sessions:
                session["_id"] = str(session["_id"])
                files = session.get("files", [])
                session["summary"] = {
                    "total": len(files),
                    "completed": len([f for f in files if f["status"] == "completed"]),
                    "failed": len([f for f in files if f["status"] == "failed"]),
                    "processing": len([f for f in files if f["status"] == "processing"]),
                    "pending": len([f for f in files if f["status"] == "pending"])
                }
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error retrieving recent upload sessions: {e}")
            return []
    
    def cleanup_old_upload_sessions(self, days_old: int = 7) -> int:
        """
        Clean up old upload sessions
        
        Args:
            days_old: Remove sessions older than this many days
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            result = self.db.upload_sessions.delete_many(
                {"started_at": {"$lt": cutoff_date}}
            )
            
            count = result.deleted_count
            if count > 0:
                logger.info(f"🧹 Cleaned up {count} old upload sessions")
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up upload sessions: {e}")
            return 0