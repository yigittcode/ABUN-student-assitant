"""
MongoDB Manager for Persistent Document Storage
Handles document storage, retrieval, and management operations in MongoDB Atlas
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
import uuid

logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    MongoDB Manager for handling persistent document storage and retrieval
    """
    
    def __init__(self, connection_url: str, db_name: str):
        """
        Initialize MongoDB connection
        
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
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_url)
            self.db = self.client[self.db_name]
            self.fs = gridfs.GridFS(self.db)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB database: {self.db_name}")
            
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
            # Index for documents collection
            self.db.documents.create_index([
                ("file_name", 1),
                ("created_at", -1)
            ])
            
            # Index for document_chunks collection  
            self.db.document_chunks.create_index([
                ("document_id", 1),
                ("chunk_index", 1)
            ])
            
            # Index for chat_history collection
            self.db.chat_history.create_index([
                ("timestamp", -1)
            ])
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
    
    # Document Management Methods
    
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
            
            # Prepare chunk documents
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
            
            # Insert all chunks
            result = self.db.document_chunks.insert_many(chunk_docs)
            count = len(result.inserted_ids)
            
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
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all stored documents metadata
        
        Returns:
            List of document metadata dictionaries
        """
        try:
            documents = list(self.db.documents.find().sort("created_at", -1))
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])
                doc["file_id"] = str(doc["file_id"])
            
            logger.info(f"📚 Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            raise
    
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
    
    def get_document_content(self, document_id: str) -> Optional[bytes]:
        """
        Get document file content by document ID
        
        Args:
            document_id: Document ID
            
        Returns:
            File content as bytes or None if not found
        """
        try:
            doc = self.get_document_by_id(document_id)
            if not doc:
                return None
            
            file_id = ObjectId(doc["file_id"])
            file_content = self.fs.get(file_id).read()
            
            logger.info(f"📄 Retrieved content for document {document_id}")
            return file_content
            
        except Exception as e:
            logger.error(f"❌ Error retrieving content for document {document_id}: {e}")
            return None
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        try:
            chunks = list(
                self.db.document_chunks
                .find({"document_id": ObjectId(document_id)})
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
    
    def get_all_chunks(self) -> List[Dict]:
        """
        Get all document chunks from all documents
        
        Returns:
            List of all document chunks
        """
        try:
            chunks = list(self.db.document_chunks.find())
            
            # Convert ObjectId to string
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
                chunk["document_id"] = str(chunk["document_id"])
            
            logger.info(f"📑 Retrieved {len(chunks)} total chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving all chunks: {e}")
            return []
    
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
    
    def remove_document_by_filename(self, filename: str) -> bool:
        """
        Remove a document by filename
        
        Args:
            filename: Document filename to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc = self.get_document_by_filename(filename)
            if not doc:
                logger.warning(f"⚠️ Document {filename} not found for removal")
                return False
            
            return self.remove_document(doc["_id"])
            
        except Exception as e:
            logger.error(f"❌ Error removing document by filename {filename}: {e}")
            return False
    
    # Chat History Methods
    
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
    
    def get_chat_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent chat history
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of chat messages
        """
        try:
            messages = list(
                self.db.chat_history
                .find()
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
    
    # Utility Methods
    
    # Upload Progress Tracking Methods
    
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
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {
                "documents_count": self.db.documents.count_documents({}),
                "chunks_count": self.db.document_chunks.count_documents({}),
                "chat_messages_count": self.db.chat_history.count_documents({}),
                "database_size": self.db.command("dbStats")["dataSize"],
                "collections": self.db.list_collection_names()
            }
            
            logger.info(f"📊 Database stats: {stats['documents_count']} docs, {stats['chunks_count']} chunks")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

    # Conversation Session Management Methods

    def create_conversation_session(self, user_email: str, initial_message: str = None) -> str:
        """
        Create a new conversation session for a user
        
        Args:
            user_email: User's email address
            initial_message: Optional initial message
            
        Returns:
            Session ID
        """
        try:
            session_data = {
                "session_id": str(uuid.uuid4()),
                "user_email": user_email,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "message_count": 0,
                "status": "active",
                "context_summary": "",
                "initial_message": initial_message
            }
            
            result = self.db.conversation_sessions.insert_one(session_data)
            session_id = session_data["session_id"]
            
            # Create index for faster queries
            try:
                self.db.conversation_sessions.create_index([
                    ("user_email", 1),
                    ("created_at", -1)
                ])
                self.db.conversation_sessions.create_index([("session_id", 1)])
            except Exception:
                pass  # Index might already exist
            
            logger.info(f"💬 Created conversation session {session_id} for {user_email}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Error creating conversation session: {e}")
            raise

    def get_conversation_session(self, session_id: str) -> Optional[Dict]:
        """
        Get conversation session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found
        """
        try:
            session = self.db.conversation_sessions.find_one({"session_id": session_id})
            if session:
                session["_id"] = str(session["_id"])
            return session
            
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation session {session_id}: {e}")
            return None

    def get_user_active_session(self, user_email: str) -> Optional[Dict]:
        """
        Get user's most recent active conversation session
        
        Args:
            user_email: User's email address
            
        Returns:
            Most recent active session or None
        """
        try:
            session = self.db.conversation_sessions.find_one(
                {"user_email": user_email, "status": "active"},
                sort=[("updated_at", -1)]
            )
            if session:
                session["_id"] = str(session["_id"])
            return session
            
        except Exception as e:
            logger.error(f"❌ Error getting active session for {user_email}: {e}")
            return None

    def store_conversation_message(self, session_id: str, question: str, answer: str, 
                                 sources: List[Dict], interaction_type: str = "text") -> str:
        """
        Store a chat message within a conversation session
        
        Args:
            session_id: Conversation session ID
            question: User question
            answer: Bot answer
            sources: List of source references
            interaction_type: Type of interaction (text, voice, stream)
            
        Returns:
            Message ID
        """
        try:
            # Store the message
            message_data = {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "interaction_type": interaction_type,
                "timestamp": datetime.utcnow()
            }
            
            result = self.db.conversation_messages.insert_one(message_data)
            message_id = str(result.inserted_id)
            
            # Update session
            self.db.conversation_sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": {"updated_at": datetime.utcnow()},
                    "$inc": {"message_count": 1}
                }
            )
            
            # Also store in legacy chat_history for backward compatibility
            self.store_chat_message(question, answer, sources, interaction_type)
            
            logger.info(f"💬 Stored conversation message {message_id} in session {session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"❌ Error storing conversation message: {e}")
            raise

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        try:
            messages = list(
                self.db.conversation_messages
                .find({"session_id": session_id})
                .sort("timestamp", 1)  # Oldest first for conversation flow
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for msg in messages:
                msg["_id"] = str(msg["_id"])
            
            logger.info(f"💬 Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation history for {session_id}: {e}")
            return []

    def get_recent_conversation_context(self, session_id: str, max_messages: int = 10, 
                                       max_tokens: int = 2000) -> str:
        """
        Get recent conversation context formatted for RAG prompt
        
        Args:
            session_id: Session ID
            max_messages: Maximum number of recent messages
            max_tokens: Maximum token limit for context
            
        Returns:
            Formatted conversation context
        """
        try:
            # Get recent messages (reverse order to get most recent first)
            messages = list(
                self.db.conversation_messages
                .find({"session_id": session_id})
                .sort("timestamp", -1)
                .limit(max_messages)
            )
            
            if not messages:
                return ""
            
            # Reverse to get chronological order
            messages.reverse()
            
            # Build context with token estimation
            context_parts = []
            total_tokens = 0
            
            for msg in messages:
                question = msg.get("question", "").strip()
                answer = msg.get("answer", "").strip()
                
                if question and answer:
                    # Estimate tokens (rough: 1 token ≈ 4 characters)
                    msg_tokens = (len(question) + len(answer)) // 4
                    
                    if total_tokens + msg_tokens > max_tokens:
                        break
                        
                    context_parts.append(f"Kullanıcı: {question}")
                    context_parts.append(f"Asistan: {answer}")
                    total_tokens += msg_tokens
            
            if context_parts:
                context = "\n".join(context_parts)
                logger.info(f"💭 Built conversation context: {len(context_parts)//2} messages, ~{total_tokens} tokens")
                return context
            else:
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error building conversation context for {session_id}: {e}")
            return ""

    def close_conversation_session(self, session_id: str, summary: str = None) -> bool:
        """
        Close a conversation session
        
        Args:
            session_id: Session ID to close
            summary: Optional conversation summary
            
        Returns:
            True if successful
        """
        try:
            update_data = {
                "status": "closed",
                "closed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            if summary:
                update_data["context_summary"] = summary
            
            result = self.db.conversation_sessions.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"🔒 Closed conversation session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error closing conversation session {session_id}: {e}")
            return False

    def cleanup_old_sessions(self, days_old: int = 30, inactive_days: int = 7) -> Dict[str, int]:
        """
        Clean up old and inactive conversation sessions
        
        Args:
            days_old: Remove sessions older than this many days
            inactive_days: Close sessions inactive for this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            now = datetime.utcnow()
            
            # Close inactive sessions
            inactive_cutoff = now - timedelta(days=inactive_days)
            inactive_result = self.db.conversation_sessions.update_many(
                {
                    "status": "active",
                    "updated_at": {"$lt": inactive_cutoff}
                },
                {
                    "$set": {
                        "status": "inactive",
                        "closed_at": now,
                        "updated_at": now
                    }
                }
            )
            
            # Remove very old sessions
            old_cutoff = now - timedelta(days=days_old)
            
            # First get session IDs to remove
            old_sessions = list(self.db.conversation_sessions.find(
                {"created_at": {"$lt": old_cutoff}},
                {"session_id": 1}
            ))
            
            old_session_ids = [s["session_id"] for s in old_sessions]
            
            # Remove messages for old sessions
            messages_result = self.db.conversation_messages.delete_many(
                {"session_id": {"$in": old_session_ids}}
            )
            
            # Remove old sessions
            sessions_result = self.db.conversation_sessions.delete_many(
                {"created_at": {"$lt": old_cutoff}}
            )
            
            stats = {
                "sessions_closed": inactive_result.modified_count,
                "sessions_deleted": sessions_result.deleted_count,
                "messages_deleted": messages_result.deleted_count
            }
            
            logger.info(f"🧹 Session cleanup: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error during session cleanup: {e}")
            return {"sessions_closed": 0, "sessions_deleted": 0, "messages_deleted": 0}

    def get_user_conversations(self, user_email: str, limit: int = 20) -> List[Dict]:
        """
        Get user's conversation sessions
        
        Args:
            user_email: User's email address
            limit: Maximum number of sessions to return
            
        Returns:
            List of conversation sessions
        """
        try:
            sessions = list(
                self.db.conversation_sessions
                .find({"user_email": user_email})
                .sort("updated_at", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string and add summary info
            for session in sessions:
                session["_id"] = str(session["_id"])
                
                # Get first and last message for preview
                first_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", 1)]
                )
                
                last_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", -1)]
                )
                
                session["first_message"] = first_msg.get("question", "") if first_msg else ""
                session["last_message"] = last_msg.get("question", "") if last_msg else ""
            
            logger.info(f"📚 Retrieved {len(sessions)} conversation sessions for {user_email}")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error retrieving user conversations for {user_email}: {e}")
            return []

    def create_anonymous_conversation_session(self, initial_message: str = None, user_identifier: str = None) -> str:
        """
        Create a new anonymous conversation session for public users
        
        Args:
            initial_message: Optional initial message
            user_identifier: Optional anonymous identifier (IP, browser fingerprint, etc.)
            
        Returns:
            Session ID
        """
        try:
            session_data = {
                "session_id": str(uuid.uuid4()),
                "user_email": user_identifier or "anonymous",
                "is_anonymous": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "message_count": 0,
                "status": "active",
                "context_summary": "",
                "initial_message": initial_message
            }
            
            result = self.db.conversation_sessions.insert_one(session_data)
            session_id = session_data["session_id"]
            
            # Create index for faster queries
            try:
                self.db.conversation_sessions.create_index([
                    ("user_email", 1),
                    ("created_at", -1)
                ])
                self.db.conversation_sessions.create_index([("session_id", 1)])
                self.db.conversation_sessions.create_index([("is_anonymous", 1)])
            except Exception:
                pass  # Index might already exist
            
            logger.info(f"💬 Created anonymous conversation session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Error creating anonymous conversation session: {e}")
            raise

    def get_anonymous_user_conversations(self, user_identifier: str, limit: int = 20) -> List[Dict]:
        """
        Get anonymous user's conversation sessions
        
        Args:
            user_identifier: Anonymous user identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of conversation sessions
        """
        try:
            sessions = list(
                self.db.conversation_sessions
                .find({"user_email": user_identifier, "is_anonymous": True})
                .sort("updated_at", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string and add summary info
            for session in sessions:
                session["_id"] = str(session["_id"])
                
                # Get first and last message for preview
                first_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", 1)]
                )
                
                last_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", -1)]
                )
                
                session["first_message"] = first_msg.get("question", "") if first_msg else ""
                session["last_message"] = last_msg.get("question", "") if last_msg else ""
            
            logger.info(f"📚 Retrieved {len(sessions)} anonymous conversation sessions for {user_identifier}")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error retrieving anonymous user conversations for {user_identifier}: {e}")
            return []

    def delete_user_session(self, session_id: str) -> bool:
        """
        Delete a specific conversation session and all its messages
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete all messages for this session
            messages_result = self.db.conversation_messages.delete_many(
                {"session_id": session_id}
            )
            
            # Delete the session itself
            session_result = self.db.conversation_sessions.delete_one(
                {"session_id": session_id}
            )
            
            success = session_result.deleted_count > 0
            if success:
                logger.info(f"🗑️ Deleted session {session_id} and {messages_result.deleted_count} messages")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting session {session_id}: {e}")
            return False

    def get_user_sessions_by_identifier(self, user_identifier: str, limit: int = 20) -> List[Dict]:
        """
        Get conversation sessions by user identifier (for anonymous users)
        
        Args:
            user_identifier: User identifier (e.g., "anon_192.168.1.1")
            limit: Maximum number of sessions to return
            
        Returns:
            List of conversation sessions
        """
        try:
            sessions = list(
                self.db.conversation_sessions
                .find({"user_email": user_identifier})
                .sort("updated_at", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string and add summary info
            for session in sessions:
                session["_id"] = str(session["_id"])
                
                # Get first and last message for preview
                first_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", 1)]
                )
                
                last_msg = self.db.conversation_messages.find_one(
                    {"session_id": session["session_id"]},
                    sort=[("timestamp", -1)]
                )
                
                session["first_message"] = first_msg.get("question", "") if first_msg else ""
                session["last_message"] = last_msg.get("question", "") if last_msg else ""
            
            logger.info(f"📚 Retrieved {len(sessions)} sessions for identifier {user_identifier}")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error retrieving sessions for {user_identifier}: {e}")
            return []

    def delete_all_user_sessions(self, user_identifier: str) -> int:
        """
        Delete ALL conversation sessions for a user identifier
        
        Args:
            user_identifier: User identifier to delete sessions for
            
        Returns:
            Number of sessions deleted
        """
        try:
            # Get all session IDs for this user
            sessions = list(self.db.conversation_sessions.find(
                {"user_email": user_identifier},
                {"session_id": 1}
            ))
            
            session_ids = [s["session_id"] for s in sessions]
            
            if not session_ids:
                return 0
            
            # Delete all messages for these sessions
            messages_result = self.db.conversation_messages.delete_many(
                {"session_id": {"$in": session_ids}}
            )
            
            # Delete all sessions for this user
            sessions_result = self.db.conversation_sessions.delete_many(
                {"user_email": user_identifier}
            )
            
            deleted_count = sessions_result.deleted_count
            if deleted_count > 0:
                logger.info(f"🗑️ Deleted {deleted_count} sessions and {messages_result.deleted_count} messages for {user_identifier}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error deleting all sessions for {user_identifier}: {e}")
            return 0