"""
IntelliDocs API Backend v2.0 - Persistent Document Storage
Advanced RAG API with continuous chatbot entity and MongoDB Atlas integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Annotated
import weaviate
from weaviate.classes.data import DataObject
from openai import AsyncOpenAI
import google.generativeai as genai
from sentence_transformers import CrossEncoder
import uvicorn
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import uuid
import os
import shutil
import jwt
from passlib.context import CryptContext
import asyncio

# Import modular components
from config import (
    OPENAI_API_KEY, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT,
    COLLECTION_NAME, EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
    MONGODB_URL, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME, JWT_SECRET_KEY, JWT_ALGORITHM, 
    JWT_EXPIRE_MINUTES, ADMIN_EMAIL, ADMIN_PASSWORD, PERSISTENT_COLLECTION_NAME
)
# Import the new modular RAG system
from rag_orchestrator import ask_question, ask_question_stream, ask_question_voice, create_optimized_embeddings
from document_processor import load_and_process_documents
from mongodb_manager import MongoDBManager
from speech_integration import SpeechProcessor


# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and clients
embedding_model = None
cross_encoder_model = None
weaviate_client = None
openai_client = None
mongodb_manager = None
speech_processor = None

# JWT Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# JWT utility functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Get password hash"""
    return pwd_context.hash(password)

def authenticate_admin(email: str, password: str) -> bool:
    """Authenticate admin user"""
    return email == ADMIN_EMAIL and password == ADMIN_PASSWORD

async def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Additional check: ensure it's admin
        if email != ADMIN_EMAIL:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Auth dependency for protected endpoints
AuthRequired = Annotated[str, Depends(verify_jwt_token)]

# Request/Response models
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    email: str

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]
    timestamp: str

class UploadResponse(BaseModel):
    status: str
    message: str
    files_processed: int
    session_id: str
    
class DocumentInfo(BaseModel):
    id: str
    file_name: str
    file_size: int
    created_at: str
    status: str
    chunks_count: Optional[int] = 0

class DocumentManagementResponse(BaseModel):
    status: str
    message: str
    documents_count: int

class DocumentChunk(BaseModel):
    chunk_index: int
    content: str
    source: str
    article: Optional[str] = None

class DocumentContent(BaseModel):
    document_id: str
    file_name: str
    file_size: int
    created_at: str
    status: str
    total_chunks: int
    full_content: str  # Tüm içerik birleştirilmiş halde
    raw_chunks: Optional[List[DocumentChunk]] = None  # İsteğe bağlı chunk detayları

class DocumentDetails(BaseModel):
    document_id: str
    file_name: str
    file_size: int
    created_at: str
    status: str
    chunks_count: int
    metadata: Dict[str, Any]
    weaviate_chunks: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global embedding_model, cross_encoder_model, weaviate_client, openai_client, mongodb_manager, speech_processor
    
    try:
        # Startup: Load models and initialize connections
        logger.info("🚀 Starting IntelliDocs API Backend v2.0...")
        
        # Load AI models
        logger.info(f"Loading Bi-Encoder model: {EMBEDDING_MODEL}")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        embedding_model = EMBEDDING_MODEL
        
        logger.info(f"Loading Cross-Encoder model: {CROSS_ENCODER_MODEL}")
        cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
        
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize MongoDB Manager
        logger.info("Connecting to MongoDB Atlas...")
        mongodb_manager = MongoDBManager(MONGODB_URL, MONGODB_DB_NAME)
        
        # Initialize Weaviate client
        logger.info("Connecting to Weaviate database...")
        weaviate_client = weaviate.connect_to_local(
            host=WEAVIATE_HOST, 
            port=WEAVIATE_PORT, 
            grpc_port=WEAVIATE_GRPC_PORT
        )
        
        # Create persistent collection in Weaviate if it doesn't exist
        await setup_persistent_collection()
        
        # Initialize Speech Processor
        logger.info("Initializing Speech Processor...")
        speech_processor = SpeechProcessor(whisper_model_name="small")
        
        # Initialize Transcript Advisor
        await initialize_transcript_advisor()
        
        # Create upload directory for temporary files
        os.makedirs("./temp_uploads", exist_ok=True)
        
        logger.info("✅ All models loaded and connections established successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown: Clean up connections
        logger.info("🛑 Shutting down IntelliDocs API v2.0...")
        if weaviate_client:
            weaviate_client.close()
        if mongodb_manager:
            mongodb_manager.close()
        if speech_processor:
            speech_processor.cleanup_temp_files()
        logger.info("All connections closed.")

async def setup_persistent_collection():
    """Setup the persistent Weaviate collection for all documents"""
    global weaviate_client
    
    try:
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            logger.info(f"🗄️ Creating persistent collection: {PERSISTENT_COLLECTION_NAME}")
            
            import weaviate.classes.config as wvc
            
            weaviate_client.collections.create(
                name=PERSISTENT_COLLECTION_NAME,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                properties=[
                    wvc.Property(name="source", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="article", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="document_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="chunk_index", data_type=wvc.DataType.INT)
                ]
            )
            logger.info("✅ Persistent collection created successfully")
        else:
            logger.info("✅ Persistent collection already exists")
    
    except Exception as e:
        logger.error(f"❌ Error setting up persistent collection: {e}")
        raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="IntelliDocs API v2.0",
    description="Advanced RAG API with persistent document storage and continuous chatbot entity",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware - Allow all origins for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - API status check"""
    return {
        "service": "IntelliDocs API v2.0",
        "version": "2.0.0",
        "status": "running",
        "description": "Advanced RAG API with persistent document storage and continuous chatbot entity",
        "features": [
            "Persistent document storage",
            "No session management",
            "Dynamic document management",
            "MongoDB Atlas integration",
            "Speech-to-Speech voice assistant",
            "Text-to-Speech responses"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global weaviate_client, embedding_model, cross_encoder_model, openai_client, mongodb_manager, speech_processor
    
    try:
        # Check if all components are loaded
        if not all([weaviate_client, embedding_model, cross_encoder_model, openai_client, mongodb_manager]):
            raise Exception("Some components are not initialized")
        
        # Get database stats
        db_stats = mongodb_manager.get_database_stats()
        
        return {
            "status": "healthy",
            "message": "All systems operational", 
            "components": {
                "weaviate": "connected",
                "embedding_model": "loaded",
                "cross_encoder": "loaded",
                "openai": "connected",
                "mongodb": "connected",
                "speech_processor": "loaded" if speech_processor else "not loaded"
            },
            "database_stats": db_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    token: AuthRequired,
    files: List[UploadFile] = File(...)
):
    """Upload and process documents to persistent storage with progress tracking"""
    global mongodb_manager
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create unique session ID for this upload
    session_id = str(uuid.uuid4())
    
    # Create temporary directory for this upload batch
    temp_dir = f"./temp_uploads/{session_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded files temporarily
    uploaded_files = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
            
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    
    if not uploaded_files:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")
    
    # Create upload session for progress tracking
    user_info = {"user_email": token, "upload_time": datetime.now().isoformat()}
    mongodb_manager.create_upload_session(session_id, uploaded_files, user_info)
    
    # Process documents in background with progress tracking
    background_tasks.add_task(
        process_uploaded_documents_with_progress, 
        session_id,
        temp_dir,
        uploaded_files
    )
    
    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded_files)} files. Processing started...",
        "files_processed": len(uploaded_files),
        "session_id": session_id  # Return session ID for progress tracking
    }

@app.post("/api/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Admin login endpoint"""
    if not authenticate_admin(login_request.email, login_request.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=JWT_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": login_request.email},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRE_MINUTES * 60,  # Convert to seconds
        email=login_request.email
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_documents(message: ChatMessage):
    """Chat with all documents in persistent storage"""
    global weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # Get persistent collection
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        # Process the question
        answer, sources_metadata = await ask_question(
            question=message.message.strip(),
            collection=collection,
            openai_client=openai_client,
            model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            domain_context=""
        )
        
        # Store chat interaction in MongoDB
        mongodb_manager.store_chat_message(
            question=message.message.strip(),
            answer=answer,
            sources=sources_metadata
        )
        
        return ChatResponse(
            response=answer,
            sources=sources_metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/api/chat/stream")
async def chat_with_documents_stream(message: ChatMessage):
    """Stream chat responses with all documents in persistent storage"""
    global weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # Get persistent collection
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        async def generate_stream():
            """Generate streaming response"""
            import json
            try:
                collected_response = ""
                sources_metadata = []
                
                # Use normal ask_question for consistent quality, then simulate streaming
                answer, sources_metadata = await ask_question(
                    question=message.message.strip(),
                    collection=collection,
                    openai_client=openai_client,
                    model=embedding_model,
                    cross_encoder_model=cross_encoder_model,
                    domain_context=""
                )
                
                # Simulate streaming by sending words one by one
                import json
                import asyncio
                
                words = answer.split()
                for i, word in enumerate(words):
                    chunk_data = json.dumps({
                        "type": "content",
                        "content": word + (" " if i < len(words) - 1 else ""),
                        "done": False
                    })
                    yield f"data: {chunk_data}\n\n"
                    
                    # Small delay for streaming effect
                    await asyncio.sleep(0.03)
                
                # Send completion
                completion_data = {
                    "type": "complete",
                    "content": "",
                    "done": True,
                    "sources": sources_metadata,
                    "full_response": answer,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
                collected_response = answer
                
                # Store chat interaction in MongoDB
                if collected_response:
                    mongodb_manager.store_chat_message(
                        question=message.message.strip(),
                        answer=collected_response,
                        sources=sources_metadata
                    )
                        
                # Send final end marker
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"❌ Error in streaming: {e}")
                error_data = {
                    "type": "error",
                    "content": f"Streaming sırasında hata oluştu: {str(e)}",
                    "done": True,
                    "sources": []
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in streaming chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing streaming question: {str(e)}")

@app.get("/api/upload/progress/{session_id}")
async def get_upload_progress(session_id: str):
    """Get upload progress for a session"""
    global mongodb_manager
    
    try:
        session = mongodb_manager.get_upload_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "progress_percentage": session["progress_percentage"],
            "summary": session["summary"],
            "files": session["files"],
            "started_at": session["started_at"].isoformat() if session["started_at"] else None,
            "completed_at": session["completed_at"].isoformat() if session["completed_at"] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting upload progress: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving upload progress: {str(e)}")

@app.get("/api/upload/sessions")
async def get_recent_upload_sessions(token: AuthRequired, limit: int = 10):
    """Get recent upload sessions"""
    global mongodb_manager
    
    try:
        sessions = mongodb_manager.get_recent_upload_sessions(limit)
        
        # Format sessions for response
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                "session_id": session["session_id"],
                "status": session["status"],
                "progress_percentage": session["progress_percentage"],
                "summary": session["summary"],
                "started_at": session["started_at"].isoformat() if session["started_at"] else None,
                "completed_at": session["completed_at"].isoformat() if session["completed_at"] else None,
                "total_files": session["total_files"]
            })
        
        return {"upload_sessions": formatted_sessions}
        
    except Exception as e:
        logger.error(f"❌ Error getting upload sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving upload sessions: {str(e)}")

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents(token: AuthRequired):
    """List all documents in persistent storage"""
    global mongodb_manager
    
    try:
        documents = mongodb_manager.get_all_documents()
        
        document_list = []
        for doc in documents:
            document_list.append(DocumentInfo(
                id=doc["_id"],
                file_name=doc["file_name"],
                file_size=doc["file_size"],
                created_at=doc["created_at"].isoformat(),
                status=doc["status"],
                chunks_count=doc.get("chunks_count", 0)
            ))
        
        return document_list
        
    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.get("/api/documents/{document_id}/content", response_model=DocumentContent)
async def get_document_content(document_id: str, token: AuthRequired, show_chunks: bool = False):
    """Get full content of a specific document as continuous text or with chunks"""
    global mongodb_manager, weaviate_client
    
    try:
        # Get document info from MongoDB
        doc = mongodb_manager.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks from MongoDB
        chunks_data = mongodb_manager.get_document_chunks(document_id)
        
        # Format chunks for internal processing
        chunks = []
        for i, chunk in enumerate(chunks_data):
            chunks.append(DocumentChunk(
                chunk_index=i,
                content=chunk.get("content", ""),
                source=chunk.get("source", doc["file_name"]),
                article=chunk.get("article", None)
            ))
        
        # Combine all chunk contents into one continuous text
        full_content = ""
        for chunk in chunks:
            content = chunk.content.strip()
            if content:
                # Add article header if available
                if chunk.article:
                    full_content += f"\n\n=== {chunk.article} ===\n"
                full_content += content + "\n"
        
        # Clean up extra whitespace
        full_content = full_content.strip()
        
        return DocumentContent(
            document_id=document_id,
            file_name=doc["file_name"],
            file_size=doc["file_size"],
            created_at=doc["created_at"].isoformat(),
            status=doc["status"],
            total_chunks=len(chunks),
            full_content=full_content,
            raw_chunks=chunks if show_chunks else None  # Only include chunks if requested
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting document content: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document content: {str(e)}")

@app.get("/api/documents/{document_id}/info", response_model=DocumentDetails)
async def get_document_details(document_id: str, token: AuthRequired):
    """Get detailed information about a specific document"""
    global mongodb_manager, weaviate_client
    
    try:
        # Get document info from MongoDB
        doc = mongodb_manager.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks count from MongoDB
        chunks_data = mongodb_manager.get_document_chunks(document_id)
        chunks_count = len(chunks_data)
        
        # Get Weaviate chunks count using simplified approach
        weaviate_chunks = 0
        if weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
            
            # Use simpler query approach - get all objects and count matches
            try:
                from weaviate.classes.query import Filter
                response_count = collection.aggregate.over_all(
                    filters=Filter.by_property("document_id").equal(document_id),
                    total_count=True
                )
                weaviate_chunks = response_count.total_count
            except Exception as weaviate_error:
                logger.warning(f"⚠️ Could not get Weaviate count for document {document_id}: {weaviate_error}")
                # Fallback: Use MongoDB chunks count
                weaviate_chunks = chunks_count
        
        return DocumentDetails(
            document_id=document_id,
            file_name=doc["file_name"],
            file_size=doc["file_size"],
            created_at=doc["created_at"].isoformat(),
            status=doc["status"],
            chunks_count=chunks_count,
            metadata=doc.get("metadata", {}),
            weaviate_chunks=weaviate_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document details: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str, token: AuthRequired):
    """Remove a document from persistent storage"""
    global mongodb_manager, weaviate_client
    
    try:
        # Get document info
        doc = mongodb_manager.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove document chunks from Weaviate
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        try:
            from weaviate.classes.query import Filter
            collection.data.delete_many(
                where=Filter.by_property("document_id").equal(document_id)
            )
            logger.info(f"✅ Removed Weaviate chunks for document: {document_id}")
        except Exception as weaviate_error:
            logger.warning(f"⚠️ Could not remove Weaviate chunks for document {document_id}: {weaviate_error}")
            # Continue with MongoDB removal even if Weaviate fails
        
        # Remove document from MongoDB
        success = mongodb_manager.remove_document(document_id)
        
        if success:
            return {
                "status": "success", 
                "message": f"Document '{doc['file_name']}' removed successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to remove document")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error removing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing document: {str(e)}")

@app.delete("/api/documents")
async def clear_all_documents(token: AuthRequired):
    """Clear all documents from persistent storage"""
    global mongodb_manager, weaviate_client
    
    try:
        # Get all documents
        documents = mongodb_manager.get_all_documents()
        
        # Clear Weaviate collection
        if weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            weaviate_client.collections.delete(PERSISTENT_COLLECTION_NAME)
            # Recreate empty collection
            await setup_persistent_collection()
        
        # Clear MongoDB documents
        count = len(documents)
        for doc in documents:
            mongodb_manager.remove_document(doc["_id"])
        
        return DocumentManagementResponse(
            status="success",
            message=f"Cleared {count} documents from persistent storage",
            documents_count=0
        )
        
    except Exception as e:
        logger.error(f"❌ Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/api/chat/history")
async def get_chat_history(token: AuthRequired, limit: int = 50):
    """Get recent chat history"""
    global mongodb_manager
    
    try:
        history = mongodb_manager.get_chat_history(limit=limit)
        return {"chat_history": history}
        
    except Exception as e:
        logger.error(f"❌ Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.delete("/api/chat/history")
async def clear_chat_history(token: AuthRequired):
    """Clear all chat history"""
    global mongodb_manager
    
    try:
        count = mongodb_manager.clear_chat_history()
        return {
            "status": "success",
            "message": f"Cleared {count} chat messages"
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

@app.get("/api/stats")
async def get_system_stats(token: AuthRequired):
    """Get system statistics"""
    global mongodb_manager, weaviate_client
    
    try:
        # Get MongoDB stats
        db_stats = mongodb_manager.get_database_stats()
        
        # Get Weaviate collection info
        weaviate_objects = 0
        if weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
            response = collection.aggregate.over_all(total_count=True)
            weaviate_objects = response.total_count
        
        return {
            "mongodb_stats": db_stats,
            "weaviate_objects": weaviate_objects,
            "persistent_collection": PERSISTENT_COLLECTION_NAME
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

async def process_uploaded_documents(temp_dir: str, uploaded_files: List[str]):
    """Process uploaded documents and add to persistent storage"""
    global mongodb_manager, weaviate_client, embedding_model
    
    try:
        logger.info(f"🔄 Processing {len(uploaded_files)} documents for persistent storage")
        
        for filename in uploaded_files:
            file_path = os.path.join(temp_dir, filename)
            
            try:
                # Check if document already exists
                existing_doc = mongodb_manager.get_document_by_filename(filename)
                if existing_doc:
                    logger.info(f"📄 Document {filename} already exists, skipping...")
                    continue
                
                # Read file content
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Store document in MongoDB with processing status
                doc_id = mongodb_manager.store_document(
                    file_name=filename, 
                    file_content=file_content,
                    metadata={"uploaded_at": datetime.utcnow().isoformat()},
                    status="processing"
                )
                
                # Process document chunks
                documents = load_and_process_documents(temp_dir, specific_files=[filename])
                
                if documents:
                    # Store chunks in MongoDB
                    chunk_count = mongodb_manager.store_document_chunks(doc_id, documents)
                    logger.info(f"📑 Stored {chunk_count} chunks for {filename}")
                    
                    # Generate embeddings and store in Weaviate
                    collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
                    
                    from weaviate.classes.data import DataObject
                    data_objects = []
                    
                    embeddings = await create_optimized_embeddings(documents, embedding_model)

                    for i, doc in enumerate(documents):
                        try:
                            vector = embeddings[i]
                            
                            # Add document metadata
                            doc_properties = dict(doc)
                            doc_properties["document_id"] = doc_id
                            doc_properties["chunk_index"] = i
                            
                            data_objects.append(DataObject(
                                properties=doc_properties, 
                                vector=vector
                            ))
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing chunk {i} for {filename}: {e}")
                            continue
                    
                    if data_objects:
                        collection.data.insert_many(data_objects)
                        logger.info(f"✅ Added {len(data_objects)} chunks to Weaviate for {filename}")
                        
                        # Update document status to processed
                        mongodb_manager.update_document_status(doc_id, "processed")
                    
                else:
                    logger.warning(f"⚠️ No chunks extracted from {filename}")
                    # Mark as failed if no chunks extracted
                    mongodb_manager.update_document_status(doc_id, "failed")
                
            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}")
                continue
        
        logger.info("🎉 Document processing completed!")
        
    except Exception as e:
        logger.error(f"❌ Critical error in document processing: {e}")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

async def process_uploaded_documents_with_progress(session_id: str, temp_dir: str, uploaded_files: List[str]):
    """Optimized document processing with parallel operations and progress tracking"""
    global weaviate_client, embedding_model, mongodb_manager
    
    logger.info(f"📦 Starting optimized document processing for session: {session_id}")
    
    try:
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        # Process files in batches for better performance
        batch_size = 3  # Process max 3 files concurrently
        
        for i in range(0, len(uploaded_files), batch_size):
            batch_files = uploaded_files[i:i+batch_size]
            
            # Process batch in parallel
            tasks = []
            for filename in batch_files:
                task = process_single_file_optimized(
                    session_id, temp_dir, filename, collection
                )
                tasks.append(task)
            
            # Wait for batch completion
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"🎉 Optimized document processing completed for session: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in optimized document processing session {session_id}: {e}")
        # Mark any remaining pending files as failed
        session = mongodb_manager.get_upload_session(session_id)
        if session:
            for file_info in session.get("files", []):
                if file_info["status"] == "pending":
                    mongodb_manager.update_upload_progress(
                        session_id, file_info["filename"], "failed",
                        error_message="Processing session failed"
                    )
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

async def process_single_file_optimized(session_id: str, temp_dir: str, filename: str, collection):
    """Process a single file with full optimization"""
    global embedding_model, mongodb_manager
    try:
        logger.info(f"🔄 Processing file: {filename}")
        # Update status to processing
        mongodb_manager.update_upload_progress(session_id, filename, "processing")

        file_path = os.path.join(temp_dir, filename)

        # Check if document already exists
        existing_doc = mongodb_manager.get_document_by_filename(filename)
        if existing_doc:
            logger.info(f"📄 Document {filename} already exists, skipping...")
            mongodb_manager.update_upload_progress(
                session_id, filename, "completed",
                document_id=existing_doc["_id"]
            )
            return

        # Parallel operations: Read file + Process document
        def read_file_content():
            with open(file_path, 'rb') as f:
                return f.read()

        def process_document():
            return load_and_process_documents(temp_dir, specific_files=[filename])

        file_content, documents_data = await asyncio.gather(
            asyncio.to_thread(read_file_content),
            asyncio.to_thread(process_document)
        )

        if not documents_data:
            mongodb_manager.update_upload_progress(
                session_id, filename, "failed",
                error_message="Failed to extract content from PDF"
            )
            return

        logger.info(f"📄 Extracted {len(documents_data)} chunks from {filename}")

        # Parallel operations: Create embeddings + Store in MongoDB
        def create_embeddings():
            # Sync embedding creation - simple and reliable
            embeddings = []
            for doc in documents_data:
                try:
                    embedding = genai.embed_content(model=EMBEDDING_MODEL, content=doc['content'])['embedding']
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to create embedding for chunk: {e}")
                    # Fallback to zero vector
                    embeddings.append([0.0] * EMBEDDING_DIMENSION)
            return embeddings

        def store_in_mongodb():
            return mongodb_manager.store_document(
                file_name=filename,
                file_content=file_content,
                metadata={"processed_at": datetime.now().isoformat()},
                status="processing"
            )

        embeddings, doc_id = await asyncio.gather(
            asyncio.to_thread(create_embeddings),
            asyncio.to_thread(store_in_mongodb)
        )

        # Store chunks in MongoDB
        chunks_count = await asyncio.to_thread(
            mongodb_manager.store_document_chunks, doc_id, documents_data
        )

        # Add to Weaviate with final document_id
        data_objects = []
        for i, (doc, embedding) in enumerate(zip(documents_data, embeddings)):
            try:
                doc_properties = dict(doc)
                doc_properties["document_id"] = doc_id
                doc_properties["chunk_index"] = i
                data_objects.append(DataObject(
                    properties=doc_properties,
                    vector=embedding
                ))
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare chunk {i} for {filename}: {e}")

        # Batch insert to Weaviate
        if data_objects:
            await asyncio.to_thread(collection.data.insert_many, data_objects)
            logger.info(f"✅ Added {len(data_objects)} chunks to Weaviate for {filename}")

        # Mark as completed
        mongodb_manager.update_upload_progress(
            session_id, filename, "completed",
            document_id=doc_id, chunks_count=chunks_count
        )

        # Update document status
        mongodb_manager.update_document_status(doc_id, "processed")

        logger.info(f"✅ Successfully processed: {filename} ({chunks_count} chunks)")

    except Exception as file_error:
        logger.error(f"❌ Error processing file {filename}: {file_error}")
        mongodb_manager.update_upload_progress(
            session_id, filename, "failed",
            error_message=str(file_error)
        )

# =================== SPEECH ENDPOINTS ===================

@app.post("/api/speech-to-speech")
async def speech_to_speech_endpoint(
    request: Request,
    audio_file: UploadFile = File(...),
    voice: str = Form("tr-TR-EmelNeural"),
    language: str = Form("tr")
):
    """Sesli Asistan: Ses → STT → RAG → TTS → Ses"""
    global speech_processor, weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # 🔍 Başlangıç kontrolü - Client hala bağlı mı?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected at start")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # Gelen ses dosyasını geçici olarak kaydet
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            temp_audio_path = tmp_file.name
            speech_processor.temp_files.append(temp_audio_path)
        
        # STT: Ses → Metin
        logger.info("🎤 Speech-to-Text conversion...")
        
        # 🔍 STT başında kontrol
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected BEFORE STT")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # STT işlemi - timeout yok, sadece disconnection check
        recognized_text = await asyncio.to_thread(
            speech_processor.speech_to_text, temp_audio_path, language
        )
        
        # 🔍 STT içinde kontrol (çünkü STT uzun sürebilir)
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected DURING STT")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        if not recognized_text:
            raise HTTPException(status_code=400, detail="Ses tanınamadı")
        
        logger.info(f"🎤 Recognized text: {recognized_text}")
        
        # 🔍 STT sonrası kontrol - Client hala var mı?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected after STT")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # RAG: Soru → Cevap (voice-specific version)
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        logger.info("🧠 Processing with RAG engine...")
        # RAG işlemi - Her step'te disconnection check
        try:
            rag_response, sources_metadata = await ask_question_voice(
                question=recognized_text.strip(),
                collection=collection,
                openai_client=openai_client,
                model=embedding_model,
                cross_encoder_model=cross_encoder_model,
                domain_context="",
                request=request  # 🔍 Request object geçiliyor
            )
        except Exception as e:
            if "Client disconnected" in str(e):
                logger.info(f"🚪 {str(e)}")
                speech_processor.cleanup_temp_files()
                raise HTTPException(status_code=499, detail="Client disconnected during RAG")
            else:
                raise e
        
        # 🔍 RAG sonrası kontrol - Client hala dinliyor mu?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected after RAG")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # TTS: Cevap → Ses
        logger.info("🔊 Text-to-Speech conversion...")
        # TTS işlemi - timeout yok, sadece disconnection check
        audio_path = await speech_processor.text_to_speech(rag_response, voice)
        
        if not audio_path:
            raise HTTPException(status_code=500, detail="TTS oluşturulamadı")
        
        # Store chat interaction in MongoDB
        mongodb_manager.store_chat_message(
            question=recognized_text.strip(),
            answer=rag_response,
            sources=sources_metadata,
            interaction_type="voice"
        )
        
        # 🔍 Final kontrol - Client hala bekliyior mu?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected before response")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # Ses dosyasını döndür
        from fastapi.responses import FileResponse
        try:
            return FileResponse(
                audio_path,
                media_type="audio/mpeg",
                filename="response.mp3",
                headers={"Cache-Control": "no-cache"}
            )
        except (ConnectionResetError, BrokenPipeError):
            logger.info("🚪 Connection broken while sending response")
            speech_processor.cleanup_temp_files()
            raise HTTPException(status_code=499, detail="Connection broken")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Speech-to-Speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/text-to-speech")
async def text_to_speech_endpoint(
    request: Request,
    text: str = Form(...),
    voice: str = Form("tr-TR-EmelNeural")
):
    """Metin → RAG → TTS"""
    global speech_processor, weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # 🔍 Başlangıç kontrolü - Client hala bağlı mı?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected at start")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # RAG: Soru → Cevap (voice-specific version)
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        logger.info(f"🧠 Processing text query: {text}")
        try:
            rag_response, sources_metadata = await ask_question_voice(
                question=text.strip(),
                collection=collection,
                openai_client=openai_client,
                model=embedding_model,
                cross_encoder_model=cross_encoder_model,
                domain_context="",
                request=request  # 🔍 Request object geçiliyor
            )
        except Exception as e:
            if "Client disconnected" in str(e):
                logger.info(f"🚪 {str(e)}")
                raise HTTPException(status_code=499, detail="Client disconnected during RAG")
            else:
                raise e
        
        # 🔍 RAG sonrası kontrol - Client hala dinliyor mu?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected after RAG")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # TTS: Cevap → Ses
        logger.info("🔊 Text-to-Speech conversion...")
        # TTS işlemi - timeout yok, sadece disconnection check  
        audio_path = await speech_processor.text_to_speech(rag_response, voice)
        
        if not audio_path:
            raise HTTPException(status_code=500, detail="TTS oluşturulamadı")
        
        # Store chat interaction in MongoDB
        mongodb_manager.store_chat_message(
            question=text.strip(),
            answer=rag_response,
            sources=sources_metadata,
            interaction_type="voice"
        )
        
        # 🔍 Final kontrol - Client hala bekliyior mu?
        if await request.is_disconnected():
            logger.info("🚪 Client disconnected before response")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # Ses dosyasını döndür
        from fastapi.responses import FileResponse
        try:
            return FileResponse(
                audio_path,
                media_type="audio/mpeg", 
                filename="response.mp3",
                headers={"Cache-Control": "no-cache"}
            )
        except (ConnectionResetError, BrokenPipeError):
            logger.info("🚪 Connection broken while sending response")
            raise HTTPException(status_code=499, detail="Connection broken")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text-to-Speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speech/voices")
async def get_speech_voices():
    """Mevcut sesleri listele"""
    global speech_processor
    
    return {
        "voices": speech_processor.get_available_voices() if speech_processor else {},
        "default": "tr-TR-EmelNeural"
    }

@app.post("/api/admin/clear-cache")
async def clear_system_cache(token: AuthRequired):
    """Clear all system caches for debugging"""
    try:
        # Import all cache objects
        from query_processor import query_analysis_cache, query_variants_cache
        from search_engine import search_cache
        from embedding_engine import embedding_cache
        from response_generator import response_cache
        
        # Clear all caches
        query_analysis_cache.clear()
        query_variants_cache.clear()
        search_cache.clear()
        embedding_cache.clear()
        response_cache.clear()
        
        return {
            "status": "success",
            "message": "All system caches cleared successfully",
            "cleared_caches": [
                "query_analysis_cache",
                "query_variants_cache", 
                "search_cache",
                "embedding_cache",
                "response_cache"
            ]
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")

# =================== TRANSCRIPT ADVISOR ENDPOINTS ===================

# Transcript Advisor global instance
transcript_advisor = None

async def initialize_transcript_advisor():
    """Initialize transcript advisor on startup"""
    global transcript_advisor, openai_client
    try:
        from transcript_advisor import IntelligentTranscriptAdvisor
        transcript_advisor = IntelligentTranscriptAdvisor(openai_client)
        logger.info("🎓 Transcript Advisor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Transcript Advisor: {e}")

class TranscriptUploadRequest(BaseModel):
    transcript_text: str

class TranscriptAnalysisResponse(BaseModel):
    status: str
    message: str
    analysis: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]

class AdvisorQuestionRequest(BaseModel):
    question: str

class AdvisorQuestionResponse(BaseModel):
    status: str
    message: str
    response: str
    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    urgency_level: str = "normal"

@app.post("/api/transcript/upload-pdf", response_model=TranscriptAnalysisResponse)
async def upload_transcript_pdf(file: UploadFile = File(...)):
    """Transcript PDF'ini yükle ve analiz et"""
    global transcript_advisor
    
    try:
        if not transcript_advisor:
            raise HTTPException(status_code=503, detail="Transcript Advisor not initialized")
        
        # PDF dosyası kontrolü
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilir")
        
        # PDF'yi oku ve metne çevir
        import pypdf
        import tempfile
        import os
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # PDF'yi oku
            reader = pypdf.PdfReader(temp_file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            if not full_text.strip():
                raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı")
            
            # Transcript'i parse et
            student_profile, courses = transcript_advisor.parse_transcript(full_text)
            
            # Kapsamlı analiz yap (async fonksiyon olduğu için await ile çağır)
            analysis = await transcript_advisor.get_comprehensive_analysis()
            
            # Öneriler ve sonraki adımlar
            recommendations = [
                f"🎯 Akademik durumunuz: {student_profile.academic_standing}",
                f"📊 Mevcut GPA: {student_profile.current_gpa:.2f}/4.0",
                f"📚 Tamamlanan ders sayısı: {student_profile.completed_courses}",
                f"💳 Toplam kredi: {student_profile.total_credits}",
                f"🎓 ECTS: {student_profile.total_ects}",
                f"📋 Hazırlık durumu: {student_profile.prep_status} ({student_profile.prep_grade} puan)"
            ]
            
            next_steps = [
                "📋 Detaylı analiz için soru sorabilirsiniz",
                "🎓 Mezuniyet durumu hakkında bilgi alabilirsiniz", 
                "📚 Ders önerileri için danışabilirsiniz",
                "🚀 Kariyer planlaması yapabilirsiniz"
            ]
            
            return TranscriptAnalysisResponse(
                status="success",
                message=f"✅ Transcript PDF başarıyla analiz edildi! {student_profile.name} için kapsamlı değerlendirme hazır.",
                analysis=analysis,
                recommendations=recommendations,
                next_steps=next_steps
            )
            
        finally:
            # Geçici dosyayı sil
            os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF transcript analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF transcript analizi sırasında hata: {str(e)}")

# =================== END TRANSCRIPT ADVISOR ENDPOINTS ===================

if __name__ == "__main__":
    print("🚀 Starting IntelliDocs API Backend v2.0...")
    print("📚 Advanced RAG API with persistent document storage")
    print("🔗 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
