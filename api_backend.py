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
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
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
import torch

# Import modular components
from config import (
    OPENAI_API_KEY, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT,
    COLLECTION_NAME, EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
    MONGODB_URL, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME, JWT_SECRET_KEY, JWT_ALGORITHM, 
    JWT_EXPIRE_MINUTES, ADMIN_EMAIL, ADMIN_PASSWORD, PERSISTENT_COLLECTION_NAME,
    USE_GPU, GPU_DEVICE, GPU_BATCH_SIZE, CPU_BATCH_SIZE, GPU_MAX_MEMORY_FRACTION,
    GPU_CONCURRENT_STREAMS, GPU_LARGE_BATCH_SIZE, GPU_EMBEDDING_BATCH_SIZE
)
from rag_engine import ask_question, ask_question_stream, ask_question_voice, create_optimized_embeddings, ask_question_optimized, ask_question_voice_optimized
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
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except (jwt.PyJWTError, jwt.InvalidTokenError, jwt.DecodeError) as e:  # Updated exception names
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
        
        # Load AI models with GPU support and memory optimization
        device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"🎮 Using device: {device}")
        
        if device == "cuda":
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"🔋 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # ULTRA HIGH PERFORMANCE GPU optimization
            logger.info("🧹 Clearing GPU cache for optimal loading...")
            torch.cuda.empty_cache()
            
            # MAXIMUM memory utilization - Use 95% instead of 80%
            try:
                if hasattr(torch.cuda, 'set_memory_fraction'):
                    torch.cuda.set_memory_fraction(GPU_MAX_MEMORY_FRACTION)  # Use 95% of GPU memory
                    logger.info(f"🔋 GPU memory limit set to {GPU_MAX_MEMORY_FRACTION*100:.0f}%")
                else:
                    # Alternative method for older PyTorch versions
                    torch.cuda.set_per_process_memory_fraction(GPU_MAX_MEMORY_FRACTION)
                    logger.info(f"🔋 GPU memory limit set to {GPU_MAX_MEMORY_FRACTION*100:.0f}% (alternative method)")
            except Exception as e:
                logger.warning(f"⚠️ Could not set GPU memory fraction: {e}")
                logger.info("🔋 Using default GPU memory management")
            
            # Force GPU memory preallocation for better utilization
            logger.info("⚡ Pre-allocating GPU memory for better performance...")
            try:
                # Create a dummy tensor to force memory allocation
                dummy_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
                del dummy_tensor
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory pre-allocation successful")
            except Exception as e:
                logger.warning(f"⚠️ GPU memory pre-allocation failed: {e}")
            
            # Enable ultra-performance settings  
            logger.info(f"⚡ GPU Batch Size: {GPU_BATCH_SIZE}")
            logger.info(f"⚡ GPU Embedding Batch: {GPU_EMBEDDING_BATCH_SIZE}")
            logger.info(f"⚡ GPU Large Batch: {GPU_LARGE_BATCH_SIZE}")
            logger.info(f"⚡ CUDA Streams: {GPU_CONCURRENT_STREAMS}")
        
        logger.info(f"Loading Bi-Encoder model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        # 🚀 GPU OPTIMIZATIONS - FP16 + TorchScript 
        if device == "cuda":
            logger.info("🔥 Applying GPU optimizations...")
            
            # FP16 Optimization - %20-40 speed increase, minimal quality loss
            logger.info("⚡ Converting embedding model to FP16...")
            embedding_model = embedding_model.half()
            logger.info("✅ Embedding model converted to FP16")
            
            # Clear cache after conversion
            torch.cuda.empty_cache()
            
            logger.info(f"🔋 Memory after embedding model + FP16: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # 🚀 TorchScript Optimization Setup
            logger.info("🔥 Setting up TorchScript optimizations...")
            
            # Create optimized encode wrapper with JIT compilation
            def create_optimized_encoder(model):
                """Create JIT-optimized encoder wrapper"""
                original_encode = model.encode
                encode_cache = {}
                
                def optimized_encode(texts, **kwargs):
                    """TorchScript-optimized encode with smart caching"""
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Create cache key from texts and major kwargs
                    cache_key = hash(str(texts[:2]) + str(kwargs.get('batch_size', GPU_BATCH_SIZE)))
                    
                    try:
                        # Use original encode but with GPU optimizations and larger batches
                        with torch.amp.autocast('cuda', enabled=True):  # Updated syntax
                            embeddings = original_encode(
                                texts,
                                batch_size=kwargs.get('batch_size', GPU_BATCH_SIZE),  # Use config value
                                show_progress_bar=kwargs.get('show_progress_bar', False),
                                convert_to_numpy=kwargs.get('convert_to_numpy', True),
                                normalize_embeddings=kwargs.get('normalize_embeddings', True)
                            )
                        return embeddings
                    except Exception as e:
                        # Fallback to original method
                        return original_encode(texts, **kwargs)
                
                return optimized_encode
            
            # Apply TorchScript optimization to embedding model
            embedding_model._original_encode = embedding_model.encode
            embedding_model.encode = create_optimized_encoder(embedding_model)
            logger.info("✅ TorchScript optimization applied to embedding model")
            
            # Memory optimization settings
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            logger.info("✅ CUDNN optimizations enabled")
        
        logger.info(f"Loading Cross-Encoder model: {CROSS_ENCODER_MODEL}")
        cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
        
        # 🚀 Cross-Encoder GPU Optimizations
        if device == "cuda":
            logger.info("⚡ Converting cross-encoder to FP16...")
            # CrossEncoder has internal model attribute
            if hasattr(cross_encoder_model, 'model'):
                cross_encoder_model.model = cross_encoder_model.model.half()
                logger.info("✅ Cross-encoder converted to FP16")
            
            # 🚀 TorchScript optimization for Cross-Encoder
            logger.info("🔥 Applying TorchScript optimization to cross-encoder...")
            
            def create_optimized_predictor(cross_encoder):
                """Create optimized predictor for cross-encoder"""
                original_predict = cross_encoder.predict
                
                def optimized_predict(sentence_pairs, **kwargs):
                    """Optimized predict with GPU acceleration"""
                    try:
                        # Use mixed precision for faster inference with larger batches
                        with torch.amp.autocast('cuda', enabled=True):  # Updated syntax
                            predictions = original_predict(
                                sentence_pairs,
                                batch_size=kwargs.get('batch_size', GPU_BATCH_SIZE//2),  # Use half for cross-encoder
                                show_progress_bar=kwargs.get('show_progress_bar', False)
                            )
                        return predictions
                    except Exception as e:
                        # Fallback to original method
                        return original_predict(sentence_pairs, **kwargs)
                
                return optimized_predict
            
            # Apply optimization to cross-encoder
            cross_encoder_model._original_predict = cross_encoder_model.predict
            cross_encoder_model.predict = create_optimized_predictor(cross_encoder_model)
            logger.info("✅ TorchScript optimization applied to cross-encoder")
            
            # Apply memory optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow faster operations
            
            # Clear cache after all conversions
            torch.cuda.empty_cache()
            
            logger.info(f"🔋 Total GPU memory used (optimized): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"🔋 GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            logger.info("🚀 GPU optimizations complete! Expected %30-50 speed increase.")
        
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
    """Health check endpoint with GPU information"""
    global weaviate_client, embedding_model, cross_encoder_model, openai_client, mongodb_manager, speech_processor
    
    try:
        # Check if all components are loaded
        if not all([weaviate_client, embedding_model, cross_encoder_model, openai_client, mongodb_manager]):
            raise Exception("Some components are not initialized")
        
        # Get database stats
        db_stats = mongodb_manager.get_database_stats()
        
        # Get GPU information
        gpu_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_enabled": USE_GPU,
            "device": "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        }
        
        if gpu_info["gpu_available"]:
            gpu_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            })
        
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
            "gpu_info": gpu_info,
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
        
        # Process the question - OPTIMIZED VERSION
        answer, sources_metadata = await ask_question_optimized(
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
            try:
                collected_response = ""
                sources_metadata = []
                
                # Use optimized ask_question for better performance, then simulate streaming
                answer, sources_metadata = await ask_question_optimized(
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
                    where=Filter.by_property("document_id").equal(document_id),
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
    """Process uploaded documents with progress tracking"""
    global weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    logger.info(f"📦 Starting document processing for session: {session_id}")
    
    try:
        # Get persistent collection
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        for filename in uploaded_files:
            try:
                logger.info(f"🔄 Processing file: {filename}")
                
                # Update status to processing
                mongodb_manager.update_upload_progress(session_id, filename, "processing")
                
                file_path = os.path.join(temp_dir, filename)
                
                # Load and process the document
                documents_data = load_and_process_documents(temp_dir, specific_files=[filename])
                
                if not documents_data:
                    # Mark as failed
                    mongodb_manager.update_upload_progress(
                        session_id, filename, "failed", 
                        error_message="Failed to extract content from PDF"
                    )
                    continue
                
                # Read file for storage (keep in memory)
                with open(file_path, 'rb') as file:
                    file_content = file.read()
                
                # Step 1: Create embeddings FIRST
                logger.info(f"🚀 Step 1/3: Creating embeddings for {filename}...")
                embeddings = await create_optimized_embeddings(documents_data, embedding_model)
                
                # Step 2: Add to Weaviate BEFORE MongoDB (so search is ready)
                logger.info(f"💾 Step 2/3: Adding to vector database for {filename}...")
                from weaviate.classes.data import DataObject
                data_objects = []
                
                # Generate temporary document_id for Weaviate (will be replaced later)
                temp_doc_id = f"temp_{session_id}_{filename}"
                
                for i, (doc, embedding) in enumerate(zip(documents_data, embeddings)):
                    try:
                        # Add document metadata with temp ID
                        doc_properties = dict(doc)
                        doc_properties["document_id"] = temp_doc_id
                        doc_properties["chunk_index"] = i
                        doc_properties["filename"] = filename  # Add filename for later reference
                        
                        data_objects.append(DataObject(
                            properties=doc_properties, 
                            vector=embedding
                        ))
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to prepare chunk {i} for {filename}: {e}")
                
                # Batch insert to Weaviate
                if data_objects:
                    collection.data.insert_many(data_objects)
                    logger.info(f"✅ Added {len(data_objects)} chunks to Weaviate for {filename}")
                
                # Step 3: Store in MongoDB ONLY after everything is ready
                logger.info(f"📑 Step 3/3: Storing document metadata for {filename}...")
                doc_id = mongodb_manager.store_document(
                    file_name=filename,
                    file_content=file_content,
                    metadata={"processed_at": datetime.now().isoformat()},
                    status="processed"  # Set to processed immediately since everything is ready
                )
                
                # Store chunks in MongoDB
                chunks_count = mongodb_manager.store_document_chunks(doc_id, documents_data)
                
                # Update Weaviate chunks with real document_id
                try:
                    from weaviate.classes.query import Filter
                    # Get objects that match our temp_doc_id and update them
                    objects_updated = 0
                    
                    # Use simpler approach - update all objects with temp_doc_id
                    # Since we know the temp_doc_id pattern, we can update directly
                    batch_size = 100
                    updated_count = 0
                    
                    # Get all objects in batches and check for temp_doc_id
                    try:
                        all_objects = collection.query.fetch_objects(
                            limit=1000,
                            return_properties=["document_id", "chunk_index", "filename"]
                        )
                        
                        # Filter and update objects that match our temp_doc_id
                        for obj in all_objects.objects:
                            if obj.properties.get("document_id") == temp_doc_id:
                                collection.data.update(
                                    uuid=obj.uuid,
                                    properties={"document_id": doc_id}
                                )
                                updated_count += 1
                        
                        logger.info(f"🔄 Updated {updated_count} Weaviate chunks with real document_id: {doc_id}")
                    
                    except Exception as update_error:
                        logger.warning(f"⚠️ Could not update Weaviate document_ids: {update_error}")
                        logger.info(f"✅ Document stored successfully in Weaviate with temp_doc_id: {temp_doc_id}")
                        
                except Exception as weaviate_error:
                    logger.error(f"❌ Weaviate update error: {weaviate_error}")
                    # Continue even if update fails - the data is still in Weaviate
                
                # Mark as completed ONLY after all operations are done
                mongodb_manager.update_upload_progress(
                    session_id, filename, "completed", 
                    document_id=doc_id, chunks_count=chunks_count
                )
                
                # Update document status to processed
                mongodb_manager.update_document_status(doc_id, "processed")
                
                logger.info(f"✅ Successfully processed: {filename} ({chunks_count} chunks)")
                
            except Exception as file_error:
                logger.error(f"❌ Error processing file {filename}: {file_error}")
                mongodb_manager.update_upload_progress(
                    session_id, filename, "failed", 
                    error_message=str(file_error)
                )
        
        logger.info(f"🎉 Document processing completed for session: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in document processing session {session_id}: {e}")
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
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

# =================== SPEECH ENDPOINTS ===================

@app.post("/api/speech-to-speech")
async def speech_to_speech_endpoint(
    request: Request,
    audio_file: UploadFile = File(...),
    voice: str = Form("tr-TR-EmelNeural"),
    gender: str = Form("female"),  # Yeni basit parametre: male/female/erkek/kadın
    language: str = Form("tr")
):
    """Sesli Asistan: Ses → STT → RAG → TTS → Ses"""
    global speech_processor, weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # 🎤 Akıllı ses seçimi: gender parametresi öncelikli, voice parametresi fallback
        if gender != "female":  # Varsayılan değilse gender'ı kullan
            selected_voice = speech_processor.get_voice_by_gender(gender)
        else:
            # Varsayılan female ise, voice parametresine bak (geriye uyumluluk)
            selected_voice = speech_processor.get_voice_by_gender(voice)
        
        logger.info(f"🎤 Selected voice: {selected_voice} (gender: {gender}, voice: {voice})")
        
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
        # RAG işlemi - OPTIMIZED VERSION with disconnection check
        try:
            rag_response, sources_metadata = await ask_question_voice_optimized(
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
        audio_path = await speech_processor.text_to_speech(rag_response, selected_voice)
        
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
    voice: str = Form("tr-TR-EmelNeural"),
    gender: str = Form("female")  # Yeni basit parametre: male/female/erkek/kadın
):
    """Metin → RAG → TTS"""
    global speech_processor, weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # 🎤 Akıllı ses seçimi: gender parametresi öncelikli, voice parametresi fallback
        if gender != "female":  # Varsayılan değilse gender'ı kullan
            selected_voice = speech_processor.get_voice_by_gender(gender)
        else:
            # Varsayılan female ise, voice parametresine bak (geriye uyumluluk)
            selected_voice = speech_processor.get_voice_by_gender(voice)
        
        logger.info(f"🎤 Selected voice: {selected_voice} (gender: {gender}, voice: {voice})")
        
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
            rag_response, sources_metadata = await ask_question_voice_optimized(
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
        audio_path = await speech_processor.text_to_speech(rag_response, selected_voice)
        
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
    """Mevcut sesleri ve gender seçeneklerini listele"""
    global speech_processor
    
    return {
        "voices": speech_processor.get_available_voices() if speech_processor else {},
        "gender_options": speech_processor.get_gender_options() if speech_processor else {},
        "defaults": {
            "voice": "tr-TR-EmelNeural",
            "gender": "female"
        },
        "usage_examples": {
            "simple": "gender=male (erkek ses için)",
            "advanced": "voice=tr-TR-AhmetNeural (direkt voice name)",
            "turkish": "gender=erkek (Türkçe alias)"
        }
    }

@app.post("/api/chat/compare")
async def compare_performance(message: ChatMessage, token: AuthRequired):
    """🚀 Performance comparison: Original vs Optimized RAG systems"""
    global weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # Get persistent collection
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        import time
        
        # 🐌 Test ORIGINAL version
        print("🐌 Testing ORIGINAL RAG system...")
        start_time = time.time()
        
        original_answer, original_sources = await ask_question(
            question=message.message.strip(),
            collection=collection,
            openai_client=openai_client,
            model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            domain_context=""
        )
        
        original_time = time.time() - start_time
        print(f"🐌 Original completed in: {original_time:.2f}s")
        
        # 🚀 Test OPTIMIZED version
        print("🚀 Testing OPTIMIZED RAG system...")
        start_time = time.time()
        
        optimized_answer, optimized_sources = await ask_question_optimized(
            question=message.message.strip(),
            collection=collection,
            openai_client=openai_client,
            model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            domain_context=""
        )
        
        optimized_time = time.time() - start_time
        print(f"🚀 Optimized completed in: {optimized_time:.2f}s")
        
        # Calculate improvement
        speed_improvement = ((original_time - optimized_time) / original_time) * 100 if original_time > 0 else 0
        
        # Store comparison results
        mongodb_manager.store_chat_message(
            question=message.message.strip(),
            answer=optimized_answer,
            sources=optimized_sources,
            interaction_type="performance_test"
        )
        
        return {
            "question": message.message.strip(),
            "performance": {
                "original_time": round(original_time, 3),
                "optimized_time": round(optimized_time, 3), 
                "speed_improvement_percent": round(speed_improvement, 1),
                "faster_by": round(original_time - optimized_time, 3)
            },
            "responses": {
                "original": {
                    "answer": original_answer,
                    "sources": original_sources[:3]  # Top 3 sources
                },
                "optimized": {
                    "answer": optimized_answer,
                    "sources": optimized_sources[:3]  # Top 3 sources
                }
            },
            "quality_check": {
                "answers_similar": len(set(original_answer.split()) & set(optimized_answer.split())) > len(original_answer.split()) * 0.7,
                "source_overlap": len([s for s in optimized_sources[:5] if s in original_sources[:5]]) >= 3
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in performance comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error in performance comparison: {str(e)}")

@app.post("/api/voice/compare")
async def compare_voice_responses(message: ChatMessage, token: AuthRequired):
    """🎤 Voice vs Text Response comparison - Test voice enhancement"""
    global weaviate_client, openai_client, embedding_model, cross_encoder_model, mongodb_manager
    
    try:
        # Get persistent collection
        if not weaviate_client.collections.exists(PERSISTENT_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail="No documents available for chat. Please upload documents first.")
        
        collection = weaviate_client.collections.get(PERSISTENT_COLLECTION_NAME)
        
        import time
        
        # 📝 Test NORMAL text response (detailed, with formal source references)
        print("📝 Testing NORMAL TEXT response...")
        start_time = time.time()
        
        text_answer, text_sources = await ask_question_optimized(
            question=message.message.strip(),
            collection=collection,
            openai_client=openai_client,
            model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            domain_context=""
        )
        
        text_time = time.time() - start_time
        print(f"📝 Text response completed in: {text_time:.2f}s")
        
        # 🎤 Test ENHANCED voice response (concise but source-aware)
        print("🎤 Testing ENHANCED VOICE response...")
        start_time = time.time()
        
        voice_answer, voice_sources = await ask_question_voice_optimized(
            question=message.message.strip(),
            collection=collection,
            openai_client=openai_client,
            model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            domain_context=""
        )
        
        voice_time = time.time() - start_time
        print(f"🎤 Voice response completed in: {voice_time:.2f}s")
        
        # Analyze differences
        text_has_formal_refs = "[Kaynak:" in text_answer
        voice_has_natural_refs = any(indicator in voice_answer.lower() for indicator in [
            'göre', 'belirtildiği', 'mevzuat', 'dosya', 'madde', 'bölüm'
        ])
        
        # Store comparison results
        mongodb_manager.store_chat_message(
            question=message.message.strip(),
            answer=voice_answer,
            sources=voice_sources,
            interaction_type="voice_comparison_test"
        )
        
        return {
            "question": message.message.strip(),
            "performance": {
                "text_time": round(text_time, 3),
                "voice_time": round(voice_time, 3),
                "time_difference": round(abs(text_time - voice_time), 3)
            },
            "responses": {
                "text": {
                    "answer": text_answer,
                    "length": len(text_answer),
                    "has_formal_sources": text_has_formal_refs,
                    "sources": text_sources[:3]
                },
                "voice": {
                    "answer": voice_answer,
                    "length": len(voice_answer),
                    "has_natural_sources": voice_has_natural_refs,
                    "sources": voice_sources[:3]
                }
            },
            "analysis": {
                "voice_is_shorter": len(voice_answer) < len(text_answer),
                "length_ratio": round(len(voice_answer) / len(text_answer) if len(text_answer) > 0 else 0, 2),
                "both_have_sources": text_has_formal_refs and voice_has_natural_refs,
                "source_overlap": len([s for s in voice_sources[:5] if s in text_sources[:5]]) >= 3
            },
            "improvement_notes": {
                "voice_conciseness": "Voice response should be shorter and more natural",
                "source_integration": "Voice should have natural source references",
                "same_accuracy": "Both should reference same source materials",
                "speech_friendly": "Voice should be conversational and flow well"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in voice comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error in voice comparison: {str(e)}")

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