#!/usr/bin/env python3
"""
Performance Debug Script
This script will help identify bottlenecks in the RAG system.
"""

import time
import asyncio
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI
from config import *

async def debug_rag_performance():
    """Debug each step of the RAG process to find bottlenecks"""
    print("🔍 DocBoss Performance Debug")
    print("=" * 50)
    
    # Test question
    question = "mühendislik bölümleri neler?"
    
    # Step 1: Check GPU
    print(f"🎮 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Step 2: Load models and measure time
    print("\n📊 Model Loading Times:")
    
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")
    
    # Embedding model loading
    start_time = time.time()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedding_load_time = time.time() - start_time
    print(f"⚡ Embedding model load: {embedding_load_time:.2f}s")
    
    # Cross-encoder loading
    start_time = time.time()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    cross_encoder_load_time = time.time() - start_time
    print(f"⚡ Cross-encoder load: {cross_encoder_load_time:.2f}s")
    
    # OpenAI client
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Step 3: Test embedding generation
    print("\n🧪 Embedding Generation Test:")
    
    test_texts = [
        "Bu bir test cümlesidir.",
        "Mühendislik bölümleri hakkında bilgi arıyorum.",
        "Üniversite bölümleri ve programları",
        question
    ]
    
    # CPU test
    if device == "cuda":
        cpu_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        start_time = time.time()
        cpu_embeddings = cpu_model.encode(test_texts)
        cpu_time = time.time() - start_time
        print(f"💻 CPU embeddings: {cpu_time:.3f}s")
        del cpu_model
    
    # GPU test
    start_time = time.time()
    gpu_embeddings = embedding_model.encode(test_texts)
    gpu_time = time.time() - start_time
    print(f"🎮 GPU embeddings: {gpu_time:.3f}s")
    
    if device == "cuda" and 'cpu_time' in locals():
        speedup = cpu_time / gpu_time
        print(f"🚀 GPU Speedup: {speedup:.2f}x")
    
    # Step 4: Test cross-encoder
    print("\n🎯 Cross-Encoder Test:")
    
    test_pairs = [
        (question, "Mühendislik bölümleri öğrencilere teknik eğitim sağlar"),
        (question, "Üniversitede çeşitli mühendislik programları bulunur"),
        (question, "Bilgisayar mühendisliği popüler bir bölümdür")
    ]
    
    start_time = time.time()
    scores = cross_encoder.predict(test_pairs)
    cross_encoder_time = time.time() - start_time
    print(f"⚡ Cross-encoder scoring: {cross_encoder_time:.3f}s")
    print(f"📊 Scores: {scores}")
    
    # Step 5: Test HyDE generation
    print("\n📝 HyDE Generation Test:")
    
    start_time = time.time()
    hyde_prompt = f"""Şu soruya verilen kapsamlı bir cevap yaz: {question}
    
    Cevap:"""
    
    response = await openai_client.chat.completions.create(
        model=HYDE_LLM_MODEL,
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.1,
        max_tokens=200
    )
    
    hyde_time = time.time() - start_time
    hyde_answer = response.choices[0].message.content.strip()
    print(f"⚡ HyDE generation: {hyde_time:.3f}s")
    print(f"📝 HyDE answer: {hyde_answer[:100]}...")
    
    # Step 6: Test final response generation
    print("\n💬 Final Response Test:")
    
    mock_context = """
    Ankara Bilim Üniversitesi'nde şu mühendislik bölümleri bulunmaktadır:
    - Bilgisayar Mühendisliği
    - Endüstri Mühendisliği  
    - Elektrik-Elektronik Mühendisliği
    """
    
    start_time = time.time()
    
    final_response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(context=mock_context, question=question)}],
        temperature=0.1,
        max_tokens=500
    )
    
    response_time = time.time() - start_time
    final_answer = final_response.choices[0].message.content.strip()
    print(f"⚡ Final response: {response_time:.3f}s")
    print(f"💬 Answer: {final_answer[:100]}...")
    
    # Summary
    print("\n📈 Performance Summary:")
    print(f"🤖 Embedding model load: {embedding_load_time:.2f}s")
    print(f"🎯 Cross-encoder load: {cross_encoder_load_time:.2f}s")
    print(f"🧠 Embedding generation: {gpu_time:.3f}s")
    print(f"⚖️ Cross-encoder scoring: {cross_encoder_time:.3f}s")
    print(f"📝 HyDE generation: {hyde_time:.3f}s")
    print(f"💬 Final response: {response_time:.3f}s")
    
    total_ai_time = hyde_time + response_time
    total_ml_time = gpu_time + cross_encoder_time
    
    print(f"\n🎯 Bottleneck Analysis:")
    print(f"🔥 Total AI API time: {total_ai_time:.2f}s")
    print(f"⚡ Total ML processing: {total_ml_time:.2f}s")
    
    if total_ai_time > total_ml_time * 2:
        print("🚨 BOTTLENECK: OpenAI API calls are the main bottleneck!")
    elif total_ml_time > 3.0:
        print("🚨 BOTTLENECK: ML processing (GPU optimization needed)")
    else:
        print("✅ Performance looks good!")

if __name__ == "__main__":
    asyncio.run(debug_rag_performance()) 
 
"""
Performance Debug Script
This script will help identify bottlenecks in the RAG system.
"""

import time
import asyncio
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI
from config import *

async def debug_rag_performance():
    """Debug each step of the RAG process to find bottlenecks"""
    print("🔍 DocBoss Performance Debug")
    print("=" * 50)
    
    # Test question
    question = "mühendislik bölümleri neler?"
    
    # Step 1: Check GPU
    print(f"🎮 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Step 2: Load models and measure time
    print("\n📊 Model Loading Times:")
    
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")
    
    # Embedding model loading
    start_time = time.time()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedding_load_time = time.time() - start_time
    print(f"⚡ Embedding model load: {embedding_load_time:.2f}s")
    
    # Cross-encoder loading
    start_time = time.time()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    cross_encoder_load_time = time.time() - start_time
    print(f"⚡ Cross-encoder load: {cross_encoder_load_time:.2f}s")
    
    # OpenAI client
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Step 3: Test embedding generation
    print("\n🧪 Embedding Generation Test:")
    
    test_texts = [
        "Bu bir test cümlesidir.",
        "Mühendislik bölümleri hakkında bilgi arıyorum.",
        "Üniversite bölümleri ve programları",
        question
    ]
    
    # CPU test
    if device == "cuda":
        cpu_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        start_time = time.time()
        cpu_embeddings = cpu_model.encode(test_texts)
        cpu_time = time.time() - start_time
        print(f"💻 CPU embeddings: {cpu_time:.3f}s")
        del cpu_model
    
    # GPU test
    start_time = time.time()
    gpu_embeddings = embedding_model.encode(test_texts)
    gpu_time = time.time() - start_time
    print(f"🎮 GPU embeddings: {gpu_time:.3f}s")
    
    if device == "cuda" and 'cpu_time' in locals():
        speedup = cpu_time / gpu_time
        print(f"🚀 GPU Speedup: {speedup:.2f}x")
    
    # Step 4: Test cross-encoder
    print("\n🎯 Cross-Encoder Test:")
    
    test_pairs = [
        (question, "Mühendislik bölümleri öğrencilere teknik eğitim sağlar"),
        (question, "Üniversitede çeşitli mühendislik programları bulunur"),
        (question, "Bilgisayar mühendisliği popüler bir bölümdür")
    ]
    
    start_time = time.time()
    scores = cross_encoder.predict(test_pairs)
    cross_encoder_time = time.time() - start_time
    print(f"⚡ Cross-encoder scoring: {cross_encoder_time:.3f}s")
    print(f"📊 Scores: {scores}")
    
    # Step 5: Test HyDE generation
    print("\n📝 HyDE Generation Test:")
    
    start_time = time.time()
    hyde_prompt = f"""Şu soruya verilen kapsamlı bir cevap yaz: {question}
    
    Cevap:"""
    
    response = await openai_client.chat.completions.create(
        model=HYDE_LLM_MODEL,
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.1,
        max_tokens=200
    )
    
    hyde_time = time.time() - start_time
    hyde_answer = response.choices[0].message.content.strip()
    print(f"⚡ HyDE generation: {hyde_time:.3f}s")
    print(f"📝 HyDE answer: {hyde_answer[:100]}...")
    
    # Step 6: Test final response generation
    print("\n💬 Final Response Test:")
    
    mock_context = """
    Ankara Bilim Üniversitesi'nde şu mühendislik bölümleri bulunmaktadır:
    - Bilgisayar Mühendisliği
    - Endüstri Mühendisliği  
    - Elektrik-Elektronik Mühendisliği
    """
    
    start_time = time.time()
    
    final_response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(context=mock_context, question=question)}],
        temperature=0.1,
        max_tokens=500
    )
    
    response_time = time.time() - start_time
    final_answer = final_response.choices[0].message.content.strip()
    print(f"⚡ Final response: {response_time:.3f}s")
    print(f"💬 Answer: {final_answer[:100]}...")
    
    # Summary
    print("\n📈 Performance Summary:")
    print(f"🤖 Embedding model load: {embedding_load_time:.2f}s")
    print(f"🎯 Cross-encoder load: {cross_encoder_load_time:.2f}s")
    print(f"🧠 Embedding generation: {gpu_time:.3f}s")
    print(f"⚖️ Cross-encoder scoring: {cross_encoder_time:.3f}s")
    print(f"📝 HyDE generation: {hyde_time:.3f}s")
    print(f"💬 Final response: {response_time:.3f}s")
    
    total_ai_time = hyde_time + response_time
    total_ml_time = gpu_time + cross_encoder_time
    
    print(f"\n🎯 Bottleneck Analysis:")
    print(f"🔥 Total AI API time: {total_ai_time:.2f}s")
    print(f"⚡ Total ML processing: {total_ml_time:.2f}s")
    
    if total_ai_time > total_ml_time * 2:
        print("🚨 BOTTLENECK: OpenAI API calls are the main bottleneck!")
    elif total_ml_time > 3.0:
        print("🚨 BOTTLENECK: ML processing (GPU optimization needed)")
    else:
        print("✅ Performance looks good!")

if __name__ == "__main__":
    asyncio.run(debug_rag_performance()) 
 
 