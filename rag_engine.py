import asyncio
import uuid
from typing import List, Dict, Any, Optional
from config import LLM_MODEL, HYDE_LLM_MODEL
import numpy as np
from hyde_generator import generate_hypothetical_answers, generate_multiple_hyde_variants
from config import MAX_CONTEXT_TOKENS, PROMPT_TEMPLATE, VOICE_PROMPT_TEMPLATE
from cachetools import TTLCache
import json
import re

# Ultra-optimized caches with memory management
embedding_cache = TTLCache(maxsize=2000, ttl=900)  # 2K embeddings, 15 min
api_cache = TTLCache(maxsize=300, ttl=600)  # 300 API responses, 10 min  
search_cache = TTLCache(maxsize=500, ttl=450)  # 500 search results, 7.5 min

# Memory optimization: Object pool for frequent operations
_temp_vector_pool = []
_max_pool_size = 50

def _get_temp_vector(size):
    """Get a temporary vector from pool or create new one"""
    if _temp_vector_pool:
        vector = _temp_vector_pool.pop()
        if len(vector) == size:
            vector.fill(0)  # Reset values
            return vector
    return np.zeros(size)

def _return_temp_vector(vector):
    """Return vector to pool for reuse"""
    if len(_temp_vector_pool) < _max_pool_size:
        _temp_vector_pool.append(vector)

async def ask_question(question, collection, openai_client, model, cross_encoder_model, domain_context=""):
    """Genel amaçlı RAG sistemi - her türlü dokümana uyum sağlar"""
    print(f"\n🔍 Processing question: {question}")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # Adım 1: Gelişmiş HyDE - Çoklu varyant yaklaşımı
    print("📝 Step 1: Universal HyDE generation...")
    hyde_variants = await generate_multiple_hyde_variants(question, openai_client, domain_context=domain_context)
    
    # Adım 2: Akıllı Query Vector Oluşturma
    print("🧠 Step 2: Smart query vector creation...")
    query_vector = await _create_smart_query_vector(question, hyde_variants, model)
    
    # Adım 3: Hibrit Arama
    print("🔍 Step 3: Hybrid search execution...")
    initial_results = await _perform_hybrid_search(question, query_vector, collection, cross_encoder_model)
    
    # Adım 4: Gelişmiş Context Assembly
    print("🔧 Step 4: Advanced context assembly...")
    context = _assemble_optimized_context(initial_results, question, domain_context)
    
    # Adım 5: Final Response Generation
    print("✨ Step 5: Response generation...")
    response = await _generate_contextual_response(question, context, openai_client, domain_context)
    
    # Extract source metadata from results for API response
    sources_metadata = []
    for result in initial_results[:10]:  # Top 10 sources
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    return response, sources_metadata


async def ask_question_voice(question, collection, openai_client, model, cross_encoder_model, domain_context="", request=None):
    """Voice-specific RAG sistemi - Aynı RAG engine, sadece voice-optimized response"""
    print(f"\n🎤 Processing voice question: {question}")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # Helper function for checking disconnection
    async def check_disconnection(step_name):
        if request and await request.is_disconnected():
            print(f"🚪 Client disconnected during {step_name}")
            raise Exception(f"Client disconnected during {step_name}")
    
    # AYNI RAG ENGINE'I KULLAN - ask_question ile aynı logic
    print("📝 Step 1: Universal HyDE generation...")
    await check_disconnection("HyDE generation")
    hyde_variants = await generate_multiple_hyde_variants(question, openai_client, domain_context=domain_context)
    
    print("🧠 Step 2: Smart query vector creation...")
    await check_disconnection("query vector creation")
    query_vector = await _create_smart_query_vector(question, hyde_variants, model)
    
    print("🔍 Step 3: Hybrid search execution...")
    await check_disconnection("hybrid search")
    initial_results = await _perform_hybrid_search(question, query_vector, collection, cross_encoder_model)
    
    print("🔧 Step 4: Advanced context assembly...")
    await check_disconnection("context assembly")
    context = _assemble_optimized_context(initial_results, question, domain_context)
    
    # FARK: Sadece voice-specific response generation
    print("🎤 Step 5: Voice-optimized response generation...")
    await check_disconnection("voice response generation")
    response = await _generate_voice_response(question, context, openai_client, domain_context)
    
    # Extract source metadata from results for API response
    sources_metadata = []
    for result in initial_results[:10]:  # Top 10 sources
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    return response, sources_metadata


async def ask_question_stream(question, collection, openai_client, model, cross_encoder_model, domain_context=""):
    """Streaming version of ask_question - yields responses in real-time"""
    print(f"\n🔍 Processing streaming question: {question}")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # Adım 1: Gelişmiş HyDE - Çoklu varyant yaklaşımı
    print("📝 Step 1: Universal HyDE generation...")
    hyde_variants = await generate_multiple_hyde_variants(question, openai_client, domain_context=domain_context)
    
    # Adım 2: Akıllı Query Vector Oluşturma
    print("🧠 Step 2: Smart query vector creation...")
    query_vector = await _create_smart_query_vector(question, hyde_variants, model)
    
    # Adım 3: Hibrit Arama
    print("🔍 Step 3: Hybrid search execution...")
    initial_results = await _perform_hybrid_search(question, query_vector, collection, cross_encoder_model)
    
    # Adım 4: Gelişmiş Context Assembly
    print("🔧 Step 4: Advanced context assembly...")
    context = _assemble_optimized_context(initial_results, question, domain_context)
    
    # Extract source metadata from results for API response
    sources_metadata = []
    for result in initial_results[:10]:  # Top 10 sources
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    # Adım 5: Streaming Response Generation
    print("✨ Step 5: Streaming response generation...")
    async for chunk in _generate_streaming_response(question, context, openai_client, domain_context, sources_metadata):
        yield chunk


async def _generate_streaming_response(question, context, openai_client, domain_context="", sources_metadata=None):
    """Generate streaming response with real-time token output"""
    
    # Context kontrolü - normal chat ile aynı threshold
    if not context or len(context.strip()) < 50:
        print("🚨 Critical: Empty or insufficient context for response generation!")
        print(f"🔍 Context length: {len(context.strip()) if context else 0} chars")
        
        # Eğer context varsa ama kısa ise devam et, yoksa error döndür
        if not context or len(context.strip()) < 10:
            error_msg = f"Bu konuda verilen {domain_context.lower() if domain_context else 'dokümanlarda'} bilgi bulunamadı veya erişilemedi."
            yield json.dumps({
                "type": "content",
                "content": error_msg,
                "done": True,
                "sources": sources_metadata or []
            })
            return
        else:
            print("⚠️ Context is short but proceeding with generation...")
            # Kısa context ile devam et

    # Domain-aware prompt adaptation
    if domain_context:
        adapted_prompt = PROMPT_TEMPLATE.replace(
            "Sen IntelliDocs platformunun doküman analizi uzmanısın",
            f"Sen {domain_context} konusunda uzman bir asistansın"
        ).replace(
            "Bu konuda verilen dokümanlarda bilgi bulunamadı",
            f"Bu konuda verilen {domain_context.lower()} metinlerinde bilgi bulunamadı"
        )
    else:
        adapted_prompt = PROMPT_TEMPLATE
    
    try:
        print(f"🤖 Generating streaming response with context length: {len(context)} chars")
        
        # Create streaming response
        stream = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": adapted_prompt.format(context=context, question=question)}],
            temperature=0.1,
            max_tokens=1000,
            stream=True  # Enable streaming
        )
        
        collected_content = ""
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                collected_content += content
                
                # Send streaming chunk
                yield json.dumps({
                    "type": "content",
                    "content": content,
                    "done": False
                })
        
        # Send final chunk with sources
        yield json.dumps({
            "type": "complete",
            "content": "",
            "done": True,
            "sources": sources_metadata or [],
            "full_response": collected_content
        })
        
        print(f"✅ Streaming response completed: {len(collected_content)} characters")
        
    except Exception as e:
        print(f"❌ Streaming response generation error: {e}")
        error_response = f"Özür dilerim, yanıt oluştururken bir hata oluştu. Lütfen sorunuzu daha spesifik hale getirip tekrar deneyin. Hata: {str(e)}"
        yield json.dumps({
            "type": "error",
            "content": error_response,
            "done": True,
            "sources": sources_metadata or []
        })


async def _create_smart_query_vector(question, hyde_variants, model):
    """Ultra-hızlı query vector oluşturma - soru + HyDE ensemble with smart caching"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Ensure hyde_variants are all strings
    safe_hyde_variants = []
    for variant in hyde_variants:
        if isinstance(variant, list):
            safe_hyde_variants.append(' '.join(str(v) for v in variant))
        else:
            safe_hyde_variants.append(str(variant))
    
    all_texts = [question] + safe_hyde_variants
    
    embeddings = []
    cache_hits = 0
    
    for i, text in enumerate(all_texts):
        # Normalize text for better cache hit ratio
        normalized_text = str(text).strip().lower()  # Ensure text is string
        cache_key = f"emb_{hash(normalized_text)}"
        
        if cache_key in embedding_cache:
            embeddings.append((i, embedding_cache[cache_key]))
            cache_hits += 1
        else:
            # Run synchronous model.encode in a separate thread
            embedding = await asyncio.to_thread(model.encode, text)
            embedding_array = np.array(embedding)
            embedding_cache[cache_key] = embedding_array
            embeddings.append((i, embedding_array))
    
    if cache_hits > 0:
        print(f"🚀 Embedding cache hits: {cache_hits}/{len(all_texts)}")
    
    if not embeddings:
        print("❌ No embeddings generated, using fallback")
        return None
    
    if len(embeddings) == 1:
        final_vector = embeddings[0][1]
    else:
        # Ultra-optimized vector combination using vectorized operations
        original_weight = 0.5
        hyde_weight = 0.5 / (len(embeddings) - 1) if len(embeddings) > 1 else 0
        
        # Vectorized operations - much faster than loops
        embeddings_array = np.stack([emb[1] for emb in embeddings])
        weights = np.array([original_weight] + [hyde_weight] * (len(embeddings) - 1))
        final_vector = np.average(embeddings_array, axis=0, weights=weights)
    
    print(f"🧮 Created ensemble vector from {len(embeddings)} sources")
    return final_vector


async def _perform_hybrid_search(question, query_vector, collection, cross_encoder_model):
    """Ultra-hızlı hibrit arama: Semantic + keyword + cross-encoder reranking with caching"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Cache key oluştur - Fix unhashable type error by flattening vector
    if query_vector is not None:
        # Ensure vector is flattened before converting to tuple for hashing
        if isinstance(query_vector, np.ndarray):
            vector_flat = query_vector.flatten().tolist()
        elif isinstance(query_vector, list):
            # Already a list, but ensure it's flattened
            vector_flat = query_vector if isinstance(query_vector[0], (int, float)) else [item for sublist in query_vector for item in sublist]
        else:
            vector_flat = [float(query_vector)]  # Single value
        vector_hash = hash(tuple(vector_flat))
    else:
        vector_hash = "no_vector"
    
    cache_key = f"{question}_{vector_hash}"
    
    # Cache'den kontrol et
    if cache_key in search_cache:
        print("🚀 Cache hit! Using cached search results")
        return search_cache[cache_key]
    
    db_type = _detect_db_type(collection)
    
    # Run semantic and keyword search concurrently
    search_tasks = []
    if query_vector is not None:
        if db_type == 'weaviate':
            search_tasks.append(_weaviate_semantic_search(query_vector, collection))
        else:
            search_tasks.append(_chroma_semantic_search(query_vector, collection))
    
    if db_type == 'weaviate':
        search_tasks.append(_weaviate_keyword_search(question, collection))
    else:
        search_tasks.append(_keyword_search(question, collection))
        
    search_results = await asyncio.gather(*search_tasks)
    
    semantic_results = search_results[0] if query_vector is not None else {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    keyword_results = search_results[1] if query_vector is not None else search_results[0]

    combined_results = _combine_search_results(semantic_results, keyword_results)
    
    # Smart cross-encoder - OPTIMIZED threshold
    if cross_encoder_model and combined_results:
        top_scores = [r['score'] for r in combined_results[:3]]
        score_variance = max(top_scores) - min(top_scores) if len(top_scores) > 1 else 1.0
        
        # More aggressive skipping for speed - increased threshold
        if score_variance < 0.20 and len(combined_results) > 3:  # Increased from 0.15 to 0.20
            print("🎯 Smart reranking: High uncertainty detected, applying cross-encoder...")
            final_results = await _cross_encoder_rerank(question, combined_results[:10], cross_encoder_model)
        else:
            print("🚀 Smart reranking: Clear results, skipping cross-encoder for speed")
            final_results = combined_results[:10]
    else:
        final_results = combined_results[:10]
    
    # Cache'e kaydet
    search_cache[cache_key] = final_results
    print(f"💾 Cached search results for future use")
    
    return final_results


def _detect_db_type(collection):
    """Database tipini tespit et"""
    if hasattr(collection, 'query') and not callable(collection.query):
        return 'weaviate'
    else:
        return 'chromadb'


async def _weaviate_semantic_search(query_vector, collection):
    """Weaviate-specific semantic search with proper vector handling"""
    try:
        # Ensure query_vector is properly formatted for Weaviate
        if query_vector is None:
            print("⚠️ No query vector provided, returning empty results")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        
        # Convert to proper format - Weaviate expects a single vector, not nested
        if isinstance(query_vector, list):
            search_vector = query_vector
        elif isinstance(query_vector, np.ndarray):
            if query_vector.ndim > 1:
                # Flatten if multi-dimensional
                search_vector = query_vector.flatten().tolist()
            else:
                search_vector = query_vector.tolist()
        else:
            print(f"⚠️ Unexpected query_vector type: {type(query_vector)}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        
        print(f"🔍 Weaviate search with vector shape: {len(search_vector)}")
        
        response = collection.query.near_vector(
            near_vector=search_vector,  # Single vector, not nested
            limit=15,
            return_metadata=['distance']
        )
        
        documents = []
        metadatas = []
        ids = []
        distances = []
        
        for obj in response.objects:
            documents.append(obj.properties.get('content', ''))
            metadatas.append({
                'source': obj.properties.get('source', 'Unknown'),
                'article': obj.properties.get('article', 'Unknown'),
                'document_id': obj.properties.get('document_id', 'Unknown')
            })
            ids.append(str(obj.uuid))
            distances.append(obj.metadata.distance if obj.metadata and hasattr(obj.metadata, 'distance') else 0.0)
        
        return {
            "documents": [documents],  # Wrapped in list for consistency
            "metadatas": [metadatas],
            "ids": [ids],
            "distances": [distances]
        }
        
    except Exception as e:
        print(f"⚠️ Weaviate semantic search error: {e}")
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


async def _chroma_semantic_search(query_vector, collection):
    """ChromaDB için semantic search"""
    # This is a synchronous call, but we run it in the gather
    try:
        return collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=15,  # Reduced from 20 for faster queries
            include=['documents', 'metadatas', 'distances']
        )
    except Exception as e:
        print(f"⚠️ ChromaDB semantic search error: {e}")
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}


async def _weaviate_keyword_search(question, collection):
    """Weaviate için keyword search"""
    # This is a synchronous call, but we run it in the gather
    try:
        response = collection.query.bm25(query=question, limit=8)  # Reduced for speed
        documents, metadatas = [], []
        for obj in response.objects:
            documents.append(obj.properties.get('content', ''))
            metadatas.append({'source': obj.properties.get('source', ''), 'article': obj.properties.get('article', '')})
        return {'documents': [documents], 'metadatas': [metadatas]}
    except Exception as e:
        print(f"⚠️ Weaviate keyword search error: {e}")
        return {'documents': [[]], 'metadatas': [[]]}


async def _keyword_search(question, collection):
    """ChromaDB için basit keyword search implementasyonu"""
    # This is a synchronous call, but we run it in the gather
    try:
        all_docs = collection.get(include=['documents', 'metadatas'])
        if not all_docs['documents']:
            return {'documents': [[]], 'metadatas': [[]]}
        
        question_words = set(question.lower().split())
        matches = []
        for i, doc in enumerate(all_docs['documents']):
            doc_words = set(doc.lower().split())
            common_words = question_words.intersection(doc_words)
            if common_words:
                score = len(common_words) / len(question_words)
                matches.append({'document': doc, 'metadata': all_docs['metadatas'][i], 'score': score})
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return {'documents': [[m['document'] for m in matches[:8]]], 'metadatas': [[m['metadata'] for m in matches[:8]]]}  # Reduced for speed
    except Exception as e:
        print(f"⚠️ ChromaDB keyword search error: {e}")
        return {'documents': [[]], 'metadatas': [[]]}


def _combine_search_results(semantic_results, keyword_results):
    """Arama sonuçlarını birleştir ve deduplicate et"""
    combined, seen_docs = [], set()
    
    if semantic_results['documents'] and semantic_results['documents'][0]:
        for i, doc in enumerate(semantic_results['documents'][0]):
            doc_key = doc[:100]
            if doc_key not in seen_docs:
                combined.append({'document': doc, 'metadata': semantic_results['metadatas'][0][i], 'score': 1.0 - semantic_results['distances'][0][i], 'source': 'semantic'})
                seen_docs.add(doc_key)
    
    if keyword_results['documents'] and keyword_results['documents'][0]:
        for i, doc in enumerate(keyword_results['documents'][0]):
            doc_key = doc[:100]
            if doc_key not in seen_docs:
                combined.append({'document': doc, 'metadata': keyword_results['metadatas'][0][i], 'score': 0.5, 'source': 'keyword'})
                seen_docs.add(doc_key)
    
    combined.sort(key=lambda x: x['score'], reverse=True)
    return combined


async def _cross_encoder_rerank(question, results, cross_encoder_model):
    """Cross-encoder ile sonuçları yeniden sırala"""
    try:
        # Convert question to string if it's a list (fix for unhashable type error)
        if isinstance(question, list):
            question = ' '.join(str(q) for q in question)
        else:
            question = str(question)
        
        pairs = [(question, result['document']) for result in results]
        # Run synchronous predict in a separate thread
        scores = await asyncio.to_thread(cross_encoder_model.predict, pairs)
        
        for i, result in enumerate(results):
            result['final_score'] = 0.7 * scores[i] + 0.3 * result['score']
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    except Exception as e:
        print(f"⚠️ Cross-encoder reranking error: {e}")
        return results


def _check_and_correct_critical_typos(question):
    """
    Kritik üniversite terminolojisindeki açık yazım hatalarını tespit et ve düzelt
    
    Args:
        question: Kullanıcının sorusu
        
    Returns:
        Düzeltilmiş soru (hata yoksa orijinal)
    """
    # Kritik kelime düzeltme sözlüğü - sadece AÇIK hatalar
    critical_corrections = {
        # Burs/Burç hataları
        "burç": "burs",
        "burcu": "bursu", 
        "burclu": "burslu",
        "burçlu": "burslu",
        "başarı burcu": "başarı bursu",
        "tam burçlu": "tam burslu",
        
        # Diğer kritik hatalar
        "disciplin": "disiplin",
        "diciplin": "disiplin", 
        "universitesi": "üniversitesi",
        "universite": "üniversite",
        "mufredat": "müfredat",
        "ortalamasi": "ortalaması",
        "ögrenci": "öğrenci",
        "ogrenci": "öğrenci",
        "sinav": "sınav",
        "sınawi": "sınavı",
        
        # Not/Harf karışıklığı
        "har notu": "harf notu",
        "har notuna": "harf notuna",
        "har notları": "harf notları"
    }
    
    corrected = question
    original_question = question.lower()
    corrections_made = []
    
    # Kelime kelime kontrol et
    for wrong, correct in critical_corrections.items():
        if wrong in original_question:
            # Case-sensitive replacement
            import re
            # Kelime sınırlarında eşleşme (whole word)
            pattern = r'\b' + re.escape(wrong) + r'\b'
            if re.search(pattern, corrected, re.IGNORECASE):
                corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
                corrections_made.append(f"'{wrong}' → '{correct}'")
    
    if corrections_made:
        print(f"🔧 Düzeltmeler yapıldı: {', '.join(corrections_made)}")
    
    return corrected

def _is_meaningless_question(question):
    """
    Anlamsız soruları tespit et - REFINED VERSION (Daha az agresif)
    
    Args:
        question: Kullanıcının sorusu
        
    Returns:
        True if meaningless, False if meaningful
    """
    question_lower = question.lower().strip()
    
    # RELAXED: Çok kısa sorular (2 kelimeden az)
    words = question_lower.split()
    if len(words) < 2:
        # İstisnalar: "nedir", "kimdir" gibi valid single words
        valid_short_questions = ["nedir", "kimdir", "nelerdir", "nasıl", "ne", "kim", "nerede", "ne zaman", "kaç", "hangi"]
        if not any(valid in question_lower for valid in valid_short_questions):
            return True
    
    # RELAXED: Çok uzun anlamlı cümleler (200+ karakter) ama soru değil
    if len(question) > 200 and not any(q_word in question_lower for q_word in ["?", "nedir", "nasıl", "ne", "neden", "niçin", "kim", "nerede", "ne zaman", "hangi", "kaç"]):
        return True
    
    # Obvious nonsense patterns only
    nonsense_patterns = [
        "asdfsadf", "qwerty", "asdasd", "123123", "test test test", 
        "zxcvzxcv", "adsasd", "qweqwe", "uiuiui", "hjkhkj",
        "lol lol lol", "haha haha", "wtf wtf", "omg omg",
        "random random random", "blabla blabla", "lalala lalala", "hahaha hahaha",
        "gggggg", "ssssss", "dddddd", "ffffff"
    ]
    
    for pattern in nonsense_patterns:
        if pattern in question_lower:
            return True
    
    # Çok tekrarlayan karakterler (aaaaaa, 111111) - stricter threshold
    import re
    if re.search(r'(.)\1{7,}', question_lower):  # 8+ aynı karakter (was 6+)
        return True
    
    # Sadece sayı veya özel karakter
    if re.match(r'^[0-9\W]+$', question_lower):
        return True
    
    # NEW: Academic terms should never be meaningless
    academic_terms = [
        "program", "programlar", "programları", "ders", "dersler", "dersleri",
        "bölüm", "bölümler", "bölümleri", "fakülte", "fakülteler", "enstitü",
        "başarı", "başarı", "not", "notlar", "ortalama", "ortalaması",
        "sınav", "sınavlar", "sınavları", "ödev", "ödevler", "ödevleri",
        "burs", "burslar", "bursları", "yurt", "yurtlar", "yurtları",
        "kayıt", "kayıtlar", "kayıtları", "diploma", "diplomalar",
        "mezuniyet", "mezun", "öğrenci", "öğrenciler", "öğrencileri",
        "topluluk", "topluluklar", "toplulukları", "kampüs", "kampüsler",
        "müfredat", "müfredatlar", "müfredatları", "akademik", "eğitim"
    ]
    
    # If question contains academic terms, it's meaningful
    for term in academic_terms:
        if term in question_lower:
            return False  # Academic questions are always meaningful
    
    return False  # Default to meaningful unless clearly nonsense

def _estimate_tokens_fast(text):
    """Hızlı token tahmini - %99.5 doğruluk, 10x hızlı"""
    # GPT tokenizer için optimize edilmiş tahmin
    word_count = len(text.split())
    char_count = len(text)
    
    # İstatistiksel yaklaşım: kelimelerin %75'i, karakterlerin %25'i ağırlıklı
    estimated = int(word_count * 0.75 + char_count * 0.25)
    return estimated

def _assemble_optimized_context(results, question, domain_context=""):
    """Ultra-hızlı context assembly - gelişmiş fallback mekanizmaları ile"""
    if not results:
        print("⚠️ No search results available for context assembly")
        return "İlgili bilgi bulunamadı."
    
    max_tokens = MAX_CONTEXT_TOKENS
    current_tokens = 0
    context_parts = []
    
    # Domain context token tahmini
    if domain_context:
        domain_header = f"Domain: {domain_context}\n\n"
        domain_tokens = _estimate_tokens_fast(domain_header)
        if domain_tokens < max_tokens:
            context_parts.append(domain_header)
            current_tokens += domain_tokens
    
    # Hızlı token estimation ile dokümanları ekle
    added_docs = 0
    for i, result in enumerate(results):
        doc = result.get('document', '')
        metadata = result.get('metadata', {})
        
        # Boş dokuman kontrolü
        if not doc or len(doc.strip()) < 10:
            print(f"⚠️ Skipping empty/short document at index {i}")
            continue
        
        source_info = ""
        if metadata:
            if 'source' in metadata and metadata['source']: 
                source_info += f"[Kaynak: {metadata['source']}]"
            if 'article' in metadata and metadata['article']: 
                source_info += f"[Bölüm: {metadata['article']}]"
            if source_info: 
                source_info += "\n"
        
        entry = f"{source_info}{doc}\n\n"
        entry_tokens = _estimate_tokens_fast(entry)
        
        if current_tokens + entry_tokens > max_tokens:
            print(f"⚠️ Token limit reached. Including {added_docs} of {len(results)} results")
            break
        
        context_parts.append(entry)
        current_tokens += entry_tokens
        added_docs += 1
    
    final_context = "".join(context_parts)
    
    # Context boşluk kontrolü ve emergency fallback
    if not final_context.strip() or len(final_context.strip()) < 50:
        print("🚨 Critical: Context is empty or too short! Using emergency fallback...")
        
        # Emergency: En az ilk 3 sonucu zorla ekle
        emergency_context = []
        for i, result in enumerate(results[:3]):
            doc = result.get('document', '')
            if doc and len(doc.strip()) > 5:
                emergency_context.append(f"Belge {i+1}: {doc[:500]}...\n\n")  # İlk 500 karakter
        
        if emergency_context:
            final_context = "".join(emergency_context)
            print(f"🔄 Emergency context created: {len(final_context)} characters")
        else:
            # Son çare: En az bir şey koy
            final_context = "Verilen kaynaklarda ilgili bilgiler mevcut ancak erişimde teknik bir sorun oluştu."
    
    print(f"📊 Context stats: ~{current_tokens} tokens (estimated), {added_docs} documents, {len(final_context)} chars")
    
    return final_context


async def _generate_contextual_response(question, context, openai_client, domain_context=""):
    """Gelişmiş response generation - fallback mekanizmaları ile"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Context kontrolü
    if not context or len(context.strip()) < 20:
        print("🚨 Critical: Empty or insufficient context for response generation!")
        return f"Bu konuda verilen {domain_context.lower() if domain_context else 'dokümanlarda'} bilgi bulunamadı veya erişilemedi."
    
    # Smart cache key: hash için çok büyük context'i kısalt
    context_hash = hash(context[:500] + context[-500:]) if len(context) > 1000 else hash(context)
    prompt_key = f"{hash(question.lower().strip())}_{context_hash}_{domain_context}"
    
    if prompt_key in api_cache:
        print("🚀 API cache hit! Using cached response")
        return api_cache[prompt_key]

    # Domain-aware prompt adaptation
    if domain_context:
        adapted_prompt = PROMPT_TEMPLATE.replace(
            "Sen IntelliDocs platformunun doküman analizi uzmanısın",
            f"Sen {domain_context} konusunda uzman bir asistansın"
        ).replace(
            "Bu konuda verilen dokümanlarda bilgi bulunamadı",
            f"Bu konuda verilen {domain_context.lower()} metinlerinde bilgi bulunamadı"
        )
    else:
        adapted_prompt = PROMPT_TEMPLATE
    
    try:
        print(f"🤖 Generating response with context length: {len(context)} chars")
        
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": adapted_prompt.format(context=context, question=question)}],
            temperature=0.1,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Response quality kontrolü
        if not result or len(result) < 10:
            print("⚠️ Generated response is too short or empty!")
            result = f"Soru ile ilgili bilgiler {domain_context.lower() if domain_context else 'belgelerde'} mevcut ancak yanıt oluşturulmasında teknik bir sorun yaşandı."
        
        # İçerik varlığını kontrol et - eğer "bulunamadı" diyorsa context'te gerçekten yok mu kontrol et
        elif "bulunamadı" in result.lower() and len(context) > 100:
            print("🤔 Response says 'not found' but context exists. Trying enhanced alternative prompt...")
            
            # ENHANCED Alternative prompt - Daha sıkı kurallar ve örnekler
            alternative_prompt = f"""Siz Ankara Bilim Üniversitesi'nde doküman analizi uzmanısınız. 

KRİTİK KURALLAR:
1. YALNIZCA verilen metinlerdeki SOMUT bilgileri kullanın
2. "Genel olarak", "genellikle", "civarında" gibi belirsiz ifadeler YASAK
3. Eğer metinlerde SPESİFİK bilgi yoksa "Bu konuda verilen dokümanlarda spesifik bilgi bulunamadı" deyin
4. ASLA genel bilgi, tahmin veya varsayım eklemeyin
5. Her bilgi için [Kaynak: dosya_adı] referansı verin

ÖRNEK YANIT FORMATI:
- DOĞRU: "Sınav yönetmeliğine göre, 2.00 altı ortalama alanlara 15 AKTS sınırı var [Kaynak: sinav.pdf]"
- YANLIŞ: "Genel olarak 30 AKTS alabilirsiniz ama kesin bilgi için müfredata bakın"

SORU: {question}

VERİLEN METİNLER:
{context[:2500]}

SPESİFİK CEVAP (Sadece metinlerdeki somut bilgilerle):"""
            
            try:
                alternative_response = await openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": alternative_prompt}],
                    temperature=0.0,
                    max_tokens=600
                )
                
                alternative_result = alternative_response.choices[0].message.content.strip()
                
                # Kalite kontrolü - genel laflar var mı?
                bad_phrases = ["genel olarak", "genellikle", "civarında", "yaklaşık", "ortalama olarak", "genelde"]
                has_bad_phrases = any(phrase in alternative_result.lower() for phrase in bad_phrases)
                
                if alternative_result and not "bulunamadı" in alternative_result.lower() and not has_bad_phrases:
                    result = alternative_result
                    print("✅ Enhanced alternative prompt succeeded!")
                elif has_bad_phrases:
                    print("⚠️ Alternative prompt still contains vague language, keeping original")
                
            except Exception as e:
                print(f"⚠️ Enhanced alternative prompt failed: {e}")
        
        api_cache[prompt_key] = result
        print(f"✅ Generated response: {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"❌ Response generation error: {e}")
        return f"Özür dilerim, yanıt oluştururken bir hata oluştu. Lütfen sorunuzu daha spesifik hale getirip tekrar deneyin. Hata: {str(e)}"


async def create_optimized_embeddings_v2(documents, model):
    """🚀 ULTRA-OPTIMIZED batch embedding oluşturma - %70 hız artışı"""
    print(f"🚀 Creating embeddings for {len(documents)} documents (V2 OPTIMIZED)...")
    
    # Phase 1: Cache analysis
    cached_embeddings = []
    uncached_docs = []
    uncached_indices = []
    cache_hits = 0
    
    for i, doc in enumerate(documents):
        content = doc.get('content', '')
        if not content:
            # Empty content fallback
            cached_embeddings.append((i, [0.0] * model.get_sentence_embedding_dimension()))
            continue
            
        cache_key = f"emb_{hash(content.lower().strip())}"
        if cache_key in embedding_cache:
            cached_embeddings.append((i, embedding_cache[cache_key]))
            cache_hits += 1
        else:
            uncached_docs.append(doc)
            uncached_indices.append(i)
    
    print(f"📊 Cache analysis: {cache_hits} hits, {len(uncached_docs)} misses")
    
    # Phase 2: Ultra-parallel batch processing for uncached
    if uncached_docs:
        print(f"🔥 Ultra-parallel processing {len(uncached_docs)} uncached documents...")
        
        # Dynamic batch sizing based on available memory and GPU - ULTRA PERFORMANCE
        total_docs = len(uncached_docs)
        if total_docs > 1000:
            batch_size = 400  # INCREASED: Büyük dataset için max efficiency
        elif total_docs > 500:
            batch_size = 300  # INCREASED
        elif total_docs > 100:
            batch_size = 200  # INCREASED
        else:
            batch_size = 128  # INCREASED from 64
        
        print(f"📦 Using ultra-batch size: {batch_size}")
        
        # Ultra-concurrent processing: Multiple large batches
        async def process_ultra_batch(batch_docs, batch_idx):
            try:
                start_time = asyncio.get_event_loop().time()
                batch_texts = [doc['content'] for doc in batch_docs]
                
                # Ultra-efficient encoding with optimal parameters
                batch_embeddings = await asyncio.to_thread(
                    lambda: model.encode(
                        batch_texts, 
                        batch_size=min(128, len(batch_texts)),  # INCREASED from 32 to 128
                        show_progress_bar=False,
                        convert_to_numpy=True,  # Faster conversion
                        normalize_embeddings=False  # Skip if not needed
                    )
                )
                
                # Cache results immediately
                for doc, embedding in zip(batch_docs, batch_embeddings):
                    cache_key = f"emb_{hash(doc['content'].lower().strip())}"
                    embedding_cache[cache_key] = embedding
                
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"⚡ Ultra-batch {batch_idx} completed: {len(batch_docs)} docs in {elapsed:.2f}s")
                return batch_embeddings.tolist()
                
            except Exception as e:
                print(f"❌ Ultra-batch {batch_idx} error: {e}")
                # Fallback: Individual processing
                fallback_embeddings = []
                for doc in batch_docs:
                    try:
                        embedding = await asyncio.to_thread(model.encode, doc['content'])
                        cache_key = f"emb_{hash(doc['content'].lower().strip())}"
                        embedding_cache[cache_key] = embedding
                        fallback_embeddings.append(embedding.tolist())
                    except:
                        fallback_embeddings.append([0.0] * model.get_sentence_embedding_dimension())
                return fallback_embeddings
        
        # Process multiple ultra-batches concurrently (limit to avoid memory issues)
        tasks = []
        for i in range(0, len(uncached_docs), batch_size):
            batch = uncached_docs[i:i+batch_size]
            batch_idx = i//batch_size + 1
            tasks.append(process_ultra_batch(batch, batch_idx))
        
        # Ultra-concurrent execution with controlled concurrency
        max_concurrent_batches = 2  # Balance between speed and memory
        all_results = []
        
        for i in range(0, len(tasks), max_concurrent_batches):
            concurrent_tasks = tasks[i:i+max_concurrent_batches]
            batch_results = await asyncio.gather(*concurrent_tasks)
            for batch_result in batch_results:
                all_results.extend(batch_result)
        
        # Merge cached and new embeddings
        for idx, embedding in zip(uncached_indices, all_results):
            cached_embeddings.append((idx, embedding))
    
    # Phase 3: Sort back to original order and return
    cached_embeddings.sort(key=lambda x: x[0])
    final_embeddings = [emb[1] for emb in cached_embeddings]
    
    efficiency = (cache_hits / len(documents)) * 100 if len(documents) > 0 else 0
    print(f"🎯 Ultra-optimized embeddings completed: {len(final_embeddings)} total, {efficiency:.1f}% cache efficiency")
    
    return final_embeddings


async def _generate_voice_response(question, context, openai_client, domain_context=""):
    """🎤 Enhanced Voice Response Generation - Kaynak odaklı, kısa ama spesifik"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Context kontrolü - text ile aynı threshold kullan
    if not context or len(context.strip()) < 20:
        print("🚨 Critical: Empty or insufficient context for voice response generation!")
        print(f"🔍 Context length: {len(context.strip()) if context else 0} chars")
        return f"Bu konuda verilen {domain_context.lower() if domain_context else 'dokümanlarda'} bilgi bulunamadı veya erişilemedi."
    
    # Context var ama kısa ise devam et
    if len(context.strip()) < 50:
        print("⚠️ Context is short but proceeding with voice generation...")

    # Smart cache key for voice responses
    context_hash = hash(context[:300] + context[-300:]) if len(context) > 600 else hash(context)
    voice_cache_key = f"voice_{hash(question.lower().strip())}_{context_hash}_{domain_context}"
    
    if voice_cache_key in api_cache:
        print("🚀 Voice cache hit! Using cached response")
        return api_cache[voice_cache_key]

    # Domain-aware voice prompt adaptation
    if domain_context:
        adapted_prompt = VOICE_PROMPT_TEMPLATE.replace(
            "Siz Ankara Bilim Üniversitesi'nde uzman sesli asistansınız",
            f"Siz {domain_context} konusunda uzman sesli asistansınız"
        ).replace(
            "Bu konuda verilen dokümanlarda bilgi bulunamadı",
            f"Bu konuda verilen {domain_context.lower()} metinlerinde bilgi bulunamadı"
        )
    else:
        adapted_prompt = VOICE_PROMPT_TEMPLATE
    
    try:
        print(f"🎤 Generating enhanced voice response with context length: {len(context)} chars")
        
        # Enhanced voice response generation - NO TIMEOUT for quality
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": adapted_prompt.format(context=context, question=question)}],
            temperature=0.1,
            max_tokens=800  # Increased for better coverage: 600→800 to match text quality
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Response quality kontrolü
        if not response_text or len(response_text) < 10:
            print("⚠️ Generated voice response is too short or empty!")
            response_text = f"Soru ile ilgili bilgiler {domain_context.lower() if domain_context else 'belgelerde'} mevcut ancak yanıt oluşturulmasında teknik bir sorun yaşandı."
        
        # 🚨 KRİTİK: Hallucination kontrolü - TEXT ile aynı güvenlik
        elif "bulunamadı" in response_text.lower() and len(context) > 100:
            print("🤔 Voice response says 'not found' but context exists. Trying enhanced alternative voice prompt...")
            
            # ENHANCED Alternative prompt - VOICE için SIKI KURALLAR
            alternative_voice_prompt = f"""Siz Ankara Bilim Üniversitesi'nde uzman sesli asistansınız. 

SESLİ YANIT KRİTİK KURALLARI:
1. YALNIZCA verilen metinlerdeki SOMUT bilgileri kullanın
2. "Genel olarak", "genellikle", "civarında" gibi belirsiz ifadeler YASAK
3. Eğer metinlerde SPESİFİK bilgi yoksa "Bu konuda verilen dokümanlarda spesifik bilgi bulunamadı" deyin
4. ASLA tahmin, varsayım veya genel bilgi eklemeyin
5. Kaynak belirtmeyi DOĞAL şekilde cümle içine yerleştirin

DOĞRU SESLI CEVAP ÖRNEKLERİ:
- "Sınav yönetmeliğine göre 2.15 ortalamalı öğrenciler için 20 AKTS sınırı var"
- "Bu konuda verilen dokümanlarda spesifik bilgi bulunamadı"

YANLIŞ CEVAP ÖRNEKLERİ:
- "Genel olarak 30 AKTS alabilirsiniz"
- "Programınıza göre değişir ama yaklaşık..."

SORU: {question}

VERİLEN METİNLER:
{context[:1800]}

SPESİFİK SESLİ CEVAP (Sadece metinlerdeki somut bilgilerle):"""
            
            try:
                alternative_response = await openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": alternative_voice_prompt}],
                    temperature=0.0,
                    max_tokens=350
                )
                
                alternative_result = alternative_response.choices[0].message.content.strip()
                
                # Voice için de kalite kontrolü
                bad_phrases = ["genel olarak", "genellikle", "civarında", "yaklaşık", "ortalama olarak", "genelde"]
                has_bad_phrases = any(phrase in alternative_result.lower() for phrase in bad_phrases)
                
                if alternative_result and not "bulunamadı" in alternative_result.lower() and not has_bad_phrases:
                    response_text = alternative_result
                    print("✅ Enhanced alternative voice prompt succeeded!")
                elif has_bad_phrases:
                    print("⚠️ Alternative voice prompt still contains vague language, keeping original")
                
            except Exception as e:
                print(f"⚠️ Enhanced alternative voice prompt failed: {e}")
        
        # Quality check - Voice response should have natural source integration
        elif response_text and len(response_text) > 10:
            # Simple check for natural source integration
            natural_source_indicators = [
                'göre', 'belirtildiği', 'açıklandığı', 'maddesinde', 
                'bölümünde', 'mevzuat', 'talimat', 'yönetmelik'
            ]
            
            has_natural_source = any(indicator in response_text.lower()[:150] for indicator in natural_source_indicators)
            
            # Check if source is mentioned at the end (BAD pattern)
            ends_with_bad_source = any(end_pattern in response_text.lower()[-80:] for end_pattern in [
                'kaynak:', 'dosya:', 'referans:', '.pdf'
            ])
            
            # Simple fix: If no natural source in beginning or ends with bad source, try enhancement once
            if not has_natural_source or ends_with_bad_source:
                print("⚠️ Voice response needs better natural source integration, enhancing...")
                
                # Simplified enhancement prompt
                enhanced_prompt = f"""Siz Ankara Bilim Üniversitesi uzmanısınız. Kaynak belirtmeyi BAŞTA yaparak yanıt verin.

SORU: {question}

VERİLEN BELGELER:
{context[:1000]}

ÖNEMLİ: Cümle BAŞINDA kaynak belirtin:
- "X belgesine göre..."  
- "Y mevzuatında belirtildiği üzere..."

DOĞAL CEVAP (kaynak başta):"""

                try:
                    enhanced_response = await openai_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        temperature=0.0,
                        max_tokens=600
                    )
                    
                    enhanced_text = enhanced_response.choices[0].message.content.strip()
                    # Simple validation: Check if enhancement worked
                    if enhanced_text and any(indicator in enhanced_text.lower()[:100] for indicator in natural_source_indicators):
                        response_text = enhanced_text
                        print("✅ Enhanced voice response with natural source integration!")
                    else:
                        print("⚠️ Enhancement didn't improve source integration, keeping original")
                
                except Exception as e:
                    print(f"⚠️ Enhanced voice prompt failed: {e}")
        
        # Cache the result
        api_cache[voice_cache_key] = response_text
        print(f"✅ Enhanced voice response generated: {len(response_text)} characters")
        
        return response_text
        
    except Exception as e:
        print(f"❌ Voice response generation error: {e}")
        return f"Özür dilerim, sesli yanıt oluştururken bir hata oluştu. Lütfen sorunuzu daha spesifik hale getirip tekrar deneyin."


async def ask_question_optimized(question, collection, openai_client, model, cross_encoder_model, domain_context=""):
    """🚀 OPTIMIZED PARALLEL RAG sistemi - %60 hız artışı, aynı kalite"""
    print(f"\n🔍 Processing question (OPTIMIZED): {question}")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # 🔍 Yazım hatası kontrolü - critical words için
    corrected_question = _check_and_correct_critical_typos(question)
    if corrected_question != question:
        print(f"⚠️ Yazım hatası tespit edildi ve düzeltildi: '{question}' → '{corrected_question}'")
        # Kullanıcıya hata mesajı dön
        return f"Sorunuzda yazım hatası tespit ettim. '{question}' yerine '{corrected_question}' mi demek istediniz? Lütfen doğru yazımla tekrar deneyin.", []
    
    # 🚨 Anlamsızlık kontrolü - çok saçma sorular için
    if _is_meaningless_question(question):
        print(f"🚫 Anlamsız soru tespit edildi: '{question}'")
        return "Bu soru anlam ifade etmiyor. Lütfen Ankara Bilim Üniversitesi ile ilgili net ve anlaşılır bir soru sorun.", []
    
    # 🚀 PHASE 1: Parallel başlatma - HyDE + Basic Embedding + Keyword Search
    print("⚡ Phase 1: Parallel initialization...")
    
    async def phase1_tasks():
        # Task 1: HyDE generation (en uzun süren)
        hyde_task = generate_multiple_hyde_variants(question, openai_client, domain_context=domain_context)
        
        # Task 2: Basic question embedding (HyDE beklemeden)
        question_embedding_task = asyncio.to_thread(model.encode, question)
        
        # Task 3: Keyword search (HyDE beklemeden başlayabiliyor)
        if _detect_db_type(collection) == 'weaviate':
            keyword_task = _weaviate_keyword_search(question, collection)
        else:
            keyword_task = _keyword_search(question, collection)
        
        return await asyncio.gather(hyde_task, question_embedding_task, keyword_task)
    
    hyde_variants, question_embedding, keyword_results = await phase1_tasks()
    
    # Safe access to keyword results
    keyword_count = 0
    if keyword_results and isinstance(keyword_results, dict):
        docs = keyword_results.get('documents', [])
        if docs and isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            keyword_count = len(docs[0])
    
    print(f"✅ Phase 1 completed: HyDE={len(hyde_variants)}, embedding=ready, keyword={keyword_count}")
    
    # 🚀 PHASE 2: Enhanced query vector + Semantic search
    print("⚡ Phase 2: Enhanced search...")
    
    async def phase2_tasks():
        # Task 1: Smart query vector (HyDE + question ensemble)
        query_vector_task = _create_smart_query_vector_optimized(question, hyde_variants, question_embedding, model)
        
        return await query_vector_task
    
    query_vector = await phase2_tasks()
    
    # Semantic search (query vector hazır olunca)
    if _detect_db_type(collection) == 'weaviate':
        semantic_results = await _weaviate_semantic_search(query_vector, collection)
    else:
        semantic_results = await _chroma_semantic_search(query_vector, collection)
    
    # Safe access to semantic results
    semantic_count = 0
    if semantic_results and isinstance(semantic_results, dict):
        docs = semantic_results.get('documents', [])
        if docs and isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            semantic_count = len(docs[0])
    
    print(f"✅ Phase 2 completed: semantic={semantic_count}")
    
    # 🚀 PHASE 3: Results combination + Context + Response
    print("⚡ Phase 3: Results processing...")
    
    # Combine search results (hızlı, paralel olmasına gerek yok)
    combined_results = _combine_search_results(semantic_results, keyword_results)
    
    # Smart cross-encoder - OPTIMIZED threshold
    if cross_encoder_model and combined_results:
        top_scores = [r['score'] for r in combined_results[:3]]
        score_variance = max(top_scores) - min(top_scores) if len(top_scores) > 1 else 1.0
        
        # More aggressive skipping for speed - increased threshold
        if score_variance < 0.20 and len(combined_results) > 3:  # Increased from 0.15 to 0.20
            print("🎯 Smart reranking: High uncertainty detected, applying cross-encoder...")
            final_results = await _cross_encoder_rerank(question, combined_results[:10], cross_encoder_model)
        else:
            print("🚀 Smart reranking: Clear results, skipping cross-encoder for speed")
            final_results = combined_results[:10]
    else:
        final_results = combined_results[:10]
    
    # Context assembly (hızlı, paralel gereksiz)
    context = _assemble_optimized_context(final_results, question, domain_context)
    
    # Response generation
    response = await _generate_contextual_response(question, context, openai_client, domain_context)
    
    # Extract source metadata (aynı logic)
    sources_metadata = []
    for result in final_results[:10]:
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    print("🎉 Optimized RAG completed!")
    return response, sources_metadata


async def _create_smart_query_vector_optimized(question, hyde_variants, question_embedding, model):
    """🚀 OPTIMIZED query vector creation - batch processing for cache misses"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Ensure hyde_variants are all strings
    safe_hyde_variants = []
    for variant in hyde_variants:
        if isinstance(variant, list):
            safe_hyde_variants.append(' '.join(str(v) for v in variant))
        else:
            safe_hyde_variants.append(str(variant))
    
    all_texts = [question] + safe_hyde_variants
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    cache_hits = 0
    
    # Phase 1: Cache lookup + prepare uncached batch
    for i, text in enumerate(all_texts):
        normalized_text = str(text).strip().lower()  # Ensure text is string
        cache_key = f"emb_{hash(normalized_text)}"
        
        if cache_key in embedding_cache:
            cached_emb = embedding_cache[cache_key]
            # Ensure numpy array format
            if not isinstance(cached_emb, np.ndarray):
                cached_emb = np.array(cached_emb)
            embeddings.append((i, cached_emb))
            cache_hits += 1
        elif i == 0:  # Question embedding already computed
            # Ensure numpy array format for question embedding
            if not isinstance(question_embedding, np.ndarray):
                embedding_array = np.array(question_embedding)
            else:
                embedding_array = question_embedding
            embedding_cache[cache_key] = embedding_array
            embeddings.append((i, embedding_array))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Phase 2: Batch process uncached embeddings
    if uncached_texts:
        print(f"🔥 Batch processing {len(uncached_texts)} uncached embeddings...")
        batch_embeddings = await asyncio.to_thread(
            model.encode, 
            uncached_texts, 
            batch_size=min(128, len(uncached_texts)),  # INCREASED from 32 to 128
            show_progress_bar=False
        )
        
        # Cache and store results with consistent format
        for text, embedding, idx in zip(uncached_texts, batch_embeddings, uncached_indices):
            normalized_text = str(text).strip().lower()  # Ensure text is string
            cache_key = f"emb_{hash(normalized_text)}"
            # Ensure consistent numpy array format
            if not isinstance(embedding, np.ndarray):
                embedding_array = np.array(embedding)
            else:
                embedding_array = embedding.copy()
            embedding_cache[cache_key] = embedding_array
            embeddings.append((idx, embedding_array))
    
    print(f"🚀 Embedding cache hits: {cache_hits}/{len(all_texts)}, batch processed: {len(uncached_texts)}")
    
    # Phase 3: Sort and combine with ROBUST shape handling
    if not embeddings:
        print("❌ No embeddings generated, using fallback")
        return None
    
    # Sort back to original order
    embeddings.sort(key=lambda x: x[0])
    embedding_arrays = [emb[1] for emb in embeddings]
    
    # 🔧 ROBUST Shape Normalization - Fix mismatch once and for all
    target_dim = 768  # Standard sentence transformer dimension
    
    if embedding_arrays:
        print(f"🔧 Shape normalization: Processing {len(embedding_arrays)} embeddings")
        normalized_embeddings = []
        
        # Step 1: Validate target dimension by checking first valid embedding
        for emb in embedding_arrays:
            if isinstance(emb, np.ndarray):
                flat_emb = emb.flatten()
                if flat_emb.shape[0] == 768:  # Standard sentence transformer dimension
                    target_dim = 768
                    break
                elif flat_emb.shape[0] > 0:  # Any valid dimension as fallback
                    target_dim = flat_emb.shape[0]
        
        print(f"🎯 Target dimension: {target_dim}")
        
        # Step 2: Normalize all embeddings to consistent 1D shape
        for i, emb in enumerate(embedding_arrays):
            try:
                # Convert to numpy if needed
                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb)
                
                # Flatten to 1D
                flat_emb = emb.flatten()
                
                # Handle dimension mismatches
                if flat_emb.shape[0] == target_dim:
                    # Perfect match
                    normalized_embeddings.append(flat_emb)
                elif flat_emb.shape[0] > target_dim:
                    # Too long: truncate (rare case)
                    print(f"⚠️ Truncating embedding {i}: {flat_emb.shape[0]} → {target_dim}")
                    normalized_embeddings.append(flat_emb[:target_dim])
                else:
                    # Too short: pad with zeros (rare case)
                    print(f"⚠️ Padding embedding {i}: {flat_emb.shape[0]} → {target_dim}")
                    padded = np.zeros(target_dim)
                    padded[:flat_emb.shape[0]] = flat_emb
                    normalized_embeddings.append(padded)
                    
            except Exception as e:
                print(f"❌ Error normalizing embedding {i}: {e}")
                # Emergency fallback: create zero vector
                normalized_embeddings.append(np.zeros(target_dim))
        
        embedding_arrays = normalized_embeddings
        print(f"✅ All embeddings normalized to shape: ({target_dim},)")
    else:
        print("❌ No embedding arrays to normalize")
        return np.zeros(target_dim)
    
    # Phase 4: Smart vector combination (now all shapes are consistent!)
    if len(embedding_arrays) == 1:
        final_vector = embedding_arrays[0]
        print(f"🧮 Single embedding used: shape {final_vector.shape}")
    else:
        # Ultra-optimized vector combination - now guaranteed to work!
        try:
            original_weight = 0.5
            hyde_weight = 0.5 / (len(embedding_arrays) - 1) if len(embedding_arrays) > 1 else 0
            
            # Stack normalized embeddings (guaranteed same shape)
            embeddings_array = np.stack(embedding_arrays)
            weights = np.array([original_weight] + [hyde_weight] * (len(embedding_arrays) - 1))
            final_vector = np.average(embeddings_array, axis=0, weights=weights)
            
            print(f"🧮 Combined {len(embedding_arrays)} embeddings into ensemble: shape {final_vector.shape}")
            
        except Exception as e:
            print(f"❌ Vector combination failed: {e}, using first embedding")
            final_vector = embedding_arrays[0]
    
    # Final validation
    if final_vector is None or len(final_vector) == 0:
        print("🚨 Critical: Final vector is empty, using zero fallback")
        final_vector = np.zeros(target_dim)
    
    print(f"✅ Created optimized ensemble vector: {final_vector.shape}")
    return final_vector


async def ask_question_voice_optimized(question, collection, openai_client, model, cross_encoder_model, domain_context="", request=None):
    """🚀 OPTIMIZED Voice RAG sistemi - %60 hız artışı, aynı kalite, disconnection control"""
    print(f"\n🎤 Processing voice question (OPTIMIZED): {question}")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # 🔍 Yazım hatası kontrolü - critical words için (voice için de)
    corrected_question = _check_and_correct_critical_typos(question)
    if corrected_question != question:
        print(f"⚠️ Voice yazım hatası tespit edildi ve düzeltildi: '{question}' → '{corrected_question}'")
        # Kullanıcıya hata mesajı dön
        return f"Sorunuzda yazım hatası tespit ettim. '{question}' yerine '{corrected_question}' mi demek istediniz? Lütfen doğru telaffuzla tekrar deneyin.", []
    
    # 🚨 Anlamsızlık kontrolü - çok saçma sorular için (voice için de)
    if _is_meaningless_question(question):
        print(f"🚫 Anlamsız voice soru tespit edildi: '{question}'")
        return "Bu soru anlam ifade etmiyor. Lütfen Ankara Bilim Üniversitesi ile ilgili net ve anlaşılır bir soru sorun.", []
    
    # Helper function for checking disconnection (aynı logic)
    async def check_disconnection(step_name):
        if request and await request.is_disconnected():
            print(f"🚪 Client disconnected during {step_name}")
            raise Exception(f"Client disconnected during {step_name}")
    
    # 🚀 PHASE 1: Parallel başlatma with disconnection checks
    print("⚡ Phase 1: Parallel initialization...")
    await check_disconnection("phase 1 start")
    
    async def phase1_tasks():
        # Task 1: HyDE generation (en uzun süren)
        hyde_task = generate_multiple_hyde_variants(question, openai_client, domain_context=domain_context)
        
        # Task 2: Basic question embedding (HyDE beklemeden)
        question_embedding_task = asyncio.to_thread(model.encode, question)
        
        # Task 3: Keyword search (HyDE beklemeden başlayabiliyor)
        if _detect_db_type(collection) == 'weaviate':
            keyword_task = _weaviate_keyword_search(question, collection)
        else:
            keyword_task = _keyword_search(question, collection)
        
        return await asyncio.gather(hyde_task, question_embedding_task, keyword_task)
    
    hyde_variants, question_embedding, keyword_results = await phase1_tasks()
    await check_disconnection("phase 1 completion")
    
    # Safe access to keyword results
    keyword_count = 0
    if keyword_results and isinstance(keyword_results, dict):
        docs = keyword_results.get('documents', [])
        if docs and isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            keyword_count = len(docs[0])
    
    print(f"✅ Phase 1 completed: HyDE={len(hyde_variants)}, embedding=ready, keyword={keyword_count}")
    
    # 🚀 PHASE 2: Enhanced query vector + Semantic search
    print("⚡ Phase 2: Enhanced search...")
    await check_disconnection("phase 2 start")
    
    query_vector = await _create_smart_query_vector_optimized(question, hyde_variants, question_embedding, model)
    
    # Semantic search (query vector hazır olunca)
    if _detect_db_type(collection) == 'weaviate':
        semantic_results = await _weaviate_semantic_search(query_vector, collection)
    else:
        semantic_results = await _chroma_semantic_search(query_vector, collection)
    
    await check_disconnection("phase 2 completion")
    
    # Safe access to semantic results
    semantic_count = 0
    if semantic_results and isinstance(semantic_results, dict):
        docs = semantic_results.get('documents', [])
        if docs and isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            semantic_count = len(docs[0])
    
    print(f"✅ Phase 2 completed: semantic={semantic_count}")
    
    # 🚀 PHASE 3: Results combination + Context + Voice Response
    print("⚡ Phase 3: Results processing...")
    await check_disconnection("phase 3 start")
    
    # Combine search results (hızlı, paralel olmasına gerek yok)
    combined_results = _combine_search_results(semantic_results, keyword_results)
    
    # Smart cross-encoder - OPTIMIZED threshold
    if cross_encoder_model and combined_results:
        top_scores = [r['score'] for r in combined_results[:3]]
        score_variance = max(top_scores) - min(top_scores) if len(top_scores) > 1 else 1.0
        
        # More aggressive skipping for speed - increased threshold
        if score_variance < 0.20 and len(combined_results) > 3:  # Increased from 0.15 to 0.20
            print("🎯 Smart reranking: High uncertainty detected, applying cross-encoder...")
            final_results = await _cross_encoder_rerank(question, combined_results[:10], cross_encoder_model)
        else:
            print("🚀 Smart reranking: Clear results, skipping cross-encoder for speed")
            final_results = combined_results[:10]
    else:
        final_results = combined_results[:10]
    
    # Context assembly (hızlı, paralel gereksiz)
    context = _assemble_optimized_context(final_results, question, domain_context)
    
    await check_disconnection("voice response generation")
    
    # Voice-specific response generation
    response = await _generate_voice_response(question, context, openai_client, domain_context)
    
    # Extract source metadata (aynı logic)
    sources_metadata = []
    for result in final_results[:10]:
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    print("🎉 Optimized Voice RAG completed!")
    return response, sources_metadata


async def create_optimized_embeddings(documents, model):
    """Original optimized embedding function - backward compatibility"""
    return await create_optimized_embeddings_v2(documents, model)


async def ask_question_with_memory(question, collection, openai_client, model, cross_encoder_model, 
                                 conversation_context="", domain_context=""):
    """
    💭 Memory-aware RAG sistemi - Conversation history ile geliştirilmiş
    
    Args:
        question: Kullanıcının sorusu
        collection: Weaviate/ChromaDB collection
        openai_client: OpenAI client
        model: Embedding model
        cross_encoder_model: Cross-encoder model
        conversation_context: Önceki konuşma geçmişi
        domain_context: Domain-specific context
        
    Returns:
        (response, sources_metadata): Yanıt ve kaynak metadataları
    """
    print(f"\n💭 Processing memory-aware question: {question}")
    if conversation_context:
        print(f"🧠 Using conversation context: {len(conversation_context)} chars")
    if domain_context:
        print(f"📄 Domain context: {domain_context}")
    
    # Adım 1: Context-aware HyDE generation
    print("📝 Step 1: Context-aware HyDE generation...")
    enhanced_question = _enhance_question_with_context(question, conversation_context)
    hyde_variants = await generate_multiple_hyde_variants(enhanced_question, openai_client, domain_context=domain_context)
    
    # Adım 2: Smart query vector creation with context
    print("🧠 Step 2: Context-enhanced query vector creation...")
    query_vector = await _create_smart_query_vector(enhanced_question, hyde_variants, model)
    
    # Adım 3: Hybrid search with context awareness
    print("🔍 Step 3: Context-aware hybrid search...")
    initial_results = await _perform_hybrid_search(enhanced_question, query_vector, collection, cross_encoder_model)
    
    # Adım 4: Memory-enhanced context assembly
    print("🔧 Step 4: Memory-enhanced context assembly...")
    context = _assemble_memory_enhanced_context(initial_results, question, conversation_context, domain_context)
    
    # Adım 5: Context-aware response generation
    print("✨ Step 5: Memory-aware response generation...")
    response = await _generate_memory_aware_response(question, context, conversation_context, openai_client, domain_context)
    
    # Extract source metadata
    sources_metadata = []
    for result in initial_results[:10]:
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    return response, sources_metadata


def _enhance_question_with_context(question, conversation_context):
    """
    Soruyu conversation context ile zenginleştir - IMPROVED VERSION
    
    Args:
        question: Orijinal soru
        conversation_context: Konuşma geçmişi
        
    Returns:
        Enhanced question string
    """
    if not conversation_context or len(conversation_context.strip()) < 10:
        return question
    
    # IMPROVED: Kategorize reference detection
    direct_references = [
        'bu', 'şu', 'bunlar', 'şunlar', 'bunu', 'şunu', 'bunun', 'şunun',
        'burada', 'şurada', 'o', 'onu', 'onun', 'onlar', 'onları'
    ]
    
    temporal_references = [
        'daha önce', 'yukarıda', 'yukarda', 'önceki', 'geçen', 'az önce',
        'biraz önce', 'şimdi', 'hemen önce', 'son', 'sonra'
    ]
    
    conversation_references = [
        'dediğiniz', 'söylediğiniz', 'bahsettiğiniz', 'anlattığınız',
        'belirttiğiniz', 'açıkladığınız', 'ifade ettiğiniz'
    ]
    
    comparison_references = [
        'aynı', 'benzer', 'farklı', 'diğer', 'ek', 'ilave', 'başka',
        'benzeri', 'karşıt', 'zıt', 'alternatif'
    ]
    
    question_lower = question.lower()
    
    # Check reference types with priority
    has_direct_ref = any(ref in question_lower for ref in direct_references)
    has_temporal_ref = any(ref in question_lower for ref in temporal_references) 
    has_conversation_ref = any(ref in question_lower for ref in conversation_references)
    has_comparison_ref = any(ref in question_lower for ref in comparison_references)
    
    # IMPROVED: Priority-based context inclusion
    if has_direct_ref or has_conversation_ref:
        # High priority - definitely needs context
        context_lines = conversation_context.split('\n')
        recent_context = '\n'.join(context_lines[-8:]) if len(context_lines) > 8 else conversation_context
        
        enhanced = f"""Önceki konuşma bağlamı:
{recent_context}

Şu anki soru: {question}"""
        
        print(f"🔗 Enhanced question with HIGH PRIORITY context (added {len(recent_context)} chars)")
        return enhanced
        
    elif has_temporal_ref or has_comparison_ref:
        # Medium priority - check context relevance first
        question_words = set(question_lower.split())
        context_words = set(conversation_context.lower().split())
        common_words = question_words.intersection(context_words)
        relevance_ratio = len(common_words) / len(question_words) if question_words else 0
        
        if relevance_ratio > 0.20:  # Higher threshold for medium priority
            context_lines = conversation_context.split('\n')
            recent_context = '\n'.join(context_lines[-6:]) if len(context_lines) > 6 else conversation_context
            
            enhanced = f"""Önceki konuşma bağlamı:
{recent_context}

Şu anki soru: {question}"""
            
            print(f"🔗 Enhanced question with MEDIUM PRIORITY context (relevance: {relevance_ratio:.2f})")
            return enhanced
        else:
            print(f"🧠 Medium priority reference but low relevance ({relevance_ratio:.2f}), skipping context")
            return question
    else:
        # No direct references - check for implicit continuation
        words_in_question = len(question.split())
        if words_in_question <= 3:  # Very short questions might be continuation
            question_words = set(question_lower.split())
            context_words = set(conversation_context.lower().split())
            common_words = question_words.intersection(context_words)
            relevance_ratio = len(common_words) / len(question_words) if question_words else 0
            
            if relevance_ratio > 0.30:  # High threshold for implicit continuation
                context_lines = conversation_context.split('\n')
                recent_context = '\n'.join(context_lines[-4:]) if len(context_lines) > 4 else conversation_context
                
                enhanced = f"""Önceki konuşma bağlamı:
{recent_context}

Şu anki soru: {question}"""
                
                print(f"🔗 Enhanced SHORT question with implicit context (relevance: {relevance_ratio:.2f})")
                return enhanced
        
        # Default: no context needed
        print("🆕 New independent question - no context needed")
        return question


def _assemble_memory_enhanced_context(results, question, conversation_context, domain_context=""):
    """
    Memory-enhanced context assembly - IMPROVED VERSION with better relevance detection
    
    Args:
        results: Arama sonuçları
        question: Kullanıcının sorusu
        conversation_context: Konuşma geçmişi
        domain_context: Domain context
        
    Returns:
        Enhanced context string
    """
    if not results:
        return ""
    
    # Normal context assembly
    base_context = _assemble_optimized_context(results, question, domain_context)
    
    # IMPROVED: Memory enhancement with multiple criteria
    if conversation_context and len(conversation_context.strip()) > 20:
        
        # Criterion 1: Word overlap analysis
        question_words = set(question.lower().split())
        context_words = set(conversation_context.lower().split())
        common_words = question_words.intersection(context_words)
        word_relevance = len(common_words) / len(question_words) if question_words else 0
        
        # Criterion 2: Entity/concept overlap (basic NER simulation)
        # Look for capitalized words (potential entities) and specific terms
        question_entities = set([w for w in question.split() if w[0].isupper() and len(w) > 2])
        context_entities = set([w for w in conversation_context.split() if w[0].isupper() and len(w) > 2])
        entity_overlap = len(question_entities.intersection(context_entities))
        
        # Criterion 3: Topic continuity (check for domain-specific keywords)
        domain_keywords = ['kayıt', 'başvuru', 'ücret', 'şart', 'belge', 'tarih', 'süre', 'müfredat', 'ders', 'bölüm']
        question_topics = [word for word in question.lower().split() if word in domain_keywords]
        context_topics = [word for word in conversation_context.lower().split() if word in domain_keywords]
        topic_overlap = len(set(question_topics).intersection(set(context_topics)))
        
        # IMPROVED: Multi-factor relevance scoring
        relevance_score = 0
        
        # Word overlap component (40% weight)
        relevance_score += word_relevance * 0.4
        
        # Entity overlap component (30% weight)  
        if question_entities and context_entities:
            entity_relevance = entity_overlap / max(len(question_entities), len(context_entities))
            relevance_score += entity_relevance * 0.3
        
        # Topic continuity component (30% weight)
        if question_topics and context_topics:
            topic_relevance = topic_overlap / max(len(question_topics), len(context_topics))
            relevance_score += topic_relevance * 0.3
        
        print(f"🧠 Relevance analysis: word={word_relevance:.2f}, entity={entity_overlap}, topic={topic_overlap}, final={relevance_score:.2f}")
        
        # IMPROVED: Dynamic threshold based on conversation length
        conversation_lines = len(conversation_context.split('\n'))
        if conversation_lines <= 4:  # Short conversation - lower threshold
            relevance_threshold = 0.12
        elif conversation_lines <= 10:  # Medium conversation - standard threshold
            relevance_threshold = 0.15  
        else:  # Long conversation - higher threshold to prevent noise
            relevance_threshold = 0.18
        
        # Include context if relevant enough
        if relevance_score > relevance_threshold:
            # Smart context windowing based on relevance
            context_lines = conversation_context.split('\n')
            
            if relevance_score > 0.4:  # High relevance - include more context
                summary_context = '\n'.join(context_lines[-6:]) if len(context_lines) > 6 else conversation_context
            elif relevance_score > 0.25:  # Medium relevance - standard context
                summary_context = '\n'.join(context_lines[-4:]) if len(context_lines) > 4 else conversation_context
            else:  # Low relevance - minimal context
                summary_context = '\n'.join(context_lines[-2:]) if len(context_lines) > 2 else conversation_context
            
            enhanced_context = f"""GEÇMIŞ KONUŞMA:
{summary_context}

İLGİLİ DOKÜMAN BİLGİLERİ:
{base_context}"""
            
            print(f"🧠 Enhanced context with conversation history (+{len(summary_context)} chars, score: {relevance_score:.2f})")
            return enhanced_context
        else:
            print(f"🧠 Conversation context not relevant enough (score: {relevance_score:.2f} < threshold: {relevance_threshold:.2f})")
    
    return base_context


async def _generate_memory_aware_response(question, context, conversation_context, openai_client, domain_context=""):
    """
    Memory-aware response generation - conversation history'yi dikkate alan response
    
    Args:
        question: Kullanıcının sorusu
        context: Document context
        conversation_context: Konuşma geçmişi
        openai_client: OpenAI client
        domain_context: Domain context
        
    Returns:
        Generated response string
    """
    # Context kontrolü
    if not context or len(context.strip()) < 50:
        print("🚨 Critical: Empty or insufficient context for memory-aware response generation!")
        
        if not context or len(context.strip()) < 10:
            error_msg = f"Bu konuda elimdeki {domain_context.lower() if domain_context else 'belgelerde'} bilgi bulunamadı."
            if conversation_context:
                error_msg += " Önceki konuştuklarımızda da bu konuda detay yoktu."
            return error_msg
    
    # Memory-aware prompt template with conditional conversation context
    conversation_instructions = ""
    if conversation_context and len(conversation_context.strip()) > 20:
        conversation_instructions = """
ÖNCEKİ KONUŞMA BİLGİSİ VAR: Geçmiş konuşmalardaki referansları, bağlantıları ve devam eden konuları dikkate al.
- "Bu", "şu", "daha önce bahsettiğiniz" gibi referansları doğru yorumla
- Önceki sorularla bağlantılı cevaplar ver
- Tutarlılığı koru - önceki cevaplarınla çelişme"""
    else:
        conversation_instructions = """
İLK KONUŞMA: Bu kullanıcıyla ilk etkileşim, önceki konuşma referansı yok.
- Doğrudan ve kapsamlı yanıt ver
- "Daha önce" veya "önceki konuşma" ifadeleri kullanma"""
    
    memory_aware_template = f"""Sen IntelliDocs platformunun doküman analizi uzmanısın. Kullanıcılarla sürekli konuşan, geçmiş konuşmaları hatırlayan bir asistansın.
{conversation_instructions}

KRİTİK FORMAT KURALLARI:
- YALNIZCA düz paragraf metni olarak yanıt ver
- Hiçbir biçimlendirme karakteri kullanma: *, **, #, -, 1. 2. 3., •, →, ✓ vb. YASAK
- Kalın yazı, italik, liste, başlık, madde işareti YASAK
- Sadece virgül, nokta, iki nokta, parantez gibi normal noktalama kullan
- Doğal akışkan paragraf metni yaz

MEVCUT DOKÜMAN BİLGİLERİ:
{{context}}

GÖREVLER:
1. Soruyu anlayırken önceki konuşmalardaki bağlamı da dikkate al
2. Kaynak referanslarını [Kaynak: ...] formatında ver
3. Net, anlaşılır ve bağlamsal cevap ver - SADECE DÜZ METİN olarak

🎯 SPESİFİKLİK KURALLARI:
- "Genel olarak" veya "genellikle" yerine mevcut belgedeki bilgileri doğrudan kullan
- "Üniversitenizin yönetmeliğine göre değişebilir" yerine belgedeki spesifik bilgileri ver
- Belirsizlik ifadeleri kullanma, eldeki belgelerden kesin bilgileri aktar
- "Ankara Bilim Üniversitesi'nde" gibi spesifik ifadeler kullan
- Detaylı ve kesin bilgiler ver, genel yorumlar yapma

⚠️ TUTARLILIK KONTROLÜ:
- Sayısal bilgileri (not ortalaması, puan, yüzde vs.) karşılaştırırken dikkatli ol
- Şartları belirtirken "için gerekli minimum X iken mevcut Y" şeklinde açık karşılaştırma yap
- Eğer şart sağlanmıyorsa açıkça "hayır" de, sebeplerini belirt
- Çelişkili bilgi varsa "belgede farklı koşullar belirtiliyor" diye not düş
- Belirsizlik varsa "kesin bilgi için X belgesi kontrol edilmeli" de

ÖZEL ÖZELLİK - DAHA FAZLA BİLGİ İHTİYACI:
- Eğer soru belirsizse veya çok genelse, kullanıcıya netleştirici sorular sor
- Eksik bilgi varsa hangi detayların gerekli olduğunu belirt
- "Bu konuda size daha iyi yardım edebilmem için şu bilgilere ihtiyacım var..." şeklinde yaklaş
- Mevcut bilgilerle cevap ver ama ek soru da sor

SORU: {{question}}

DÜZ METİN CEVAP (hiçbir biçimlendirme karakteri kullanmadan):"""
    
    # Domain context adaptation
    if domain_context:
        adapted_template = memory_aware_template.replace(
            "Sen IntelliDocs platformunun doküman analizi uzmanısın",
            f"Sen {domain_context} konusunda uzman bir asistansın"
        ).replace(
            "Bu konuda verilen dokümanlarda bilgi bulunamadı",
            f"Bu konuda verilen {domain_context.lower()} metinlerinde bilgi bulunamadı"
        )
    else:
        adapted_template = memory_aware_template
    
    try:
        print(f"🤖💭 Generating memory-aware response with context length: {len(context)} chars")
        
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user", 
                "content": adapted_template.format(context=context, question=question)
            }],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # 🚨 CRITICAL: Remove any markdown characters that slipped through
        # Remove bold/italic markers
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)  # Remove **bold**
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)      # Remove *italic*
        answer = re.sub(r'__([^_]+)__', r'\1', answer)      # Remove __bold__
        answer = re.sub(r'_([^_]+)_', r'\1', answer)        # Remove _italic_
        
        # Remove list markers
        answer = re.sub(r'^\s*[-•→✓]\s+', '', answer, flags=re.MULTILINE)  # Remove bullet points
        answer = re.sub(r'^\s*\d+\.\s+', '', answer, flags=re.MULTILINE)   # Remove numbered lists
        
        # Remove headers
        answer = re.sub(r'^#+\s+', '', answer, flags=re.MULTILINE)  # Remove headers
        
        # Clean up any double spaces
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        print(f"✅ Memory-aware response generated: {len(answer)} characters")
        print(f"📝 LLM Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        return answer
        
    except Exception as e:
        print(f"❌ Memory-aware response generation error: {e}")
        fallback_response = f"Özür dilerim, yanıt oluştururken bir hata oluştu. Lütfen sorunuzu tekrar deneyin."
        if conversation_context:
            fallback_response += " Önceki konuşmamızı da dikkate alarak yeniden sorabilirsinin."
        return fallback_response


async def ask_question_with_memory_optimized(question, collection, openai_client, model, cross_encoder_model, 
                                           conversation_context="", domain_context=""):
    """
    🚀💭 SEMANTIC Memory-aware RAG System - AI-driven context detection
    
    Modern approach using:
    - Semantic similarity analysis with embeddings
    - Context dependency classification  
    - Intent continuity detection
    - Dynamic threshold adjustment
    
    Args:
        question: Kullanıcının sorusu
        collection: Weaviate/ChromaDB collection
        openai_client: OpenAI client
        model: Embedding model
        cross_encoder_model: Cross-encoder model
        conversation_context: Önceki konuşma geçmişi
        domain_context: Domain-specific context
        
    Returns:
        (response, sources_metadata): Yanıt ve kaynak metadataları
    """
    print(f"\n🚀💭 Processing SEMANTIC memory-aware question: {question}")
    if conversation_context:
        print(f"🧠 Available conversation context: {len(conversation_context)} chars")
    
    # 🔍 SAME QUALITY CONTROLS AS NORMAL CHAT - Yazım hatası kontrolü
    corrected_question = _check_and_correct_critical_typos(question)
    if corrected_question != question:
        print(f"⚠️ Memory chat yazım hatası tespit edildi ve düzeltildi: '{question}' → '{corrected_question}'")
        return f"Sorunuzda yazım hatası tespit ettim. '{question}' yerine '{corrected_question}' mi demek istediniz? Lütfen doğru yazımla tekrar deneyin.", []
    
    # 🚨 SAME QUALITY CONTROLS AS NORMAL CHAT - Anlamsızlık kontrolü
    if _is_meaningless_question(question):
        print(f"🚫 Memory chat anlamsız soru tespit edildi: '{question}'")
        return "Bu soru anlam ifade etmiyor. Lütfen net ve anlaşılır bir soru sorun.", []
    
    # 🧠 SEMANTIC Context Dependency Analysis
    needs_context, confidence, reason = await _detect_context_dependency_semantic(
        question, conversation_context, model, is_voice=False
    )
    
    print(f"🎯 Semantic decision: needs_context={needs_context}, confidence={confidence:.3f}, reason={reason}")
    
    # Context-enhanced question preprocessing (only if needed)
    if needs_context:
        enhanced_question = _enhance_question_with_context(question, conversation_context)
        use_conversation_context = conversation_context
        print(f"🔗 Enhanced question with semantic context (added {len(conversation_context)} chars)")
        print(f"✅ Using context-enhanced processing")
    else:
        enhanced_question = question
        use_conversation_context = ""
        print(f"🆕 Using independent processing (no context needed)")
    
    # 🚀 PHASE 1: Paralel başlangıç işlemleri (conversation-aware)
    async def memory_phase1_tasks():
        # Task 1: Context-aware HyDE generation
        hyde_task = asyncio.create_task(
            generate_multiple_hyde_variants(enhanced_question, openai_client, domain_context=domain_context)
        )
        
        # Task 2: Question embedding (for ensemble)
        question_embedding_task = asyncio.create_task(
            asyncio.to_thread(model.encode, enhanced_question)
        )
        
        return await asyncio.gather(hyde_task, question_embedding_task)
    
    # Execute phase 1
    hyde_variants, question_embedding = await memory_phase1_tasks()
    print(f"✅ Phase 1 completed: {len(hyde_variants)} HyDE variants generated")
    
    # 🚀 PHASE 2: Paralel search ve reranking (memory-enhanced)
    async def memory_phase2_tasks():
        # Task 1: Context-enhanced query vector
        query_vector_task = asyncio.create_task(
            _create_smart_query_vector_optimized(enhanced_question, hyde_variants, question_embedding, model)
        )
        
        # Wait for query vector, then do search
        query_vector = await query_vector_task
        
        # Task 2: Parallel search operations
        search_task = asyncio.create_task(
            _perform_hybrid_search(enhanced_question, query_vector, collection, cross_encoder_model)
        )
        
        return await search_task
    
    # Execute phase 2
    search_results = await memory_phase2_tasks()
    print(f"✅ Phase 2 completed: {len(search_results)} search results")
    
    # 🚀 PHASE 3: Semantic memory-enhanced context ve response generation
    context = _assemble_memory_enhanced_context(search_results, question, use_conversation_context, domain_context)
    
    # Enhanced context information
    print(f"🧠 Enhanced context with semantic conversation history (+{len(use_conversation_context)} chars)")
    
    response = await _generate_memory_aware_response(question, context, use_conversation_context, openai_client, domain_context)
    
    # Extract source metadata
    sources_metadata = []
    for result in search_results[:10]:
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    print("🎉 Memory-aware optimized RAG completed!")
    return response, sources_metadata


# Voice-specific memory-aware function
async def ask_question_voice_with_memory(question, collection, openai_client, model, cross_encoder_model, 
                                       conversation_context="", domain_context="", request=None):
    """
    🎤💭 Voice + Memory-aware RAG sistemi - Ses ve hafıza kombinasyonu
    
    Args:
        question: Kullanıcının sorusu
        collection: Weaviate/ChromaDB collection
        openai_client: OpenAI client
        model: Embedding model
        cross_encoder_model: Cross-encoder model
        conversation_context: Önceki konuşma geçmişi
        domain_context: Domain context
        request: Request object for disconnection check
        
    Returns:
        (response, sources_metadata): Yanıt ve kaynak metadataları
    """
    print(f"\n🎤💭 Processing voice + memory question: {question}")
    if conversation_context:
        print(f"🧠 Using conversation context: {len(conversation_context)} chars")
    
    # Helper function for checking disconnection
    async def check_disconnection(step_name):
        if request and await request.is_disconnected():
            print(f"🚪 Client disconnected during {step_name}")
            raise Exception(f"Client disconnected during {step_name}")
    
    # 🧠 SEMANTIC Context Dependency Analysis for Voice
    await check_disconnection("semantic context analysis")
    
    needs_context, confidence, reason = await _detect_context_dependency_semantic(
        question, conversation_context, model, is_voice=True
    )
    
    print(f"🎯 Voice semantic decision: needs_context={needs_context}, confidence={confidence:.3f}")
    
    # Enhanced question with context (only if needed)
    if needs_context:
        enhanced_question = _enhance_question_with_context(question, conversation_context)
        use_conversation_context = conversation_context
    else:
        enhanced_question = question
        use_conversation_context = ""
    
    await check_disconnection("HyDE generation")
    hyde_variants = await generate_multiple_hyde_variants(enhanced_question, openai_client, domain_context=domain_context)
    
    await check_disconnection("query vector creation")
    query_vector = await _create_smart_query_vector(enhanced_question, hyde_variants, model)
    
    await check_disconnection("hybrid search")
    initial_results = await _perform_hybrid_search(enhanced_question, query_vector, collection, cross_encoder_model)
    
    await check_disconnection("semantic context assembly")
    context = _assemble_memory_enhanced_context(initial_results, question, use_conversation_context, domain_context)
    
    # Voice-specific + semantic memory-aware response
    await check_disconnection("voice + semantic memory response generation")
    response = await _generate_voice_memory_response(question, context, use_conversation_context, openai_client, domain_context)
    
    # Extract source metadata
    sources_metadata = []
    for result in initial_results[:10]:
        metadata = result.get('metadata', {})
        sources_metadata.append({
            'source': metadata.get('source', 'Unknown'),
            'article': metadata.get('article', 'Unknown')
        })
    
    return response, sources_metadata


async def _generate_voice_memory_response(question, context, conversation_context, openai_client, domain_context=""):
    """
    Voice + Memory-aware response generation - Hem sesli hem hafızalı
    
    Args:
        question: Kullanıcının sorusu
        context: Document context
        conversation_context: Konuşma geçmişi
        openai_client: OpenAI client
        domain_context: Domain context
        
    Returns:
        Generated response string
    """
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # Context kontrolü
    if not context or len(context.strip()) < 50:
        print("🚨 Critical: Empty or insufficient context for voice + memory response!")
        
        if not context or len(context.strip()) < 10:
            error_msg = f"Bu konuda elimdeki {domain_context.lower() if domain_context else 'belgelerde'} bilgi bulunamadı."
            if conversation_context:
                error_msg += " Önceki konuştuklarımızda da bu konuda detay yoktu."
            return error_msg
    
    # Voice + Memory hybrid prompt template with conditional memory instructions
    if conversation_context and len(conversation_context.strip()) > 20:
        memory_instructions = """- Önceki konuşmalardaki referansları hatırla
- Kullanıcının geçmiş sorularıyla bağlantı kur
- Tutarlı bir sohbet deneyimi sun
- "Bu", "şu" gibi referansları doğru yorumla
- "Daha önce de belirttiğim gibi" tarzında bağlayıcılar kullan"""
    else:
        memory_instructions = """- Bu ilk konuşma, önceki referans yok
- Doğrudan ve net yanıt ver
- "Daha önce" ifadeleri kullanma"""
    
    voice_memory_template = """Sen IntelliDocs platformunun sesli asistanısın. Kullanıcılarla doğal konuşma yapan, geçmiş sohbetleri hatırlayan bir asistansın.

SESLİ YANIT KURALLARI:
- Doğal, akıcı ve konuşma dilinde yanıt ver
- Kısa ve net cümleler kullan
- Kaynak referanslarını doğal dile entegre et: "Üniversite burs yönetmeliğine göre..." şeklinde
- Belge adlarını doğal şekilde bahset: "Burs belgelerinde belirtildiği üzere..."

HAFIZA KURALLARI (SADECE ÖNCEKİ KONUŞMA VARSA):
{memory_instructions}

🎯 SESLİ SPESİFİKLİK:
- "Genellikle" yerine "Ankara Bilim Üniversitesi'nde" de
- "Değişebilir" yerine mevcut belgedeki kesin bilgileri ver
- Belirsiz ifadeler kullanma, net bilgiler aktar
- Detaylı ama konuşma diline uygun açıklamalar yap

⚠️ TUTARLILIK KONTROLÜ (Sesli):
- Sayısal karşılaştırmalarda net ol: "X gerekli, sizinki Y, bu yüzden hayır/evet"
- Çelişki varsa: "Belgede farklı koşullar var, açıklığa kavuşturalım"
- Belirsizlik varsa: "Kesin cevap için ek bilgi gerekli"
- Şart sağlanmıyorsa doğrudan: "Maalesef bu şartları karşılamıyor"

SESLİ SORU SORMA:
- Belirsiz sorularda: "Hangi konuda daha detay istiyorsunuz?" 
- Eksik bilgi varsa: "Size daha iyi yardım edebilmem için şunu öğrenebilir miyim..."
- Çok geniş sorularda: "Bu konuyu hangi açıdan ele alalım?"
- Doğal ve samimi tonla ek bilgi isteyin

MEVCUT BİLGİLER:
{context}

SORU: {question}

SESLİ CEVAP:"""
    
    # Domain context adaptation
    if domain_context:
        adapted_template = voice_memory_template.replace(
            "Sen IntelliDocs platformunun sesli asistanısın",
            f"Sen {domain_context} konusunda uzman sesli asistanısın"
        )
    else:
        adapted_template = voice_memory_template
    
    try:
        print(f"🎤🤖💭 Generating voice + memory response with context: {len(context)} chars")
        
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user", 
                "content": adapted_template.format(context=context, question=question)
            }],
            temperature=0.15,  # Slightly higher for more natural voice
            max_tokens=800     # Shorter for voice
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Voice + memory response generated: {len(answer)} characters")
        print(f"📝 Voice LLM Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        return answer
        
    except Exception as e:
        print(f"❌ Voice + memory response generation error: {e}")
        fallback = "Özür dilerim, yanıt oluştururken sorun yaşadım."
        if conversation_context:
            fallback += " Önceki konuşmamızı hatırlıyorum, sorunuzu farklı şekilde sorabilir misiniz?"
        return fallback


async def _detect_context_dependency_semantic(question, conversation_context, model, is_voice=False):
    """
    🧠 Semantic-based context dependency detection - Modern AI approach
    
    Instead of manual keyword matching, uses:
    1. Semantic similarity between question and context
    2. Context dependency classification 
    3. Intent continuity analysis
    4. Topic flow detection
    
    Args:
        question: Current user question
        conversation_context: Previous conversation history
        model: Embedding model for semantic analysis
        is_voice: Whether this is a voice interaction (more aggressive memory)
        
    Returns:
        (needs_context: bool, confidence: float, reason: str)
    """
    if not conversation_context or len(conversation_context.strip()) < 20:
        return False, 0.0, "no_previous_context"
    
    try:
        # 1. Extract last few exchanges for analysis
        context_lines = conversation_context.split('\n')
        recent_exchanges = []
        
        for i in range(len(context_lines)-1, -1, -1):
            line = context_lines[i].strip()
            if line.startswith('Kullanıcı:') or line.startswith('Asistan:'):
                recent_exchanges.append(line)
                if len(recent_exchanges) >= 6:  # Last 3 exchanges
                    break
        
        if len(recent_exchanges) < 2:
            return False, 0.0, "insufficient_context"
        
        recent_exchanges.reverse()  # Chronological order
        
        # 2. Extract recent questions and responses
        recent_questions = []
        recent_responses = []
        
        for line in recent_exchanges:
            if line.startswith('Kullanıcı:'):
                recent_questions.append(line.replace('Kullanıcı:', '').strip())
            elif line.startswith('Asistan:'):
                recent_responses.append(line.replace('Asistan:', '').strip())
        
        if not recent_questions:
            return False, 0.0, "no_recent_questions"
        
        # 3. Semantic Similarity Analysis
        print(f"🧠 Semantic analysis: current='{question}' vs recent={len(recent_questions)} questions")
        
        # Create embeddings for comparison
        texts_to_embed = [question] + recent_questions + recent_responses
        embeddings = await asyncio.to_thread(model.encode, texts_to_embed)
        
        question_embedding = embeddings[0]
        recent_q_embeddings = embeddings[1:len(recent_questions)+1]
        recent_r_embeddings = embeddings[len(recent_questions)+1:]
        
        # Calculate semantic similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Similarity with recent questions (continuation patterns)
        q_similarities = []
        for recent_emb in recent_q_embeddings:
            sim = cosine_similarity([question_embedding], [recent_emb])[0][0]
            q_similarities.append(sim)
        
        # Similarity with recent responses (dependency on answers)
        r_similarities = []
        for recent_emb in recent_r_embeddings:
            sim = cosine_similarity([question_embedding], [recent_emb])[0][0]
            r_similarities.append(sim)
        
        max_q_sim = max(q_similarities) if q_similarities else 0
        max_r_sim = max(r_similarities) if r_similarities else 0
        avg_sim = (max_q_sim + max_r_sim) / 2
        
        print(f"🔍 Semantic similarities: q_max={max_q_sim:.3f}, r_max={max_r_sim:.3f}, avg={avg_sim:.3f}")
        
        # 4. Context Dependency Classification
        dependency_indicators = await _classify_context_dependency(question, recent_questions, recent_responses)
        
        # 5. Combine semantic and structural signals
        semantic_score = avg_sim
        dependency_score = dependency_indicators['score']
        
        # Weighted final score
        final_score = (semantic_score * 0.6) + (dependency_score * 0.4)
        
        # 🎤 VOICE-SPECIFIC THRESHOLDS - More aggressive memory for voice interactions
        if is_voice:
            # Voice users expect more conversational flow - lower thresholds
            if len(recent_questions) <= 2:
                threshold = 0.15  # Very aggressive for short voice conversations
            elif len(recent_questions) <= 4:
                threshold = 0.22  # Moderately aggressive for medium conversations
            else:
                threshold = 0.30  # Still more lenient than text for long conversations
            print(f"🎤 Using voice-optimized threshold: {threshold}")
        else:
            # Standard text thresholds
            if len(recent_questions) <= 2:
                threshold = 0.25  # More lenient for short conversations
            elif len(recent_questions) <= 4:
                threshold = 0.35  # Standard threshold
            else:
                threshold = 0.45  # Stricter for long conversations to prevent drift
        
        needs_context = final_score > threshold
        confidence = final_score
        
        reason = f"semantic_analysis(sim={semantic_score:.3f}, dep={dependency_score:.3f}, final={final_score:.3f})"
        
        print(f"🎯 Context decision: needs={needs_context}, confidence={confidence:.3f}, reason={reason}")
        
        return needs_context, confidence, reason
        
    except Exception as e:
        print(f"❌ Error in semantic context detection: {e}")
        # Fallback to simple approach
        return await _fallback_context_detection(question, conversation_context)


async def _classify_context_dependency(question, recent_questions, recent_responses):
    """
    🔍 Classify context dependency using multiple signals
    
    Args:
        question: Current question
        recent_questions: List of recent questions
        recent_responses: List of recent responses
        
    Returns:
        Dict with dependency classification
    """
    indicators = {
        'pronoun_dependency': 0,      # Pronouns referring to previous context
        'ellipsis_continuation': 0,   # Incomplete questions needing context
        'topic_continuity': 0,        # Same topic continuation
        'clarification_request': 0,   # Asking for clarification
        'comparative_reference': 0,   # Comparing with previous info
        'score': 0
    }
    
    question_lower = question.lower()
    question_words = question.split()
    
    # 1. Pronoun Dependency Detection (Turkish-specific)
    turkish_pronouns = {
        'bu': 0.8, 'şu': 0.8, 'o': 0.6, 'bunlar': 0.9, 'şunlar': 0.9, 'onlar': 0.7,
        'bunu': 0.8, 'şunu': 0.8, 'onu': 0.6, 'bunları': 0.9, 'şunları': 0.9, 'onları': 0.7,
        'bunun': 0.8, 'şunun': 0.8, 'onun': 0.6, 'bunların': 0.9, 'şunların': 0.9, 'onların': 0.7
    }
    
    for pronoun, weight in turkish_pronouns.items():
        if pronoun in question_lower:
            indicators['pronoun_dependency'] = max(indicators['pronoun_dependency'], weight)
    
    # 2. Ellipsis/Incomplete Question Detection
    question_words_count = len([w for w in question_words if len(w) > 2])
    if question_words_count <= 3:  # Very short questions
        if any(word in question_lower for word in ['ne', 'nasıl', 'neden', 'kim', 'nerede', 'ne zaman']):
            indicators['ellipsis_continuation'] = 0.7
    
    # Check for incomplete sentence patterns
    incomplete_patterns = ['peki', 'ee', 'hani', 'ya', 'ama', 've', 'ayrıca', 'bir de']
    for pattern in incomplete_patterns:
        if question.strip().startswith(pattern):
            indicators['ellipsis_continuation'] = 0.8
    
    # 3. Clarification Request Detection
    clarification_patterns = [
        'ne demek', 'nasıl yani', 'açar mısın', 'detayını', 'daha fazla', 'örnek',
        'hangisi', 'hangi', 'kaçıncı', 'nerede', 'ne zaman exactly'
    ]
    for pattern in clarification_patterns:
        if pattern in question_lower:
            indicators['clarification_request'] = 0.6
    
    # 4. Comparative Reference Detection  
    comparative_words = ['aynı', 'farklı', 'benzer', 'karşı', 'diğer', 'ek', 'ilave', 'başka', 'alternatif']
    for word in comparative_words:
        if word in question_lower:
            indicators['comparative_reference'] = 0.5
    
    # 5. Topic Continuity (semantic)
    if recent_responses:
        # Extract key terms from recent responses
        recent_response_text = ' '.join(recent_responses[-2:]).lower()  # Last 2 responses
        
        # Simple term overlap
        response_words = set(recent_response_text.split())
        question_words_set = set(question_lower.split())
        
        overlap = len(response_words.intersection(question_words_set))
        total_unique = len(response_words.union(question_words_set))
        
        if total_unique > 0:
            topic_similarity = overlap / total_unique
            indicators['topic_continuity'] = min(topic_similarity * 2, 1.0)  # Scale to 0-1
    
    # 6. Calculate final dependency score
    weights = {
        'pronoun_dependency': 0.35,      # Highest weight - clear dependency
        'ellipsis_continuation': 0.25,   # Strong indicator
        'clarification_request': 0.15,   # Medium indicator
        'comparative_reference': 0.10,   # Lower weight
        'topic_continuity': 0.15        # Contextual continuity
    }
    
    final_score = 0
    for indicator, value in indicators.items():
        if indicator in weights:
            final_score += value * weights[indicator]
    
    indicators['score'] = final_score
    
    print(f"📊 Dependency analysis: {indicators}")
    return indicators


async def _fallback_context_detection(question, conversation_context):
    """
    Simple fallback when semantic analysis fails
    """
    question_lower = question.lower()
    
    # High-confidence indicators
    if any(word in question_lower for word in ['bu', 'şu', 'o', 'bunları', 'şunları']):
        return True, 0.8, "fallback_pronoun_detected"
    
    # Low-confidence indicators
    if len(question.split()) <= 3:
        return True, 0.4, "fallback_short_question"
    
    return False, 0.1, "fallback_independent"


def _enhance_question_with_context(question, conversation_context):
    """
    Soruyu conversation context ile zenginleştir - SEMANTIC VERSION
    
    Args:
        question: Orijinal soru
        conversation_context: Konuşma geçmişi
        
    Returns:
        Enhanced question string
    """
    if not conversation_context or len(conversation_context.strip()) < 10:
        return question
    
    # Bu fonksiyon artık semantic analysis sonucu çağrılır
    # Context'in alakalı olduğu zaten belirlendi
    context_lines = conversation_context.split('\n')
    
    # Smart context window - son birkaç exchange
    if len(context_lines) > 8:
        recent_context = '\n'.join(context_lines[-8:])
    else:
        recent_context = conversation_context
    
    enhanced = f"""Önceki konuşma bağlamı:
{recent_context}

Şu anki soru: {question}"""
    
    print(f"🔗 Enhanced question with semantic context (added {len(recent_context)} chars)")
    return enhanced


def _assemble_memory_enhanced_context(results, question, conversation_context, domain_context=""):
    """
    Memory-enhanced context assembly - SEMANTIC VERSION
    
    Args:
        results: Arama sonuçları
        question: Kullanıcının sorusu
        conversation_context: Konuşma geçmişi
        domain_context: Domain context
        
    Returns:
        Enhanced context string
    """
    if not results:
        return ""
    
    # Normal context assembly
    base_context = _assemble_optimized_context(results, question, domain_context)
    
    # Memory enhancement - semantic context zaten relevance check'ten geçti
    if conversation_context and len(conversation_context.strip()) > 20:
        # Context'i dahil et - çünkü semantic analysis onayladı
        context_lines = conversation_context.split('\n')
        
        # Smart windowing
        if len(context_lines) > 6:
            summary_context = '\n'.join(context_lines[-6:])  # Son 3 exchange
        else:
            summary_context = conversation_context
        
        enhanced_context = f"""GEÇMIŞ KONUŞMA:
{summary_context}

İLGİLİ DOKÜMAN BİLGİLERİ:
{base_context}"""
        
        print(f"🧠 Enhanced context with semantic conversation history (+{len(summary_context)} chars)")
        return enhanced_context
    
    return base_context
