"""
🚀 Enhanced Semantic Embedding Engine Module
Advanced semantic-aware embedding generation with content intelligence
"""

import asyncio
import google.generativeai as genai
import numpy as np
from typing import List, Dict, Optional
from cachetools import TTLCache
from config import EMBEDDING_DIMENSION
import re

# Embedding cache for performance
embedding_cache = TTLCache(maxsize=3000, ttl=1200)  # Increased cache, 20 min TTL

# Memory optimization: Object pool for frequent operations
_temp_vector_pool = []
_max_pool_size = 100

class SemanticEmbeddingEngine:
    """Advanced semantic embedding engine with content intelligence"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        # Content type detection patterns
        self.content_patterns = {
            'legal_article': [
                r'MADDE\s+\d+',
                r'Article\s+\d+',
                r'\(\d+\).*[a-z]\)',  # Legal subsections
            ],
            'procedural': [
                r'başvuru.*yapılır',
                r'işlem.*gerçekleştirilir',
                r'adımlar.*takip edilir',
                r'süreç.*aşağıdaki gibidir'
            ],
            'definitional': [
                r'.*nedir\?',
                r'.*tanımı',
                r'.*anlamı',
                r'.*ifade.*eder'
            ],
            'conditional': [
                r'eğer.*ise',
                r'durumunda.*',
                r'halinde.*',
                r'koşulunda.*'
            ],
            'numerical': [
                r'\d+\s*(gün|ay|yıl)',
                r'%\d+',
                r'\d+\s*(TL|lira)',
                r'\d+\.\d+\s*(not|ortalama)'
            ]
        }
    
    def detect_content_type(self, text: str) -> str:
        """Detect the semantic type of content"""
        text_lower = text.lower()
        
        # Count matches for each content type
        type_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                type_scores[content_type] = score
        
        if not type_scores:
            return 'general'
        
        # Return the type with highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def enhance_text_for_embedding(self, text: str, content_type: str = None) -> str:
        """Enhance text for better semantic embedding"""
        if not content_type:
            content_type = self.detect_content_type(text)
        
        enhanced_text = text
        
        # Content-type specific enhancement
        if content_type == 'legal_article':
            # Add semantic markers for legal content
            enhanced_text = f"[LEGALsemantic] {enhanced_text}"
            
        elif content_type == 'procedural':
            # Add process indicators
            enhanced_text = f"[PROCEDUREsemantic] {enhanced_text}"
            
        elif content_type == 'definitional':
            # Add definition markers
            enhanced_text = f"[DEFINITIONsemantic] {enhanced_text}"
            
        elif content_type == 'conditional':
            # Add conditional logic markers
            enhanced_text = f"[CONDITIONALsemantic] {enhanced_text}"
            
        elif content_type == 'numerical':
            # Add numerical context
            enhanced_text = f"[NUMERICALsemantic] {enhanced_text}"
        
        # Domain-specific enhancement for Turkish educational content
        if any(term in text.lower() for term in ['üniversite', 'öğrenci', 'ders', 'sınav', 'burs']):
            enhanced_text = f"[EDUCATIONALsemantic] {enhanced_text}"
        
        return enhanced_text
    
    def _get_temp_vector(self, size: int) -> np.ndarray:
        """Get a temporary vector from pool or create new one"""
        if _temp_vector_pool:
            vector = _temp_vector_pool.pop()
            if len(vector) == size:
                vector.fill(0)  # Reset values
                return vector
        return np.zeros(size)
    
    def _return_temp_vector(self, vector: np.ndarray):
        """Return vector to pool for reuse"""
        if len(_temp_vector_pool) < _max_pool_size:
            _temp_vector_pool.append(vector)
    
    async def create_semantic_query_embeddings(self, queries: List[str]) -> List[np.ndarray]:
        """Create semantically enhanced embeddings for queries"""
        
        print(f"🧠 Creating semantic embeddings for {len(queries)} queries")
        
        embeddings = []
        cache_hits = 0
        
        # Enhance queries for better semantic understanding
        enhanced_queries = []
        for query in queries:
            content_type = self.detect_content_type(query)
            enhanced_query = self.enhance_text_for_embedding(query, content_type)
            enhanced_queries.append(enhanced_query)
        
        # Check cache and generate embeddings
        for i, (original_query, enhanced_query) in enumerate(zip(queries, enhanced_queries)):
            normalized_query = original_query.strip().lower()
            cache_key = f"semantic_query_emb_{hash(normalized_query)}"
            
            if cache_key in embedding_cache:
                embeddings.append(embedding_cache[cache_key])
                cache_hits += 1
            else:
                # Generate embedding with enhancement
                embedding = genai.embed_content(model=self.embedding_model, content=enhanced_query)['embedding']
                embedding_array = np.array(embedding)
                embedding_cache[cache_key] = embedding_array
                embeddings.append(embedding_array)
        
        if cache_hits > 0:
            print(f"🚀 Semantic embedding cache hits: {cache_hits}/{len(queries)}")
        
        return embeddings
    
    # Backward compatibility
    async def create_query_embeddings(self, queries: List[str]) -> List[np.ndarray]:
        """Backward compatible method"""
        return await self.create_semantic_query_embeddings(queries)
    
    async def create_smart_query_vector(self, primary_query: str, hyde_variants: List[str]) -> np.ndarray:
        """Create smart query vector from primary query + HyDE variants with semantic awareness"""
        
        all_texts = [primary_query] + hyde_variants
        
        # Detect content types for adaptive weighting
        content_types = [self.detect_content_type(text) for text in all_texts]
        
        embeddings = await self.create_semantic_query_embeddings(all_texts)
        
        if not embeddings:
            print("❌ No embeddings generated, returning zero vector")
            return np.zeros(EMBEDDING_DIMENSION)
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Adaptive weighting based on content type
        primary_weight = 0.6  # Increased primary weight
        hyde_weights = []
        
        for i, content_type in enumerate(content_types[1:], 1):
            # Weight based on content type relevance
            if content_type == 'definitional':
                weight = 0.15  # Definitions are important
            elif content_type == 'procedural':
                weight = 0.12  # Procedures are valuable
            elif content_type == 'legal_article':
                weight = 0.10  # Legal content is specific
            else:
                weight = 0.08  # General content
            
            hyde_weights.append(weight)
        
        # Normalize weights
        total_hyde_weight = sum(hyde_weights)
        if total_hyde_weight > 0:
            hyde_weights = [w * (0.4 / total_hyde_weight) for w in hyde_weights]
        
        # Vectorized operations for performance
        embeddings_array = np.stack(embeddings)
        weights = np.array([primary_weight] + hyde_weights)
        final_vector = np.average(embeddings_array, axis=0, weights=weights)
        
        print(f"🧮 Created semantic-aware query vector from {len(embeddings)} sources")
        return final_vector
    
    async def create_optimized_document_embeddings(self, documents: List[Dict], batch_size: int = 120) -> List[List[float]]:
        """Ultra-optimized semantic document embedding creation"""
        
        print(f"🧠 Creating semantic document embeddings for {len(documents)} documents")
        
        # Adaptive batch size based on document count and content complexity
        if len(documents) > 800:
            batch_size = 150
        elif len(documents) > 400:
            batch_size = 120
        else:
            batch_size = 80
        
        print(f"📦 Using adaptive batch size: {batch_size}")
        
        # Enhance documents for better semantic representation
        enhanced_documents = []
        for doc in documents:
            content = doc.get('content', '')
            
            # Detect content type
            content_type = self.detect_content_type(content)
            
            # Enhance based on document metadata if available
            structure_type = doc.get('structure_type', 'general')
            hierarchy_level = doc.get('hierarchy_level', 1)
            
            # Create enhanced content
            enhanced_content = self.enhance_text_for_embedding(content, content_type)
            
            # Add structural context
            if structure_type == 'article':
                enhanced_content = f"[ARTICLE_L{hierarchy_level}] {enhanced_content}"
            elif structure_type == 'section':
                enhanced_content = f"[SECTION_L{hierarchy_level}] {enhanced_content}"
            elif structure_type == 'subsection':
                enhanced_content = f"[SUBSECTION_L{hierarchy_level}] {enhanced_content}"
            
            enhanced_documents.append(enhanced_content)
        
        # Process in parallel batches with semantic awareness
        async def process_semantic_batch(batch: List[str], batch_idx: int) -> List[List[float]]:
            try:
                batch_embeddings = await asyncio.to_thread(
                    lambda: genai.embed_content(model=self.embedding_model, content=batch, task_type="retrieval_document")['embedding']
                )
                print(f"⚡ Completed semantic batch {batch_idx}")
                return batch_embeddings.tolist()
            except Exception as e:
                print(f"❌ Semantic batch {batch_idx} error: {e}")
                # Fallback: process individually with semantic enhancement
                fallback_embeddings = []
                for enhanced_content in batch:
                    try:
                        embedding = genai.embed_content(model=self.embedding_model, content=enhanced_content, task_type="retrieval_document")['embedding']
                        fallback_embeddings.append(embedding)
                    except:
                        # Ultimate fallback: zero vector
                        dim = 768
                        fallback_embeddings.append([0.0] * dim)
                return fallback_embeddings
        
        # Create batches
        batches = []
        for i in range(0, len(enhanced_documents), batch_size):
            batch = enhanced_documents[i:i+batch_size]
            batches.append(batch)
        
        # Process batches with optimized concurrency control
        embeddings = []
        chunk_size = 4  # Increased concurrency for better performance
        
        for i in range(0, len(batches), chunk_size):
            batch_chunk = batches[i:i+chunk_size]
            tasks = [
                process_semantic_batch(batch, i + j + 1) 
                for j, batch in enumerate(batch_chunk)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            for batch_result in batch_results:
                embeddings.extend(batch_result)
        
        print(f"🎯 Generated {len(embeddings)} semantic embeddings with content-aware enhancement")
        return embeddings
    
    # Backward compatibility
    async def create_optimized_embeddings(self, documents: List[Dict], batch_size: int = 120) -> List[List[float]]:
        """Backward compatible method"""
        return await self.create_optimized_document_embeddings(documents, batch_size)
    
    def compute_semantic_similarity_scores(self, query_vector: np.ndarray, 
                                         doc_vectors: List[np.ndarray]) -> List[float]:
        """Compute semantic similarity scores with content awareness"""
        
        if not doc_vectors:
            return []
        
        # Stack all document vectors
        doc_matrix = np.stack(doc_vectors)
        
        # Normalize vectors for better semantic comparison
        query_norm = query_vector / np.linalg.norm(query_vector)
        doc_norms = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        
        # Compute cosine similarities in batch
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities.tolist()
    
    

# Backward compatibility class
class EmbeddingEngine(SemanticEmbeddingEngine):
    """Backward compatible class name"""
    pass 