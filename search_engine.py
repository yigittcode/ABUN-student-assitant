"""
🔍 Search Engine Module
Semantic, keyword search operations with advanced re-ranking
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from cachetools import TTLCache

# Search result cache
search_cache = TTLCache(maxsize=1000, ttl=450)  # 7.5 min cache

class SearchEngine:
    """Advanced search engine with hybrid search capabilities"""
    
    def __init__(self, cross_encoder_model=None):
        self.cross_encoder_model = cross_encoder_model
        
        # Domain-specific query expansion patterns for Turkish university content
        self.query_expansions = {
            'ortak dersler': [
                'ortak zorunlu dersler',
                'atatürk ilkeleri ve inkılap tarihi',
                'türk dili',
                'yabancı dil',
                'temel bilimler dersleri',
                'matematik fizik kimya biyoloji',
                'ortak koordinatörlük'
            ],
            'burs': [
                'burs türleri',
                'başarı bursu',
                'ihtiyaç bursu', 
                'ösym bursu',
                'tam burslu',
                'kısmi burslu',
                'indirim oranları'
            ],
            'yurt': [
                'öğrenci yurdu',
                'anlaşmalı yurtlar',
                'barınma imkanları',
                'konaklama seçenekleri',
                'yurt fiyatları',
                'yurt başvuru'
            ],
            'kayıt': [
                'öğrenci kayıt',
                'kayıt yenileme',
                'akademik kayıt',
                'ders kayıt',
                'kayıt işlemleri',
                'kayıt tarihleri'
            ],
            'sınav': [
                'final sınavı',
                'ara sınav',
                'sınav tarihleri',
                'sınav sistemi',
                'değerlendirme kriterleri',
                'not sistemi'
            ]
        }
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """Expand query with domain-specific terms"""
        
        query_lower = query.lower()
        expanded_terms = [query]  # Always include original
        
        # Check for expansion patterns
        for key_pattern, expansions in self.query_expansions.items():
            if key_pattern in query_lower:
                print(f"   🔍 Expanding '{key_pattern}' with {len(expansions)} domain terms")
                expanded_terms.extend(expansions)
        
        # Additional intelligent expansions based on query content
        if 'birinci sınıf' in query_lower and 'ders' in query_lower:
            expanded_terms.extend([
                'zorunlu dersler',
                'ortak zorunlu dersler',
                'müfredat dersleri',
                'temel eğitim dersleri'
            ])
            
        if 'çekilme' in query_lower or 'bırakma' in query_lower:
            expanded_terms.extend([
                'ders çekilme',
                'dersten çekilme',
                'kayıt iptal',
                'ders bırakma koşulları'
            ])
        
        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_expansions.append(term)
        
        if len(unique_expansions) > 1:
            print(f"   📝 Query expanded from 1 to {len(unique_expansions)} terms")
            
        return unique_expansions[:8]  # Limit to prevent overload
    
    async def execute_multi_search(self, queries: List[str], query_vectors: List[np.ndarray], 
                                 collection, use_reranking: bool = True) -> List[Dict]:
        """Execute multiple queries in parallel and combine results"""
        
        print(f"🔍 Executing multi-search for {len(queries)} queries")
        
        # Execute all searches in parallel
        search_tasks = []
        for i, query in enumerate(queries):
            query_vector = query_vectors[i] if i < len(query_vectors) else None
            task = self._perform_single_hybrid_search(query, query_vector, collection)
            search_tasks.append(task)
        
        # Wait for all searches to complete
        all_results = await asyncio.gather(*search_tasks)
        
        # Combine and deduplicate results
        combined_results = self._combine_multi_search_results(all_results)
        
        # Smart re-ranking: Only for complex queries or uncertain results
        if use_reranking and self.cross_encoder_model and combined_results:
            primary_query = queries[0]  # Use first query as primary
            
            # Skip re-ranking for simple, high-confidence results  
            if len(queries) <= 4 and len(combined_results) <= 8:
                top_scores = [r['score'] for r in combined_results[:3]]
                if len(top_scores) > 1:
                    score_variance = max(top_scores) - min(top_scores)
                    if score_variance > 0.15:  # High confidence, skip re-ranking
                        print("🚀 Skipping re-ranking (high confidence results)")
                        return combined_results[:15]
            
            reranked_results = await self._intelligent_rerank(primary_query, combined_results)
            return reranked_results
        
        return combined_results[:25]  # Return top 25 results for better coverage
    
    async def _perform_single_hybrid_search(self, question: str, query_vector: Optional[np.ndarray], 
                                          collection) -> List[Dict]:
        """Perform enhanced hybrid search with query expansion for better coverage"""
        
        # Create cache key
        vector_hash = hash(tuple(query_vector.tolist())) if query_vector is not None else "no_vector"
        cache_key = f"search_{hash(question)}_{vector_hash}"
        
        # Check cache
        if cache_key in search_cache:
            return search_cache[cache_key]
        
        print(f"🔍 Enhanced hybrid search for: '{question}'")
        
        # Step 1: Expand query terms for better coverage
        expanded_queries = self._expand_query_terms(question)
        
        db_type = self._detect_db_type(collection)
        
        # Step 2: Execute semantic and keyword search in parallel for all expanded terms
        search_tasks = []
        
        # Original semantic search with query vector
        if query_vector is not None:
            if db_type == 'weaviate':
                search_tasks.append(self._weaviate_semantic_search(query_vector, collection))
            else:
                search_tasks.append(self._chroma_semantic_search(query_vector, collection))
        
        # Keyword search for original query
        if db_type == 'weaviate':
            search_tasks.append(self._weaviate_keyword_search(question, collection))
        else:
            search_tasks.append(self._keyword_search(question, collection))
        
        # Additional keyword searches for expanded terms (if any)
        if len(expanded_queries) > 1:
            print(f"   🔎 Performing {len(expanded_queries)-1} additional expansion searches")
            for expanded_query in expanded_queries[1:]:  # Skip original query
                if db_type == 'weaviate':
                    search_tasks.append(self._weaviate_keyword_search(expanded_query, collection))
                else:
                    search_tasks.append(self._keyword_search(expanded_query, collection))
        
        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks)
        
        # Step 3: Combine results from all searches
        combined_semantic = search_results[0] if query_vector is not None else {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Combine all keyword results
        all_keyword_results = {'documents': [[]], 'metadatas': [[]]}
        
        start_idx = 1 if query_vector is not None else 0
        for keyword_result in search_results[start_idx:]:
            if keyword_result['documents'] and keyword_result['documents'][0]:
                all_keyword_results['documents'][0].extend(keyword_result['documents'][0])
                all_keyword_results['metadatas'][0].extend(keyword_result['metadatas'][0])
        
        # Step 4: Enhanced result combination with expansion boost
        combined_results = self._combine_search_results_with_expansion(
            combined_semantic, all_keyword_results, len(expanded_queries) - 1
        )
        
        # Cache results
        search_cache[cache_key] = combined_results
        
        print(f"   ✅ Enhanced search completed: {len(combined_results)} results")
        return combined_results
    
    def _detect_db_type(self, collection):
        """Detect database type (Weaviate vs ChromaDB)"""
        if hasattr(collection, 'query') and not callable(collection.query):
            return 'weaviate'
        else:
            return 'chromadb'
    
    async def _weaviate_semantic_search(self, query_vector: np.ndarray, collection):
        """Weaviate semantic search"""
        try:
            response = collection.query.near_vector(
                near_vector=query_vector.tolist(),
                limit=20,  # Increased for better coverage
                return_metadata=['distance']
            )
            
            documents, metadatas, distances = [], [], []
            for obj in response.objects:
                documents.append(obj.properties.get('content', ''))
                metadatas.append({
                    'source': obj.properties.get('source', ''), 
                    'article': obj.properties.get('article', '')
                })
                distances.append(obj.metadata.distance if obj.metadata.distance else 0.5)
            
            return {'documents': [documents], 'metadatas': [metadatas], 'distances': [distances]}
        except Exception as e:
            print(f"⚠️ Weaviate semantic search error: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    async def _chroma_semantic_search(self, query_vector: np.ndarray, collection):
        """ChromaDB semantic search"""
        try:
            return collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=20,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"⚠️ ChromaDB semantic search error: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    async def _weaviate_keyword_search(self, question: str, collection):
        """Weaviate keyword search"""
        try:
            response = collection.query.bm25(query=question, limit=8)
            documents, metadatas = [], []
            for obj in response.objects:
                documents.append(obj.properties.get('content', ''))
                metadatas.append({
                    'source': obj.properties.get('source', ''), 
                    'article': obj.properties.get('article', '')
                })
            return {'documents': [documents], 'metadatas': [metadatas]}
        except Exception as e:
            print(f"⚠️ Weaviate keyword search error: {e}")
            return {'documents': [[]], 'metadatas': [[]]}
    
    async def _keyword_search(self, question: str, collection):
        """ChromaDB keyword search implementation"""
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
            return {
                'documents': [[m['document'] for m in matches[:8]]], 
                'metadatas': [[m['metadata'] for m in matches[:8]]]
            }
        except Exception as e:
            print(f"⚠️ ChromaDB keyword search error: {e}")
            return {'documents': [[]], 'metadatas': [[]]}
    
    def _combine_search_results(self, semantic_results: Dict, keyword_results: Dict) -> List[Dict]:
        """Combine semantic and keyword search results"""
        combined, seen_docs = [], set()
        
        # Add semantic results first (higher priority)
        if semantic_results['documents'] and semantic_results['documents'][0]:
            for i, doc in enumerate(semantic_results['documents'][0]):
                doc_key = doc[:100]  # Use first 100 chars as key
                if doc_key not in seen_docs and doc.strip():
                    combined.append({
                        'document': doc,
                        'metadata': semantic_results['metadatas'][0][i],
                        'score': 1.0 - semantic_results['distances'][0][i],
                        'source': 'semantic'
                    })
                    seen_docs.add(doc_key)
        
        # Add keyword results
        if keyword_results['documents'] and keyword_results['documents'][0]:
            for i, doc in enumerate(keyword_results['documents'][0]):
                doc_key = doc[:100]
                if doc_key not in seen_docs and doc.strip():
                    combined.append({
                        'document': doc,
                        'metadata': keyword_results['metadatas'][0][i],
                        'score': 0.6,  # Fixed score for keyword matches
                        'source': 'keyword'
                    })
                    seen_docs.add(doc_key)
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined
    
    def _combine_multi_search_results(self, all_results: List[List[Dict]]) -> List[Dict]:
        """Combine results from multiple searches with intelligent deduplication"""
        
        combined = {}  # Use dict to handle scoring for duplicates
        
        for result_set in all_results:
            for result in result_set:
                doc_key = result['document'][:100]  # Use first 100 chars as key
                
                if doc_key in combined:
                    # Boost score for documents found in multiple searches
                    combined[doc_key]['score'] = max(combined[doc_key]['score'], result['score'])
                    combined[doc_key]['multi_match'] = True
                else:
                    result['multi_match'] = False
                    combined[doc_key] = result
        
        # Convert back to list and apply multi-match boost
        final_results = []
        for result in combined.values():
            if result['multi_match']:
                result['score'] *= 1.2  # 20% boost for multi-match
            final_results.append(result)
        
        # Sort by enhanced score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"🔗 Combined {len(final_results)} unique results from multi-search")
        return final_results
    
    async def _intelligent_rerank(self, question: str, results: List[Dict]) -> List[Dict]:
        """Intelligent re-ranking using cross-encoder with smart triggering"""
        
        if not self.cross_encoder_model or len(results) <= 3:
            return results[:25]
        
        # Check if re-ranking is needed (score variance analysis)
        top_scores = [r['score'] for r in results[:5]]
        if len(top_scores) > 1:
            score_variance = max(top_scores) - min(top_scores)
            if score_variance < 0.15:  # More aggressive re-ranking threshold
                print("🎯 Applying intelligent re-ranking (uncertainty detected)")
                
                try:
                    # Apply cross-encoder to top 20 results
                    rerank_candidates = results[:20]
                    pairs = [(question, result['document']) for result in rerank_candidates]
                    
                    # Run cross-encoder in thread to avoid blocking
                    ce_scores = await asyncio.to_thread(self.cross_encoder_model.predict, pairs)
                    
                    # Apply weighted scoring (70% cross-encoder, 30% original)
                    for i, result in enumerate(rerank_candidates):
                        original_score = result['score']
                        ce_score = float(ce_scores[i])
                        result['final_score'] = 0.7 * ce_score + 0.3 * original_score
                        result['reranked'] = True
                    
                    # Add remaining results without re-ranking
                    for result in results[20:]:
                        result['final_score'] = result['score']
                        result['reranked'] = False
                    
                    # Sort by final score
                    results.sort(key=lambda x: x['final_score'], reverse=True)
                    
                    print(f"✅ Re-ranking completed for {len(rerank_candidates)} candidates")
                    
                except Exception as e:
                    print(f"⚠️ Re-ranking failed: {e}")
            else:
                print("🚀 Skipping re-ranking (clear ranking detected)")
        
        return results[:15] 

    def _combine_search_results_with_expansion(self, semantic_results: Dict, keyword_results: Dict, 
                                             expansion_count: int) -> List[Dict]:
        """Enhanced result combination that boosts scores for expansion matches"""
        combined, seen_docs = [], set()
        
        # Add semantic results first (highest priority)
        if semantic_results['documents'] and semantic_results['documents'][0]:
            for i, doc in enumerate(semantic_results['documents'][0]):
                doc_key = doc[:100]  # Use first 100 chars as key
                if doc_key not in seen_docs and doc.strip():
                    score = 1.0 - semantic_results['distances'][0][i]
                    combined.append({
                        'document': doc,
                        'metadata': semantic_results['metadatas'][0][i],
                        'score': score,
                        'source': 'semantic'
                    })
                    seen_docs.add(doc_key)
        
        # Add keyword results with expansion awareness
        if keyword_results['documents'] and keyword_results['documents'][0]:
            for i, doc in enumerate(keyword_results['documents'][0]):
                doc_key = doc[:100]
                if doc_key not in seen_docs and doc.strip():
                    # Base score for keyword matches
                    base_score = 0.6
                    
                    # Boost score if we had query expansions (indicates specialized search)
                    if expansion_count > 0:
                        expansion_boost = min(0.2, expansion_count * 0.05)  # Up to 20% boost
                        final_score = base_score + expansion_boost
                    else:
                        final_score = base_score
                    
                    combined.append({
                        'document': doc,
                        'metadata': keyword_results['metadatas'][0][i],
                        'score': final_score,
                        'source': f'keyword{"_expanded" if expansion_count > 0 else ""}'
                    })
                    seen_docs.add(doc_key)
        
        # Sort by enhanced score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        if expansion_count > 0:
            print(f"   🎯 Applied expansion boost to {len([r for r in combined if 'expanded' in r['source']])} results")
        
        return combined 