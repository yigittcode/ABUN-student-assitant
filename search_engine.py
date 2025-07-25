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
            'rektör': [
                'rektör',
                'üniversite rektörü', 
                'rektörlük',
                'üniversite başkanı',
                'yönetim kurulu başkanı',
                'akademik yönetim',
                'prof dr',
                'dekan',
                'rektörüdür',
                'yavuz demir',
                'prof yavuz',
                'rektörü kimdir',
                'rektörü kim',
                'özgeçmiş',
                'görev yaptığı',
                'atanan prof',
                'görevde',
                'başkan'
            ],
            'yönetim': [
                'rektör',
                'rektörlük',
                'yönetim kurulu',
                'senato',
                'mütevelli heyet',
                'dekan',
                'başkan'
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
        """Perform enhanced hybrid search with parallel execution of all search types"""
        
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
        
        # Step 2: Execute ALL search operations in parallel
        search_tasks = []
        
        # Original semantic search with query vector
        if query_vector is not None:
            if db_type == 'weaviate':
                search_tasks.append(self._weaviate_semantic_search(query_vector, collection))
            else:
                search_tasks.append(self._chroma_semantic_search(query_vector, collection))
        
        # Parallel keyword searches for ALL expanded queries
        for expanded_query in expanded_queries:
            if db_type == 'weaviate':
                search_tasks.append(self._weaviate_keyword_search(expanded_query, collection))
            else:
                search_tasks.append(self._keyword_search(expanded_query, collection))
        
        # Execute ALL searches in parallel - maximum concurrency
        print(f"⚡ Executing {len(search_tasks)} search operations in parallel")
        all_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Filter out failed searches and combine results
        valid_results = []
        for i, result in enumerate(all_search_results):
            if isinstance(result, Exception):
                print(f"⚠️ Search task {i} failed: {result}")
            elif result:
                valid_results.extend(result)
        
        # Step 3: Intelligent deduplication and scoring
        combined_results = self._smart_deduplicate_and_score(valid_results, question)
        
        # Cache results
        search_cache[cache_key] = combined_results
        
        print(f"✅ Combined {len(valid_results)} raw results into {len(combined_results)} unique results")
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
                    score = 1.0 - semantic_results['distances'][0][i]
                    
                    # Apply biographical content boost
                    if self._contains_biographical_info(doc):
                        score *= 1.4  # 40% boost for biographical content
                        print(f"   📊 Applied biographical boost to semantic result (score: {score:.3f})")
                    
                    combined.append({
                        'document': doc,
                        'metadata': semantic_results['metadatas'][0][i],
                        'score': score,
                        'source': 'semantic'
                    })
                    seen_docs.add(doc_key)
        
        # Add keyword results
        if keyword_results['documents'] and keyword_results['documents'][0]:
            for i, doc in enumerate(keyword_results['documents'][0]):
                doc_key = doc[:100]
                if doc_key not in seen_docs and doc.strip():
                    score = 0.6  # Fixed score for keyword matches
                    
                    # Apply biographical content boost
                    if self._contains_biographical_info(doc):
                        score *= 1.4  # 40% boost for biographical content
                        print(f"   📊 Applied biographical boost to keyword result (score: {score:.3f})")
                    
                    combined.append({
                        'document': doc,
                        'metadata': keyword_results['metadatas'][0][i],
                        'score': score,
                        'source': 'keyword'
                    })
                    seen_docs.add(doc_key)
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined
    
    def _contains_biographical_info(self, doc: str) -> bool:
        """Check if document contains biographical information rather than procedural references"""
        doc_lower = doc.lower()
        
        # Biographical indicators
        biographical_indicators = [
            'rektörüdür',           # "is the rector"
            'özgeçmiş',             # "biography"
            'doğdu',                # "was born"
            'atanan prof',          # "appointed professor"
            'görev yaptığı',        # "served at"
            'prof dr',              # "Prof. Dr."
            'yavuz',                # rector's name
            'demir',                # rector's name
            'manchester',           # university where he studied
            'oxford',               # university where he worked
            'samsun',               # birthplace
            'kitap',                # "book" (publications)
            'makale',               # "article" (publications)
            'tamamladı',            # "completed"
            'yılında',              # "in year"
            'haziran',              # "June" (appointment month)
            'bilim üniversitesi rektörü olarak'  # "as rector of science university"
        ]
        
        # Procedural indicators (lower priority)
        procedural_indicators = [
            'yürütür',              # "executes"
            'hükümlerini',          # "provisions"
            'yönerge',              # "directive"
            'yönetmelik',           # "regulation"
            'madde',                # "article" (legal)
            'bu kanun',             # "this law"
            'usul',                 # "procedure"
            'esaslar',              # "principles"
        ]
        
        # Count biographical vs procedural indicators
        biographical_score = sum(1 for indicator in biographical_indicators if indicator in doc_lower)
        procedural_score = sum(1 for indicator in procedural_indicators if indicator in doc_lower)
        
        # Boost if biographical content is dominant
        return biographical_score > procedural_score or biographical_score >= 2
    
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
                    
                    # Apply biographical content boost
                    if self._contains_biographical_info(doc):
                        score *= 1.4  # 40% boost for biographical content
                        print(f"   📊 Applied biographical boost to semantic result (score: {score:.3f})")
                    
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
                    
                    # Apply biographical content boost
                    if self._contains_biographical_info(doc):
                        final_score *= 1.4  # 40% boost for biographical content
                        print(f"   📊 Applied biographical boost to keyword result (score: {final_score:.3f})")
                    
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

    def _smart_deduplicate_and_score(self, results: List[Dict], question: str) -> List[Dict]:
        """Smart deduplication with relevance-based scoring"""
        if not results:
            return []
        
        # Group by content hash to identify duplicates
        content_groups = {}
        for result in results:
            content = result.get('content', '').strip()
            if not content:
                continue
                
            # Create content hash for deduplication
            content_hash = hash(content[:500])  # Use first 500 chars for grouping
            
            if content_hash not in content_groups:
                content_groups[content_hash] = []
            content_groups[content_hash].append(result)
        
        # Select best result from each group and boost scores
        deduplicated = []
        for content_hash, group in content_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Multiple results with same content - pick best score and boost it
                best_result = max(group, key=lambda x: x.get('score', 0))
                
                # Boost score for multiple source confirmation
                boost_factor = min(1.2, 1.0 + (len(group) - 1) * 0.05)
                best_result['score'] = best_result.get('score', 0) * boost_factor
                best_result['source_count'] = len(group)  # Track multiple sources
                
                deduplicated.append(best_result)
        
        # Sort by score and return top results
        deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
        return deduplicated[:30]  # Increased result limit for better coverage 