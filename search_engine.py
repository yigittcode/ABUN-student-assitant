"""
🔍 Search Engine Module
Semantic, keyword search operations with advanced re-ranking
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from cachetools import TTLCache

# Search result cache
search_cache = TTLCache(maxsize=2000, ttl=300)  # 5 min cache for search results

class SearchEngine:
    """Advanced search engine with hybrid search capabilities"""
    
    def __init__(self, cross_encoder_model=None):
        self.cross_encoder_model = cross_encoder_model
        
        # Add search result cache instance
        self.search_result_cache = TTLCache(maxsize=1000, ttl=180)  # 3 min cache for complete search results
        
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

        }
        
        # Progressive search widening configuration
        self.search_levels = {
            'level_1_exact': {
                'description': 'Exact specific search',
                'broadening_factor': 0,
                'semantic_weight': 0.7,
                'keyword_weight': 0.3
            },
            'level_2_moderate': {
                'description': 'Moderate broadening',
                'broadening_factor': 1,
                'semantic_weight': 0.6,
                'keyword_weight': 0.4
            },
            'level_3_broad': {
                'description': 'Broad semantic search',
                'broadening_factor': 2,
                'semantic_weight': 0.8,
                'keyword_weight': 0.2
            },
            'level_4_maximum': {
                'description': 'Maximum generalization',
                'broadening_factor': 3,
                'semantic_weight': 0.9,
                'keyword_weight': 0.1
            }
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
        """Enhanced multi-search with progressive widening fallback"""
        
        print(f"\n" + "="*80)
        print(f"🚀 PROGRESSIVE SEARCH ENGINE ACTIVATED!")
        print(f"🔍 Starting for {len(queries)} queries")
        print(f"📋 Queries: {[q[:40]+'...' for q in queries]}")
        print(f"🎯 Search levels configured: {list(self.search_levels.keys())}")
        print(f"="*80)
        
        try:
            # Try progressive search levels until we get sufficient results
            for level_name, level_config in self.search_levels.items():
                print(f"🎯 LEVEL {level_name}: {level_config['description']}")
                
                try:
                    # Adjust queries based on broadening factor
                    adjusted_queries = self._broaden_queries(queries, level_config['broadening_factor'])
                    
                    # Execute search with level-specific weights
                    results = await self._execute_search_level(
                        adjusted_queries, query_vectors, collection, level_config, use_reranking
                    )
                    
                    # Evaluate result quality
                    quality_score = self._evaluate_search_quality(results, queries[0])
                    print(f"   📊 Quality score: {quality_score:.2f} (Results: {len(results)})")
                    
                    # If we have sufficient quality results, return them
                    if quality_score >= 0.3 or level_name == 'level_4_maximum':
                        print(f"✅ SUCCESS: Found sufficient results at {level_config['description']}")
                        return results
                    
                    print(f"⚠️ Quality insufficient, escalating to next level...")
                    
                except Exception as e:
                    print(f"❌ Level {level_name} failed: {e}")
                    continue
            
            # If we get here, all levels failed
            print(f"❌ PROGRESSIVE SEARCH FAILED: All levels exhausted")
            return results if 'results' in locals() else []
            
        except Exception as e:
            print(f"❌ PROGRESSIVE SEARCH EXCEPTION: {e}")
            print(f"🔄 Falling back to legacy multi-search...")
            
            # Emergency fallback to old system
            return await self._legacy_multi_search(queries, query_vectors, collection, use_reranking)

    async def _legacy_multi_search(self, queries: List[str], query_vectors: List[np.ndarray], 
                                 collection, use_reranking: bool) -> List[Dict]:
        """Legacy multi-search implementation as emergency fallback"""
        print(f"🔄 LEGACY FALLBACK: Executing basic multi-search for {len(queries)} queries")
        
        try:
            # Execute all searches in parallel (old method)
            search_tasks = []
            for i, query in enumerate(queries):
                query_vector = query_vectors[i] if i < len(query_vectors) else None
                task = self._perform_single_hybrid_search(query, query_vector, collection)
                search_tasks.append(task)
            
            # Wait for all searches to complete
            all_results = await asyncio.gather(*search_tasks)
            
            # Combine results using old method
            combined_results = self._combine_multi_search_results(all_results)
            
            # Apply basic re-ranking if needed
            if use_reranking and self.cross_encoder_model and len(combined_results) > 5:
                try:
                    combined_results = await self._intelligent_rerank(queries[0], combined_results)
                except Exception as e:
                    print(f"   ⚠️ Legacy re-ranking failed: {e}")
            
            print(f"✅ LEGACY FALLBACK: Returned {len(combined_results)} results")
            return combined_results[:15]
            
        except Exception as e:
            print(f"❌ LEGACY FALLBACK FAILED: {e}")
            return []  # Ultimate fallback - empty results

    def _broaden_queries(self, queries: List[str], broadening_factor: int) -> List[str]:
        """Intelligently broaden queries based on broadening factor"""
        if broadening_factor == 0:
            return queries  # No broadening - exact search
        
        broadened = []
        for query in queries:
            broadened.append(query)  # Always include original
            
            if broadening_factor >= 1:
                # Level 1: Remove question words and specificity
                simplified = self._simplify_query(query)
                if simplified != query:
                    broadened.append(simplified)
            
            if broadening_factor >= 2:
                # Level 2: Extract core concepts
                core_concepts = self._extract_core_concepts(query)
                broadened.extend(core_concepts)
            
            if broadening_factor >= 3:
                # Level 3: Maximum generalization - extract main topic
                main_topic = self._extract_main_topic(query)
                if main_topic:
                    broadened.append(main_topic)
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in broadened:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        print(f"   🔄 Broadened {len(queries)} queries to {len(unique_queries)} queries")
        return unique_queries

    def _simplify_query(self, query: str) -> str:
        """Remove question words and make query more searchable"""
        # Remove common question patterns
        simplified = query.lower()
        
        # Remove question words
        question_words = ['var mı', 'nedir', 'kimdir', 'nelerdir', 'hangi', 'nasıl', 'ne zaman', 'nerede']
        for qw in question_words:
            simplified = simplified.replace(qw, '').strip()
        
        # Remove question marks and clean up
        simplified = simplified.replace('?', '').strip()
        
        # If too short, return original
        if len(simplified.split()) < 2:
            return query
            
        return simplified

    def _extract_core_concepts(self, query: str) -> List[str]:
        """Extract core concepts from query for broader search"""
        query_lower = query.lower()
        concepts = []
        
        # Core concept mappings
        concept_mappings = {
            'sporcu': ['burs', 'sporcu', 'atletik'],
            'burs': ['burs türleri', 'finansal destek', 'öğrenci yardımı'],
            'sınav': ['değerlendirme', 'ölçme', 'test'],
            'kayıt': ['öğrenci işleri', 'akademik kayıt', 'kayıt işlemleri'],
            'yurt': ['barınma', 'konaklama', 'öğrenci yurdu'],
            'ders': ['akademik program', 'müfredat', 'eğitim']
        }
        
        for key, related_concepts in concept_mappings.items():
            if key in query_lower:
                concepts.extend(related_concepts)
        
        return concepts[:3]  # Limit to prevent over-broadening

    def _extract_main_topic(self, query: str) -> str:
        """Extract the main topic for maximum generalization"""
        query_lower = query.lower()
        
        # Main topic extraction
        if any(word in query_lower for word in ['sporcu', 'burs']):
            return 'burs'
        elif any(word in query_lower for word in ['sınav', 'değerlendirme']):
            return 'sınav'
        elif any(word in query_lower for word in ['ders', 'müfredat']):
            return 'ders'
        elif any(word in query_lower for word in ['kayıt', 'işlem']):
            return 'kayıt'
        elif any(word in query_lower for word in ['yurt', 'barınma']):
            return 'yurt'
        elif any(word in query_lower for word in ['rektör', 'yönetim']):
            return 'yönetim'
        
        return None

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
            combined_semantic, all_keyword_results, len(expanded_queries) - 1, question
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
    
    def _combine_search_results(self, semantic_results: Dict, keyword_results: Dict, query: str = "") -> List[Dict]:
        """Combine semantic and keyword search results with semantic boosting"""
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
                    
                    # Apply semantic boosting for critical keywords
                    if query:
                        score = self._apply_semantic_boosting(doc, query, score)
                    
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
                    
                    # Apply semantic boosting for critical keywords
                    if query:
                        score = self._apply_semantic_boosting(doc, query, score)
                    
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

    def _apply_semantic_boosting(self, doc: str, query: str, base_score: float) -> float:
        """Advanced semantic boosting for critical keyword matches"""
        doc_lower = doc.lower()
        query_lower = query.lower()
        boosted_score = base_score
        
        # Critical time/day keywords - MASSIVE boost for exact matches
        time_keywords = {
            'cumartesi': 2.5,      # Saturday - critical for exam scheduling
            'pazar': 2.5,          # Sunday 
            'hafta sonu': 2.0,     # Weekend
            'tatil günü': 2.0,     # Holiday
            'resmi tatil': 1.8,    # Official holiday
            'pazartesi': 1.3,      # Monday
            'salı': 1.3,           # Tuesday  
            'çarşamba': 1.3,       # Wednesday
            'perşembe': 1.3,       # Thursday
            'cuma': 1.3,           # Friday
        }
        
        # Critical procedure keywords - Strong boost
        procedure_keywords = {
            'zorunlu hallerde': 2.2,  # In mandatory cases
            'olağandışı durumlarda': 2.0,  # In extraordinary circumstances
            'rektörlük onayı': 1.8,   # Rectorship approval
            'yönetim kurulu': 1.6,    # Management board
            'senato kararı': 1.6,     # Senate decision
            'madde': 1.4,             # Article (legal reference)
        }
        
        # Subject-specific keywords - Moderate boost
        subject_keywords = {
            'sınav': 1.5,             # Exam
            'final': 1.4,             # Final exam
            'ara sınav': 1.4,         # Midterm
            'bütünleme': 1.4,         # Make-up exam
            'değerlendirme': 1.3,     # Assessment
            'ölçme': 1.3,             # Measurement
        }
        
        # Apply time keyword boosting
        for keyword, boost_factor in time_keywords.items():
            if keyword in query_lower and keyword in doc_lower:
                boosted_score *= boost_factor
                print(f"   🚀 CRITICAL TIME BOOST: '{keyword}' found - score boosted by {boost_factor}x")
        
        # Apply procedure keyword boosting  
        for keyword, boost_factor in procedure_keywords.items():
            if keyword in query_lower and keyword in doc_lower:
                boosted_score *= boost_factor
                print(f"   📋 PROCEDURE BOOST: '{keyword}' found - score boosted by {boost_factor}x")
        
        # Apply subject keyword boosting
        for keyword, boost_factor in subject_keywords.items():
            if keyword in query_lower and keyword in doc_lower:
                boosted_score *= boost_factor
                print(f"   📚 SUBJECT BOOST: '{keyword}' found - score boosted by {boost_factor}x")
        
        # Compound keyword boosting - Extra boost for multiple critical terms
        critical_matches = 0
        all_keywords = {**time_keywords, **procedure_keywords, **subject_keywords}
        
        for keyword in all_keywords.keys():
            if keyword in query_lower and keyword in doc_lower:
                critical_matches += 1
                
        if critical_matches >= 2:
            compound_boost = 1.0 + (critical_matches * 0.2)  # 20% per additional match
            boosted_score *= compound_boost
            print(f"   ⚡ COMPOUND BOOST: {critical_matches} matches - additional {compound_boost}x boost")
        
        # Exact phrase matching - Highest priority
        query_phrases = [phrase.strip() for phrase in query_lower.split(',') if len(phrase.strip()) > 3]
        for phrase in query_phrases:
            if phrase in doc_lower and len(phrase) > 5:
                boosted_score *= 1.8
                print(f"   🎯 EXACT PHRASE BOOST: '{phrase}' found - score boosted by 1.8x")
        
        return min(boosted_score, 3.0)  # Cap at 3.0 to prevent over-boosting
    
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
                                             expansion_count: int, query: str = "") -> List[Dict]:
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
                    
                    # Apply semantic boosting for critical keywords
                    if query:
                        score = self._apply_semantic_boosting(doc, query, score)
                    
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
                    
                    # Apply semantic boosting for critical keywords
                    if query:
                        final_score = self._apply_semantic_boosting(doc, query, final_score)
                    
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

    async def _execute_search_level(self, queries: List[str], query_vectors: List[np.ndarray], 
                                   collection, level_config: Dict, use_reranking: bool) -> List[Dict]:
        """Execute search at specific level with configured weights"""
        
        # Adjust number of queries based on level
        max_queries = min(len(queries), 6 - level_config['broadening_factor'])
        selected_queries = queries[:max_queries]
        
        print(f"   🔍 Level search with {len(selected_queries)} queries: {[q[:30]+'...' for q in selected_queries]}")
        
        # Execute searches for this level
        all_results = []
        for i, query in enumerate(selected_queries):
            query_vector = query_vectors[i] if i < len(query_vectors) else None
            
            # Use existing single search but with level-specific adjustments
            results = await self._perform_single_hybrid_search(query, query_vector, collection)
            
            # Apply level-specific scoring immediately
            for result in results:
                if result.get('source') == 'semantic':
                    result['score'] *= level_config['semantic_weight']
                else:
                    result['score'] *= level_config['keyword_weight']
            
            all_results.extend(results)
        
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_and_score(all_results, level_config)
        
        # Apply re-ranking if requested and beneficial
        if use_reranking and self.cross_encoder_model and len(unique_results) > 5:
            try:
                unique_results = await self._intelligent_rerank(selected_queries[0], unique_results)
            except Exception as e:
                print(f"   ⚠️ Re-ranking failed at level search: {e}")
        
        return unique_results[:15]

    def _evaluate_search_quality(self, results: List[Dict], original_query: str) -> float:
        """Evaluate the quality of search results"""
        if not results:
            return 0.0
        
        # Basic quality metrics
        total_score = 0.0
        
        # 1. Number of results (more is better, up to a point)
        count_score = min(len(results) / 10, 1.0) * 0.3
        
        # 2. Average relevance score
        if results:
            avg_score = sum(r.get('score', 0) for r in results) / len(results)
            relevance_score = avg_score * 0.5
        else:
            relevance_score = 0.0
        
        # 3. Keyword coverage in top results
        query_words = set(original_query.lower().split())
        top_results = results[:5]
        keyword_matches = 0
        
        for result in top_results:
            doc_words = set(result.get('document', '').lower().split())
            if query_words.intersection(doc_words):
                keyword_matches += 1
        
        coverage_score = (keyword_matches / len(top_results)) * 0.2 if top_results else 0.0
        
        total_score = count_score + relevance_score + coverage_score
        return min(total_score, 1.0)

    def _deduplicate_and_score(self, results: List[Dict], level_config: Dict) -> List[Dict]:
        """Remove duplicates and apply level-specific scoring adjustments"""
        seen_docs = set()
        unique_results = []
        
        for result in results:
            doc_key = result.get('document', '')[:100]
            if doc_key not in seen_docs and doc_key.strip():
                # Adjust score based on level configuration
                adjusted_score = result.get('score', 0)
                
                # Apply semantic/keyword weight adjustments
                if result.get('source') == 'semantic':
                    adjusted_score *= level_config['semantic_weight']
                else:
                    adjusted_score *= level_config['keyword_weight']
                
                result['score'] = adjusted_score
                unique_results.append(result)
                seen_docs.add(doc_key)
        
        # Sort by adjusted score
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_results 