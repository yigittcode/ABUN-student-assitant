"""
🎪 RAG Orchestrator Module
Main RAG coordination with multi-query strategy
"""

import asyncio
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from query_processor import QueryProcessor, QueryAnalysis, QueryVariants
from search_engine import SearchEngine
from context_assembler import ContextAssembler
from response_generator import ResponseGenerator
from embedding_engine import SemanticEmbeddingEngine
from hyde_generator import generate_multiple_hyde_variants

class RAGOrchestrator:
    """Main RAG orchestrator with advanced multi-query strategy"""
    
    def __init__(self, openai_client, embedding_model, cross_encoder_model=None):
        self.openai_client = openai_client
        
        # Initialize all engines
        self.query_processor = QueryProcessor(openai_client)
        self.search_engine = SearchEngine(cross_encoder_model)
        self.context_assembler = ContextAssembler()
        self.response_generator = ResponseGenerator(openai_client)
        self.embedding_engine = SemanticEmbeddingEngine(embedding_model)
    
    async def ask_question(self, question: str, collection, domain_context: str = "", 
                          user_context: Optional[Dict] = None) -> Tuple[str, List[Dict]]:
        """Main RAG function with multi-query strategy"""
        
        print(f"\n🎪 RAG Orchestrator: Processing question: {question}")
        if domain_context:
            print(f"📄 Domain context: {domain_context}")
        
        # Phase 1-2: Parallel Query Processing & HyDE Generation
        print("🔍 Phase 1-2: Parallel Multi-Query Analysis & HyDE Generation")
        analysis_task = self.query_processor.process_multi_query(question, user_context)
        hyde_task = generate_multiple_hyde_variants(question, self.openai_client, domain_context)
        
        (analysis, variants), hyde_variants = await asyncio.gather(analysis_task, hyde_task)
        
        # Phase 3: Multi-Vector Creation  
        print("🧮 Phase 3: Multi-Vector Creation")
        all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
        query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
        
        # Phase 4: Multi-Search Execution
        print("🔍 Phase 4: Multi-Search Execution")
        search_results = await self.search_engine.execute_multi_search(
            all_queries, query_vectors, collection, use_reranking=True
        )
        
        # Phase 5: Intelligent Context Assembly
        print("🔧 Phase 5: Intelligent Context Assembly")
        assembled_context = await self.context_assembler.assemble_intelligent_context_async(
            search_results, question, user_context, 
            sub_questions=analysis.sub_questions if analysis.is_complex else None
        )
        
        # Phase 6: Enhanced Response Generation
        print("✨ Phase 6: Enhanced Response Generation")
        response = await self.response_generator.generate_contextual_response(
            question, assembled_context, domain_context, user_context
        )
        
        # Extract source metadata for API response
        sources_metadata = self._extract_source_metadata(search_results)
        
        print(f"🎉 RAG processing complete!")
        return response, sources_metadata
    
    async def ask_question_voice(self, question: str, collection, domain_context: str = "",
                                user_context: Optional[Dict] = None, request=None) -> Tuple[str, List[Dict]]:
        """Voice-optimized RAG with disconnection handling"""
        
        print(f"\n🎤 Voice RAG: Processing question: {question}")
        
        # Helper function for disconnection check
        async def check_disconnection(step_name: str):
            if request and await request.is_disconnected():
                print(f"🚪 Client disconnected during {step_name}")
                raise Exception(f"Client disconnected during {step_name}")
        
        try:
            # Phase 1-2: Parallel Query Processing & HyDE Generation (with disconnection check)
            await check_disconnection("parallel analysis")
            analysis_task = self.query_processor.process_multi_query(question, user_context)
            hyde_task = generate_multiple_hyde_variants(question, self.openai_client, domain_context)
            
            (analysis, variants), hyde_variants = await asyncio.gather(analysis_task, hyde_task)
            
            # Phase 3: Multi-Vector Creation
            await check_disconnection("vector creation")
            all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
            query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
            
            # Phase 4: Multi-Search Execution
            await check_disconnection("multi-search")
            search_results = await self.search_engine.execute_multi_search(
                all_queries, query_vectors, collection, use_reranking=True
            )
            
            # Phase 5: Context Assembly
            await check_disconnection("context assembly")
            assembled_context = await self.context_assembler.assemble_intelligent_context_async(
                search_results, question, user_context,
                sub_questions=analysis.sub_questions if analysis.is_complex else None
            )
            
            # Phase 6: Voice Response Generation
            await check_disconnection("voice response generation")
            response = await self.response_generator.generate_voice_response(
                question, assembled_context, domain_context, user_context
            )
            
            sources_metadata = self._extract_source_metadata(search_results)
            
            print(f"🎤 Voice RAG processing complete!")
            return response, sources_metadata
            
        except Exception as e:
            if "Client disconnected" in str(e):
                raise e
            else:
                print(f"❌ Voice RAG error: {e}")
                raise e
    
    async def ask_question_stream(self, question: str, collection, domain_context: str = "",
                                user_context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """Streaming RAG with real-time response generation"""
        
        print(f"\n🌊 Streaming RAG: Processing question: {question}")
        
        try:
            # Execute RAG pipeline up to context assembly with parallel processing
            analysis_task = self.query_processor.process_multi_query(question, user_context)
            hyde_task = generate_multiple_hyde_variants(question, self.openai_client, domain_context)
            
            (analysis, variants), hyde_variants = await asyncio.gather(analysis_task, hyde_task)
            
            all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
            query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
            
            search_results = await self.search_engine.execute_multi_search(
                all_queries, query_vectors, collection, use_reranking=True
            )
            
            assembled_context = await self.context_assembler.assemble_intelligent_context_async(
                search_results, question, user_context,
                sub_questions=analysis.sub_questions if analysis.is_complex else None
            )
            
            sources_metadata = self._extract_source_metadata(search_results)
            
            # Stream response generation
            async for chunk in self.response_generator.generate_streaming_response(
                question, assembled_context, domain_context, user_context
            ):
                # Add sources to completion chunk
                if '"type": "complete"' in chunk:
                    import json
                    chunk_data = json.loads(chunk)
                    chunk_data["sources"] = sources_metadata
                    chunk = json.dumps(chunk_data)
                
                yield chunk
            
        except Exception as e:
            print(f"❌ Streaming RAG error: {e}")
            import json
            error_response = {
                "type": "error",
                "content": f"Streaming sırasında hata oluştu: {str(e)}",
                "done": True,
                "sources": []
            }
            yield json.dumps(error_response)
    
    def _prepare_all_queries(self, original_question: str, analysis: QueryAnalysis, 
                           variants: QueryVariants, hyde_variants: List[str]) -> List[str]:
        """Prepare comprehensive query list from all sources"""
        
        print(f"🔗 Preparing comprehensive query list...")
        
        all_queries = [original_question]
        print(f"   📋 Starting with original: '{original_question}'")
        
        # Add decomposed sub-questions if complex
        if analysis.is_complex:
            print(f"   🧩 Adding {len(analysis.sub_questions)} sub-questions:")
            for i, sub_q in enumerate(analysis.sub_questions):
                print(f"      {i+1}. '{sub_q}'")
            all_queries.extend(analysis.sub_questions)
        else:
            print(f"   ✅ Query not complex - no sub-questions to add")
        
        # Add query variants
        variant_list = variants.all_variants()
        print(f"   🔄 Adding {len(variant_list)} query variants:")
        for i, variant in enumerate(variant_list):
            print(f"      {i+1}. '{variant}'")
        all_queries.extend(variant_list)
        
        # Add HyDE variants
        print(f"   📝 Adding {len(hyde_variants)} HyDE variants:")
        for i, hyde in enumerate(hyde_variants):
            print(f"      {i+1}. '{hyde[:80]}{'...' if len(hyde) > 80 else ''}'")
        all_queries.extend(hyde_variants)
        
        print(f"   📊 Total queries before dedup: {len(all_queries)}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        duplicates_found = 0
        
        for query in all_queries:
            query_normalized = query.strip().lower()
            if query_normalized not in seen and len(query.strip()) > 5:
                seen.add(query_normalized)
                unique_queries.append(query)
            else:
                duplicates_found += 1
        
        print(f"   🧹 Removed {duplicates_found} duplicates")
        
        # Aggressive performance optimization: Limit to 6 queries max
        final_queries = unique_queries[:6]
        if len(unique_queries) > 6:
            print(f"   ⚡ Limited to {len(final_queries)} queries for performance")
        
        print(f"   ✅ Final query set ({len(final_queries)} queries):")
        for i, query in enumerate(final_queries):
            print(f"      {i+1}. '{query[:80]}{'...' if len(query) > 80 else ''}'")
        
        return final_queries
    
    def _extract_source_metadata(self, search_results: List[Dict]) -> List[Dict]:
        """Extract source metadata for API response"""
        
        sources_metadata = []
        seen_sources = set()
        
        for result in search_results[:10]:  # Top 10 sources
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            article = metadata.get('article', 'Unknown')
            
            source_key = f"{source}|{article}"
            if source_key not in seen_sources:
                sources_metadata.append({
                    'source': source,
                    'article': article
                })
                seen_sources.add(source_key)
        
        return sources_metadata
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        
        return {
            "embedding_cache": self.embedding_engine.cache_stats(),
            "search_cache_size": len(self.search_engine.search_cache) if hasattr(self.search_engine, 'search_cache') else 0,
            "response_cache_size": len(self.response_generator.response_cache) if hasattr(self.response_generator, 'response_cache') else 0,
            "query_analysis_cache": len(self.query_processor.query_analysis_cache) if hasattr(self.query_processor, 'query_analysis_cache') else 0,
            "query_variants_cache": len(self.query_processor.query_variants_cache) if hasattr(self.query_processor, 'query_variants_cache') else 0
        }

# Backward compatibility functions for existing API
async def ask_question(question, collection, openai_client, model, cross_encoder_model, domain_context=""):
    """Backward compatibility wrapper"""
    orchestrator = RAGOrchestrator(openai_client, model, cross_encoder_model)
    return await orchestrator.ask_question(question, collection, domain_context)

async def ask_question_voice(question, collection, openai_client, model, cross_encoder_model, domain_context="", request=None):
    """Backward compatibility wrapper for voice"""
    orchestrator = RAGOrchestrator(openai_client, model, cross_encoder_model)
    return await orchestrator.ask_question_voice(question, collection, domain_context, request=request)

async def ask_question_stream(question, collection, openai_client, model, cross_encoder_model, domain_context=""):
    """Backward compatibility wrapper for streaming"""
    orchestrator = RAGOrchestrator(openai_client, model, cross_encoder_model)
    async for chunk in orchestrator.ask_question_stream(question, collection, domain_context):
        yield chunk

# Keep the optimized embeddings function for document processing
async def create_optimized_embeddings(documents, model):
    """Backward compatibility for document processing"""
    embedding_engine = SemanticEmbeddingEngine(model) # Changed from EmbeddingEngine to SemanticEmbeddingEngine
    return await embedding_engine.create_optimized_embeddings(documents) 