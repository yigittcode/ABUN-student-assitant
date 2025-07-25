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
        """Main RAG function with isolated sub-question processing"""
        
        print(f"\n🎪 RAG Orchestrator: Processing question: {question}")
        if domain_context:
            print(f"📄 Domain context: {domain_context}")
        
        # Phase 1-2: Parallel Query Processing & HyDE Generation
        print("🔍 Phase 1-2: Parallel Multi-Query Analysis & HyDE Generation")
        analysis_task = self.query_processor.process_multi_query(question, user_context)
        hyde_task = generate_multiple_hyde_variants(question, self.openai_client, domain_context)
        
        (analysis, variants), hyde_variants = await asyncio.gather(analysis_task, hyde_task)
        
        # CRITICAL ARCHITECTURAL CHANGE: Isolated sub-question processing
        if analysis.is_complex and analysis.sub_questions:
            print(f"🎯 ISOLATED PROCESSING: {len(analysis.sub_questions)} sub-questions")
            
            # CRITICAL FIX: Process all sub-questions in PARALLEL using asyncio.gather
            print(f"🚀 PARALLEL PROCESSING: Starting {len(analysis.sub_questions)} sub-questions simultaneously")
            
            # Create parallel tasks for all sub-questions
            sub_question_tasks = []
            for i, sub_question in enumerate(analysis.sub_questions):
                print(f"   📝 Preparing task {i+1}: '{sub_question}'")
                task = self._process_individual_question(
                    sub_question, collection, domain_context, user_context, 
                    hyde_variants, f"Sub-Q{i+1}"
                )
                sub_question_tasks.append(task)
            
            # Execute all sub-questions in PARALLEL
            print(f"⚡ EXECUTING {len(sub_question_tasks)} sub-questions in parallel...")
            parallel_results = await asyncio.gather(*sub_question_tasks)
            
            # Process results
            sub_contexts = []
            all_sources = []
            
            for i, (sub_context, sub_sources) in enumerate(parallel_results):
                sub_question = analysis.sub_questions[i]
                print(f"   ✅ Sub-Q{i+1} completed: {len(sub_context)} chars, {len(sub_sources)} sources")
                
                sub_contexts.append({
                    "question": sub_question,
                    "context": sub_context,
                    "topic_id": f"topic_{i}"
                })
                all_sources.extend(sub_sources)
            
            print(f"🎯 PARALLEL EXECUTION COMPLETE: All {len(analysis.sub_questions)} sub-questions processed simultaneously!")
            
            # Phase 5: Intelligent Multi-Context Assembly
            print(f"\n🔧 Phase 5: Multi-Context Assembly from {len(sub_contexts)} isolated contexts")
            assembled_context = await self._assemble_multi_contexts(
                sub_contexts, question, user_context
            )
            
        else:
            # Simple question - use existing pipeline
            print("🎯 SIMPLE PROCESSING: Single pipeline")
            
            # Phase 3: Multi-Vector Creation  
            print("🧮 Phase 3: Multi-Vector Creation")
            all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
            query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
            
            # Phase 4: Multi-Search Execution
            print("🔍 Phase 4: Multi-Search Execution")
            search_results = await self.search_engine.execute_multi_search(
                all_queries, query_vectors, collection, use_reranking=True
            )
            
            # Phase 5: Simple Context Assembly
            print("🔧 Phase 5: Simple Context Assembly")
            assembled_context = await self.context_assembler.assemble_intelligent_context_async(
                search_results, question, user_context, sub_questions=None
            )
            
            all_sources = self._extract_source_metadata(search_results)
        
        # Phase 6: Enhanced Response Generation
        print("✨ Phase 6: Enhanced Response Generation")
        response = await self.response_generator.generate_contextual_response(
            question, assembled_context, domain_context, user_context
        )
        
        print(f"🎉 RAG processing complete!")
        return response, all_sources
    
    async def _process_individual_question(self, question: str, collection, domain_context: str,
                                         user_context: Optional[Dict], hyde_variants: List[str],
                                         prefix: str = "") -> Tuple[str, List[Dict]]:
        """Process a single question with complete isolation - full pipeline"""
        
        print(f"  🧠 {prefix}: Starting isolated pipeline for '{question}'")
        
        # Step 1: Generate variants for this specific question
        _, individual_variants = await self.query_processor.process_multi_query(question, user_context)
        
        # Step 2: Prepare queries for this question only
        individual_queries = [question]
        individual_queries.extend(individual_variants.all_variants()[:3])  # Limit variants for performance
        individual_queries.extend(hyde_variants[:1])  # Add one HyDE variant
        
        # Remove duplicates
        seen = set()
        unique_queries = []
        for query in individual_queries:
            query_norm = query.strip().lower()
            if query_norm not in seen and len(query.strip()) > 3:
                seen.add(query_norm)
                unique_queries.append(query)
        
        print(f"  📋 {prefix}: Using {len(unique_queries)} queries: {[q[:40]+'...' for q in unique_queries]}")
        
        # Step 3: Create embeddings for this question's queries
        query_vectors = await self.embedding_engine.create_query_embeddings(unique_queries)
        
        # Step 4: Execute isolated search
        search_results = await self.search_engine.execute_multi_search(
            unique_queries, query_vectors, collection, use_reranking=True
        )
        
        # Step 5: Assemble context for this specific question
        context = await self.context_assembler.assemble_intelligent_context_async(
            search_results, question, user_context, sub_questions=None  # No sub-questions for individual processing
        )
        
        # Step 6: Extract sources
        sources = self._extract_source_metadata(search_results)
        
        print(f"  ✅ {prefix}: Generated {len(context)} chars context with {len(sources)} sources")
        
        return context, sources
    
    async def _assemble_multi_contexts(self, sub_contexts: List[Dict], original_question: str,
                                     user_context: Optional[Dict] = None) -> str:
        """Intelligently assemble multiple isolated contexts into a cohesive response context"""
        
        print(f"  🔗 Assembling {len(sub_contexts)} isolated contexts")
        
        # Simple but effective assembly strategy
        assembled_parts = []
        
        for i, ctx_data in enumerate(sub_contexts):
            question = ctx_data["question"]
            context = ctx_data["context"]
            topic_id = ctx_data["topic_id"]
            
            if context and len(context.strip()) > 50:  # Only include substantial contexts
                # Add topic header for organization
                topic_header = f"\n[KONU {i+1}: {question.upper()}]\n"
                assembled_parts.append(topic_header + context)
                print(f"    ✅ Topic {i+1}: {len(context)} chars - '{question[:40]}...'")
            else:
                print(f"    ⚠️ Topic {i+1}: Insufficient context - '{question[:40]}...'")
        
        if not assembled_parts:
            print(f"    ❌ No substantial contexts found, using fallback")
            return "Bu konularda verilen dokümanlarda yeterli bilgi bulunamadı."
        
        # Combine all contexts with clear separation
        final_context = "\n\n".join(assembled_parts)
        
        print(f"  🎯 Final assembled context: {len(final_context)} chars from {len(assembled_parts)} topics")
        
        return final_context
    
    async def ask_question_voice(self, question: str, collection, domain_context: str = "",
                                user_context: Optional[Dict] = None, request=None) -> Tuple[str, List[Dict]]:
        """Voice-optimized RAG with isolated sub-question processing and disconnection handling"""
        
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
            
            await check_disconnection("query preparation")
            
            # ISOLATED PROCESSING: Same architecture as ask_question
            if analysis.is_complex and analysis.sub_questions:
                print(f"🎯 VOICE ISOLATED PROCESSING: {len(analysis.sub_questions)} sub-questions")
                
                # CRITICAL FIX: Process all sub-questions in PARALLEL
                print(f"🚀 VOICE PARALLEL PROCESSING: Starting {len(analysis.sub_questions)} sub-questions simultaneously")
                
                # Create parallel tasks for all sub-questions
                sub_question_tasks = []
                for i, sub_question in enumerate(analysis.sub_questions):
                    await check_disconnection(f"preparing sub-question {i+1}")
                    print(f"   📝 Voice preparing task {i+1}: '{sub_question}'")
                    task = self._process_individual_question(
                        sub_question, collection, domain_context, user_context, 
                        hyde_variants, f"Voice-Sub-Q{i+1}"
                    )
                    sub_question_tasks.append(task)
                
                await check_disconnection("parallel execution")
                
                # Execute all sub-questions in PARALLEL
                print(f"⚡ VOICE EXECUTING {len(sub_question_tasks)} sub-questions in parallel...")
                parallel_results = await asyncio.gather(*sub_question_tasks)
                
                # Process results
                sub_contexts = []
                all_sources = []
                
                for i, (sub_context, sub_sources) in enumerate(parallel_results):
                    sub_question = analysis.sub_questions[i]
                    print(f"   ✅ Voice Sub-Q{i+1} completed: {len(sub_context)} chars, {len(sub_sources)} sources")
                    
                    sub_contexts.append({
                        "question": sub_question,
                        "context": sub_context,
                        "topic_id": f"topic_{i}"
                    })
                    all_sources.extend(sub_sources)
                
                print(f"🎯 VOICE PARALLEL EXECUTION COMPLETE: All {len(analysis.sub_questions)} sub-questions processed simultaneously!")
                
                await check_disconnection("multi-context assembly")
                
                # Phase 5: Intelligent Multi-Context Assembly
                print(f"\n🔧 Voice Phase 5: Multi-Context Assembly from {len(sub_contexts)} isolated contexts")
                assembled_context = await self._assemble_multi_contexts(
                    sub_contexts, question, user_context
                )
                
            else:
                # Simple question processing
                print("🎯 VOICE SIMPLE PROCESSING: Single pipeline")
                
                await check_disconnection("vector creation")
                all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
                query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
                
                await check_disconnection("multi-search")
                search_results = await self.search_engine.execute_multi_search(
                    all_queries, query_vectors, collection, use_reranking=True
                )
                
                await check_disconnection("context assembly")
                assembled_context = await self.context_assembler.assemble_intelligent_context_async(
                    search_results, question, user_context, sub_questions=None
                )
                
                all_sources = self._extract_source_metadata(search_results)
            
            # Phase 6: Voice Response Generation
            await check_disconnection("voice response generation")
            response = await self.response_generator.generate_voice_response(
                question, assembled_context, domain_context, user_context
            )
            
            print(f"🎤 Voice RAG processing complete!")
            return response, all_sources
            
        except Exception as e:
            if "Client disconnected" in str(e):
                raise e
            else:
                print(f"❌ Voice RAG error: {e}")
                raise e
    
    async def ask_question_stream(self, question: str, collection, domain_context: str = "",
                                user_context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """Streaming RAG with isolated sub-question processing for real-time response generation"""
        
        print(f"\n🌊 Streaming RAG: Processing question: {question}")
        
        try:
            # Execute RAG pipeline up to context assembly with isolated processing
            analysis_task = self.query_processor.process_multi_query(question, user_context)
            hyde_task = generate_multiple_hyde_variants(question, self.openai_client, domain_context)
            
            (analysis, variants), hyde_variants = await asyncio.gather(analysis_task, hyde_task)
            
            # ISOLATED PROCESSING: Same architecture as other methods
            if analysis.is_complex and analysis.sub_questions:
                print(f"🎯 STREAM ISOLATED PROCESSING: {len(analysis.sub_questions)} sub-questions")
                
                # CRITICAL FIX: Process all sub-questions in PARALLEL
                print(f"🚀 STREAM PARALLEL PROCESSING: Starting {len(analysis.sub_questions)} sub-questions simultaneously")
                
                # Create parallel tasks for all sub-questions
                sub_question_tasks = []
                for i, sub_question in enumerate(analysis.sub_questions):
                    print(f"   📝 Stream preparing task {i+1}: '{sub_question}'")
                    task = self._process_individual_question(
                        sub_question, collection, domain_context, user_context, 
                        hyde_variants, f"Stream-Sub-Q{i+1}"
                    )
                    sub_question_tasks.append(task)
                
                # Execute all sub-questions in PARALLEL
                print(f"⚡ STREAM EXECUTING {len(sub_question_tasks)} sub-questions in parallel...")
                parallel_results = await asyncio.gather(*sub_question_tasks)
                
                # Process results
                sub_contexts = []
                
                for i, (sub_context, sub_sources) in enumerate(parallel_results):
                    sub_question = analysis.sub_questions[i]
                    print(f"   ✅ Stream Sub-Q{i+1} completed: {len(sub_context)} chars, {len(sub_sources)} sources")
                    
                    sub_contexts.append({
                        "question": sub_question,
                        "context": sub_context,
                        "topic_id": f"topic_{i}"
                    })
                
                print(f"🎯 STREAM PARALLEL EXECUTION COMPLETE: All {len(analysis.sub_questions)} sub-questions processed simultaneously!")
                
                # Phase 5: Intelligent Multi-Context Assembly
                print(f"\n🔧 Stream Phase 5: Multi-Context Assembly from {len(sub_contexts)} isolated contexts")
                assembled_context = await self._assemble_multi_contexts(
                    sub_contexts, question, user_context
                )
                
            else:
                # Simple question processing
                print("🎯 STREAM SIMPLE PROCESSING: Single pipeline")
                
                all_queries = self._prepare_all_queries(question, analysis, variants, hyde_variants)
                query_vectors = await self.embedding_engine.create_query_embeddings(all_queries)
                
                search_results = await self.search_engine.execute_multi_search(
                    all_queries, query_vectors, collection, use_reranking=True
                )
                
                assembled_context = await self.context_assembler.assemble_intelligent_context_async(
                    search_results, question, user_context, sub_questions=None
                )
            
            # Generate streaming response
            print("✨ Streaming Phase 6: Real-time Response Generation")
            async for chunk in self.response_generator.generate_streaming_response(
                question, assembled_context, domain_context, user_context
            ):
                yield chunk
                
        except Exception as e:
            print(f"❌ Streaming RAG error: {e}")
            yield f"Error: {str(e)}"
    
    def _prepare_all_queries(self, original_question: str, analysis: QueryAnalysis, 
                           variants: QueryVariants, hyde_variants: List[str]) -> List[str]:
        """Prepare comprehensive query list for SIMPLE queries only (no sub-questions)"""
        
        print(f"🔗 Preparing simple query list...")
        
        all_queries = [original_question]
        print(f"   📋 Starting with original: '{original_question}'")
        
        # NOTE: Sub-questions are NOT added here anymore - they're handled in isolated processing
        print(f"   ✅ Simple query processing - no sub-questions to add")
        
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
        
        if duplicates_found > 0:
            print(f"   🧹 Removed {duplicates_found} duplicates")
        
        # Limit queries for performance in simple mode
        if len(unique_queries) > 8:
            unique_queries = unique_queries[:8]
            print(f"   ⚡ Limited to {len(unique_queries)} queries for performance")
        
        print(f"   ✅ Final simple query set ({len(unique_queries)} queries):")
        for i, query in enumerate(unique_queries):
            print(f"      {i+1}. '{query}'")
        
        return unique_queries
    
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