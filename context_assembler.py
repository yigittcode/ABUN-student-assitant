"""
🔧 Context Assembler Module
Intelligent context assembly with multi-search fusion and balanced multi-topic handling
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from config import MAX_CONTEXT_TOKENS

# Context management constants
CHARS_PER_TOKEN = 3.5  # Conservative estimate for Turkish text
MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN * 0.9)  # 90% safety margin

@dataclass
class ContextSegment:
    """Individual context segment with metadata"""
    content: str
    source: str
    article: str
    relevance_score: float
    search_origin: str  # semantic, keyword, etc.
    topic_relevance: Dict[str, float] = None  # NEW: Topic-specific relevance scores
    
    def __post_init__(self):
        if self.topic_relevance is None:
            self.topic_relevance = {}

class ContextAssembler:
    """Intelligent context assembly with multi-search fusion"""
    
    def __init__(self):
        # Domain-specific weight adjustments
        self.domain_weights = {
            'madde': 1.3,  # Legal articles are important
            'burs': 1.2,   # Scholarship info is critical
            'ders': 1.2,   # Course info is important
            'sınav': 1.1,  # Exam info is relevant
            'kayıt': 1.1   # Registration info is relevant
        }
    
    def assemble_intelligent_context(self, search_results: List[Dict], original_query: str, 
                                   user_context: Optional[Dict] = None, 
                                   sub_questions: Optional[List[str]] = None) -> str:
        """Assemble context from multi-search results with intelligent fusion and balanced multi-topic handling"""
        
        if not search_results:
            print("⚠️ No search results available for context assembly")
            return "İlgili bilgi bulunamadı."
        
        print(f"🔧 Assembling context from {len(search_results)} search results")
        if sub_questions:
            print(f"📝 Multi-topic mode: {len(sub_questions)} topics detected")
        
        # Step 1: Convert to context segments
        context_segments = self._create_context_segments(search_results)
        
        # Step 2: Apply relevance scoring (with topic-specific scoring if complex)
        if sub_questions and len(sub_questions) > 1:
            scored_segments = self._score_multi_topic_relevance(context_segments, original_query, sub_questions)
        else:
            scored_segments = self._score_relevance(context_segments, original_query)
        
        # Step 3: Detect and handle contradictions
        resolved_segments = self._resolve_contradictions(scored_segments, original_query)
        
        # Step 4: Apply user context if available
        if user_context:
            personalized_segments = self._apply_user_context(resolved_segments, user_context, original_query)
        else:
            personalized_segments = resolved_segments
        
        # Step 5: Assemble final context with balanced multi-topic handling
        if sub_questions and len(sub_questions) > 1:
            final_context = self._assemble_balanced_multi_topic_context(
                personalized_segments, original_query, sub_questions
            )
        else:
            final_context = self._assemble_final_context(personalized_segments, original_query)
        
        print(f"📊 Context assembly complete: {len(final_context)} chars, segments used for balanced representation")
        
        return final_context
    
    def _create_context_segments(self, search_results: List[Dict]) -> List[ContextSegment]:
        """Convert search results to context segments"""
        
        segments = []
        for result in search_results:
            document = result.get('document', '')
            metadata = result.get('metadata', {})
            
            if not document or len(document.strip()) < 10:
                continue
                
            segment = ContextSegment(
                content=document,
                source=metadata.get('source', 'Unknown'),
                article=metadata.get('article', 'Unknown'),
                relevance_score=result.get('score', 0.0),
                search_origin=result.get('source', 'unknown')
            )
            segments.append(segment)
        
        return segments
    
    def _score_relevance(self, segments: List[ContextSegment], query: str) -> List[ContextSegment]:
        """Apply advanced relevance scoring"""
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for segment in segments:
            content_lower = segment.content.lower()
            
            # Base relevance score
            base_score = segment.relevance_score
            
            # Domain-specific boost
            domain_boost = 1.0
            for domain, weight in self.domain_weights.items():
                if domain in content_lower:
                    domain_boost = max(domain_boost, weight)
            
            # Query term coverage boost
            content_terms = set(content_lower.split())
            term_overlap = len(query_terms.intersection(content_terms))
            term_coverage = term_overlap / len(query_terms) if query_terms else 0
            coverage_boost = 1.0 + (term_coverage * 0.3)  # Up to 30% boost
            
            # Source reliability boost
            source_boost = 1.0
            if segment.search_origin == 'semantic':
                source_boost = 1.1  # Semantic matches are more reliable
            elif segment.search_origin == 'keyword':
                source_boost = 1.0  # Standard boost for keyword
            
            # Calculate final relevance score
            segment.relevance_score = base_score * domain_boost * coverage_boost * source_boost
        
        # Sort by relevance score
        segments.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return segments
    
    def _score_multi_topic_relevance(self, segments: List[ContextSegment], 
                                   original_query: str, sub_questions: List[str]) -> List[ContextSegment]:
        """Score segments for multiple topics to ensure balanced representation"""
        
        print(f"🎯 Multi-topic relevance scoring for {len(sub_questions)} topics")
        
        # First, score each segment for each topic
        for segment in segments:
            segment.topic_relevance = {}
            
            for i, sub_question in enumerate(sub_questions):
                topic_score = self._calculate_topic_relevance(segment.content, sub_question)
                segment.topic_relevance[f"topic_{i}"] = topic_score
                
                # Log high-relevance segments for debugging
                if topic_score > 0.7:
                    print(f"   🎯 High relevance for topic {i}: {topic_score:.2f} - {segment.content[:80]}...")
        
        # Apply traditional relevance scoring as well
        scored_segments = self._score_relevance(segments, original_query)
        
        return scored_segments
    
    def _calculate_topic_relevance(self, content: str, topic_query: str) -> float:
        """Enhanced topic relevance calculation with semantic matching"""
        
        content_lower = content.lower()
        topic_lower = topic_query.lower()
        
        # Enhanced keyword sets for better matching
        topic_terms = []
        semantic_groups = []
        
        if 'yurt' in topic_lower:
            # Primary terms
            topic_terms.extend(['yurt', 'anlaşmalı', 'konaklama', 'barınma'])
            # Semantic expansion for dormitory content
            semantic_groups.extend([
                ['öğrenci', 'yurdu', 'yurtları'],  # "öğrenci yurdu"
                ['kız', 'erkek'],  # gender-specific dorms
                ['duru', 'mayıs', 'eyüboğlu', 'eylül', 'ardıçlar', 'güneş', 'çalapkulu'],  # dorm names
                ['tandoğan', 'bahçelievler', 'balgat'],  # locations
                ['üniversitemiz', 'üniversitesi'],  # university references
                ['mevcut', 'bulunmaktadır', 'sunmaktadır']  # availability verbs
            ])
            
        if 'burs' in topic_lower:
            topic_terms.extend(['burs', 'ösym', 'indirim', 'ücretsiz', 'scholarship'])
            semantic_groups.extend([
                ['yerleştirmeye', 'dayalı', 'yerleşen'],
                ['muafiyet', 'karşılıksız', 'başarı'],
                ['%100', '%75', '%50', '%25', 'yüzde'],  # percentage terms
                ['yarıyıl', 'süre', 'devam']
            ])
            
        if 'ders' in topic_lower:
            topic_terms.extend(['ders', 'kurs', 'eğitim', 'müfredat'])
            semantic_groups.extend([
                ['çekilme', 'bırakma', 'vazgeçme'],
                ['danışman', 'önerisi', 'izin']
            ])
        
        # Add query-specific terms
        topic_words = set(re.findall(r'\b\w+\b', topic_lower))
        topic_terms.extend([word for word in topic_words if len(word) > 2])
        
        # Calculate base relevance score
        relevance_score = 0.0
        
        # Primary term matching (higher weight)
        for term in topic_terms:
            if term in content_lower:
                if term in ['yurt', 'burs', 'ösym']:
                    relevance_score += 0.4  # Increased weight for key terms
                elif term in ['anlaşmalı', 'indirim', 'öğrenci']:
                    relevance_score += 0.3  # Medium importance terms
                else:
                    relevance_score += 0.2  # Regular terms
        
        # Semantic group matching (for context understanding)
        for group in semantic_groups:
            group_matches = sum(1 for term in group if term in content_lower)
            if group_matches > 0:
                # Bonus for semantic context (scaled by matches in group)
                semantic_bonus = min(0.3, group_matches * 0.1)
                relevance_score += semantic_bonus
        
        # Content length normalization (longer content gets slight boost if relevant)
        if relevance_score > 0.3 and len(content) > 200:
            relevance_score += 0.1
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _resolve_contradictions(self, segments: List[ContextSegment], query: str) -> List[ContextSegment]:
        """Detect and resolve contradictions in context"""
        
        # Look for common contradiction patterns
        contradiction_patterns = [
            (r'alamaz', r'alabilir'),  # can't get vs can get
            (r'yasak', r'izin'),       # forbidden vs allowed
            (r'%100.*burs', r'başarı.*burs'),  # full scholarship vs achievement scholarship
            (r'tek.*ders.*çekil', r'ders.*çekil')  # single course withdrawal contradictions
        ]
        
        resolved_segments = []
        
        for segment in segments:
            content_lower = segment.content.lower()
            is_contradictory = False
            
            # Check for contradictions with higher-scored segments
            for resolved_segment in resolved_segments:
                if resolved_segment.relevance_score > segment.relevance_score:
                    resolved_content_lower = resolved_segment.content.lower()
                    
                    # Check contradiction patterns
                    for pattern1, pattern2 in contradiction_patterns:
                        if (re.search(pattern1, content_lower) and re.search(pattern2, resolved_content_lower)) or \
                           (re.search(pattern2, content_lower) and re.search(pattern1, resolved_content_lower)):
                            print(f"🔍 Contradiction detected, favoring higher-scored segment")
                            is_contradictory = True
                            break
                    
                    if is_contradictory:
                        break
            
            if not is_contradictory:
                resolved_segments.append(segment)
        
        return resolved_segments
    
    def _apply_user_context(self, segments: List[ContextSegment], user_context: Dict, 
                          query: str) -> List[ContextSegment]:
        """Apply user-specific context to prioritize relevant information"""
        
        # Extract user attributes
        student_type = user_context.get('student_type', '')
        program = user_context.get('program', '')
        semester = user_context.get('semester', '')
        
        personalized_segments = []
        
        for segment in segments:
            content_lower = segment.content.lower()
            personalization_boost = 1.0
            
            # Boost based on student type
            if student_type.lower() in content_lower:
                personalization_boost *= 1.3
            
            # Boost based on program
            if program.lower() in content_lower:
                personalization_boost *= 1.2
            
            # Apply user-specific rules
            if 'tam burslu' in student_type.lower() and 'başarı bursu' in query.lower():
                # For full scholarship students asking about achievement scholarship
                if 'tam burslu' in content_lower and ('alamaz' in content_lower or 'yasakl' in content_lower):
                    personalization_boost *= 1.5  # Boost exclusion rules
            
            if 'tek ders' in query.lower() or 'sadece.*ders' in query.lower():
                # For single course scenarios
                if 'tek ders' in content_lower or 'minimum' in content_lower:
                    personalization_boost *= 1.4
            
            # Apply boost
            segment.relevance_score *= personalization_boost
            personalized_segments.append(segment)
        
        # Re-sort after personalization
        personalized_segments.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return personalized_segments
    
    def _assemble_final_context(self, segments: List[ContextSegment], query: str) -> str:
        """Assemble final context with smart token management for simple queries"""
        
        print(f"📝 Smart context assembly for simple query")
        
        available_tokens = MAX_CONTEXT_TOKENS - 200  # Reserve for formatting
        context_parts = []
        used_tokens = 0
        added_segments = 0
        
        # Smart token allocation strategy for simple queries
        for segment in segments:
            segment_text = self._format_segment(segment)
            segment_tokens = self._estimate_tokens_fast(segment_text)
            
            # Check if we can add this segment
            if used_tokens + segment_tokens <= available_tokens:
                context_parts.append(segment_text)
                used_tokens += segment_tokens
                added_segments += 1
            else:
                # Try smart truncation for remaining high-quality segments
                if added_segments < 2:  # Ensure at least 2 segments for simple queries
                    remaining_tokens = available_tokens - used_tokens
                    if remaining_tokens > 300:  # Minimum viable truncation
                        truncated = self._smart_truncate_segment(segment, remaining_tokens)
                        if truncated:
                            context_parts.append(truncated)
                            added_segments += 1
                break
        
        final_context = "".join(context_parts)
        
        # Enhanced fallback for simple queries
        if not final_context.strip() or len(final_context.strip()) < 50:
            print("🚨 Triggering enhanced emergency fallback for simple query")
            return self._emergency_simple_query_context(segments, query)
        
        print(f"   ✅ Simple query context: {added_segments} segments, {used_tokens} tokens")
        return final_context
    
    def _emergency_simple_query_context(self, segments: List[ContextSegment], query: str) -> str:
        """Emergency context assembly for simple queries when normal assembly fails"""
        
        print(f"🚑 Emergency context recovery for simple query: {query}")
        
        # Extract query keywords for targeted search
        query_keywords = self._extract_topic_keywords(query)
        print(f"   🔍 Query keywords: {query_keywords}")
        
        # Find segments that match query keywords
        relevant_segments = []
        for segment in segments:
            content_lower = segment.content.lower()
            keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
            if keyword_matches > 0:
                relevant_segments.append((segment, keyword_matches))
        
        if not relevant_segments:
            print("   ⚠️ No keyword matches found, using top segments")
            relevant_segments = [(seg, 1) for seg in segments[:3]]
        
        # Sort by keyword matches
        relevant_segments.sort(key=lambda x: x[1], reverse=True)
        
        # Build emergency context
        emergency_parts = []
        emergency_tokens = 0
        max_emergency_tokens = 4000  # Conservative limit for emergency
        
        for segment, matches in relevant_segments:
            if emergency_tokens > max_emergency_tokens:
                break
                
            # Truncate aggressively for emergency context
            emergency_content = self._smart_truncate_segment(segment, 800)  # 800 tokens max per segment
            if emergency_content:
                emergency_parts.append(emergency_content)
                emergency_tokens += self._estimate_tokens_fast(emergency_content)
                
            if len(emergency_parts) >= 5:  # Limit emergency segments
                break
        
        final_emergency_context = "".join(emergency_parts)
        
        if final_emergency_context.strip():
            print(f"   ✅ Emergency recovery successful: {len(emergency_parts)} segments")
            return final_emergency_context
        else:
            print(f"   ❌ Emergency recovery failed")
            return "Bu konuda verilen dokümanlarda ilgili bilgiler mevcut ancak sistem yanıt oluştururken teknik bir sorunla karşılaştı. Lütfen sorunuzu farklı şekilde ifade ederek tekrar deneyin."
    
    def _assemble_balanced_multi_topic_context(self, segments: List[ContextSegment], 
                                             original_query: str, sub_questions: List[str]) -> str:
        """Assemble context ensuring balanced representation for all topics"""
        
        print(f"⚖️ Assembling balanced context for {len(sub_questions)} topics")
        
        # Group segments by their strongest topic affinity
        topic_segments = defaultdict(list)
        unassigned_segments = []
        
        for segment in segments:
            if not segment.topic_relevance:
                unassigned_segments.append(segment)
                continue
            
            # Find the topic this segment is most relevant to
            best_topic = max(segment.topic_relevance.items(), key=lambda x: x[1])
            
            # CRITICAL FIX: Force assignment based on source file matching
            force_assigned = False
            
            # Force yurt.pdf → yurt topic
            if 'yurt.pdf' in segment.source.lower():
                for i, sub_q in enumerate(sub_questions):
                    if 'yurt' in sub_q.lower():
                        topic_segments[f"topic_{i}"].append((segment, 1.0))  # Force high score
                        force_assigned = True
                        print(f"   🔧 FORCE: yurt.pdf → topic {i}")
                        break
            
            # Force burs.pdf → burs topic  
            elif 'burs.pdf' in segment.source.lower():
                for i, sub_q in enumerate(sub_questions):
                    if 'burs' in sub_q.lower() or 'ösym' in sub_q.lower():
                        topic_segments[f"topic_{i}"].append((segment, 1.0))  # Force high score
                        force_assigned = True
                        print(f"   🔧 FORCE: burs.pdf → topic {i}")
                        break
            
            if not force_assigned:
                if best_topic[1] > 0.15:  # Lowered threshold for more inclusive assignment
                    topic_segments[best_topic[0]].append((segment, best_topic[1]))
                else:
                    # Still check if segment has ANY relevance to any topic
                    any_relevance = any(score > 0.1 for score in segment.topic_relevance.values())
                    if any_relevance:
                        # Assign to best topic even with low score
                        topic_segments[best_topic[0]].append((segment, best_topic[1]))
                    else:
                        unassigned_segments.append(segment)
        
        # Topic distribution summary
        for topic_id, segs in topic_segments.items():
            topic_idx = int(topic_id.split('_')[1])
            print(f"   📝 Topic {topic_idx} ({sub_questions[topic_idx][:30]}...): {len(segs)} segments")
        
        # Calculate adaptive token allocation based on semantic chunking
        available_tokens = MAX_CONTEXT_TOKENS - 300  # Reserve for formatting
        num_topics = len(sub_questions)
        
        # Adaptive allocation for better semantic coverage
        if num_topics == 2:
            # 88% for topics (44% each), 12% for unassigned
            base_tokens_per_topic = int(available_tokens * 0.44)
        elif num_topics == 3:
            # 90% for topics (30% each), 10% for unassigned  
            base_tokens_per_topic = int(available_tokens * 0.30)
        elif num_topics == 4:
            # 92% for topics (23% each), 8% for unassigned
            base_tokens_per_topic = int(available_tokens * 0.23)
        else:
            # Fallback for unusual topic counts
            base_tokens_per_topic = available_tokens // (num_topics + 1)
        
        print(f"   💰 Semantic-aware token allocation: {base_tokens_per_topic} per topic (total: {available_tokens})")
        
        context_parts = []
        used_tokens = 0
        
        # Allocate tokens for each topic
        for i, sub_question in enumerate(sub_questions):
            topic_id = f"topic_{i}"
            topic_segs = topic_segments.get(topic_id, [])
            
            if not topic_segs:
                print(f"   ⚠️ No segments found for topic {i}: {sub_question}")
                continue
            
            # Sort by topic-specific relevance
            topic_segs.sort(key=lambda x: x[1], reverse=True)
            
            # Add best segments for this topic (LLM-friendly format)
            if 'yurt' in sub_question.lower():
                topic_context = f"\n[YURT BİLGİLERİ]:\n"
            elif 'burs' in sub_question.lower():
                topic_context = f"\n[BURS BİLGİLERİ]:\n"
            elif 'ders' in sub_question.lower():
                topic_context = f"\n[DERS BİLGİLERİ]:\n"
            else:
                # Generic topic header
                topic_word = sub_question.split()[0].upper() if sub_question else "BİLGİ"
                topic_context = f"\n[{topic_word} HAKKINDA]:\n"
            topic_tokens = self._estimate_tokens_fast(topic_context)
            
            segments_added = 0
            min_segments_per_topic = 2  # MINIMUM 2 segments per topic
            max_segments_per_topic = 4  # MAXIMUM 4 segments to avoid overwhelming
            
            for segment, topic_score in topic_segs:
                segment_text = self._format_segment(segment)
                segment_tokens = self._estimate_tokens_fast(segment_text)
                
                # Check if we've reached our maximums
                if segments_added >= max_segments_per_topic:
                    break
                
                # More generous fitting logic
                if topic_tokens + segment_tokens > base_tokens_per_topic:
                    # If we haven't reached minimum, try truncation
                    if segments_added < min_segments_per_topic:
                        remaining_tokens = base_tokens_per_topic - topic_tokens
                        if remaining_tokens > 200:  # Only truncate if meaningful space left
                            truncated = self._smart_truncate_segment(segment, remaining_tokens)
                            if truncated:
                                topic_context += truncated
                                segments_added += 1
                                topic_tokens += self._estimate_tokens_fast(truncated)
                    break
                
                topic_context += segment_text
                topic_tokens += segment_tokens
                segments_added += 1
            
            if segments_added > 0:
                context_parts.append(topic_context)
                used_tokens += topic_tokens
                print(f"   ✅ Topic {i}: {segments_added} segments, {topic_tokens} tokens")
            else:
                print(f"   ⚠️ Topic {i}: No segments fit in token limit")
        
        # Add remaining high-quality unassigned segments if space allows
        remaining_tokens = available_tokens - used_tokens
        if remaining_tokens > 100 and unassigned_segments:
            print(f"   🔄 Adding general segments with {remaining_tokens} remaining tokens")
            
            unassigned_segments.sort(key=lambda x: x.relevance_score, reverse=True)
            for segment in unassigned_segments:
                segment_text = self._format_segment(segment)
                segment_tokens = self._estimate_tokens_fast(segment_text)
                
                if segment_tokens > remaining_tokens:
                    break
                
                context_parts.append(segment_text)
                remaining_tokens -= segment_tokens
        
        final_context = "\n".join(context_parts)
        
        # Enhanced validation: Check if each topic has meaningful representation
        missing_topics = []
        for i, sub_question in enumerate(sub_questions):
            topic_keywords = self._extract_topic_keywords(sub_question)
            
            # Check for keyword presence AND meaningful content length
            has_keywords = any(keyword in final_context.lower() for keyword in topic_keywords)
            
            # Count content related to this topic
            topic_content_chars = 0
            for keyword in topic_keywords:
                if keyword in final_context.lower():
                    # Count surrounding context (simple heuristic)
                    topic_content_chars += len(keyword) * 10  # Rough estimate
            
            # More generous validation for multi-topic - focus on meaningful content
            content_threshold = 300 if num_topics <= 2 else 200  # More generous for 2-topic queries
            if not has_keywords or topic_content_chars < content_threshold:
                missing_topics.append(i)
                print(f"   ⚠️ Topic {i} underrepresented: keywords={has_keywords}, content_chars={topic_content_chars}, threshold={content_threshold}")
        
        if missing_topics:
            print(f"   🚨 Missing/underrepresented topics: {missing_topics}")
            
            # FINAL SOLUTION: Smart fallback to simple query logic for missing topics
            for missing_idx in missing_topics:
                missing_topic = sub_questions[missing_idx]
                
                # Try simple query approach for this specific topic
                print(f"   🔄 Trying simple query approach for topic: {missing_topic}")
                simple_context = self._emergency_simple_query_context(segments, missing_topic)
                
                if simple_context and len(simple_context.strip()) > 100:
                    # If simple approach found good content, use it (LLM-friendly format)
                    if 'yurt' in missing_topic.lower():
                        topic_header = f"\n[YURT BİLGİLERİ EK]:\n"
                    elif 'burs' in missing_topic.lower():
                        topic_header = f"\n[BURS BİLGİLERİ EK]:\n"
                    elif 'ders' in missing_topic.lower():
                        topic_header = f"\n[DERS BİLGİLERİ EK]:\n"
                    else:
                        topic_word = missing_topic.split()[0].upper() if missing_topic else "BİLGİ"
                        topic_header = f"\n[{topic_word} EK BİLGİ]:\n"
                    
                    final_context += topic_header + simple_context
                    print(f"   ✅ Simple query fallback successful for topic {missing_idx}")
                else:
                    print(f"   ❌ Simple query fallback failed for topic {missing_idx}")
        
        return final_context
    
    def _extract_topic_keywords(self, topic_query: str) -> List[str]:
        """Extract key identifying words from a topic query"""
        keywords = []
        query_lower = topic_query.lower()
        
        # Domain-specific keyword mapping
        if 'yurt' in query_lower:
            keywords.extend(['yurt', 'anlaşmalı'])
        if 'burs' in query_lower:
            keywords.extend(['burs', 'ösym'])
        if 'ders' in query_lower:
            keywords.extend(['ders', 'kurs'])
        
        # Add all significant words from query
        words = re.findall(r'\b\w+\b', query_lower)
        keywords.extend([w for w in words if len(w) > 3])
        
        return list(set(keywords))
    

    
    def _smart_truncate_segment(self, segment: ContextSegment, max_tokens: int) -> str:
        """Intelligently truncate a segment to fit token limit while preserving meaning"""
        
        max_chars = int(max_tokens * CHARS_PER_TOKEN * 0.8)  # Conservative estimate
        content = segment.content
        
        if len(content) <= max_chars:
            return self._format_segment(segment)
        
        # Try to truncate at sentence boundaries
        sentences = content.split('.')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '.') > max_chars:
                break
            truncated += sentence + '.'
        
        if not truncated.strip():
            # Fallback: hard truncation with ellipsis
            truncated = content[:max_chars-3] + "..."
        
        # Create truncated segment
        truncated_segment = ContextSegment(
            content=truncated,
            source=segment.source,
            article=segment.article,
            relevance_score=segment.relevance_score,
            search_origin=segment.search_origin
        )
        
        return self._format_segment(truncated_segment)
    
    def _format_segment(self, segment: ContextSegment) -> str:
        """Format a segment with source information"""
        source_info = ""
        if segment.source and segment.source != 'Unknown':
            source_info += f"[Kaynak: {segment.source}]"
        if segment.article and segment.article != 'Unknown':
            source_info += f"[Bölüm: {segment.article}]"
        if source_info:
            source_info += "\n"
        
        return f"{source_info}{segment.content}\n\n"
    
    def _estimate_tokens_fast(self, text: str) -> int:
        """Fast token estimation for context management"""
        # GPT tokenizer approximation: ~75% word count + 25% char count
        word_count = len(text.split())
        char_count = len(text)
        estimated = int(word_count * 0.75 + char_count * 0.25)
        return estimated
    
    def create_query_context_summary(self, assembled_context: str, query: str) -> Dict:
        """Create a summary of the assembled context for debugging/monitoring"""
        
        return {
            'context_length': len(assembled_context),
            'estimated_tokens': self._estimate_tokens_fast(assembled_context),
            'has_sources': '[Kaynak:' in assembled_context,
            'has_articles': '[Bölüm:' in assembled_context,
            'query_terms_found': [
                term for term in query.lower().split() 
                if term in assembled_context.lower()
            ]
        } 