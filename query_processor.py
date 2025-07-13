"""
🧠 Query Processor Module
Advanced query decomposition, expansion, and multi-query strategy
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from cachetools import TTLCache

# Cache for query analysis and variants
query_analysis_cache = TTLCache(maxsize=500, ttl=600)  # 10 min cache
query_variants_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min cache

@dataclass
class QueryAnalysis:
    """Query complexity analysis result"""
    is_complex: bool
    sub_questions: List[str]
    relationships: List[str]
    primary_intent: str
    confidence: float

@dataclass  
class QueryVariants:
    """Query expansion variants"""
    original: str
    paraphrases: List[str]
    domain_variants: List[str]
    context_variants: List[str]
    type_variants: List[str]
    
    def all_variants(self) -> List[str]:
        """Get all query variants as a flat list"""
        all_vars = [self.original]
        all_vars.extend(self.paraphrases)
        all_vars.extend(self.domain_variants)
        all_vars.extend(self.context_variants)
        all_vars.extend(self.type_variants)
        # Remove duplicates while preserving order
        seen = set()
        unique_vars = []
        for var in all_vars:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)
        return unique_vars[:8]  # Limit to 8 variants max

class QueryProcessor:
    """Advanced query processing with multi-query strategy"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
        # Domain-specific synonyms for better query expansion
        self.domain_synonyms = {
            'ders': ['kurs', 'ders', 'mufredat', 'eğitim'],
            'çekilme': ['çekilme', 'bırakma', 'vazgeçme', 'ayrılma'],
            'burs': ['burs', 'indirim', 'destek', 'yardım'],
            'başarı': ['başarı', 'not', 'ortalama', 'performans'],
            'tam burslu': ['tam burslu', '%100 burslu', 'tam indirimli'],
            'öğrenci': ['öğrenci', 'talebe', 'kayıtlı']
        }
    
    async def analyze_query_complexity(self, query: str) -> QueryAnalysis:
        """Analyze if query contains multiple questions or complex logic"""
        
        # Check cache first
        cache_key = f"analysis_{hash(query.lower().strip())}"
        if cache_key in query_analysis_cache:
            cached_result = query_analysis_cache[cache_key]
            print(f"🚀 Using cached analysis for query")
            return cached_result
        
        analysis_prompt = f"""Aşağıdaki soruyu analiz et ve JSON formatında yanıt ver:

Bu sorunun:
1. Birden fazla alt soru içerip içermediğini
2. Hangi alt sorulara bölünebileceğini
3. Sorular arası mantıksal ilişkiyi
4. Ana amacını

belirle.

ÖNEMLI: Basit bilgi talepleri (örn: "X hakkında bilgi ver") COMPLEX değildir!

Soru: "{query}"

Kesinlikle sadece JSON formatında yanıt ver:
{{
    "is_complex": true/false,
    "sub_questions": ["alt soru 1", "alt soru 2"],
    "relationships": ["and", "conditional", "exclusive"],
    "primary_intent": "ana_amaç",
    "confidence": 0.95
}}"""
        
        print(f"🔍 Analyzing query complexity...")
        print(f"   Query: '{query}'")
        print(f"   Sending prompt to OpenAI...")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            print(f"   📥 Raw OpenAI response: {response_text}")
            
            # Extract JSON from response (handle markdown formatting)
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
            
            print(f"   📋 Extracted JSON: {json_text}")
                
            analysis_data = json.loads(json_text)
            print(f"   🔍 Parsed analysis: {analysis_data}")
            
            result = QueryAnalysis(
                is_complex=analysis_data.get('is_complex', False),
                sub_questions=analysis_data.get('sub_questions', []),
                relationships=analysis_data.get('relationships', []),
                primary_intent=analysis_data.get('primary_intent', 'general'),
                confidence=analysis_data.get('confidence', 0.5)
            )
            
            # Detailed logging
            print(f"   ✅ Analysis Result:")
            print(f"      • Complex: {result.is_complex}")
            print(f"      • Sub-questions ({len(result.sub_questions)}): {result.sub_questions}")
            print(f"      • Relationships: {result.relationships}")
            print(f"      • Intent: {result.primary_intent}")
            print(f"      • Confidence: {result.confidence}")
            
            # Validation check - warn if simple query marked as complex
            if result.is_complex and self._is_simple_info_request(query):
                print(f"   ⚠️ WARNING: Simple info request marked as complex!")
                print(f"      • Query seems like basic info request")
                print(f"      • Consider adjusting analysis logic")
            
            # Cache result
            query_analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"   ❌ Query analysis failed: {e}")
            print(f"   🔄 Using fallback simple analysis")
            # Fallback: simple analysis
            return QueryAnalysis(
                is_complex=False,
                sub_questions=[query],
                relationships=['single'],
                primary_intent='general',
                confidence=0.3
            )
    
    def _is_simple_info_request(self, query: str) -> bool:
        """Check if query is a simple information request"""
        query_lower = query.lower().strip()
        
        # Simple info request patterns
        simple_patterns = [
            r'.*hakkında bilgi.*',
            r'.*nedir\?*$',
            r'.*anlat.*',
            r'.*açıkla.*',
            r'.*what is.*',
            r'.*tell me about.*',
            r'.*explain.*'
        ]
        
        for pattern in simple_patterns:
            if re.match(pattern, query_lower):
                return True
                
        return False
    
    async def generate_query_variants(self, original_query: str, user_context: Optional[Dict] = None) -> QueryVariants:
        """Generate multiple variants of the same query for comprehensive search"""
        
        # Check cache first
        cache_key = f"variants_{hash(original_query.lower().strip())}"
        if cache_key in query_variants_cache:
            cached_result = query_variants_cache[cache_key]
            print(f"🚀 Using cached variants for query")
            return cached_result
        
        print(f"🔄 Generating query variants...")
        print(f"   Original query: '{original_query}'")
        print(f"   User context: {user_context}")
        
        # Smart performance optimization based on query complexity
        is_likely_complex = (
            len(original_query.split()) > 8 or 
            ' ve ' in original_query or  # Turkish "and"
            ' and ' in original_query or
            '?' in original_query[:-1]  # Multiple questions
        )
        
        if is_likely_complex:
            # Full variant generation for complex queries (quality priority)
            print(f"   🔍 Full variant generation: {len(original_query.split())} words")
            
            # Generate all variants in parallel
            paraphrases_task = self._generate_paraphrases(original_query)
            domain_variants_task = self._generate_domain_variants(original_query)
            context_variants_task = self._generate_context_variants(original_query, user_context)
            type_variants_task = self._generate_type_variants(original_query)
            
            paraphrases, domain_variants, context_variants, type_variants = await asyncio.gather(
                paraphrases_task,
                domain_variants_task, 
                context_variants_task,
                type_variants_task
            )
        else:
            # Lightweight generation for simple queries (speed priority)
            print(f"   ⚡ Lightweight variant generation: {len(original_query.split())} words")
            
            # Only generate domain and type variants for simple queries
            domain_variants_task = self._generate_domain_variants(original_query)
            type_variants_task = self._generate_type_variants(original_query)
            
            domain_variants, type_variants = await asyncio.gather(
                domain_variants_task,
                type_variants_task
            )
            paraphrases = []
            context_variants = []
        
        print(f"   📝 Paraphrases ({len(paraphrases)}): {paraphrases}")
        print(f"   🏷️ Domain variants ({len(domain_variants)}): {domain_variants}")
        print(f"   👤 Context variants ({len(context_variants)}): {context_variants}")
        print(f"   ❓ Type variants ({len(type_variants)}): {type_variants}")
        
        result = QueryVariants(
            original=original_query,
            paraphrases=paraphrases,
            domain_variants=domain_variants,
            context_variants=context_variants,
            type_variants=type_variants
        )
        
        all_variants = result.all_variants()
        print(f"   ✅ Total unique variants: {len(all_variants)}")
        print(f"   📋 Final variant list: {all_variants}")
        
        # Cache result
        query_variants_cache[cache_key] = result
        
        return result
    
    async def _generate_paraphrases(self, query: str) -> List[str]:
        """Generate semantic paraphrases of the query"""
        
        paraphrase_prompt = f"""Aşağıdaki soruyu 3 farklı şekilde ifade et. Anlam aynı kalsın, sadece kelimeler değişsin, kullanılabilecek farklı kelimeler kullanın:

Orijinal: "{query}"

Sadece paraphrase'ları listele, açıklama yapma:
1. 
2. 
3. """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": paraphrase_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            text = response.choices[0].message.content.strip()
            
            # Extract numbered list
            paraphrases = []
            for line in text.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    paraphrase = re.sub(r'^\d+\.\s*', '', line).strip()
                    if paraphrase and paraphrase != query:
                        paraphrases.append(paraphrase)
            
            return paraphrases[:3]  # Max 3 paraphrases
            
        except Exception as e:
            print(f"⚠️ Paraphrase generation failed: {e}")
            return []
    
    async def _generate_domain_variants(self, query: str) -> List[str]:
        """Generate domain-specific variants using synonyms"""
        
        variants = []
        query_lower = query.lower()
        
        # Apply domain synonym replacements
        for term, synonyms in self.domain_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym != term:
                        variant = query_lower.replace(term, synonym)
                        if variant != query_lower:
                            # Capitalize first letter
                            variant = variant.capitalize()
                            variants.append(variant)
        
        return variants[:3]  # Max 3 domain variants
    
    async def _generate_context_variants(self, query: str, user_context: Optional[Dict]) -> List[str]:
        """Generate context-aware variants based on user information"""
        
        if not user_context:
            return []
            
        variants = []
        
        # Add user-specific context to query
        if 'student_type' in user_context:
            student_type = user_context['student_type']
            context_variant = f"{student_type} öğrenci olarak {query.lower()}"
            variants.append(context_variant.capitalize())
        
        if 'program' in user_context:
            program = user_context['program']
            context_variant = f"{program} programında {query.lower()}"
            variants.append(context_variant.capitalize())
            
        return variants[:2]  # Max 2 context variants
    
    async def _generate_type_variants(self, query: str) -> List[str]:
        """Generate different question type variants"""
        
        variants = []
        
        # Convert to different question types
        if '?' not in query:
            variants.append(f"{query}?")
        
        # Add explicit question words if missing
        if not any(word in query.lower() for word in ['ne', 'nasıl', 'hangi', 'kim', 'nerede', 'ne zaman']):
            if 'mümkün' in query.lower():
                variants.append(f"Nasıl {query.lower().replace('mümkün mü', 'yapabilirim')}")
            elif 'kurallar' in query.lower():
                variants.append(f"Hangi kurallar {query.lower().split('kurallar')[0]}kuralları için geçerli")
        
        return variants[:2]  # Max 2 type variants
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        
        # Basic normalization
        normalized = query.strip().lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Standardize question marks
        if not normalized.endswith('?'):
            normalized += '?'
            
        return normalized
    
    async def process_multi_query(self, original_query: str, user_context: Optional[Dict] = None) -> Tuple[QueryAnalysis, QueryVariants]:
        """Process query with both decomposition and expansion"""
        
        print(f"🔍 Processing multi-query for: {original_query}")
        
        # Execute analysis and variant generation in parallel
        analysis_task = self.analyze_query_complexity(original_query)
        variants_task = self.generate_query_variants(original_query, user_context)
        
        analysis, variants = await asyncio.gather(analysis_task, variants_task)
        
        print(f"📊 Multi-query processing complete:")
        print(f"   • Complex: {analysis.is_complex}")
        print(f"   • Sub-questions: {len(analysis.sub_questions)}")
        print(f"   • Variants: {len(variants.all_variants())}")
        
        return analysis, variants 