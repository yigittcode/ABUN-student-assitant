"""
✨ Response Generator Module
Advanced contextual and voice response generation
"""

import asyncio
import json
from typing import Optional, Dict, AsyncGenerator
from cachetools import TTLCache
from config import LLM_MODEL, PROMPT_TEMPLATE, VOICE_PROMPT_TEMPLATE

# Response cache for faster repeated queries
response_cache = TTLCache(maxsize=300, ttl=600)  # 10 min cache

class ResponseGenerator:
    """Advanced response generator with context intelligence"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def generate_contextual_response(self, question: str, context: str, 
                                         domain_context: str = "", 
                                         user_context: Optional[Dict] = None) -> str:
        """Generate contextual response with enhanced logic"""
        
        # Validate context quality
        if not context or len(context.strip()) < 20:
            print("🚨 Context validation failed - insufficient content")
            return self._generate_fallback_response(domain_context)
        
        # Check cache
        cache_key = self._create_cache_key(question, context, domain_context, user_context)
        if cache_key in response_cache:
            print("🚀 Response cache hit!")
            return response_cache[cache_key]
        
        print(f"🤖 Generating contextual response (context: {len(context)} chars)")
        
        # Prepare enhanced prompt
        enhanced_prompt = self._prepare_enhanced_prompt(question, context, domain_context, user_context)
        
        try:
            # Debug: Log prompt details for multi-topic queries
            if "ve " in question and len(context) > 1000:
                print("🔍 DEBUG: Multi-topic query detected")
                print(f"   📝 Question: {question}")
                print(f"   📊 Context length: {len(context)} chars")
                print(f"   📋 Context structure check:")
                if "[YURT BİLGİLERİ]" in context:
                    print("     ✅ Found [YURT BİLGİLERİ] section")
                if "[BURS BİLGİLERİ]" in context:
                    print("     ✅ Found [BURS BİLGİLERİ] section")
                print(f"   🎯 Context preview: {context[:500]}...")
            
            response = await self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate response quality
            validated_result = self._validate_response_quality(result, question, context, domain_context)
            
            # Cache successful response
            response_cache[cache_key] = validated_result
            
            print(f"✅ Generated contextual response: {len(validated_result)} characters")
            return validated_result
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return self._generate_error_response(str(e), domain_context)
    
    async def generate_voice_response(self, question: str, context: str, 
                                    domain_context: str = "",
                                    user_context: Optional[Dict] = None) -> str:
        """Generate voice-optimized response"""
        
        # Validate context (same as contextual response)
        if not context or len(context.strip()) < 20:
            print("🚨 Voice response: Context validation failed")
            return self._generate_fallback_response(domain_context, is_voice=True)
        
        print(f"🎤 Generating voice response (context: {len(context)} chars)")
        
        # Prepare voice-specific prompt
        voice_prompt = self._prepare_voice_prompt(question, context, domain_context, user_context)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": voice_prompt}],
                temperature=0.1,
                max_tokens=400  # Shorter for voice
            )
            
            result = response.choices[0].message.content.strip()
            
            print(f"✅ Generated voice response: {len(result)} characters")
            return result
            
        except Exception as e:
            print(f"❌ Voice response generation error: {e}")
            return self._generate_error_response(str(e), domain_context, is_voice=True)
    
    async def generate_streaming_response(self, question: str, context: str, 
                                        domain_context: str = "",
                                        user_context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response with real-time output"""
        
        # Validate context
        if not context or len(context.strip()) < 20:
            error_msg = self._generate_fallback_response(domain_context)
            yield json.dumps({
                "type": "content",
                "content": error_msg,
                "done": True
            })
            return
        
        print(f"🌊 Generating streaming response (context: {len(context)} chars)")
        
        # Prepare enhanced prompt
        enhanced_prompt = self._prepare_enhanced_prompt(question, context, domain_context, user_context)
        
        try:
            stream = await self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.1,
                max_tokens=1000,
                stream=True
            )
            
            collected_content = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    
                    # Stream chunk
                    yield json.dumps({
                        "type": "content",
                        "content": content,
                        "done": False
                    })
            
            # Stream completion
            yield json.dumps({
                "type": "complete",
                "content": "",
                "done": True,
                "full_response": collected_content
            })
            
            print(f"✅ Streaming response completed: {len(collected_content)} characters")
            
        except Exception as e:
            print(f"❌ Streaming response error: {e}")
            error_response = self._generate_error_response(str(e), domain_context)
            yield json.dumps({
                "type": "error",
                "content": error_response,
                "done": True
            })
    
    def _prepare_enhanced_prompt(self, question: str, context: str, domain_context: str, 
                               user_context: Optional[Dict]) -> str:
        """Prepare enhanced prompt with user context integration"""
        
        # Start with base prompt
        if domain_context:
            adapted_prompt = PROMPT_TEMPLATE.replace(
                "Siz Ankara Bilim Üniversitesi'nde doküman analizi uzmanısınız",
                f"Siz {domain_context} konusunda uzman bir asistansınız"
            ).replace(
                "Bu konuda verilen dokümanlarda bilgi bulunamadı",
                f"Bu konuda verilen {domain_context.lower()} metinlerinde bilgi bulunamadı"
            )
        else:
            adapted_prompt = PROMPT_TEMPLATE
        
        # Add user context awareness if available
        if user_context:
            user_context_addition = self._create_user_context_addition(user_context, question)
            if user_context_addition:
                adapted_prompt += f"\n\nKULLANICI DURUMU:\n{user_context_addition}\n"
        
        # Add logical reasoning enhancement
        logical_enhancement = self._add_logical_reasoning_instructions(question, context)
        if logical_enhancement:
            adapted_prompt += f"\n\nMANTIKSAL ÇIKARIM:\n{logical_enhancement}\n"
        
        return adapted_prompt.format(context=context, question=question)
    
    def _prepare_voice_prompt(self, question: str, context: str, domain_context: str,
                            user_context: Optional[Dict]) -> str:
        """Prepare voice-specific prompt"""
        
        if domain_context:
            adapted_prompt = VOICE_PROMPT_TEMPLATE.replace(
                "Siz Ankara Bilim Üniversitesi'nde sesli asistansınız",
                f"Siz {domain_context} konusunda uzman sesli asistansınız"
            )
        else:
            adapted_prompt = VOICE_PROMPT_TEMPLATE
        
        return adapted_prompt.format(context=context, question=question)
    
    def _create_user_context_addition(self, user_context: Dict, question: str) -> str:
        """Create user context addition for enhanced personalization"""
        
        additions = []
        
        student_type = user_context.get('student_type', '').lower()
        program = user_context.get('program', '')
        semester = user_context.get('semester', '')
        
        if student_type:
            additions.append(f"Kullanıcı {student_type} öğrencidir.")
        
        if program:
            additions.append(f"Program: {program}")
        
        if semester:
            additions.append(f"Dönem: {semester}")
        
        # Add specific logic for common scenarios
        if 'tam burslu' in student_type and 'başarı bursu' in question.lower():
            additions.append("ÖNEMLI: Tam burslu öğrenciler başarı bursu alamazlar - bu kuralı vurgula.")
        
        if 'tek ders' in question.lower() or 'sadece' in question.lower():
            additions.append("ÖNEMLI: Tek derse kayıtlı öğrenciler için özel kuralları kontrol et.")
        
        return "\n".join(additions) if additions else ""
    
    def _add_logical_reasoning_instructions(self, question: str, context: str) -> str:
        """Add logical reasoning instructions based on question type"""
        
        instructions = []
        
        # Detect question patterns that need logical reasoning
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['mümkün', 'alabilir', 'yapabilir']):
            instructions.append("Bu soru izin/yasak durumu soruyor - koşulları ve kısıtlamaları açık bir şekilde belirt.")
        
        if 'tek' in question_lower or 'sadece' in question_lower:
            instructions.append("Bu soru tek/sadece durumu içeriyor - minimum gereksinimler ve istisnalar varsa belirt.")
        
        if 'burs' in question_lower:
            instructions.append("Burs sorusu - hangi tür öğrencilerin hangi bursları alabileceğini/alamayacağını açık belirt.")
        
        if 'çekilme' in question_lower or 'bırakma' in question_lower:
            instructions.append("Ders çekilme sorusu - çekilme koşulları ve kısıtlamalarını detaylandır.")
        
        return "\n".join(instructions) if instructions else ""
    
    def _validate_response_quality(self, response: str, question: str, context: str, 
                                 domain_context: str) -> str:
        """Validate and potentially improve response quality"""
        
        if not response or len(response.strip()) < 10:
            print("⚠️ Response too short, generating fallback")
            return self._generate_fallback_response(domain_context)
        
        # Enhanced validation for multi-topic responses
        if "bulunamadı" in response.lower() and len(context) > 100:
            print("🚨 CRITICAL: Response says 'not found' but context exists!")
            print(f"   📊 Context length: {len(context)} chars")
            print(f"   🔍 Context preview: {context[:300]}...")
            
            # Check for specific content in context
            context_lower = context.lower()
            found_topics = []
            if 'yurt' in context_lower and ('anlaşmalı' in context_lower or 'öğrenci' in context_lower):
                found_topics.append("YURT")
            if 'burs' in context_lower and ('ösym' in context_lower or '%' in context_lower):
                found_topics.append("BURS")
                
            if found_topics:
                print(f"   ⚠️ Context contains information about: {found_topics}")
                print("   🔧 This indicates a prompt engineering issue!")
            # Could trigger alternative generation here if needed
        
        return response
    
    def _generate_fallback_response(self, domain_context: str = "", is_voice: bool = False) -> str:
        """Generate fallback response when context is insufficient"""
        
        if is_voice:
            return f"Bu konuda verilen {domain_context.lower() if domain_context else 'dokümanlarda'} bilgi bulunamadı."
        else:
            return f"Bu konuda verilen {domain_context.lower() if domain_context else 'dokümanlarda'} bilgi bulunamadı veya erişilemedi."
    
    def _generate_error_response(self, error: str, domain_context: str = "", is_voice: bool = False) -> str:
        """Generate user-friendly error response"""
        
        if is_voice:
            return "Özür dilerim, yanıt oluştururken bir hata oluştu. Lütfen tekrar deneyin."
        else:
            return f"Özür dilerim, yanıt oluştururken bir hata oluştu. Lütfen sorunuzu daha spesifik hale getirip tekrar deneyin."
    
    def _create_cache_key(self, question: str, context: str, domain_context: str,
                         user_context: Optional[Dict]) -> str:
        """Create cache key for response caching"""
        
        # Create a hash-based key from inputs
        context_hash = hash(context[:500] + context[-500:]) if len(context) > 1000 else hash(context)
        user_hash = hash(str(user_context)) if user_context else 0
        
        return f"{hash(question.lower().strip())}_{context_hash}_{hash(domain_context)}_{user_hash}" 