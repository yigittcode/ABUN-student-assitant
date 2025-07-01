from config import HYDE_LLM_MODEL
import asyncio
import re

class AdvancedHydeGenerator:
    """Gelişmiş HyDE: Direkt referanslar + karmaşık senaryolar dahil"""
    
    # Gelişmiş kategori sistemi - direkt referanslar dahil
    ENHANCED_CATEGORIES = {
        'direct_reference': {
            'keywords': ['madde', 'article', 'bölüm', 'chapter', 'section', 'paragraf', 'paragraph', 'fıkra'],
            'patterns': [r'\bmadde\s*\d+', r'\barticle\s*\d+', r'\bbölüm\s*\d+', r'\bsection\s*\d+'],
            'prompt_tr': "Bu soruya ilgili madde/bölüm/kısım içeriğini tam olarak açıklayarak yanıt ver:",
            'prompt_en': "Answer this by explaining the exact content of the referenced article/section:"
        },
        'scenario_based': {
            'keywords': ['durumunda', 'halinde', 'olursa', 'case', 'scenario', 'situation', 'eğer', 'if', 'varsayım', 'assume'],
            'patterns': [r'eğer.*ise', r'if.*then', r'durumunda.*ne', r'halinde.*nasıl'],
            'prompt_tr': "Bu senaryoya dayalı soruya mevzuat ve prosedürleri referans alarak detaylı yanıt ver:",
            'prompt_en': "Answer this scenario-based question with detailed reference to regulations and procedures:"
        },
        'comparative': {
            'keywords': ['fark', 'difference', 'karşı', 'vs', 'versus', 'arasında', 'between', 'karşılaştır', 'compare'],
            'patterns': [r'\bve\b.*\barasında', r'\bversus\b', r'\bkarşı\b'],
            'prompt_tr': "Bu karşılaştırma sorusuna her iki durumu da açıklayarak yanıt ver:",
            'prompt_en': "Answer this comparison question by explaining both situations:"
        },
        'factual': {
            'keywords': ['nedir', 'ne', 'what', 'kimdir', 'tanım', 'definition', 'açıkla', 'explain'],
            'patterns': [r'\bnedir\b', r'\bwhat\s+is\b'],
            'prompt_tr': "Bu soruya net tanım ve açıklama vererek yanıt ver:",
            'prompt_en': "Answer this with clear definition and explanation:"
        },
        'procedural': {
            'keywords': ['nasıl', 'how', 'adım', 'step', 'yöntem', 'method', 'prosedür', 'procedure'],
            'patterns': [r'\bnasıl\b', r'\bhow\s+to\b'],
            'prompt_tr': "Bu soruya adım adım talimatlarla yanıt ver:",
            'prompt_en': "Answer this with step-by-step instructions:"
        },
        'listing': {
            'keywords': ['neler', 'hangi', 'which', 'listele', 'list', 'çeşit', 'types', 'türler'],
            'patterns': [r'\bneler\b', r'\bwhich\s+are\b'],
            'prompt_tr': "Bu soruya madde madde liste halinde yanıt ver:",
            'prompt_en': "Answer this with a detailed numbered list:"
        },
        'general': {
            'keywords': [],  # Fallback kategori
            'patterns': [],
            'prompt_tr': "Bu soruya kapsamlı ve ayrıntılı şekilde yanıt ver:",
            'prompt_en': "Answer this comprehensively and in detail:"
        }
    }
    
    @classmethod
    def detect_category(cls, question):
        """Gelişmiş kategori tespiti: keyword + pattern matching"""
        question_lower = question.lower()
        
        # Her kategori için skor hesapla
        category_scores = {}
        
        for category, config in cls.ENHANCED_CATEGORIES.items():
            if category == 'general':
                continue  # General skip et, fallback olarak kullan
            
            score = 0
            
            # Keyword matching
            keyword_score = sum(1 for keyword in config['keywords'] if keyword in question_lower)
            
            # Pattern matching (daha yüksek ağırlık)
            pattern_score = 0
            for pattern in config['patterns']:
                if re.search(pattern, question_lower):
                    pattern_score += 2  # Pattern match daha önemli
            
            score = keyword_score + pattern_score
            
            if score > 0:
                category_scores[category] = score
        
        # En yüksek skoru alan kategoriyi döndür
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            print(f"🎯 Enhanced category detected: {best_category} (score: {category_scores[best_category]})")
            return best_category
        else:
            print("🔄 No specific category detected, using 'general' fallback")
            return 'general'
        
        
    
    @classmethod
    def detect_language(cls, question):
        """Gelişmiş dil tespit et"""
        tr_indicators = ['nedir', 'nasıl', 'neler', 'hangi', 'ne', 'kaç', 'madde', 'durumunda', 'halinde']
        en_indicators = ['what', 'how', 'which', 'when', 'where', 'why', 'article', 'case', 'situation']
        
        tr_count = sum(1 for word in tr_indicators if word in question.lower())
        en_count = sum(1 for word in en_indicators if word in question.lower())
        
        return 'turkish' if tr_count >= en_count else 'english'


async def generate_enhanced_hyde(question, client, domain_context=""):
    """Gelişmiş HyDE: Kalite korunarak hızlandırılmış"""
    
    # Convert question to string if it's a list (fix for unhashable type error)
    if isinstance(question, list):
        question = ' '.join(str(q) for q in question)
    else:
        question = str(question)
    
    # OPTIMIZATION 1: Smart HyDE caching
    hyde_cache_key = f"hyde_{hash(question.lower().strip())}_{domain_context}"
    
    # Check cache first
    from rag_engine import api_cache
    if hyde_cache_key in api_cache:
        print("🚀 HyDE cache hit! Using cached result")
        return api_cache[hyde_cache_key]
    
    # OPTIMIZATION 2: Smart category detection - skip complex analysis for simple questions
    simple_patterns = [
        'nedir', 'what is', 'kimdir', 'neler', 'nasıl', 'how to',
        'kaç', 'how many', 'var mı', 'is there', 'bulunur mu',
        'hangi', 'which', 'ne zaman', 'when', 'nerede', 'where'
    ]
    is_simple = any(pattern in question.lower() for pattern in simple_patterns)
    
    if is_simple and len(question.split()) <= 5:
        # For simple questions, skip HyDE and use question directly
        print("⚡ Simple question detected - skipping HyDE for speed")
        result = [question]
        api_cache[hyde_cache_key] = result
        return result
    
    # For complex questions, use optimized HyDE
    generator = AdvancedHydeGenerator()
    category = generator.detect_category(question)
    language = generator.detect_language(question)
    
    # OPTIMIZATION 3: Shorter, more focused prompts for speed
    config = generator.ENHANCED_CATEGORIES[category]
    prompt_key = 'prompt_tr' if language == 'turkish' else 'prompt_en'
    base_prompt = config[prompt_key]
    
    # Domain context ekle (optimized)
    if domain_context:
        if language == 'turkish':
            context_prompt = f"[{domain_context}] "
        else:
            context_prompt = f"[{domain_context}] "
        base_prompt = context_prompt + base_prompt
    
    # Direkt referans kategori ise özel yaklaşım
    if category == 'direct_reference':
        ref_match = re.search(r'madde\s*(\d+)', question.lower())
        if ref_match:
            ref_num = ref_match.group(1)
            base_prompt += f" MADDE {ref_num} odaklı yanıt."
    
    # OPTIMIZATION 4: Shorter prompt for faster generation
    final_prompt = f"{base_prompt}\n\nSoru: {question}\n\nKısa yanıt:"
    
    print(f"🚀 Optimized HyDE - Category: {category}, Simple: {is_simple}")
    
    try:
        response = await client.chat.completions.create(
            model=HYDE_LLM_MODEL, 
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.1, 
            max_tokens=150  # Reduced for speed: 300→150
        )
        
        hyde_answer = response.choices[0].message.content.strip()
        print(f"✅ Generated optimized HyDE: {hyde_answer[:60]}...")
        
        result = [hyde_answer] if hyde_answer else [question]
        
        # Cache the result
        api_cache[hyde_cache_key] = result
        
        return result
    
    except Exception as e:
        print(f"❌ HyDE generation error: {e}")
        # Fallback: Soruyu direkt kullan
        print("🔄 Using question as fallback HyDE")
        fallback_result = [question]
        api_cache[hyde_cache_key] = fallback_result
        return fallback_result


# Backward compatibility için - yeni gelişmiş sistemi kullan
async def generate_multiple_hyde_variants(question, client, domain_context=""):
    """Ana API - gelişmiş HyDE sistemi"""
    return await generate_enhanced_hyde(question, client, domain_context)


async def generate_hypothetical_answers(question, client, n=1, domain_context=""):
    """Eski API ile uyumlu wrapper"""
    result = await generate_enhanced_hyde(question, client, domain_context)
    return result if result else [question]  # Fallback eklendi