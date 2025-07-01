"""🎤 Speech Integration Module
RAG API'ye entegre edilebilir Speech-to-Speech modülü
Whisper STT + Edge TTS
"""

import asyncio
import whisper
import edge_tts
import pygame
import tempfile
import os
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

class SpeechProcessor:
    """Speech işlemlerini yapan ana sınıf"""
    
    def __init__(self, whisper_model_name: str = "small"):
        """
        Speech Processor başlatma
        
        Args:
            whisper_model_name: Whisper model adı (tiny, base, small, medium, large)
        """
        print("🎤 Speech Processor başlatılıyor...")
        
        # Pygame başlat (TTS için)
        pygame.mixer.init()
        
        # Whisper modeli yükle
        print(f"🤖 Whisper {whisper_model_name} modeli yükleniyor...")
        self.whisper_model = whisper.load_model(whisper_model_name)
        print("✅ Whisper hazır!")
        
        # Geçici dosya listesi
        self.temp_files = []
        
        # Türkçe ses seçenekleri
        self.turkish_voices = {
            "emel": "tr-TR-EmelNeural",      # Kadın ses
            "ahmet": "tr-TR-AhmetNeural",    # Erkek ses
        }
        
        # Basit gender mapping
        self.gender_voices = {
            "female": "tr-TR-EmelNeural",    # Varsayılan kadın ses
            "male": "tr-TR-AhmetNeural",     # Erkek ses seçeneği
            "kadın": "tr-TR-EmelNeural",     # Türkçe alias
            "erkek": "tr-TR-AhmetNeural"     # Türkçe alias
        }
        
        # TTS optimizasyon sözlüğü
        self.text_optimizations = {
            'Dr.': 'Doktor',
            'Prof.': 'Profesör', 
            'YÖK': 'Yükseköğretim Kurulu',
            'ÖSYM': 'Öğrenci Seçme ve Yerleştirme Merkezi',
            'gerekiyor': 'gerekmektedir',
            'olacak': 'olacaktır',
            'yapılır': 'yapılmaktadır'
        }
    
    def speech_to_text(self, audio_file_path: str, language: str = "tr") -> str:
        """
        Ses dosyasını metne çevir
        
        Args:
            audio_file_path: Ses dosyası yolu
            language: Dil kodu (tr, en, vs.)
            
        Returns:
            Tanınan metin
        """
        try:
            # NOT: STT işlemi cancellable değil, ama hızlı olması gerekiyor
            result = self.whisper_model.transcribe(audio_file_path, language=language)
            text = result["text"].strip()
            print(f"🎤 STT Sonuç: '{text}'")
            return text
        except Exception as e:
            print(f"❌ STT Hatası: {e}")
            return ""
    
    def optimize_text_for_tts(self, text: str) -> str:
        """
        Metni TTS için optimize et - Özellikle voice kullanımı için temizle
        
        Args:
            text: Ham metin
            
        Returns:
            Optimize edilmiş metin
        """
        optimized = text
        
        # Kısaltmaları ve kelime optimizasyonlarını uygula
        for old, new in self.text_optimizations.items():
            optimized = optimized.replace(old, new)
        
        # Voice kullanımı için madde işaretlerini ve formatlamayı temizle
        import re
        
        # Madde işaretlerini kaldır (1., 2., •, -, vs.)
        optimized = re.sub(r'^\s*[•\-\*]\s*', '', optimized, flags=re.MULTILINE)
        optimized = re.sub(r'^\s*\d+\.\s*', '', optimized, flags=re.MULTILINE)
        
        # Çok fazla yeni satırı temizle
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # Başta ve sonda boşluk temizle
        optimized = optimized.strip()
        
        return optimized
    
    async def text_to_speech(self, text: str, voice: str = "tr-TR-EmelNeural") -> str:
        """
        Metni sese çevir
        
        Args:
            text: Seslendirilecek metin
            voice: Kullanılacak ses
            
        Returns:
            Oluşturulan ses dosyasının yolu
        """
        try:
            # Metni optimize et
            optimized_text = self.optimize_text_for_tts(text)
            
            # Geçici ses dosyası oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                temp_path = tmp_file.name
                self.temp_files.append(temp_path)
            
            # Edge TTS ile ses oluştur
            communicate = edge_tts.Communicate(optimized_text, voice)
            await communicate.save(temp_path)
            
            print(f"🔊 TTS Başarılı: {len(optimized_text)} karakter seslendirildi")
            return temp_path
            
        except Exception as e:
            print(f"❌ TTS Hatası: {e}")
            return ""
    
    async def speech_to_speech(self, audio_file_path: str, rag_response: str, 
                             voice: str = "tr-TR-EmelNeural", language: str = "tr") -> Dict[str, Any]:
        """
        Tam Speech-to-Speech işlemi
        
        Args:
            audio_file_path: Giriş ses dosyası
            rag_response: RAG sisteminden gelen cevap metni
            voice: TTS için kullanılacak ses
            language: STT için dil
            
        Returns:
            İşlem sonucu
        """
        try:
            # 1. STT: Ses → Metin
            recognized_text = self.speech_to_text(audio_file_path, language)
            
            if not recognized_text:
                return {
                    "success": False,
                    "error": "Ses tanınamadı",
                    "recognized_text": "",
                    "audio_path": ""
                }
            
            # 2. TTS: RAG Cevabı → Ses
            audio_path = await self.text_to_speech(rag_response, voice)
            
            if not audio_path:
                return {
                    "success": False,
                    "error": "TTS oluşturulamadı", 
                    "recognized_text": recognized_text,
                    "audio_path": ""
                }
            
            return {
                "success": True,
                "recognized_text": recognized_text,
                "rag_response": rag_response,
                "audio_path": audio_path,
                "voice_used": voice
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recognized_text": "",
                "audio_path": ""
            }
    
    def cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()
    
    def get_voice_by_gender(self, gender: str) -> str:
        """
        Gender parametresinden voice name döndür
        
        Args:
            gender: "male", "female", "erkek", "kadın" veya direkt voice name
            
        Returns:
            Voice name (tr-TR-EmelNeural formatında)
        """
        gender_lower = gender.lower().strip()
        
        # Önce gender mapping'e bak
        if gender_lower in self.gender_voices:
            return self.gender_voices[gender_lower]
        
        # Sonra voice name mapping'e bak
        if gender_lower in self.turkish_voices:
            return self.turkish_voices[gender_lower]
        
        # Direkt voice name mi? (tr-TR-... formatında)
        if gender.startswith("tr-TR-"):
            return gender
        
        # Hiçbiri değilse varsayılan kadın ses
        return "tr-TR-EmelNeural"
    
    def get_available_voices(self) -> Dict[str, str]:
        """Mevcut Türkçe sesleri döndür"""
        return self.turkish_voices.copy()
    
    def get_gender_options(self) -> Dict[str, str]:
        """Basit gender seçenekleri döndür"""
        return {
            "female": "Kadın ses (Emel - varsayılan)",
            "male": "Erkek ses (Ahmet)",
            "kadın": "Kadın ses (Emel)",
            "erkek": "Erkek ses (Ahmet)"
        } 