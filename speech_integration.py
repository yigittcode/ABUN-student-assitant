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
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch

class SpeechProcessor:
    """Ses işleme sınıfı - STT ve TTS"""
    
    def __init__(self, whisper_model_name: str = "small"):
        print("🎤 Speech Processor başlatılıyor...")
        self.temp_files: List[str] = []
        
        # Force Whisper to use CPU to prevent CUDA OOM
        print("🤖 Whisper modeli CPU'da yükleniyor...")
        device = "cpu"  # Force CPU usage
        self.whisper_model = whisper.load_model(whisper_model_name, device=device)
        print(f"✅ Whisper {whisper_model_name} modeli CPU'da yüklendi")
        
        # TTS için Azure Speech Service kullanacağız
        self.tts_available = True
        
        # Pygame başlat (TTS için)
        pygame.mixer.init()
        
        # Türkçe ses seçenekleri
        self.turkish_voices = {
            "emel": "tr-TR-EmelNeural",      # Kadın ses
            "ahmet": "tr-TR-AhmetNeural",    # Erkek ses
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
    
    def get_available_voices(self) -> Dict[str, str]:
        """Mevcut Türkçe sesleri döndür"""
        return self.turkish_voices.copy() 