#!/usr/bin/env python3
"""
Document Viewer Tool
Ankra Bilim Üniversitesi DocBoss sistemi için dokuman görüntüleme aracı
"""

import requests
import json
from typing import Optional, List, Dict
import sys
from datetime import datetime

class DocumentViewer:
    def __init__(self, base_url: str = "http://localhost:8000", admin_email: str = "ankarabilim@edu.tr", admin_password: str = "ankarabilim"):
        self.base_url = base_url
        self.admin_email = admin_email
        self.admin_password = admin_password
        self.access_token = None
        
    def login(self) -> bool:
        """Admin olarak giriş yap"""
        try:
            response = requests.post(
                f"{self.base_url}/api/login",
                json={"email": self.admin_email, "password": self.admin_password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                print(f"✅ Başarıyla giriş yapıldı: {self.admin_email}")
                return True
            else:
                print(f"❌ Giriş başarısız: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Giriş hatası: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Authorization header'ı döndür"""
        if not self.access_token:
            raise Exception("Önce giriş yapmalısınız!")
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def list_documents(self) -> List[Dict]:
        """Tüm dokumanları listele"""
        try:
            response = requests.get(
                f"{self.base_url}/api/documents",
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Dokuman listesi alınamadı: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Dokuman listesi hatası: {e}")
            return []
    
    def get_document_content(self, document_id: str) -> Optional[Dict]:
        """Belirli bir dokuman içeriğini al"""
        try:
            response = requests.get(
                f"{self.base_url}/api/documents/{document_id}/content",
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"❌ Dokuman bulunamadı: {document_id}")
                return None
            else:
                print(f"❌ Dokuman içeriği alınamadı: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Dokuman içerik hatası: {e}")
            return None
    
    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """Belirli bir dokuman hakkında detaylı bilgi al"""
        try:
            response = requests.get(
                f"{self.base_url}/api/documents/{document_id}/info",
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"❌ Dokuman bulunamadı: {document_id}")
                return None
            else:
                print(f"❌ Dokuman bilgisi alınamadı: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Dokuman bilgi hatası: {e}")
            return None
    
    def show_documents_menu(self):
        """Dokumanları interaktif menü ile göster"""
        print("\n" + "="*50)
        print("📚 DOKUMAN GÖRÜNTÜLEYICI")
        print("="*50)
        
        # Dokumanları listele
        documents = self.list_documents()
        
        if not documents:
            print("❌ Hiç dokuman bulunamadı!")
            return
        
        print(f"\n📋 Toplam {len(documents)} dokuman bulundu:\n")
        
        for i, doc in enumerate(documents, 1):
            created_date = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
            print(f"{i:2d}. {doc['file_name']}")
            print(f"    📁 ID: {doc['id']}")
            print(f"    📊 Boyut: {self._format_file_size(doc['file_size'])}")
            print(f"    🗓️  Tarih: {created_date.strftime('%d.%m.%Y %H:%M')}")
            print(f"    📄 Chunk sayısı: {doc['chunks_count']}")
            print(f"    ⚡ Durum: {doc['status']}")
            print()
        
        # Kullanıcıdan seçim al
        while True:
            try:
                choice = input(f"Hangi dokumanı incelemek istiyorsunuz? (1-{len(documents)}, 'q' çıkış): ")
                
                if choice.lower() == 'q':
                    break
                    
                index = int(choice) - 1
                if 0 <= index < len(documents):
                    self.show_document_content(documents[index])
                    break
                else:
                    print(f"❌ Geçersiz seçim! 1-{len(documents)} arası bir sayı girin.")
                    
            except ValueError:
                print("❌ Geçersiz giriş! Sayı girin veya 'q' ile çıkın.")
            except KeyboardInterrupt:
                print("\n\n👋 Çıkış yapılıyor...")
                break
    
    def show_document_content(self, doc_info: Dict):
        """Seçilen dokumanın içeriğini göster"""
        doc_id = doc_info['id']
        
        print(f"\n{'='*60}")
        print(f"📖 DOKUMAN: {doc_info['file_name']}")
        print(f"{'='*60}")
        
        # Detaylı bilgileri al
        details = self.get_document_info(doc_id)
        if details:
            print(f"📊 Detaylar:")
            print(f"   • Boyut: {self._format_file_size(details['file_size'])}")
            print(f"   • MongoDB Chunk'ları: {details['chunks_count']}")
            print(f"   • Weaviate Chunk'ları: {details['weaviate_chunks']}")
            print(f"   • Durum: {details['status']}")
            print()
        
        # İçeriği al
        content = self.get_document_content(doc_id)
        if not content:
            print("❌ Dokuman içeriği alınamadı!")
            return
        
        print(f"📄 DOKUMAN İÇERİĞİ:")
        print(f"   • Toplam {content['total_chunks']} chunk'tan birleştirildi")
        print(f"   • Toplam karakter sayısı: {len(content['full_content']):,}")
        print(f"\n{'-'*60}")
        print("📝 İÇERİK:")
        print(f"{'-'*60}\n")
        
        # Tam içeriği göster
        full_text = content['full_content']
        
        # Çok uzun metinler için sayfalama
        lines = full_text.split('\n')
        lines_per_page = 40  # Her sayfada 40 satır
        
        if len(lines) <= lines_per_page:
            # Kısa doküman - direkt göster
            print(full_text)
        else:
            # Uzun doküman - sayfalama ile göster
            total_pages = (len(lines) + lines_per_page - 1) // lines_per_page
            current_page = 1
            
            while current_page <= total_pages:
                print(f"\n--- SAYFA {current_page}/{total_pages} ---\n")
                
                start_line = (current_page - 1) * lines_per_page
                end_line = min(start_line + lines_per_page, len(lines))
                
                for line in lines[start_line:end_line]:
                    print(line)
                
                if current_page < total_pages:
                    choice = input(f"\n🔄 Devam etmek için Enter, çıkmak için 'q': ")
                    if choice.lower() == 'q':
                        break
                    current_page += 1
                else:
                    break
        
        print(f"\n{'-'*60}")
        print("✅ Dokuman görüntüleme tamamlandı!")
        input("\nAna menüye dönmek için Enter'a basın...")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Dosya boyutunu okunabilir formatta göster"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

def main():
    """Ana fonksiyon"""
    print("🚀 DocBoss Dokuman Görüntüleyici başlatılıyor...")
    
    # Dokuman görüntüleyiciyi başlat
    viewer = DocumentViewer()
    
    # Giriş yap
    if not viewer.login():
        sys.exit(1)
    
    try:
        # Ana menüyü göster
        viewer.show_documents_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main() 