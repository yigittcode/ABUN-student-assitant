#!/usr/bin/env python3
"""
🎓 Akıllı Transcript Danışman Modülü v2.0
Modern AI tabanlı transcript analizi ve öğrenci danışmanlığı sistemi
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import asyncio
from datetime import datetime

# --- Ders kodu normalize fonksiyonu ---
def normalize_code(code: str) -> str:
    return code.replace(' ', '').replace('_', '').upper()

# --- Not normalize fonksiyonu ---
def normalize_grade(grade: str) -> str:
    return grade.replace(' ', '').upper()

@dataclass
class CourseRecord:
    """Ders kaydı"""
    code: str
    name: str
    credit: int
    grade: str
    ects: int
    semester: str
    grade_point: float = 0.0
    
    def __post_init__(self):
        self.grade_point = self._calculate_grade_point()
    
    def _calculate_grade_point(self) -> float:
        """Harf notunu 4.0 sistemine çevir"""
        grade_map = {
            'AA': 4.0, 'BA': 3.5, 'BB': 3.0, 'CB': 2.5,
            'CC': 2.0, 'DC': 1.5, 'DD': 1.0, 'FF': 0.0,
            'U': 0.0, 'W': 0.0, 'I': 0.0, 'P': 0.0
        }
        return grade_map.get(self.grade, 0.0)

@dataclass
class StudentProfile:
    """Öğrenci profili"""
    name: str
    student_no: str
    faculty: str
    department: str
    enrollment_date: str
    current_gpa: float
    total_credits: int
    total_ects: int
    completed_courses: int
    failed_courses: int
    average_grade: float
    academic_standing: str
    prep_status: str = "Bilinmeyen"  # Hazırlık durumu
    prep_grade: int = 0  # Hazırlık notu
    
    def __post_init__(self):
        self.academic_standing = self._determine_academic_standing()
    
    def _determine_academic_standing(self) -> str:
        """Akademik durumu belirle"""
        if self.current_gpa >= 3.5:
            return "🏆 Mükemmel"
        elif self.current_gpa >= 3.0:
            return "✅ İyi"
        elif self.current_gpa >= 2.5:
            return "⚠️ Orta"
        elif self.current_gpa >= 2.0:
            return "🔻 Düşük"
        else:
            return "❌ Kritik"

@dataclass
class AdvisorResponse:
    """Danışman yanıtı"""
    message: str
    analysis: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    resources: Optional[List[Dict]] = None
    urgency_level: str = "normal"
    
    def format_response(self) -> str:
        """Yanıtı güzel formatlı şekilde hazırla"""
        response = f"{self.message.strip()}\n"
        
        if self.recommendations:
            response += "\nÖneriler:\n" + "\n".join(f"- {rec}" for rec in self.recommendations)
        
        if self.next_steps:
            response += "\nSonraki Adımlar:\n" + "\n".join(f"- {step}" for step in self.next_steps)
        
        if self.resources:
            response += "\nKaynaklar:\n" + "\n".join(f"- {resource.get('title', 'Kaynak')}: {resource.get('description', '')}" for resource in self.resources)
        
        return response.strip()

class IntelligentTranscriptAdvisor:
    """Akıllı Transcript Danışman v2.0"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.conversation_history = []
        self.current_student = None
        self.current_courses = []
        
        # Genel bilgi veritabanı
        self.knowledge_base = {
            'graduation_requirements': {
                'min_gpa': 2.0,
                'min_credits': 120,
                'min_ects': 180,
                'min_courses': 35,
                'core_courses': ['CENG_101', 'CENG_102', 'CENG_201', 'CENG_202', 'CENG_301', 'CENG_302', 'CENG_401'],
                'math_requirements': ['MATH_101', 'MATH_102', 'MATH_201'],
                'physics_requirements': ['PHYS_101', 'PHYS_102'],
                'english_requirements': ['ENG_101', 'ENG_102'],
                'mandatory_courses': ['HIST_101', 'HIST_102', 'TURK_101', 'TURK_102']
            },
            'course_categories': {
                'OC': {'name': 'Operational Course', 'required': 2, 'description': 'Fakülte zorunlu seçmeli dersler'},
                'CMM': {'name': 'İletişim ve Medya', 'required': 2, 'description': 'İletişim becerileri dersleri'},
                'CENG': {'name': 'Bilgisayar Mühendisliği', 'required': 25, 'description': 'Ana branş dersleri'},
                'MATH': {'name': 'Matematik', 'required': 4, 'description': 'Matematik dersleri'},
                'PHYS': {'name': 'Fizik', 'required': 2, 'description': 'Fizik dersleri'},
                'ENG': {'name': 'İngilizce', 'required': 2, 'description': 'İngilizce dersleri'}
            },
            'grade_info': {
                'passing_grades': ['AA', 'BA', 'BB', 'CB', 'CC', 'DC', 'DD', 'S', 'P'],
                'failing_grades': ['FF', 'U'],
                'incomplete_grades': ['I', 'W'],
                'satisfactory_grades': ['P'],
                'grade_meanings': {
                    'AA': 'Mükemmel (4.0)',
                    'BA': 'Pek İyi (3.5)',
                    'BB': 'İyi (3.0)',
                    'CB': 'Orta-İyi (2.5)',
                    'CC': 'Orta (2.0)',
                    'DC': 'Geçer-Orta (1.5)',
                    'DD': 'Geçer (1.0)',
                    'FF': 'Başarısız (0.0)',
                    'U': 'Devamsızlık (0.0)',
                    'I': 'Eksik',
                    'W': 'Çekildi',
                    'P': 'Geçti',
                    'S': 'Geçti'
                }
            }
        }
        
        # Seçmeli ders önerileri
        self.elective_recommendations = {
            'web_development': {
                'title': '🌐 Web Geliştirme',
                'description': 'Modern web teknolojileri üzerine odaklanma',
                'courses': ['HTML/CSS', 'JavaScript', 'React', 'Node.js', 'Database Systems'],
                'career_paths': ['Frontend Developer', 'Backend Developer', 'Full Stack Developer'],
                'market_demand': 'Yüksek'
            },
            'ai_ml': {
                'title': '🤖 Yapay Zeka ve Makine Öğrenmesi',
                'description': 'AI ve ML teknolojileri üzerine uzmanlaşma',
                'courses': ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Computer Vision'],
                'career_paths': ['AI Engineer', 'Data Scientist', 'ML Engineer'],
                'market_demand': 'Çok Yüksek'
            },
            'mobile_development': {
                'title': '📱 Mobil Uygulama Geliştirme',
                'description': 'iOS ve Android platformları için uygulama geliştirme',
                'courses': ['Android Development', 'iOS Development', 'React Native', 'Flutter'],
                'career_paths': ['Mobile Developer', 'iOS Developer', 'Android Developer'],
                'market_demand': 'Yüksek'
            },
            'cybersecurity': {
                'title': '🔒 Siber Güvenlik',
                'description': 'Bilgi güvenliği ve siber güvenlik uzmanlaşması',
                'courses': ['Network Security', 'Cryptography', 'Ethical Hacking', 'Security Analysis'],
                'career_paths': ['Security Analyst', 'Cybersecurity Engineer', 'Penetration Tester'],
                'market_demand': 'Çok Yüksek'
            },
            'data_science': {
                'title': '📊 Veri Bilimi',
                'description': 'Büyük veri analizi ve istatistiksel modelleme',
                'courses': ['Data Mining', 'Statistics', 'Big Data', 'Data Visualization'],
                'career_paths': ['Data Scientist', 'Data Analyst', 'Business Intelligence Developer'],
                'market_demand': 'Çok Yüksek'
            }
        }

    def parse_transcript(self, transcript_text: str) -> Tuple[StudentProfile, List[CourseRecord]]:
        """Transcript'i parse et ve analiz et"""
        try:
            # Öğrenci bilgilerini çıkar
            student_info = self._extract_student_info(transcript_text)
            
            # Ders kayıtlarını çıkar
            courses = self._extract_course_records(transcript_text)
            
            # Öğrenci profilini oluştur
            profile = self._create_student_profile(student_info, courses)
            
            # Global değişkenleri güncelle
            self.current_student = profile
            self.current_courses = courses
            
            return profile, courses
            
        except Exception as e:
            raise Exception(f"Transcript parse hatası: {str(e)}")

    def _extract_student_info(self, text: str) -> Dict:
        """Öğrenci bilgilerini çıkar - Yeni PDF formatı için güncellendi"""
        info = {}
        
        # Satır satır analiz için metni böl
        lines = text.split('\n')
        
        # İsim - Yeni format için satır bazlı çıkarma
        # İlk olarak doğrudan büyük harflerle yazılmış isim ara (yeni format)
        name_found = False
        for line in lines:
            line = line.strip()
            if re.match(r'^[A-ZÇĞIİÖŞÜ]+\s+[A-ZÇĞIİÖŞÜ]+\s+[A-ZÇĞIİÖŞÜ]+$', line):
                info['name'] = line
                name_found = True
                break
        
        if not name_found:
            # Alternatif: Geleneksel regex
            name_match = re.search(r'(?:Ad[ı]?\s*Soyad[ı]?|Name)\s*[:\-]?\s*([A-ZÇĞIİÖŞÜ\s]+)', text, re.IGNORECASE)
            if name_match:
                info['name'] = name_match.group(1).strip()
            else:
                info['name'] = "Bilinmeyen"
        
        # Öğrenci numarası - Yeni format için güncellenmiş
        # İlk olarak doğrudan 9 haneli numara ara
        no_found = False
        for line in lines:
            line = line.strip()
            if re.match(r'^\d{9}$', line):
                info['student_no'] = line
                no_found = True
                break
        
        if not no_found:
            # Alternatif: Geleneksel regex
            no_match = re.search(r'(?:Öğrenci\s*No|Student\s*No|No)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
            if no_match:
                info['student_no'] = no_match.group(1)
            else:
                info['student_no'] = "Bilinmeyen"
        
        # Fakülte - Yeni format için güncellenmiş
        # İlk olarak "Fakültesi" ile biten satırları ara
        faculty_found = False
        for line in lines:
            line = line.strip()
            if 'Fakültesi' in line and len(line) > 10:
                info['faculty'] = line
                faculty_found = True
                break
        
        if not faculty_found:
            # Alternatif: Geleneksel regex
            faculty_match = re.search(r'(?:Fakülte|Faculty)\s*[:\-]?\s*([A-ZÇĞIİÖŞÜa-zçğıiöşü\s]+)', text, re.IGNORECASE)
            if faculty_match:
                info['faculty'] = faculty_match.group(1).strip()
            else:
                info['faculty'] = "Mühendislik ve Mimarlık Fakültesi"
        
        # Bölüm - Yeni format için güncellenmiş
        # İlk olarak "Mühendisliği" ile biten satırları ara
        dept_found = False
        for line in lines:
            line = line.strip()
            if 'Mühendisliği' in line and '(' in line and ')' in line:
                info['department'] = line
                dept_found = True
                break
        
        if not dept_found:
            # Alternatif: Geleneksel regex
            dept_match = re.search(r'(?:Bölüm|Department)\s*[:\-]?\s*([A-ZÇĞIİÖŞÜa-zçğıiöşü\s\(\)]+)', text, re.IGNORECASE)
            if dept_match:
                info['department'] = dept_match.group(1).strip()
            else:
                info['department'] = "Bilgisayar Mühendisliği"
        
        # Kayıt tarihi - Yeni format için güncellenmiş
        date_match = re.search(r'(?:Kayıt\s*Tarihi|Registration\s*Date)\s*[:\-]?\s*(\d{2}[./-]\d{2}[./-]\d{4})', text, re.IGNORECASE)
        if not date_match:
            # Alternatif: Tarih formatında olan satırları ara
            for line in lines:
                line = line.strip()
                if re.match(r'^\d{2}\.\d{2}\.\d{4}$', line):
                    info['enrollment_date'] = line
                    break
            if 'enrollment_date' not in info:
                info['enrollment_date'] = "Bilinmeyen"
        else:
            info['enrollment_date'] = date_match.group(1)
        
        # Hazırlık durumu ve notu (yeni alan)
        prep_match = re.search(r'Hazırlık\s*Durumu:\s*([A-ZÇĞIİÖŞÜa-zçğıiöşü\s]+)\s*Hazırlık\s*Notu:\s*(\d+)', text, re.IGNORECASE)
        if prep_match:
            info['prep_status'] = prep_match.group(1).strip()
            info['prep_grade'] = int(prep_match.group(2))
        else:
            info['prep_status'] = "Bilinmeyen"
            info['prep_grade'] = 0
        
        # GPA/AGNO/ANO doğrudan transcript'ten okunmaya çalışılır
        gpa_regex = re.compile(r'(?:AGNO|ANO|GPA|Genel Not Ortalaması)[^\d\n]*(\d{1,2}[\.,]\d{1,3})', re.IGNORECASE)
        gpa_found = None
        for line in lines:
            match = gpa_regex.search(line)
            if match:
                gpa_found = match.group(1).replace(',', '.')
            else:
                # Satırda birden fazla sayı varsa, son ondalıklı sayıyı al
                numbers = re.findall(r'(\d{1,2}[\.,]\d{1,3})', line)
                if any(key in line for key in ['AGNO', 'ANO', 'GPA', 'Genel Not Ortalaması']) and numbers:
                    gpa_found = numbers[-1].replace(',', '.')
        if gpa_found:
            try:
                info['gpa_from_transcript'] = float(gpa_found)
            except Exception:
                info['gpa_from_transcript'] = None
        else:
            info['gpa_from_transcript'] = None
        
        return info

    def _extract_course_records(self, text: str) -> List[CourseRecord]:
        """Ders kayıtlarını çıkar - Yeni PDF formatı için güncellendi"""
        courses = []
        
        # Yeni PDF formatı için regex
        # Format: DERS_KODU  DERS_ADI KREDI HARF_NOTU AKTS T_AKTS
        course_pattern = r'([A-Z]{2,6}\s+\d{3})\s+([A-ZÇĞIİÖŞÜa-zçğıiöşü\s\-\.\(\)&,\/]+?)\s+(\d+)\s+([A-Z]{1,2})\s*(\d+)\s*(\d+)(?:\s*,\s*(\d+))?'
        
        # Dönem bilgilerini de çıkar
        semester_pattern = r'(\d{4}-\d{4})\s*([A-ZÇĞIİÖŞÜa-zçğıiöşü\s]*(?:Dönem|Dönemi|Güz|Bahar))'
        
        # Metni dönemlere böl
        semester_matches = list(re.finditer(semester_pattern, text, re.IGNORECASE))
        
        # Her dönem için dersleri çıkar
        for i, semester_match in enumerate(semester_matches):
            semester_year = semester_match.group(1)
            semester_term = semester_match.group(2).strip()
            
            # Bu dönemin başlangıcı ve bitişi
            start_pos = semester_match.end()
            end_pos = len(text)
            
            if i + 1 < len(semester_matches):
                end_pos = semester_matches[i + 1].start()
            
            semester_text = text[start_pos:end_pos]
            
            # Bu dönemdeki dersleri bul
            course_matches = re.finditer(course_pattern, semester_text, re.MULTILINE)
            
            for match in course_matches:
                try:
                    code = match.group(1).strip()
                    name = match.group(2).strip()
                    credit = int(match.group(3))
                    grade = match.group(4).strip()
                    
                    # AKTS ve T.AKTS bilgilerini al
                    akts_str = match.group(5)
                    t_akts_str = match.group(6)
                    
                    # Virgülden sonra gelen kısım varsa (örn: 2,51)
                    if ',' in akts_str:
                        akts_parts = akts_str.split(',')
                        akts = int(akts_parts[0])
                        decimal_part = akts_parts[1] if len(akts_parts) > 1 else '0'
                    else:
                        akts = int(akts_str)
                    
                    ects = int(t_akts_str)
                    
                    # Dönem bilgisini birleştir
                    semester = f"{semester_year} {semester_term}"
                    
                    course = CourseRecord(
                        code=code,
                        name=name,
                        credit=credit,
                        grade=grade,
                        ects=ects,
                        semester=semester
                    )
                    courses.append(course)
                    
                except (ValueError, IndexError) as e:
                    print(f"Course parsing error: {e} for match: {match.groups()}")
                    continue
        
        # Eğer dönem bazlı çıkarma başarısız olursa, genel regex kullan
        if not courses:
            print("Dönem bazlı çıkarma başarısız, genel regex deneniyor...")
            matches = re.finditer(course_pattern, text, re.MULTILINE)
            
            for match in matches:
                try:
                    code = match.group(1).strip()
                    name = match.group(2).strip()
                    credit = int(match.group(3))
                    grade = match.group(4).strip()
                    akts = int(match.group(5))
                    ects = int(match.group(6))
                    
                    course = CourseRecord(
                        code=code,
                        name=name,
                        credit=credit,
                        grade=grade,
                        ects=ects,
                        semester="Bilinmeyen"
                    )
                    courses.append(course)
                    
                except (ValueError, IndexError) as e:
                    continue
        
        return courses

    def _create_student_profile(self, student_info: Dict, courses: List[CourseRecord]) -> StudentProfile:
        """Öğrenci profilini oluştur"""
        total_credits = sum(c.credit for c in courses if c.grade in self.knowledge_base['grade_info']['passing_grades'])
        total_ects = sum(c.ects for c in courses if c.grade in self.knowledge_base['grade_info']['passing_grades'])
        completed_courses = len([c for c in courses if c.grade in self.knowledge_base['grade_info']['passing_grades']])
        failed_courses = len([c for c in courses if c.grade in self.knowledge_base['grade_info']['failing_grades']])
        # GPA doğrudan transcript'ten okunmuşsa onu kullan
        if student_info.get('gpa_from_transcript') is not None:
            current_gpa = student_info['gpa_from_transcript']
        else:
            # FF dahil tüm dersler hesaba katılır
            total_points = sum(c.grade_point * c.credit for c in courses)
            total_credit_hours = sum(c.credit for c in courses)
            current_gpa = total_points / total_credit_hours if total_credit_hours > 0 else 0.0
        valid_grades = [c.grade_point for c in courses if c.grade_point > 0]
        average_grade = sum(valid_grades) / len(valid_grades) if valid_grades else 0.0
        if current_gpa >= 3.5:
            academic_standing = "Mükemmel"
        elif current_gpa >= 3.0:
            academic_standing = "İyi"
        elif current_gpa >= 2.5:
            academic_standing = "Orta"
        elif current_gpa >= 2.0:
            academic_standing = "Düşük"
        else:
            academic_standing = "Kritik"
        return StudentProfile(
            name=student_info['name'],
            student_no=student_info['student_no'],
            faculty=student_info['faculty'],
            department=student_info['department'],
            enrollment_date=student_info['enrollment_date'],
            current_gpa=current_gpa,
            total_credits=total_credits,
            total_ects=total_ects,
            completed_courses=completed_courses,
            failed_courses=failed_courses,
            average_grade=average_grade,
            academic_standing=academic_standing,
            prep_status=student_info.get('prep_status', 'Bilinmeyen'),
            prep_grade=student_info.get('prep_grade', 0)
        )

    # --- Soru sorma fonksiyonlarını kaldır ---
    # ask_question ve ilgili fonksiyonlar silinecek (burada gösterilmiyor, koddan kaldırılacak)

    # --- get_comprehensive_analysis fonksiyonunu tamamen AI destekli ve kişisel hale getir ---
    async def get_comprehensive_analysis(self) -> Dict:
        if not self.current_student:
            return {"error": "Transcript analiz edilmemiş"}

        # OC ve CMM derslerini ve notlarını listele
        oc_courses = [(c.code, c.name, c.grade) for c in self.current_courses if normalize_code(c.code).startswith('OC')]
        cmm_courses = []
        for c in self.current_courses:
            if normalize_code(c.code).startswith('CMM'):
                print(f"[DEBUG] CMM dersi bulundu: {c.code} | {c.name} | {c.grade}")
                cmm_courses.append((c.code, c.name, c.grade))

        oc_completed = len([c for c in oc_courses if normalize_grade(c[2]) in self.knowledge_base['grade_info']['passing_grades']])
        cmm_completed = len([c for c in cmm_courses if normalize_grade(c[2]) in self.knowledge_base['grade_info']['passing_grades']])
        oc_required = self.knowledge_base['course_categories']['OC']['required']
        cmm_required = self.knowledge_base['course_categories']['CMM']['required']
        oc_status = 'Tamamlandı ✅' if oc_completed >= oc_required else f'Eksik ({oc_completed}/{oc_required})'
        cmm_status = 'Tamamlandı ✅' if cmm_completed >= cmm_required else f'Eksik ({cmm_completed}/{cmm_required})'

        # Zayıf dersler (notu 2.0'ın altı, S ve P hariç)
        weak_courses = [(c.code, c.name, c.grade, c.grade_point) for c in self.current_courses if c.grade_point < 2.0 and normalize_grade(c.grade) not in ['S', 'P']]

        # Bağımlı ders senaryosu: Bitirme projesi için gerekli dersler
        required_for_graduation = [
            ('CENG 332', 'Computer Architecture'),
            ('CENG 301', 'Operating Systems'),
            ('CENG 351', 'Database Management Systems')
        ]
        taken_codes = [normalize_code(c.code) for c in self.current_courses if normalize_grade(c.grade) in self.knowledge_base['grade_info']['passing_grades']]
        can_take_graduation_project = all(any(normalize_code(code) == tc for tc in taken_codes) for code, _ in required_for_graduation)
        missing_for_graduation = [name for code, name in required_for_graduation if not any(normalize_code(code) == tc for tc in taken_codes)]

        # AI prompt hazırla
        prompt = f"""
Sen bir üniversite transcript danışmanısın. Aşağıda bir öğrencinin profili, aldığı dersler, notları ve akademik durumu var. Lütfen aşağıdaki başlıklarda kişisel, motive edici, gelişim ve yol haritası odaklı, uzun ve detaylı bir analiz ve öneri raporu hazırla:

- Aldığı OC (Operational Course) dersleri ve notları
- Aldığı CMM dersleri ve notları
- OC ve CMM mezuniyet şartı tamamlanmış mı?
- Başarısız dersler (notu 2.0'ın altı, S ve P hariç, yani CC altı) ve her biri için kişisel çalışma/iyileştirme önerileri
- Bitirme projesi için gerekli dersler alınmış mı? Eksikse hangileri?
- Kişisel seçmeli ders önerileri (ve nedenleri, öğrencinin geçmişi ve ilgi alanına göre)
- Aşağıda örnek bir seçmeli ders havuzu oluştur. Tüm seçmeli dersleri DD olarak listele. Sana göre en uygun olanları seçip, nedenlerini belirt ve bunları ayrıca öneri olarak sun.
- Genel motivasyonel ve yol gösterici kapanış, örnek bir yol haritası ve gelişim planı öner.

Yanıtı başlıksız, sadece paragraflar halinde, emoji kullanarak ve kolay okunur şekilde yaz. Hiçbir yerde başlık, markdown, kalın yazı veya özel karakter kullanma. Her bölümü ayrı paragraf olarak döndür: summary, elective_suggestions, weak_course_advice, graduation_status.

Öğrenci Profili:
Adı: {self.current_student.name}
Bölüm: {self.current_student.department}
GPA: {self.current_student.current_gpa:.2f}
Toplam Kredi: {self.current_student.total_credits}
Akademik Durum: {self.current_student.academic_standing}
Hazırlık Durumu: {self.current_student.prep_status} ({self.current_student.prep_grade})

Aldığı OC dersleri: {', '.join([f'{code} {name} ({grade})' for code, name, grade in oc_courses]) or 'Yok'}
Aldığı CMM dersleri: {', '.join([f'{code} {name} ({grade})' for code, name, grade in cmm_courses]) or 'Yok'}
OC mezuniyet şartı: {oc_status}
CMM mezuniyet şartı: {cmm_status}

Başarısız dersler: {chr(10).join([f'- {code} {name} ({grade}, {point:.2f})' for code, name, grade, point in weak_courses]) or 'Yok'}

Bitirme projesi için gerekli dersler: {'Alınmış, başvurabilir.' if can_take_graduation_project else 'Eksik: ' + ', '.join(missing_for_graduation)}

Lütfen tüm analiz ve önerileri kişisel, motive edici ve yol gösterici bir dille yaz. Seçmeli ders önerilerini öğrencinin geçmişi ve ilgi alanına göre gerekçelendir. Her bölümü ayrı paragraf olarak döndür: summary, elective_suggestions, weak_course_advice, graduation_status.
"""
        # AI'dan yanıt al
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sen uzman bir akademik danışmansın. Öğrencilere samimi, destekleyici ve pratik öneriler veriyorsun."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            ai_text = response.choices[0].message.content
            # Her bölümü ayır (her paragraf bir section)
            sections = [s.strip() for s in ai_text.split('\n\n') if s.strip()]
            summary = sections[0] if len(sections) > 0 else ""
            elective_suggestions = sections[1] if len(sections) > 1 else ""
            weak_course_advice = sections[2] if len(sections) > 2 else ""
            graduation_status = sections[3] if len(sections) > 3 else ""
        except Exception as e:
            summary = f"⚠️ AI yanıtı alınamadı: {str(e)}. Lütfen daha sonra tekrar deneyin."
            elective_suggestions = weak_course_advice = graduation_status = ""

        return {
            'student_profile': asdict(self.current_student),
            'oc_courses': oc_courses,
            'cmm_courses': cmm_courses,
            'oc_status': oc_status,
            'cmm_status': cmm_status,
            'weak_courses': weak_courses,
            'can_take_graduation_project': can_take_graduation_project,
            'missing_for_graduation_project': missing_for_graduation,
            'summary': summary,
            'elective_suggestions': elective_suggestions,
            'weak_course_advice': weak_course_advice,
            'graduation_status': graduation_status
        } 