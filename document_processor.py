import os
import re
import pypdf
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import MAX_CHARS_PER_CHUNK
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
import asyncio

class SemanticDocumentProcessor:
    """Advanced semantic document processor with content-aware chunking"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHARS_PER_CHUNK,
            chunk_overlap=300,  # Increased overlap for better continuity
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
        )
        
        # Semantic patterns for Turkish documents
        self.structure_patterns = {
            'section': [
                r'BÖLÜM\s+([IVX]+|\d+)(?:\s*[-–—]\s*(.+?))?',
                r'KISIM\s+([IVX]+|\d+)(?:\s*[-–—]\s*(.+?))?',
                r'CHAPTER\s+(\d+)(?:\s*[-–—]\s*(.+?))?'
            ],
            'article': [
                r'MADDE\s+(\d+)(?:\s*[-–—]\s*(.+?))?',
                r'Article\s+(\d+)(?:\s*[-–—]\s*(.+?))?',
                r'Madde\s+(\d+)(?:\s*[-–—]\s*(.+?))?'
            ],
            'subsection': [
                r'\((\d+)\)',  # (1), (2), etc.
                r'([a-z])\)',  # a), b), etc.
                r'(\d+)\)',    # 1), 2), etc.
            ],
            'list_item': [
                r'[-•▪▫◦▸▹▪]\s+',  # Bullet points
                r'\d+\.\s+',       # Numbered lists
                r'[a-z]\.\s+',     # Letter lists
            ]
        }
    
    def detect_document_structure(self, text: str) -> Dict[str, List[Tuple[int, str]]]:
        """Detect hierarchical structure in document"""
        structure = {
            'sections': [],
            'articles': [],
            'subsections': [],
            'lists': []
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Check for sections
            for pattern in self.structure_patterns['section']:
                match = re.match(pattern, line_clean, re.IGNORECASE)
                if match:
                    structure['sections'].append((i, line_clean))
                    break
            
            # Check for articles
            for pattern in self.structure_patterns['article']:
                match = re.match(pattern, line_clean, re.IGNORECASE)
                if match:
                    structure['articles'].append((i, line_clean))
                    break
            
            # Check for subsections
            for pattern in self.structure_patterns['subsection']:
                match = re.match(pattern, line_clean)
                if match:
                    structure['subsections'].append((i, line_clean))
                    break
            
            # Check for lists
            for pattern in self.structure_patterns['list_item']:
                match = re.match(pattern, line_clean)
                if match:
                    structure['lists'].append((i, line_clean))
                    break
        
        return structure
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.embedding_model:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def create_semantic_chunks(self, text: str, structure: Dict) -> List[Dict]:
        """Create semantically coherent chunks based on document structure"""
        chunks = []
        
        # If document has clear structure, use it
        if structure['articles'] or structure['sections']:
            chunks.extend(self._create_structured_chunks(text, structure))
        else:
            # Fallback to content-based semantic chunking
            chunks.extend(self._create_content_based_chunks(text))
        
        return chunks
    
    def _create_structured_chunks(self, text: str, structure: Dict) -> List[Dict]:
        """Create chunks based on detected document structure"""
        chunks = []
        lines = text.split('\n')
        
        # Smart primary structure selection
        sections = structure['sections']
        articles = structure['articles']
        
        # If we have many articles but few sections, prefer articles
        if len(articles) > len(sections) * 3:
            primary_markers = articles
            structure_type = 'article'
        elif sections:
            primary_markers = sections
            structure_type = 'section'
        elif articles:
            primary_markers = articles
            structure_type = 'article'
        else:
            return self._create_content_based_chunks(text)
        
        print(f"📊 Using {structure_type} markers: {len(primary_markers)} items")
        
        for i, (line_idx, marker_text) in enumerate(primary_markers):
            # Determine chunk boundaries
            start_line = line_idx
            
            # Find end of this section/article
            if i + 1 < len(primary_markers):
                end_line = primary_markers[i + 1][0]
            else:
                end_line = len(lines)
            
            # Extract content
            section_lines = lines[start_line:end_line]
            section_text = '\n'.join(section_lines).strip()
            
            if not section_text:
                continue
            
            # If section is too large, split it further
            if len(section_text) > MAX_CHARS_PER_CHUNK * 1.5:
                sub_chunks = self._split_large_section(section_text, marker_text)
                chunks.extend(sub_chunks)
                print(f"📄 Split large {structure_type}: {marker_text[:50]}... into {len(sub_chunks)} sub-chunks")
            else:
                chunks.append({
                    'content': section_text,
                    'structure_type': structure_type,
                    'title': marker_text,
                    'hierarchy_level': 1
                })
        
        return chunks
    
    def _split_large_section(self, text: str, title: str) -> List[Dict]:
        """Split large sections into smaller semantic chunks with multiple strategies"""
        chunks = []
        
        # Strategy 1: Try paragraph-based splitting first
        paragraphs = text.split('\n\n')
        
        # If we have many paragraphs, use paragraph-based splitting
        if len(paragraphs) > 3:
            current_chunk = ""
            chunk_count = 1
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if adding this paragraph would exceed limit
                potential_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
                
                if len(potential_chunk) > MAX_CHARS_PER_CHUNK and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'content': current_chunk.strip(),
                        'structure_type': 'subsection',
                        'title': f"{title} (Bölüm {chunk_count})",
                        'hierarchy_level': 2
                    })
                    
                    current_chunk = paragraph
                    chunk_count += 1
                else:
                    current_chunk = potential_chunk
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'structure_type': 'subsection', 
                    'title': f"{title} (Bölüm {chunk_count})",
                    'hierarchy_level': 2
                })
        
        # Strategy 2: If paragraphs didn't work well, try sentence-based splitting
        if not chunks or any(len(chunk['content']) > MAX_CHARS_PER_CHUNK * 1.2 for chunk in chunks):
            print(f"⚠️ Paragraph splitting failed for {title[:50]}..., trying sentence-based")
            chunks = []
            
            # Split by sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = ""
            chunk_count = 1
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed limit
                potential_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
                
                if len(potential_chunk) > MAX_CHARS_PER_CHUNK and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'content': current_chunk.strip(),
                        'structure_type': 'subsection',
                        'title': f"{title} (Bölüm {chunk_count})",
                        'hierarchy_level': 2
                    })
                    
                    current_chunk = sentence
                    chunk_count += 1
                else:
                    current_chunk = potential_chunk
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'structure_type': 'subsection',
                    'title': f"{title} (Bölüm {chunk_count})",
                    'hierarchy_level': 2
                })
        
        # Strategy 3: If sentence splitting still creates large chunks, use RecursiveCharacterTextSplitter
        if not chunks or any(len(chunk['content']) > MAX_CHARS_PER_CHUNK * 1.2 for chunk in chunks):
            print(f"⚠️ Sentence splitting failed for {title[:50]}..., using character-based splitting")
            
            fallback_chunks = self.text_splitter.split_text(text)
            chunks = []
            
            for i, chunk_text in enumerate(fallback_chunks):
                chunks.append({
                    'content': chunk_text,
                    'structure_type': 'subsection',
                    'title': f"{title} (Bölüm {i+1})",
                    'hierarchy_level': 2
                })
        
        return chunks
    
    def _create_content_based_chunks(self, text: str) -> List[Dict]:
        """Create chunks based on content similarity and natural boundaries"""
        # Use RecursiveCharacterTextSplitter as base
        base_chunks = self.text_splitter.split_text(text)
        
        if not base_chunks:
            return []
        
        # If we have an embedding model, merge similar adjacent chunks
        if self.embedding_model and len(base_chunks) > 1:
            merged_chunks = self._merge_similar_chunks(base_chunks)
        else:
            merged_chunks = base_chunks
        
        # Convert to structured format
        structured_chunks = []
        for i, chunk in enumerate(merged_chunks):
            # Try to extract a meaningful title from the chunk
            title = self._extract_chunk_title(chunk)
            
            structured_chunks.append({
                'content': chunk,
                'structure_type': 'content',
                'title': title or f"Tam Metin (Bölüm {i+1})",
                'hierarchy_level': 1
            })
        
        return structured_chunks
    
    def _merge_similar_chunks(self, chunks: List[str], similarity_threshold: float = 0.6) -> List[str]:
        """Merge adjacent chunks that are semantically similar"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # Calculate similarity between current and next chunk
            similarity = self.semantic_similarity(current_chunk[-500:], next_chunk[:500])
            
            # Merge if similar and combined size is reasonable
            combined_size = len(current_chunk) + len(next_chunk)
            
            if similarity > similarity_threshold and combined_size <= MAX_CHARS_PER_CHUNK * 1.3:
                current_chunk = current_chunk + '\n\n' + next_chunk
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        # Add the last chunk
        merged.append(current_chunk)
        return merged
    
    def _extract_chunk_title(self, chunk: str) -> str:
        """Extract a meaningful title from chunk content"""
        lines = chunk.split('\n')
        
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if not line:
                continue
            
            # Check if it looks like a title (short, capitalized, etc.)
            if (len(line) < 100 and 
                (line.isupper() or 
                 line.startswith(('MADDE', 'BÖLÜM', 'KISIM', 'Article', 'Chapter')) or
                 re.match(r'^[A-Z][a-z].*[^.]$', line))):
                return line
        
        # Fallback: use first meaningful sentence
        sentences = re.split(r'[.!?]', chunk)
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            if 20 <= len(sentence) <= 100:
                return sentence[:80] + "..." if len(sentence) > 80 else sentence
        
        return None

def _process_single_file_semantic(file_path, processor):
    """Process a single file with semantic awareness"""
    filename = os.path.basename(file_path)
    docs = []
    
    try:
        reader = pypdf.PdfReader(file_path)
        full_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text
        
        if not full_text.strip():
            print(f"Warning: No text extracted from {filename}, using filename as content")
            full_text = f"Document: {filename}\nContent could not be extracted from this PDF file."

        # Detect document structure
        structure = processor.detect_document_structure(full_text)
        
        # Create semantic chunks
        semantic_chunks = processor.create_semantic_chunks(full_text, structure)
        
        # Convert to standard format
        for chunk_data in semantic_chunks:
            docs.append({
                "source": filename,
                "article": chunk_data['title'],
                "content": chunk_data['content'],
                "structure_type": chunk_data['structure_type'],
                "hierarchy_level": chunk_data['hierarchy_level']
            })
        
        print(f"📄 {filename}: {len(docs)} semantic chunks, structure: {len(structure['articles'])} articles, {len(structure['sections'])} sections")
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        # Fallback chunk
        docs.append({
            "source": filename,
            "article": "Error Recovery",
            "content": f"Document: {filename}\nProcessing error: {str(e)}",
            "structure_type": "error",
            "hierarchy_level": 0
        })
    
    return docs

def load_and_process_documents(directory_path, specific_files=None, embedding_model=None):
    """Load and process documents with semantic awareness"""
    all_docs = []
    
    if not os.path.exists(directory_path):
        return []
    
    # Initialize semantic processor
    processor = SemanticDocumentProcessor(embedding_model)
    
    filenames_to_process = specific_files if specific_files is not None else [
        f for f in os.listdir(directory_path) if f.endswith(".pdf")
    ]
    
    if not filenames_to_process:
        return []

    file_paths = [os.path.join(directory_path, f) for f in filenames_to_process]
    
    print(f"🧠 Processing {len(file_paths)} documents with semantic chunking...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(_process_single_file_semantic, path, processor): path 
            for path in file_paths
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), 
                          total=len(file_paths), desc="Semantic Processing"):
            all_docs.extend(future.result())
            
    print(f"✅ Semantic processing complete: {len(all_docs)} total chunks")
    return all_docs 