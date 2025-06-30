import os
import re
import pypdf
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import MAX_CHARS_PER_CHUNK
import concurrent.futures

def _process_single_file(file_path, text_splitter):
    filename = os.path.basename(file_path)
    docs = []
    full_text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            full_text += page.extract_text() or ""
        
        if not full_text.strip():
            print(f"Warning: No text extracted from {filename}, using filename as content")
            full_text = f"Document: {filename}\nContent could not be extracted from this PDF file."

        articles = re.split(r'(MADDE\s+\d+\s*–)', full_text)
        if len(articles) > 1:
            for i in range(1, len(articles), 2):
                article_title = articles[i].strip().replace("–", "").strip()
                article_content = articles[i+1].strip()
                if not article_content: continue
                document_text = f"{article_title}\n{article_content}"
                if len(document_text) <= MAX_CHARS_PER_CHUNK:
                    docs.append({"source": filename, "article": article_title, "content": document_text})
                else:
                    chunks = text_splitter.split_text(document_text)
                    for chunk_index, chunk in enumerate(chunks):
                        docs.append({"source": filename, "article": f"{article_title} (Bölüm {chunk_index+1})", "content": chunk})
        else:
            if len(full_text) > MAX_CHARS_PER_CHUNK:
                chunks = text_splitter.split_text(full_text)
                for chunk_index, chunk in enumerate(chunks):
                    docs.append({"source": filename, "article": f"Tam Metin (Bölüm {chunk_index+1})", "content": chunk})
            else:
                docs.append({"source": filename, "article": "Tam Metin", "content": full_text})
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
    return docs

def load_and_process_documents(directory_path, specific_files=None):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS_PER_CHUNK, chunk_overlap=200)
    if not os.path.exists(directory_path): return []
    
    filenames_to_process = specific_files if specific_files is not None else [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    if not filenames_to_process: return []

    file_paths = [os.path.join(directory_path, f) for f in filenames_to_process]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(_process_single_file, path, text_splitter): path for path in file_paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(file_paths), desc="Processing Documents"):
            all_docs.extend(future.result())
            
    return all_docs 