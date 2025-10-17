import pdfplumber
from typing import List

def extract_text_from_pdf(path:str)->str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n\n".join(texts)

def chunk_text(text:str,chunk_size:int=1000,overlap:int=200)->List[str]:
    words = text.split()
    chunks = []
    i=0
    while i<len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i+=chunk_size-overlap
    return chunks