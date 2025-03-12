# utils/pdf_processor.py
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(pdf_bytes):
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    return docs
