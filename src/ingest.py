from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict
from .config import PDF_DIR

def load_pdfs(pdf_dir: str = PDF_DIR) -> List[Dict]:
    docs = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

        docs.append({
            "id": pdf_path.stem,
            "path": str(pdf_path),
            "text": full_text
        })
    return docs
