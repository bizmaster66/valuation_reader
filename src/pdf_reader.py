from pypdf import PdfReader

def extract_pages(pdf_path: str):
    """
    반환: [{"page": 1, "text": "..."}, ...]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages
