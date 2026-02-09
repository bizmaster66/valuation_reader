from typing import List

from docx import Document


DEFAULT_HEADINGS = [
    "Executive Summary",
    "Overview",
    "Highlights",
    "Achievement(Key Metrics & Performance)",
    "Funding Plan",
]


def extract_headings_from_sample(docx_bytes: bytes) -> List[str]:
    if not docx_bytes:
        return DEFAULT_HEADINGS
    doc = Document(docx_bytes)
    headings = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        if text in DEFAULT_HEADINGS and text not in headings:
            headings.append(text)
    return headings or DEFAULT_HEADINGS
