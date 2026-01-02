import os
import re

PAGE_RE = re.compile(r"^p(\d{3})\.txt$")

def build_fulltext_from_pages_dir(pages_dir: str) -> str:
    files = []
    for name in os.listdir(pages_dir):
        m = PAGE_RE.match(name)
        if m:
            files.append((int(m.group(1)), os.path.join(pages_dir, name)))

    files.sort(key=lambda x: x[0])

    chunks = []
    for page_no, path in files:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        chunks.append(f"[PAGE {page_no:03d}]\n{txt}\n")
        chunks.append("-" * 60 + "\n")

    return "\n".join(chunks).strip()
