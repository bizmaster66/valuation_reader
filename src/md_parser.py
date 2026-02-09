import re
from typing import Dict


def extract_company_name(md_text: str) -> str:
    # 1) **회사명:** 우선
    for line in (md_text or "").splitlines():
        if "**회사명:**" in line:
            name = line.split("**회사명:**", 1)[1].strip()
            return name.strip("* ").strip()

    # 2) 제목에서 추출: "# 1. (주)관악연구소_IR자료.pdf 분석 보고서"
    m = re.search(r"^#\\s+.*?\\s*(.+?)_IR자료", md_text or "", re.MULTILINE)
    if m:
        return m.group(1).strip()

    m = re.search(r"^#\\s+(.+?)\\s+분석\\s+보고서", md_text or "", re.MULTILINE)
    if m:
        return m.group(1).strip()

    return "Unknown"


def extract_ceo_name(md_text: str) -> str:
    for line in (md_text or "").splitlines():
        if "**대표자:**" in line or "**대표:**" in line:
            if "**대표자:**" in line:
                name = line.split("**대표자:**", 1)[1].strip()
            else:
                name = line.split("**대표:**", 1)[1].strip()
            return name.strip("* ").strip()
    return "Unknown"


def normalize_company_for_filename(company: str) -> str:
    s = re.sub(r"[\\\\/\\:*?\"<>|]+", "_", company or "")
    return s.strip() or "unknown_company"


def build_ir_text(md_text: str, limit_chars: int = 120000) -> str:
    return (md_text or "")[:limit_chars]
