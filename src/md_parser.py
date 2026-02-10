import re
from typing import Dict, List


def _clean_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"[\"'()\\[\\]<>]+", "", s)
    s = re.sub(r"\\s{2,}", " ", s).strip()
    return s


def _candidates_from_filename(filename: str) -> List[str]:
    if not filename:
        return []
    base = re.sub(r"\\.[a-zA-Z0-9]+$", "", filename)
    base = re.sub(r"^\\s*\\d+\\.?\\s*", "", base)
    base = base.replace("_IR자료", "").replace("IR자료", "")
    base = base.replace("분석보고서", "").replace("분석 보고서", "")
    base = base.strip("_- ").strip()
    return [_clean_name(base)] if base else []


def _candidates_from_text(md_text: str) -> List[str]:
    candidates = []
    for line in (md_text or "").splitlines():
        if "**회사명:**" in line:
            name = line.split("**회사명:**", 1)[1].strip()
            candidates.append(_clean_name(name))
        if "회사명:" in line:
            name = line.split("회사명:", 1)[1].strip()
            candidates.append(_clean_name(name))
        if "기업명:" in line:
            name = line.split("기업명:", 1)[1].strip()
            candidates.append(_clean_name(name))
        if "대표이사" in line or "CEO" in line or "기업개요" in line:
            # 주변에 주식회사/㈜/법인명 패턴 추출
            m = re.search(r"(주식회사\\s*[^\\s,]+|㈜\\s*[^\\s,]+|\\(주\\)\\s*[^\\s,]+)", line)
            if m:
                candidates.append(_clean_name(m.group(1)))
    # 제목에서 추출: "# 1. (주)관악연구소_IR자료.pdf 분석 보고서"
    m = re.search(r"^#\\s+.*?\\s*(.+?)_IR자료", md_text or "", re.MULTILINE)
    if m:
        candidates.append(_clean_name(m.group(1)))
    m = re.search(r"^#\\s+(.+?)\\s+분석\\s+보고서", md_text or "", re.MULTILINE)
    if m:
        candidates.append(_clean_name(m.group(1)))
    return [c for c in candidates if c]


def extract_company_candidates(md_text: str, filename: str = "") -> List[str]:
    candidates = []
    candidates.extend(_candidates_from_filename(filename))
    candidates.extend(_candidates_from_text(md_text))
    # de-dup preserving order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def extract_company_name(md_text: str, filename: str = "") -> str:
    candidates = extract_company_candidates(md_text, filename)
    return candidates[0] if candidates else "Unknown"


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
