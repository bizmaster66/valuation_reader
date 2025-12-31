# src/pdf_vision.py
import json
from typing import List, Dict, Any

from google.genai import types


def needs_vision(pages: List[Dict[str, Any]], min_chars_total: int = 800, min_nonempty_pages: int = 2) -> bool:
    """
    pypdf 텍스트 추출 결과가 너무 빈약하면(스캔본/이미지 PDF 가능성) True
    """
    total_chars = sum(len((p.get("text") or "").strip()) for p in pages)
    nonempty_pages = sum(1 for p in pages if (p.get("text") or "").strip())
    return total_chars < min_chars_total or nonempty_pages < min_nonempty_pages


def gemini_pdf_ocr_text(
    client,
    pdf_bytes: bytes,
    model_name: str = "gemini-2.5-flash",
    max_chars: int = 90000
) -> str:
    """
    PDF 자체를 Gemini에 넣고(문서 이해+OCR) 텍스트를 [PAGE n] 포맷으로 직접 받는다.
    JSON 파싱을 제거해서 안정적으로 만든 버전.
    """
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    prompt = (
        "너는 IR PDF를 문서 이해+OCR로 읽는다.\n"
        "출력은 반드시 아래 포맷을 지켜라(설명 금지):\n"
        "[PAGE 1]\n"
        "....\n\n"
        "[PAGE 2]\n"
        "....\n\n"
        "규칙:\n"
        "- 가능한 원문 텍스트를 최대한 그대로.\n"
        "- 표/차트에서 숫자/단위가 보이면 같이 포함.\n"
        "- 너무 길면 핵심 문장과 숫자 중심으로 요약.\n"
        "- 페이지 번호를 가능한 한 유지.\n"
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt, pdf_part],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    text = (resp.text or "").strip()
    return text[:max_chars]


def gemini_pdf_visual_insights(
    client,
    pdf_bytes: bytes,
    model_name: str = "gemini-2.5-flash",
    max_chars: int = 20000
) -> str:
    """
    PDF 내 표/차트/그래프/도표에서 핵심 지표/추세/인사이트만 추출해
    짧은 보조 컨텍스트로 반환(마크다운).
    """
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    prompt = (
        "너는 IR PDF의 표/차트/그래프/도표를 읽고 투자자 관점에서 핵심 수치/추세만 추출한다.\n"
        "출력은 마크다운 텍스트로.\n"
        "- 반드시 '페이지 번호'를 함께 적어라(가능한 범위에서).\n"
        "- 매출/성장률/사용자지표/단가/전환/리텐션/파이프라인/단위경제성 등 숫자가 보이면 우선 추출.\n"
        "- 과장 금지. 보이지 않으면 '확인 불가'라고 쓴다.\n"
        "형식:\n"
        "## Visual Insights\n"
        "- (p.X) 지표: 값 / 해석\n"
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt, pdf_part],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    text = (resp.text or "").strip()
    return text[:max_chars]
