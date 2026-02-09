# -*- coding: utf-8 -*-
from typing import List, Dict

SYSTEM_PROMPT = """
당신은 스타트업 창업자가 IR 데크를 발표하는 상황에서 "발표 스크립트형 Full Text"를 작성합니다.

목표:
- 이 텍스트만 읽어도 IR 전체의 논리(무엇을 설득하려는지)가 이해되어야 합니다.
- 각 페이지(슬라이드)의 핵심 메시지, 도해/도표/차트/그래프가 말하는 의미를 자연어로 설명해야 합니다.
- 슬라이드 간 연결 문장을 넣어 스토리라인(Problem→Solution→Market→Traction→BM→GTM→Team→Financials)을 자연스럽게 만드세요.

엄격 규칙(중요):
- 이미지/슬라이드에 명시되지 않은 숫자/고객사/매출/계약을 추정하거나 만들어내지 마세요.
- OCR 텍스트가 불완전할 수 있으니, 불확실하면 "슬라이드에서 식별 불가"라고 명시하세요.
- 차트/표 설명은 (무엇을 비교/측정하는지 → 핵심 트렌드 → 눈에 보이는 수치(있으면) → 왜 중요한지) 순서로 쓰세요.
- 각 슬라이드 블록 끝에는 반드시 [p.XXX] 형태의 페이지 앵커를 남기세요.

출력 형식:
- 맨 앞: IR 전체 요약(3~7문장) + 목차(섹션 흐름)
- 본문: 섹션별로 슬라이드 해설(발표 스크립트 톤, 문장형)
"""

def _format_pages(pages: List[Dict], max_chars: int = 90000) -> str:
    chunks = []
    for p in pages:
        no = p.get("page")
        try:
            no_i = int(no)
        except Exception:
            continue
        txt = (p.get("text") or "").strip()
        if not txt:
            # 텍스트가 비어도 페이지 앵커는 남겨서 "빈 페이지/이미지 중심"으로 처리 가능
            chunks.append(f"[p.{no_i:03d}]\n(텍스트 식별 불가/이미지 중심 슬라이드)\n")
            continue
        chunks.append(f"[p.{no_i:03d}]\n{txt}\n")
    body = "\n".join(chunks)
    return body[:max_chars]

def build_fulltext_v2_script(
    client,
    pages: List[Dict],
    visual_insights: str = "",
    model_name: str = "gemini-2.5-flash",
    max_chars: int = 90000,
) -> str:
    """
    pages: [{"page": 1, "text": "..."}, ...]
    visual_insights: (옵션) PDF 전체에서 추출한 차트/도표/도해 보조 설명 텍스트
    """
    ocr_body = _format_pages(pages, max_chars=max_chars)

    prompt = SYSTEM_PROMPT + "\n\n" + "OCR_TEXT_BY_PAGE:\n" + ocr_body
    if visual_insights and visual_insights.strip():
        prompt += "\n\nVISUAL_INSIGHTS:\n" + visual_insights.strip()

    # google-genai client 호환 (여러 형태 방어)
    resp = None
    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        resp = client.models.generate_content(model=model_name, contents=[prompt])
    elif hasattr(client, "generate_content"):
        resp = client.generate_content(model=model_name, contents=[prompt])
    else:
        raise RuntimeError("Gemini client interface not supported: cannot call generate_content")

    text = getattr(resp, "text", "") or ""
    return text.strip() or "(Full Text v2 생성 실패: 빈 결과)"
