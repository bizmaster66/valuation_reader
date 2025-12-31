import json
from typing import Dict, List, Any, Tuple
from google.genai import types

def extract_sources_from_grounding(resp) -> List[Dict[str, str]]:
    sources = []
    try:
        cand = resp.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if not gm:
            return []
        chunks = getattr(gm, "grounding_chunks", None) or []
        for ch in chunks:
            web = getattr(ch, "web", None)
            if not web:
                continue
            url = getattr(web, "uri", None) or ""
            title = getattr(web, "title", None) or ""
            if url:
                sources.append({"title": title, "url": url})
    except Exception:
        return []
    # 중복 제거
    seen = set()
    uniq = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            uniq.append(s)
    return uniq

def generate_company_profile(
    client,
    company_name: str,
    ceo_name: str,
    model_name: str = "gemini-2.5-flash"
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    회사명/대표자명 기준으로 Google Search tool을 사용해 사실을 수집하고,
    회사 정보를 JSON으로 정리한다.

    ⚠️ 중요: tool(google_search)을 쓸 때는 response_mime_type="application/json"을 쓰면 400 에러가 날 수 있어
    => JSON은 프롬프트로 강제하고, 응답 text를 파싱하는 방식 사용
    """
    google_tool = types.Tool(google_search=types.GoogleSearch())

    prompt = f"""
너는 스타트업 리서치 애널리스트다.
아래 회사에 대해 'Google Search 기반'으로 사실을 수집하고, 회사 정보를 정의하라.

회사명: {company_name}
대표자: {ceo_name}

엄격 규칙:
- 추정/환각 금지. 확인 불가하면 "확인 불가"라고 써라.
- 가능한 경우, 공식 홈페이지를 우선 확인하라.
- 뉴스/보도자료는 "최근 1개월 중심"으로 요약하되, 날짜가 불명확하면 명시하라.
- 출력은 JSON ONLY(설명 텍스트 금지)
- JSON 앞/뒤에 어떠한 문장도 쓰지 말 것.

출력 JSON 스키마(키 이름 고정):
{{
  "company_name": "{company_name}",
  "ceo_name": "{ceo_name}",
  "official_homepage": "URL 또는 확인 불가",
  "company_overview": "기업 개요 요약(5~8문장)",
  "business_areas": ["주요 사업영역", "..."],
  "products_services": ["주요 제품/서비스", "..."],
  "business_model_hypothesis": "BM을 개념적으로 정리(수익원/고객/채널/가치제안)",
  "industry_keywords": ["산업 키워드 5~10개"],
  "recent_updates_1m": [
    {{
      "date": "YYYY-MM-DD 또는 확인 불가",
      "title": "이슈 제목",
      "summary": "핵심 내용 2~3문장"
    }}
  ],
  "notes": "불확실/누락 포인트(있으면)"
}}
""".strip()

    # ✅ tool 사용 시 response_mime_type 지정하지 않음
    cfg = types.GenerateContentConfig(
        tools=[google_tool],
        temperature=0.2,
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg
    )

    text = (resp.text or "").strip()

    # JSON 파싱(앞뒤 군더더기 대비)
    try:
        data = json.loads(text) if text else {}
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        data = json.loads(text[s:e+1]) if (s != -1 and e != -1 and e > s) else {}

    sources = extract_sources_from_grounding(resp)
    return data, sources


def extract_industry_keywords(profile_data: Dict[str, Any]) -> List[str]:
    raw = profile_data.get("industry_keywords", []) or []
    clean = [k for k in raw if k and "확인 불가" not in str(k)]
    return clean[:10] if clean else ["tech", "platform"]

def generate_industry_report(client, keywords: List[str], model_name: str = "gemini-2.5-flash"):
    google_tool = types.Tool(google_search=types.GoogleSearch())
    kw_str = ", ".join(keywords[:10])
    prompt = f"""
너는 투자 리서치 애널리스트다.
아래 키워드로 산업 리포트를 작성하라(검색 기반).

키워드: {kw_str}

- 시장 규모/성장률(가능한 경우 숫자)
- 경쟁/규제/리스크
- 투자 동향
- 객관적 문체

출력은 마크다운으로.
""".strip()

    cfg = types.GenerateContentConfig(
        tools=[google_tool],
        response_mime_type="text/plain",
        temperature=0.2,
    )
    resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
    text = (resp.text or "").strip()
    sources = extract_sources_from_grounding(resp)
    return text, sources
