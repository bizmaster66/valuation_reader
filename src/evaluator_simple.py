# src/evaluator_simple.py
import json
from typing import Dict, Any

SECTIONS = [
    "문제 정의",
    "솔루션 & 제품",
    "시장 분석",
    "비즈니스 모델",
    "경쟁 우위",
    "성장 전략",
    "팀 역량",
    "재무 계획",
    "리스크 관리",
]


def safe_json_load(text: str) -> Dict[str, Any]:
    """
    모델이 JSON 앞/뒤에 텍스트를 붙이거나 JSON을 여러 개 출력해도
    '첫 번째 완전한 JSON 객체'만 안전하게 파싱한다.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return {"error": "No JSON object found", "raw": text}

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception as e:
                        return {"error": f"JSON parse failed: {e}", "raw": text, "candidate": candidate}

    return {"error": "Unclosed JSON object", "raw": text}


def build_overall_prompt(
    company: str,
    ceo: str,
    bm: str,
    industry: str,
    stage: str,
    weights_100: Dict[str, float],
    ir_text_with_pages: str,
    company_profile_json: Dict[str, Any],
) -> str:
    """
    ✅ 섹션 점수 10점 척도(0~10)
    ✅ 개선안(section_improvements)은 모든 항목에 항상 작성
    ✅ 7점 이상: 칭찬 + 개선(부드러운 톤)
       7점 미만: 부족한 점을 먼저 + 개선(명확한 톤)
    """
    header = (
        "너는 'AI 심사역'이다.\n"
        "목표: 수많은 IR 중 사람이 읽을 가치(Review Pass)와 추천 가능(Recommend Pass)을 가늠할 수 있도록, "
        "IR의 '작성 완성도/검토 효율' 관점에서 평가한다.\n"
        "규칙:\n"
        "- IR에 없는 내용은 '없음/확인 불가'로 처리(추정 금지)\n"
        "- 각 섹션 점수는 0~10점(정수 또는 소수 가능)\n"
        "- section_improvements(개선안)는 모든 섹션에 항상 작성해야 한다(점수와 무관)\n"
        "- 7점 이상: 강점을 칭찬하면서도 개선 포인트를 '보완하면 더 좋아지는' 톤으로 제시\n"
        "- 7점 미만: 부족한 점을 먼저 명확히 지적하고 개선안을 제시\n"
        "- 출력은 JSON ONLY. JSON 앞뒤에 문장/코드블록/마크다운 금지.\n\n"
    )

    schema = (
        "JSON 스키마:\n"
        "{\n"
        '  "one_line_pitch": "...",\n'
        '  "section_scores": {"문제 정의": 0, "솔루션 & 제품": 0, "시장 분석": 0, "비즈니스 모델": 0, "경쟁 우위": 0, "성장 전략": 0, "팀 역량": 0, "재무 계획": 0, "리스크 관리": 0},\n'
        '  "section_analysis": {"문제 정의": "...", ...},\n'
        '  "section_improvements": {"문제 정의": "...", ...},\n'
        '  "key_strengths": "...",\n'
        '  "needs_improvement": "...",\n'
        '  "final_opinion": "...",\n'
        '  "storyline": "..."\n'
        "}\n"
    )

    inp = {
        "company_name": company,
        "ceo_name": ceo,
        "bm": bm,
        "industry": industry,
        "stage": stage,
        "weights_100": weights_100,  # 엔진에서 산출된 가중치(합 100) 참고용
        "company_profile": company_profile_json,
        "ir_text_with_pages": ir_text_with_pages[:90000],
        "sections": SECTIONS,
        "note": "각 섹션 점수는 0~10으로 산정. 개선안은 무조건 포함."
    }

    return header + schema + "\n입력:\n" + json.dumps(inp, ensure_ascii=False)


def run_overall_evaluation(
    client,
    model_name: str,
    company: str,
    ceo: str,
    bm: str,
    industry: str,
    stage: str,
    weights_100: Dict[str, float],
    ir_text_with_pages: str,
    company_profile_json: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_overall_prompt(
        company=company,
        ceo=ceo,
        bm=bm,
        industry=industry,
        stage=stage,
        weights_100=weights_100,
        ir_text_with_pages=ir_text_with_pages,
        company_profile_json=company_profile_json,
    )
    resp = client.models.generate_content(model=model_name, contents=prompt)
    return safe_json_load((resp.text or "").strip())
