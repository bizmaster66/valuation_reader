import json
from typing import Any, Dict, List

from google.genai import types


def safe_json_load(text: str) -> Dict[str, Any]:
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
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception as e:
                        return {"error": f"JSON parse failed: {e}", "raw": text, "candidate": candidate}

    return {"error": "Unclosed JSON object", "raw": text}


def build_eval_prompt(
    company: str,
    ceo: str,
    sections: List[str],
    questions_by_section: Dict[str, List[str]],
    stage_rules: Dict[str, Any],
    knowledge_text: str,
    md_text: str,
    headings: List[str],
    total_score_max: int = 90,
) -> str:
    header = (
        "역할: 당신은 VC의 'AI 심사역'이다.\n"
        "목표: 업로드된 IR 자료를 분석/평가하여 투자자용 요약 및 추천 리포트와 상세 피드백을 생성한다.\n"
        "규칙:\n"
        "- 웹 검색 금지. IR 문서 텍스트와 제공된 지식만 사용.\n"
        "- 없는 내용은 '확인 불가'로 표기하고 추정 금지.\n"
        "- 9개 항목(문제 정의~리스크 관리)만 평가.\n"
        "- 항목 점수는 0~10점, 총점은 합계(최대 90점).\n"
        "- 80점 이상/미만에 따라 보고서 톤을 일관되게 달리한다.\n"
        "- 결과물 2/3은 한국어 기준 3,000~4,000자 분량.\n"
        "- 항목별 피드백은 3~5문장, 종합 피드백은 15~20문장.\n"
        "- 항목별 질문 리스트는 최소 5개씩.\n"
        "- 페이지 표시는 (p.xx) 형태로 포함.\n"
        "- 출력은 JSON ONLY.\n\n"
    )

    schema = (
        "JSON 스키마:\n"
        "{\n"
        '  "company_name": "...",\n'
        '  "ceo_name": "...",\n'
        '  "stage_estimate": "...",\n'
        '  "total_score_90": 0,\n'
        '  "section_scores": {"문제 정의": 0, ...},\n'
        '  "labels": ["근거 부족형", "검증 미흡형", "논리 단절형"],\n'
        '  "investor_report": {\n'
        f'    "{headings[0]}": "...",\n'
        f'    "{headings[1]}": "...",\n'
        f'    "{headings[2]}": ["...", "..."],\n'
        f'    "{headings[3]}": "...",\n'
        f'    "{headings[4]}": "...",\n'
        '    "Recommendation": "..." \n'
        "  },\n"
        '  "feedback_report": {\n'
        '    "overall_summary": "...",\n'
        '    "sections": {\n'
        '      "문제 정의": {\n'
        '        "score_0_10": 0,\n'
        '        "strengths": "...",\n'
        '        "weaknesses": "...",\n'
        '        "improvements": "...",\n'
        '        "investor_questions": "...",\n'
        '        "risks_expectations": "..."\n'
        "      }, ...\n"
        "    },\n"
        '    "priorities": "...",\n'
        '    "investor_type_strategy": "...",\n'
        '    "stage_guidelines": "...",\n'
        '    "pitch_faq_strategy": "...",\n'
        '    "visual_suggestions": "..."\n'
        "  }\n"
        "}\n"
    )

    prompt = header + schema
    prompt += "\n[투자 단계 추정 룰]\n" + json.dumps(stage_rules, ensure_ascii=False) + "\n"
    prompt += "\n[항목별 질문]\n" + json.dumps(questions_by_section, ensure_ascii=False) + "\n"
    prompt += "\n[지식 문서 요약/근거]\n" + (knowledge_text or "")[:120000] + "\n"
    prompt += "\n[IR 문서]\n" + (md_text or "")[:150000]
    prompt += f"\n\n[메타]\ncompany={company}\nceo={ceo}\nsections={sections}\n"
    prompt += f"total_score_max={total_score_max}\n"
    return prompt


def run_evaluation(
    client,
    model_name: str,
    prompt: str,
) -> Dict[str, Any]:
    cfg = types.GenerateContentConfig(temperature=0.0, top_p=0.1, top_k=1)
    resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
    return safe_json_load((resp.text or "").strip())
