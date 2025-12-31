# src/evaluator_v2.py
import json
from typing import Dict, Any, List

CRITERIA = [
    "problem_definition",
    "solution_product",
    "market",
    "business_model",
    "competitive_advantage",
    "growth_strategy",
    "team",
    "financial_plan",
    "risk_management",
]

KOR_LABEL = {
    "problem_definition": "문제 정의",
    "solution_product": "솔루션 & 제품",
    "market": "시장 분석",
    "business_model": "비즈니스 모델",
    "competitive_advantage": "경쟁 우위",
    "growth_strategy": "성장 전략",
    "team": "팀 역량",
    "financial_plan": "재무 계획",
    "risk_management": "리스크 관리",
}

CAP_BY_EVIDENCE = {
    "L0": 2.5,
    "L1": 3.0,
    "L2": 4.0,
    "L3": 5.0,
}

BM_CATEGORIES = [
    "subscription",
    "usage",
    "transaction_take_rate",
    "payment_processing_fee",
    "success_fee_brokerage",
    "advertising",
    "commerce_margin",
    "licensing_royalty",
    "data_api_sales",
    "professional_services",
    "unknown_not_disclosed",
]

ARCHETYPES = [
    "deeptech_milestone",
    "hardware_manufacturing",
    "traction_software",
    "marketplace_platform",
    "unclear",
]

def safe_json_first_object(text: str) -> Dict[str, Any]:
    """
    모델이 JSON 뒤에 글을 붙여도 첫 JSON object만 파싱.
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
        return {"error": "No JSON object", "raw": text}

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
                    cand = text[start:i+1]
                    try:
                        return json.loads(cand)
                    except Exception as e:
                        return {"error": f"JSON parse failed: {e}", "raw": text, "candidate": cand}

    return {"error": "Unclosed JSON", "raw": text}

def clamp_score_1_to_5(x: float) -> float:
    # 0점 금지 → 최소 1.0
    return max(1.0, min(5.0, float(x)))

def apply_cap(raw_score: float, evidence_level: str) -> float:
    cap = CAP_BY_EVIDENCE.get(evidence_level, 2.5)
    return min(clamp_score_1_to_5(raw_score), cap)

def build_prompt(company_name: str, stage: str, ir_text: str, archetype_hint: str = "", language: str = "ko") -> str:
    """
    LLM에게는 '분류/증거수준/원점수(raw_score)/근거/요청'만 맡기고,
    cap 적용(final_score)은 파이썬이 수행.
    """
    header = (
        "[ROLE]\n"
        "너는 “AI 심사역(보수적 평가 모드)”이다.\n"
        "IR 텍스트만을 근거로 ①archetype 분류, ②BM(Revenue Mechanism) 분류, ③9개 항목 raw_score(1~5)와 evidence_level(L0~L3)을 산출한다.\n"
        "중요: IR에 없는 내용은 추정하지 말고 '근거 없음/불명확'으로 표시한다.\n"
        "단, 점수는 0점을 주지 말고 raw_score는 1~5 범위에서 산정하라.\n"
        "cap(final_score)은 파이썬이 적용하므로 너는 raw_score와 evidence_level만 정확히 내라.\n\n"
    )

    bm_list = "\n".join([f"- {x}" for x in BM_CATEGORIES])
    arch_list = "\n".join([f"- {x}" for x in ARCHETYPES])

    spec = (
        f"[INPUT]\n"
        f"- company_name: {company_name}\n"
        f"- stage: {stage}\n"
        f"- archetype_hint(optional): {archetype_hint}\n"
        f"- language: {language}\n\n"
        f"[BM Revenue Mechanism categories]\n{bm_list}\n\n"
        f"[Archetype categories]\n{arch_list}\n\n"
        f"[CRITERIA]\n"
        + "\n".join([f"- {c} ({KOR_LABEL[c]})" for c in CRITERIA]) + "\n\n"
        "Evidence Level:\n"
        "- L0: 주장만 있음\n"
        "- L1: 내부 근거(요약/범위)\n"
        "- L2: 외부 근거(LOI/PoC/인증/특허 목록 등)\n"
        "- L3: 독립 검증(상용계약/감사수치/인허가 승인 등)\n\n"
        "OUTPUT은 JSON ONLY. JSON 앞뒤 문장 금지.\n"
        "JSON 스키마(키 이름 고정):\n"
        "{\n"
        ' "archetype": {"primary": "...", "secondary_candidates": ["..."], "confidence": 0.0, "evidence_snippets": ["..."], "missing_info": ["..."]},\n'
        ' "bm_revenue_mechanism": {"primary": "...", "secondary_candidates": ["..."], "confidence": 0.0, "evidence_snippets": ["..."], "missing_info_to_confirm": ["..."]},\n'
        ' "scores": {\n'
        '   "problem_definition": {"evidence_level":"L0","raw_score":3.0,"confidence":0.5,"evidence_snippets":["..."],"missing_info_to_upgrade":["..."]},\n'
        '   ... (9개 모두)\n'
        ' },\n'
        ' "data_requests_minimum": ["..."]\n'
        "}\n\n"
        "[IR_TEXT]\n"
    )

    return header + spec + ir_text

def run_llm_eval(client, model_name: str, company_name: str, stage: str, ir_text: str, archetype_hint: str = "") -> Dict[str, Any]:
    prompt = build_prompt(company_name, stage, ir_text, archetype_hint=archetype_hint)
    resp = client.models.generate_content(model=model_name, contents=prompt)
    return safe_json_first_object((resp.text or "").strip())

def postprocess_caps(llm_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    cap 적용 + 0점 금지 + Unknown BM이면 BM/재무 상한 3.0 적용.
    """
    out = dict(llm_json) if isinstance(llm_json, dict) else {}
    scores = out.get("scores", {}) or {}

    bm = (out.get("bm_revenue_mechanism", {}) or {}).get("primary", "") or ""
    bm_unknown = (bm == "unknown_not_disclosed")

    final_scores: Dict[str, float] = {}
    for c in CRITERIA:
        item = scores.get(c, {}) or {}
        ev = (item.get("evidence_level") or "L0").strip()
        raw = item.get("raw_score", 1.0)
        try:
            raw = float(raw)
        except Exception:
            raw = 1.0

        fs = apply_cap(raw, ev)

        # BM unknown이면 BM/재무는 최대 3.0
        if bm_unknown and c in ["business_model", "financial_plan"]:
            fs = min(fs, 3.0)

        final_scores[c] = round(fs, 2)

        # scores 구조에 final_score를 추가(LLM이 안 낸 값)
        item["final_score"] = final_scores[c]
        scores[c] = item

    out["scores"] = scores
    out["_final_scores_1_to_5"] = final_scores  # WeightEngine 입력용
    out["_bm_unknown_cap_applied"] = bool(bm_unknown)
    return out
