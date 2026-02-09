import os
import glob
import shutil
import time
import json
import hashlib
from datetime import datetime
import io
import inspect
import math
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

from src.gemini_client import get_client
from src.storage import append_history, load_history
from src.startup_analyzer_adapter import generate_company_profile

from src.evaluator_simple import run_overall_evaluation
from src.dataset_logger import upsert_ai_output, load_ai_outputs, AI_OUTPUT_PATH
from src.pdf_vision import gemini_pdf_visual_insights

from src.pdf_ocr_pages import ocr_pdf_all_pages
from src.weight_engine import WeightEngine
from src.fulltext_script_v2 import build_fulltext_v2_script
from src.fulltext_from_cache import build_fulltext_from_pages_dir

# -----------------------------
# 0) 기준 섹션 및 영문 매핑
# -----------------------------
SECTIONS_KOR = [
    "문제 정의", "솔루션 & 제품", "시장 분석", "비즈니스 모델", 
    "경쟁 우위", "성장 전략", "팀 역량", "재무 계획", "리스크 관리"
]

KOR_TO_ENG = {
    "문제 정의": "Problem",
    "솔루션 & 제품": "Solution",
    "시장 분석": "Market",
    "비즈니스 모델": "BM",
    "경쟁 우위": "Competitiveness",
    "성장 전략": "Growth",
    "팀 역량": "Team",
    "재무 계획": "Finance",
    "리스크 관리": "Risk",
}

KOR_TO_CRITERION = {
    "문제 정의": "problem_definition",
    "솔루션 & 제품": "solution_product",
    "시장 분석": "market",
    "비즈니스 모델": "business_model",
    "경쟁 우위": "competitive_advantage",
    "성장 전략": "growth_strategy",
    "팀 역량": "team",
    "재무 계획": "financial_plan",
    "리스크 관리": "risk_management",
}

# -----------------------------
# 1) Helpers (WeightEngine 매핑 및 유틸리티)
# -----------------------------
def map_stage_for_engine(stage_raw: str) -> str:
    s = (stage_raw or "").strip().lower().replace("_", " ").replace("-", " ")
    if any(x in s for x in ["b+", "b 이상", "b plus"]): return "series_b"
    if any(x in s for x in ["c+", "c 이상"]): return "series_c"
    if any(x in s for x in ["seed", "preseed"]): return "seed"
    if "pre a" in s: return "pre_a"
    if "series a" in s or s == "a": return "series_a"
    if "series b" in s or s == "b": return "series_b"
    if "series c" in s or s == "c": return "series_c"
    return "seed"

def map_industry_for_engine(industry_raw: str):
    s = (industry_raw or "").strip().lower()
    if any(x in s for x in ["바이오", "헬스", "의료"]): return "bio_healthcare"
    if any(x in s for x in ["딥테크", "로봇", "반도체"]): return "deeptech"
    if "saas" in s or "b2b" in s: return "b2b_saas"
    return None

def map_bm_for_engine(bm_raw: str):
    s = (bm_raw or "").strip().lower()
    if "구독" in s or "subscription" in s: return "subscription_saas"
    if "마켓" in s or "marketplace" in s: return "transaction_marketplace"
    return None

def save_uploaded_pdf(uploaded) -> str:
    os.makedirs("data/uploads", exist_ok=True)
    path = os.path.join("data/uploads", uploaded.name)
    with open(path, "wb") as f: f.write(uploaded.getbuffer())
    return path

def build_packed_text(pages, limit_chars: int = 60000) -> str:
    parts = [f"[PAGE {p['page']}]\n{(p.get('text') or '').strip()}" for p in pages if (p.get('text') or '').strip()]
    return "\n\n".join(parts)[:limit_chars]

def safe_json_load(text: str):
    try: return json.loads(text)
    except:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1]) if s != -1 and e != -1 else {"error": "JSON parse failed"}

def extract_company_and_ceo(client, packed_text: str) -> dict:
    prompt = "IR PDF에서 'company_name', 'ceo_name'을 찾아 JSON으로 출력하라. 근거 페이지와 인용(evidence) 포함."
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt + "\n" + packed_text)
    return safe_json_load(resp.text)

def classify_bm_industry_stage(client, company, ceo, packed_text, profile) -> dict:
    prompt = "아래 정보를 바탕으로 business_model, industry, stage를 JSON으로 추천하라.\n" + packed_text[:5000]
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return safe_json_load(resp.text)

# -----------------------------
# 2) Radar Chart (영문, Weight 반영)
# -----------------------------
def radar_chart(scores_dict, weights_dict):
    # 가중치가 0인 항목 제외
    active_labels = [k for k in SECTIONS_KOR if weights_dict.get(KOR_TO_CRITERION[k], 0) > 0]
    if not active_labels: return None
    
    values = [float(scores_dict.get(k, 0) or 0) for k in active_labels]
    eng_labels = [KOR_TO_ENG.get(k, k) for k in active_labels]
    
    N = len(values)
    angles = [2 * math.pi * n / N for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(eng_labels, fontsize=9)
    ax.set_ylim(0, 10)
    return fig

# -----------------------------
# 3) UI 및 메인 로직
# -----------------------------
show_debug = False
st.set_page_config(page_title="AI IR Analyzer", layout="wide")
st.title("AI 심사역 powered by MARK")

with st.sidebar:
    st.header("옵션")
    use_company_profile = st.toggle("외부 데이터(뉴스/홈페이지) 활용", value=True)
    use_visual_insights = st.toggle("시각 인사이트 추출", value=True)
    show_debug = st.toggle("디버그 정보", value=False)
    dpi = st.slider("OCR DPI", 180, 300, 220)

uploaded_files = st.file_uploader("IR PDF 업로드", type=["pdf"], accept_multiple_files=True)
run_btn = st.button("분석 실행", type="primary", disabled=not uploaded_files)

if "rows" not in st.session_state: st.session_state.rows = []
if "view_mode" not in st.session_state: st.session_state.view_mode = "list"

if run_btn:
    engine = WeightEngine()
    client = get_client()
    last_report_path = None
    
    for up in uploaded_files:
        file_key = hashlib.md5(up.getvalue()).hexdigest()[:10]
        pdf_path = save_uploaded_pdf(up)
        ocr_cache_dir = f"data/ocr_cache/{file_key}"
        
        # OCR 실행
        pages = ocr_pdf_all_pages(client=client, pdf_path=pdf_path, cache_dir=ocr_cache_dir, dpi=dpi)
        packed_text = build_packed_text(pages)
        fulltext = build_fulltext_from_pages_dir(os.path.join(ocr_cache_dir, "pages"))
        
        # 시각 정보 결합
        vis_text = ""
        if use_visual_insights:
            vis_text = gemini_pdf_visual_insights(client, up.getvalue())
            packed_text += f"\n\n[Visual Insights]\n{vis_text}"

        # 객관적 스크립트 생성 (요청 사항 반영)
        # 심사역의 주관적 시선 없이 IR 자료 내용을 충실히 설명하도록 지시
        fulltext_v2 = build_fulltext_v2_script(
            client=client, 
            pages=[{"page": p["page"], "text": p["text"]} for p in pages],
            visual_insights=vis_text,
            model_name="gemini-2.5-flash"
        )
        
        # 기업 정보 및 평가
        ext = extract_company_and_ceo(client, packed_text)
        company, ceo = ext.get("company_name", "Unknown"), ext.get("ceo_name", "Unknown")
        
        prof = generate_company_profile(client, company, ceo)[0] if use_company_profile else {}
        cls = classify_bm_industry_stage(client, company, ceo, packed_text, prof)
        
        # 가중치 엔진 적용
        s_key, i_key, b_key = map_stage_for_engine(cls.get("stage")), map_industry_for_engine(cls.get("industry")), map_bm_for_engine(cls.get("business_model"))
        weights_100 = engine.compute_weights(stage=s_key, industry=i_key, business_model=b_key)
        
        eval_json = run_overall_evaluation(client=client, company=company, ceo=ceo, weights_100=weights_100, ir_text_with_pages=packed_text, company_profile_json=prof)
        
        # 결과 저장
        out_dir = f"data/outputs/{company}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/01_fulltext.txt", "w", encoding="utf-8") as f: f.write(fulltext_v2)
        
        report_payload = {
            "company_name": company, "total_100": engine.evaluate(scores_1_to_5={KOR_TO_CRITERION[k]: float(eval_json['section_scores'].get(k, 0))/2.0 for k in SECTIONS_KOR}, stage=s_key)["overall_100_after_gates"],
            "eval": eval_json, "weights_100": weights_100, "output_path": out_dir
        }
        with open(f"{out_dir}/report.json", "w", encoding="utf-8") as f: json.dump(report_payload, f, ensure_ascii=False)
        
        append_history({"company_name": company, "total_score": report_payload["total_100"], "output_path": out_dir})
        st.success(f"{company} 분석 완료")
        last_report_path = os.path.join(out_dir, "report.json")

    if last_report_path:
        st.session_state.selected_report_path = last_report_path
        st.session_state.view_mode = "detail"
        st.rerun()

# -----------------------------
# 4) 리포트 렌더링 (차트 및 다운로드 버튼)
# -----------------------------
def render_report(report):
    st.header(f"📊 {report['company_name']} 분석 결과")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = radar_chart(report['eval']['section_scores'], report['weights_100'])
        if fig: st.pyplot(fig)
        st.metric("종합 점수", f"{report['total_100']:.1f}")

    with col2:
        st.subheader("객관적 IR 설명 스크립트")
        script_path = os.path.join(report['output_path'], "01_fulltext.txt")
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()
            st.text_area("Script Content", script_content, height=300)
            st.download_button("📥 스크립트(.md) 다운로드", script_content, file_name=f"{report['company_name']}_IR_Script.md")

if st.session_state.view_mode == "detail":
    if st.button("← 목록으로"): 
        st.session_state.view_mode = "list"
        st.rerun()
    with open(st.session_state.selected_report_path, "r", encoding="utf-8") as f:
        render_report(json.load(f))
else:
    hist = load_history()
    for i, r in hist.tail(10).iterrows():
        if st.button(f"보기: {r['company_name']} ({r['total_score']})", key=f"btn_{i}"):
            st.session_state.selected_report_path = os.path.join(r['output_path'], "report.json")
            st.session_state.view_mode = "detail"
            st.rerun()
