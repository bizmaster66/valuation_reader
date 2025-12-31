import os
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
from src.pdf_reader import extract_pages
from src.storage import append_history, load_history
from src.startup_analyzer_adapter import generate_company_profile

from src.evaluator_simple import run_overall_evaluation  # evaluator는 섹션 점수/분석/개선안/요약 생성
from src.dataset_logger import upsert_ai_output, load_ai_outputs, AI_OUTPUT_PATH
from src.pdf_vision import needs_vision, gemini_pdf_ocr_text, gemini_pdf_visual_insights

from src.weight_engine import WeightEngine


# -----------------------------
# 0) 기준 섹션(9개)
# -----------------------------
SECTIONS_KOR = [
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
# 1) Helpers: WeightEngine 매핑
# -----------------------------
def map_stage_for_engine(stage_raw: str) -> str:
    """
    WeightEngine normalize_stage가 alias를 처리하지만,
    B+ / B 이상 같은 변형은 안전하게 여기서 먼저 정규화.
    """
    s = (stage_raw or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ").strip()

    if "b+" in s or "b 이상" in s or "b이상" in s or "b plus" in s:
        return "series_b"
    if "c+" in s or "c 이상" in s or "c이상" in s:
        return "series_c"

    if "seed" in s or "preseed" in s or "pre seed" in s or "pre-seed" in s:
        return "seed"
    if "pre a" in s or "pre-a" in s or "prea" in s:
        return "pre_a"
    if "series a" in s or s == "a":
        return "series_a"
    if "series b" in s or s == "b":
        return "series_b"
    if "series c" in s or s == "c":
        return "series_c"
    if "pre ipo" in s or "pre-ipo" in s:
        return "pre_ipo"
    if "ipo" in s:
        return "ipo"

    return "seed"


def map_industry_for_engine(industry_raw: str):
    """
    industry 멀티플라이어 키(예: b2b_saas, deeptech 등)로 매핑.
    매핑 실패 시 None → 멀티플라이어 미적용.
    """
    s = (industry_raw or "").strip().lower()

    if "바이오" in s or "헬스" in s or "의료" in s:
        return "bio_healthcare"
    if "딥테크" in s or "로봇" in s or "소재" in s or "반도체" in s:
        return "deeptech"
    if "제조" in s or "소부장" in s:
        return "manufacturing_sobu"
    if "플랫폼" in s or "마켓" in s or "마켓플레이스" in s:
        return "platform_marketplace"
    if "핀테크" in s or "금융" in s:
        return "fintech_regulated"
    if "커머스" in s or "쇼핑" in s or "d2c" in s:
        return "d2c_commerce"
    if "saas" in s or "b2b" in s:
        return "b2b_saas"

    return None


def map_bm_for_engine(bm_raw: str):
    """
    business model multipliers 키(subscription_saas 등)로 매핑.
    매핑 실패 시 None → 멀티플라이어 미적용.
    """
    s = (bm_raw or "").strip().lower()

    if "구독" in s or "subscription" in s:
        return "subscription_saas"
    if "usage" in s or "사용량" in s:
        return "usage_based"
    if "marketplace" in s or "거래" in s or "마켓" in s:
        return "transaction_marketplace"
    if "enterprise" in s or "엔터프라이즈" in s or "세일즈" in s:
        return "enterprise_sales"

    return None


# -----------------------------
# 2) Helpers: 파일/텍스트/JSON
# -----------------------------
def save_uploaded_pdf(uploaded) -> str:
    os.makedirs("data/uploads", exist_ok=True)
    path = os.path.join("data/uploads", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def write_md(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def build_packed_text(pages, limit_chars: int = 60000) -> str:
    parts = []
    for p in pages:
        t = (p.get("text") or "").strip()
        if t:
            parts.append(f"[PAGE {p['page']}]\n{t}")
    packed = "\n\n".join(parts)
    return packed[:limit_chars]


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        s = (text or "").find("{")
        e = (text or "").rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
        return {"error": "JSON parse failed", "raw": text}


def extract_company_and_ceo(client, packed_text: str) -> dict:
    prompt = (
        "너는 IR PDF를 읽고 '기업명'과 '대표자 성명'을 찾아야 한다.\n"
        "추정하지 말고, 문서에 명시된 근거 페이지와 짧은 인용을 함께 제시하라.\n"
        "출력은 JSON ONLY: company_name, ceo_name, evidence[{page, quote}]\n\n"
        "IR 텍스트:\n"
    ) + packed_text
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return safe_json_load((resp.text or "").strip())


def classify_bm_industry_stage(client, company: str, ceo: str, packed_text: str, company_profile: dict) -> dict:
    ctx = {
        "company_name": company,
        "ceo_name": ceo,
        "ir_text_excerpt": packed_text[:8000],
        "company_profile": company_profile,
    }
    prompt = (
        "너는 스타트업 IR 심사역이다.\n"
        "아래 입력을 바탕으로 비즈니스 모델(BM), 산업 분야, 투자유치 단계(Stage)를 추천하라.\n"
        "추정이 어렵다면 '확인 불가'라고 쓰고 이유를 짧게 써라.\n"
        "출력은 JSON ONLY: business_model, industry, stage, reason\n\n"
        "business_model 예: SaaS/플랫폼/제조/딥테크/바이오/커머스/콘텐츠/기타\n"
        "industry 예: 모빌리티/헬스케어/핀테크/AI/교육/리테일/기타\n"
        "stage 예: Pre-seed/Seed/Pre-A/Series A/Series B+/확인 불가\n\n"
        "입력:\n"
    ) + json.dumps(ctx, ensure_ascii=False)

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return safe_json_load((resp.text or "").strip())


def detect_scale_div(section_scores: dict) -> float:
    """
    evaluator가 0~10을 내면 /2 해서 1~5 스케일로 변환해야 WeightEngine에 넣기 좋음.
    0~5를 내면 그대로(1.0).
    """
    mx = 0.0
    for k in SECTIONS_KOR:
        try:
            mx = max(mx, float(section_scores.get(k, 0) or 0))
        except Exception:
            pass
    return 2.0 if mx > 5.5 else 1.0


# -----------------------------
# 3) 엑셀 컬럼(샘플 포맷)
# -----------------------------
COLUMNS = [
    "분석 일시", "회사명", "한줄 피치", "종합 점수", "투자 추천", "분석 단계", "분석 관점",
    "문제 정의 (점수)", "문제 정의 (분석)", "문제 정의 (개선안)",
    "솔루션 & 제품 (점수)", "솔루션 & 제품 (분석)", "솔루션 & 제품 (개선안)",
    "시장 분석 (점수)", "시장 분석 (분석)", "시장 분석 (개선안)",
    "비즈니스 모델 (점수)", "비즈니스 모델 (분석)", "비즈니스 모델 (개선안)",
    "경쟁 우위 (점수)", "경쟁 우위 (분석)", "경쟁 우위 (개선안)",
    "성장 전략 (점수)", "성장 전략 (분석)", "성장 전략 (개선안)",
    "팀 역량 (점수)", "팀 역량 (분석)", "팀 역량 (개선안)",
    "재무 계획 (점수)", "재무 계획 (분석)", "재무 계획 (개선안)",
    "리스크 관리 (점수)", "리스크 관리 (분석)", "리스크 관리 (개선안)",
    "핵심 강점", "개선 필요", "최종 의견", "스토리라인"
]


# -----------------------------
# 4) Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI 심사역 (Lite+Engine)", layout="wide")
st.title("AI 심사역 powered by MARK")

st.sidebar.header("옵션")
use_company_profile = st.sidebar.toggle("회사 설명문(홈페이지/뉴스) 생성", value=True)
use_vision_fallback = st.sidebar.toggle("스캔 PDF OCR(vision) 자동 사용", value=True)
use_visual_insights = st.sidebar.toggle("표/차트/그래프 인사이트 추출", value=True)
force_ocr = st.sidebar.toggle("강제 OCR(테스트)", value=False)
show_debug = st.sidebar.toggle("디버그 표시", value=False)

uploaded_files = st.file_uploader("IR PDF 업로드(최대 10개)", type=["pdf"], accept_multiple_files=True)
run_btn = st.button("분석 실행", type="primary", disabled=not uploaded_files)

if "result_cache" not in st.session_state:
    st.session_state.result_cache = {}
if "rows" not in st.session_state:
    st.session_state.rows = []

if "selected_report_path" not in st.session_state:
    st.session_state.selected_report_path = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "list"  # "list" or "detail"


# -----------------------------
# 5) Main run
# -----------------------------
if run_btn:
    engine = WeightEngine()
    client = get_client()
    st.session_state.rows = []

    # evaluator_simple 시그니처(버전 호환)
    sig = inspect.signature(run_overall_evaluation)
    has_weights_100 = "weights_100" in sig.parameters
    has_weights = "weights" in sig.parameters

    for up in uploaded_files[:10]:
        file_key = hashlib.md5(up.getvalue()).hexdigest()[:10]
        if file_key in st.session_state.result_cache:
            st.session_state.rows.append(st.session_state.result_cache[file_key])
            continue

        # PDF 읽기
        pdf_bytes = up.getvalue()
        pdf_path = save_uploaded_pdf(up)
        pages = extract_pages(pdf_path)

        packed_text_pypdf = build_packed_text(pages, limit_chars=60000)

        total_chars = sum(len((p.get("text") or "").strip()) for p in pages)
        nonempty_pages = sum(1 for p in pages if (p.get("text") or "").strip())
        need_v = needs_vision(pages)

        packed_text = packed_text_pypdf
        ocr_text = ""
        ocr_error = ""

        if use_vision_fallback and (force_ocr or need_v):
            try:
                ocr_text = gemini_pdf_ocr_text(client, pdf_bytes, model_name="gemini-2.5-flash", max_chars=90000)
            except Exception as e:
                ocr_error = str(e)

            if force_ocr:
                packed_text = ocr_text if ocr_text else packed_text_pypdf
            else:
                if len(ocr_text.strip()) > max(500, len(packed_text_pypdf.strip())):
                    packed_text = ocr_text

        if use_visual_insights:
            try:
                vis = gemini_pdf_visual_insights(client, pdf_bytes, model_name="gemini-2.5-flash", max_chars=12000)
                if vis and vis.strip():
                    packed_text = (packed_text + "\n\n" + vis)[:90000]
            except Exception:
                pass

        if show_debug:
            st.sidebar.markdown("### [PDF 추출 디버그]")
            st.sidebar.caption(f"file: {up.name}")
            st.sidebar.caption(f"pypdf total_chars={total_chars}, nonempty_pages={nonempty_pages}, needs_vision={need_v}")
            st.sidebar.caption(f"pypdf packed len={len(packed_text_pypdf)}")
            st.sidebar.caption(f"ocr len={len(ocr_text)}")
            if ocr_error:
                st.sidebar.error(f"OCR error: {ocr_error}")
            st.sidebar.caption(f"final packed len={len(packed_text)}")

        # 기업/대표 추출
        extracted = extract_company_and_ceo(client, packed_text)
        company = extracted.get("company_name") or "unknown_company"
        ceo = extracted.get("ceo_name") or "unknown_ceo"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"data/outputs/{company}/{ts}"

        write_md(
            f"{out_dir}/00_extract.md",
            f"# 기업명/대표자 추출\n- 파일: {up.name}\n\n"
            f"- 기업명: {company}\n- 대표자: {ceo}\n\n"
            f"근거:\n{json.dumps(extracted.get('evidence', []), ensure_ascii=False, indent=2)}\n"
        )

        # 회사 설명문(홈페이지/뉴스 기반)
        profile = {}
        profile_sources = []
        if use_company_profile and company != "unknown_company":
            try:
                profile, profile_sources = generate_company_profile(client, company, ceo)
                write_md(
                    f"{out_dir}/01_company_profile.md",
                    f"# 회사 설명문(홈페이지/뉴스 기반)\n- 회사명: {company}\n- 대표자: {ceo}\n\n"
                    f"```json\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n```\n\n"
                    f"출처:\n{json.dumps(profile_sources, ensure_ascii=False, indent=2)}\n"
                )
            except Exception as e:
                profile = {"error": str(e)}

        # BM/산업/단계 추출
        cls = classify_bm_industry_stage(client, company, ceo, packed_text, profile if isinstance(profile, dict) else {})
        bm = (cls.get("business_model") or "확인 불가").strip()
        industry = (cls.get("industry") or "확인 불가").strip()
        stage = (cls.get("stage") or "확인 불가").strip()

        # ✅ WeightEngine 가중치(합 100)
        stage_key = map_stage_for_engine(stage)
        industry_key = map_industry_for_engine(industry)
        bm_key = map_bm_for_engine(bm)
        weights_100 = engine.compute_weights(stage=stage_key, industry=industry_key, business_model=bm_key)

        # ✅ evaluator 실행(버전 호환)
        if has_weights_100:
            eval_json = run_overall_evaluation(
                client=client,
                model_name="gemini-2.5-flash",
                company=company,
                ceo=ceo,
                bm=bm,
                industry=industry,
                stage=stage,
                weights_100=weights_100,
                ir_text_with_pages=packed_text,
                company_profile_json=profile if isinstance(profile, dict) else {},
            )
        elif has_weights:
            weights_1 = {k: v / 100.0 for k, v in weights_100.items()}
            eval_json = run_overall_evaluation(
                client=client,
                model_name="gemini-2.5-flash",
                company=company,
                ceo=ceo,
                bm=bm,
                industry=industry,
                stage=stage,
                weights=weights_1,
                ir_text_with_pages=packed_text,
                company_profile_json=profile if isinstance(profile, dict) else {},
            )
        else:
            eval_json = run_overall_evaluation(
                client=client,
                model_name="gemini-2.5-flash",
                company=company,
                ceo=ceo,
                bm=bm,
                industry=industry,
                stage=stage,
                ir_text_with_pages=packed_text,
                company_profile_json=profile if isinstance(profile, dict) else {},
            )

        section_scores = eval_json.get("section_scores", {}) or {}
        scale_div = detect_scale_div(section_scores)

        # ✅ WeightEngine 입력(1~5)
        scores_1_to_5 = {}
        for kor, crit in KOR_TO_CRITERION.items():
            try:
                scores_1_to_5[crit] = float(section_scores.get(kor, 0) or 0) / scale_div
            except Exception:
                scores_1_to_5[crit] = 0.0

        engine_out = engine.evaluate(
            scores_1_to_5=scores_1_to_5,
            stage=stage_key,
            industry=industry_key,
            business_model=bm_key,
            apply_gating=True,
        )
        total_100 = float(engine_out["overall_100_after_gates"])
        recommend = "YES" if total_100 >= 80 else "NO"

        # ---- report.json 저장(상세 화면 렌더용) ----
        report_payload = {
            "file_name": up.name,
            "company_name": company,
            "ceo_name": ceo,
            "bm": bm,
            "industry": industry,
            "stage": stage,
	    "total_100": float(total_100),
	    "recommend": recommend,
	    "weights_100": weights_100,    # WeightEngine compute_weights 결과(합 100)
	    "engine_out": engine_out,      # gates_applied 포함
	    "eval": eval_json,             # section_scores/analysis/improvements/요약
        }
        with open(f"{out_dir}/report.json", "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)


        # ---- ai_outputs.xlsx 업서트(학습 데이터 누적) ----
        row_out = {
            "file_name": up.name,
            "company_name": company,
            "ceo_name": ceo,
            "bm": bm,
            "industry": industry,
            "stage": stage,

            # evaluator 출력 점수 스케일(0~5 또는 0~10)을 그대로 저장
            "score_problem": float(section_scores.get("문제 정의", 0) or 0),
            "score_solution": float(section_scores.get("솔루션 & 제품", 0) or 0),
            "score_market": float(section_scores.get("시장 분석", 0) or 0),
            "score_business_model": float(section_scores.get("비즈니스 모델", 0) or 0),
            "score_competition": float(section_scores.get("경쟁 우위", 0) or 0),
            "score_growth": float(section_scores.get("성장 전략", 0) or 0),
            "score_team": float(section_scores.get("팀 역량", 0) or 0),
            "score_finance": float(section_scores.get("재무 계획", 0) or 0),
            "score_risk": float(section_scores.get("리스크 관리", 0) or 0),

            "ai_raw_total_score": float(total_100),
            "ai_recommend_raw": recommend,

            "model_name": "gemini-2.5-flash",
            "prompt_version": "v_engine_applied",
        }
        upsert_ai_output(row_out)

        # 04_overall.md
        write_md(
            f"{out_dir}/04_overall.md",
            f"# 종합 평가(요약)\n- 회사명: {company}\n- 대표자: {ceo}\n- BM: {bm}\n- 산업: {industry}\n- 단계: {stage}\n"
            f"- 총점(0~100): {total_100}\n- 추천(80+): {recommend}\n\n"
            f"## 한줄 피치\n{eval_json.get('one_line_pitch', '')}\n\n"
            f"## 최종 의견\n{eval_json.get('final_opinion', '')}\n"
        )

        # 엑셀 row 구성
        row = {c: "" for c in COLUMNS}
        row["분석 일시"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["회사명"] = company
        row["한줄 피치"] = eval_json.get("one_line_pitch", "")
        row["종합 점수"] = float(total_100)
        row["투자 추천"] = recommend
        row["분석 단계"] = stage
        row["분석 관점"] = "투자자 관점"

        analysis = eval_json.get("section_analysis", {}) or {}
        improvements = eval_json.get("section_improvements", {}) or {}

        def fill(sec_name: str, col_prefix: str):
            s = float(section_scores.get(sec_name, 0) or 0)
            row[f"{col_prefix} (점수)"] = s
            row[f"{col_prefix} (분석)"] = analysis.get(sec_name, "")
            row[f"{col_prefix} (개선안)"] = improvements.get(sec_name, "")

        fill("문제 정의", "문제 정의")
        fill("솔루션 & 제품", "솔루션 & 제품")
        fill("시장 분석", "시장 분석")
        fill("비즈니스 모델", "비즈니스 모델")
        fill("경쟁 우위", "경쟁 우위")
        fill("성장 전략", "성장 전략")
        fill("팀 역량", "팀 역량")
        fill("재무 계획", "재무 계획")
        fill("리스크 관리", "리스크 관리")

        row["핵심 강점"] = eval_json.get("key_strengths", "")
        row["개선 필요"] = eval_json.get("needs_improvement", "")
        row["최종 의견"] = eval_json.get("final_opinion", "")
        row["스토리라인"] = eval_json.get("storyline", "")

        st.session_state.result_cache[file_key] = row
        st.session_state.rows.append(row)

        append_history({
            "company_name": company,
            "ceo_name": ceo,
            "bm": bm,
            "industry": industry,
            "stage": stage,
            "total_score": float(total_100),
            "recommendation": recommend,
            "file_name": up.name,
            "output_path": out_dir
        })


# -----------------------------
# Results table + download
# -----------------------------
if st.session_state.rows:
    st.subheader("결과 테이블(샘플 엑셀 컬럼 동일)")
    df = pd.DataFrame(st.session_state.rows, columns=COLUMNS)
    st.dataframe(df, use_container_width=True, height=420)

    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    st.download_button(
        "엑셀 다운로드",
        data=buf.getvalue(),
        file_name="InnoForest_Detailed_Report_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("## 히스토리(기존 history.xlsx)")
hist = load_history()
st.dataframe(hist, use_container_width=True)

st.markdown("## AI Outputs (튜닝용 데이터셋)")
try:
    df_ai = load_ai_outputs()
    st.dataframe(df_ai, use_container_width=True, height=260)

    buf2 = io.BytesIO()
    df_ai.to_excel(buf2, index=False)
    st.download_button(
        "ai_outputs.xlsx 다운로드",
        data=buf2.getvalue(),
        file_name="ai_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(f"저장 경로: {AI_OUTPUT_PATH}")
except Exception as e:
    st.warning(f"AI Outputs 표시 실패: {e}")

# -----------------------------
# 상세 리포트 렌더 함수
# -----------------------------
def radar_chart(scores_dict, labels_kor):
    values = [float(scores_dict.get(k, 0) or 0) for k in labels_kor]
    N = len(values)
    angles = [2 * math.pi * n / N for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_kor, fontsize=8)
    return fig

def render_report(report: dict):
    eval_json = report.get("eval", {}) or {}
    section_scores = eval_json.get("section_scores", {}) or {}
    section_analysis = eval_json.get("section_analysis", {}) or {}
    section_improvements = eval_json.get("section_improvements", {}) or {}

    weights_100 = report.get("weights_100", {}) or {}
    engine_out = report.get("engine_out", {}) or {}
    gates = engine_out.get("gates_applied", []) or []

    st.markdown(f"## {report.get('company_name','')} — 분석 리포트")
    st.caption(
        f"file: {report.get('file_name','')} | stage: {report.get('stage','')} | "
        f"bm: {report.get('bm','')} | industry: {report.get('industry','')}"
    )
    st.metric("총점(0~100)", report.get("total_100", ""))
    st.caption(f"추천(80+): {report.get('recommend','')}")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Visual Score Balance")
        fig = radar_chart(section_scores, SECTIONS_KOR)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Executive Summary")
        st.write(eval_json.get("final_opinion", "") or "")

        if gates:
            st.subheader("Gatekeeper's Verdict")
            for g in gates[:5]:
                st.write(f"- **{g.get('gate','')}**: {g.get('note','')}")
                st.caption(f"criterion={g.get('criterion')} | before={g.get('before_100')} → after={g.get('after_100')}")

    with right:
        st.subheader("Detailed Analysis")
        for k in SECTIONS_KOR:
            crit_key = KOR_TO_CRITERION.get(k)
            w = float(weights_100.get(crit_key, 0) or 0)
            s = float(section_scores.get(k, 0) or 0)
            weighted = round(s * (w / 100.0), 2)

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {k}")
                with c2:
                    st.write(f"**{s}**")
                    st.caption(f"w={w:.1f}%")

                st.caption(f"Weighted: {weighted}")

                st.markdown("**ANALYSIS**")
                st.write(section_analysis.get(k, "") or "")

                st.markdown("**IMPROVEMENT**")
                st.write(section_improvements.get(k, "") or "")

# -----------------------------
# 상세 화면 모드면 먼저 보여주고 종료
# -----------------------------
if st.session_state.view_mode == "detail":
    if st.button("← 리스트로 돌아가기"):
        st.session_state.view_mode = "list"
        st.session_state.selected_report_path = None
        st.rerun()

    rp = st.session_state.selected_report_path
    if not rp or not os.path.exists(rp):
        st.error("report.json을 찾을 수 없습니다.")
    else:
        with open(rp, "r", encoding="utf-8") as f:
            report = json.load(f)
        render_report(report)

    st.stop()

# -----------------------------
# 리스트(히스토리 기반) + '분석결과 보기' 버튼
# -----------------------------
st.markdown("## 분석 파일 리스트")
hist = load_history()

if hist.empty:
    st.info("히스토리가 없습니다.")
else:
    # 최신순(최근 50개만)
    view = hist.tail(50).reset_index(drop=True)

    for i, r in view.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 2])
        with col1:
            st.write(f"**{r.get('company_name','')}**  ({r.get('file_name','')})")
        with col2:
            st.write(f"Score: {r.get('total_score','')}")
        with col3:
            st.write(f"Rec: {r.get('recommendation','')}")
        with col4:
            st.write(r.get("stage", ""))
        with col5:
            if st.button("분석결과 보기", key=f"view_{i}"):
                report_path = os.path.join(str(r.get("output_path","")), "report.json")
                st.session_state.selected_report_path = report_path
                st.session_state.view_mode = "detail"
                st.rerun()
