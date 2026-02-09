import streamlit as st

from src.drive_pipeline import run_drive_evaluation
from src.config_loader import load_yaml


st.set_page_config(page_title="AI 심사역", layout="wide")
st.title("AI 심사역")

with st.sidebar:
    st.header("Drive 설정")
    folder_id = st.text_input("Google Drive 폴더 ID", value="")
    rules = load_yaml("config/eval_rules.yaml")
    knowledge_cfg = rules.get("knowledge_sources", {})
    default_ir_id = knowledge_cfg.get("ir_strategy_file_id", "")
    default_docx_id = knowledge_cfg.get("investor_report_sample_file_id", "")
    local_ir_path = knowledge_cfg.get("local_ir_strategy_path", "")
    local_docx_path = knowledge_cfg.get("local_investor_report_sample_path", "")
    ir_strategy_file_id = st.text_input("IR 전략 PDF 파일 ID (옵션)", value=default_ir_id)
    report_sample_file_id = st.text_input("투자자용 요약 샘플 DOCX 파일 ID (옵션)", value=default_docx_id)
    model_name = st.text_input("Gemini 모델", value="gemini-2.5-flash")
    if st.checkbox("Secrets 상태 보기", value=False):
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []
        st.write("Secrets keys:", keys)
        st.write("has gcp_service_account:", "gcp_service_account" in keys)
        st.write("has gemini_api_key:", "gemini_api_key" in keys)
        if "gcp_service_account" in keys:
            sa = st.secrets.get("gcp_service_account")
            st.write("gcp_service_account type:", type(sa).__name__)

run_btn = st.button("분석 실행", type="primary", disabled=not folder_id)

if "results" not in st.session_state:
    st.session_state.results = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

if run_btn:
    with st.spinner("Drive에서 파일을 읽고 평가 중입니다..."):
        results, result_folder_id, status_rows = run_drive_evaluation(
            folder_id=folder_id,
            model_name=model_name,
            ir_strategy_file_id=ir_strategy_file_id,
            sample_docx_id=report_sample_file_id,
            local_ir_strategy_path=local_ir_path,
            local_sample_docx_path=local_docx_path,
        )
    st.session_state.results = results
    st.session_state.selected_idx = None
    st.session_state.status_rows = status_rows
    completed = sum(1 for r in status_rows if r["status"] == "completed")
    total = len(status_rows)
    progress = completed / total if total else 0
    st.progress(progress)
    st.success(f"완료: {completed}/{total}개 처리. 결과 폴더 ID: {result_folder_id}")

if "status_rows" not in st.session_state:
    st.session_state.status_rows = []

status_rows = st.session_state.status_rows
if status_rows:
    st.subheader("처리 상태")
    status_map = {
        "completed": "완료",
        "failed": "실패",
        "already_processed": "이미 처리됨",
    }
    table = [
        {
            "파일명": r["filename"],
            "회사명": r.get("company_name", ""),
            "상태": status_map.get(r["status"], r["status"]),
            "에러": r.get("error", ""),
        }
        for r in status_rows
    ]
    st.dataframe(table, use_container_width=True)

results = st.session_state.results
if results:
    st.subheader("평가 결과 목록")
    table = [
        {
            "회사명": r.get("company_name"),
            "총점(90)": r.get("total_score_90"),
            "단계 추정": r.get("stage_estimate"),
            "원본 파일": r.get("source_filename"),
        }
        for r in results
    ]
    st.dataframe(table, use_container_width=True)

    for i, r in enumerate(results):
        if st.button(f"보기: {r.get('company_name')} ({r.get('total_score_90')})", key=f"view_{i}"):
            st.session_state.selected_idx = i

    if st.session_state.selected_idx is not None:
        r = results[st.session_state.selected_idx]
        eval_json = r.get("eval", {})
        st.subheader(f"{r.get('company_name')} 상세")
        st.write(f"총점(90): {r.get('total_score_90')}")
        st.write(f"단계 추정: {r.get('stage_estimate')}")

        st.markdown("### 항목별 점수")
        st.json(eval_json.get("section_scores", {}))

        st.markdown("### 투자자용 요약 및 추천")
        st.json(eval_json.get("investor_report", {}))

        st.markdown("### 상세 피드백")
        st.json(eval_json.get("feedback_report", {}))
else:
    st.info("폴더 ID를 입력하고 분석을 실행하세요.")
