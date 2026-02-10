import streamlit as st

from src.drive_pipeline import run_drive_evaluation
from src.config_loader import load_yaml


st.set_page_config(page_title="ID Deck Analyzer powered by MARK", layout="wide")
st.title("ID Deck Analyzer powered by MARK")

rules = load_yaml("config/eval_rules.yaml")
knowledge_cfg = rules.get("knowledge_sources", {})
local_ir_path = knowledge_cfg.get("local_ir_strategy_path", "")
local_docx_path = knowledge_cfg.get("local_investor_report_sample_path", "")
model_name = "gemini-2.5-flash"
ir_strategy_file_id = ""
report_sample_file_id = ""

col1, col2 = st.columns([3, 1])
with col1:
    folder_id = st.text_input("Google Drive 폴더 ID", value="")
with col2:
    difficulty_mode = st.selectbox(
        "평가 난이도",
        options=["critical", "neutral", "positive"],
        index=0,
    )

run_btn = st.button("분석 실행", type="primary", disabled=not folder_id)

if "results" not in st.session_state:
    st.session_state.results = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

def _on_progress_factory(progress_bar, summary_box):
    def _cb(counts):
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        progress = completed / total if total else 0
        progress_bar.progress(progress)
        summary_box.write(
            f"총 {counts.get('total',0)}개 / 미처리 {counts.get('pending',0)}개 / 이미 처리됨 {counts.get('already_processed',0)}개 / 완료 {completed}개 / 실패 {counts.get('failed',0)}개"
        )

    return _cb


if run_btn:
    progress_bar = st.progress(0.0)
    summary_box = st.empty()
    on_progress = _on_progress_factory(progress_bar, summary_box)

    with st.spinner("Drive에서 파일을 읽고 평가 중입니다..."):
        results, result_folder_id, status_rows, counts = run_drive_evaluation(
            folder_id=folder_id,
            model_name=model_name,
            ir_strategy_file_id=ir_strategy_file_id,
            sample_docx_id=report_sample_file_id,
            local_ir_strategy_path=local_ir_path,
            local_sample_docx_path=local_docx_path,
            progress_cb=on_progress,
            difficulty_mode=difficulty_mode,
        )
    st.session_state.results = results
    st.session_state.selected_idx = None
    st.session_state.status_rows = status_rows
    st.session_state.counts = counts
    completed = counts.get("completed", 0)
    total = counts.get("total", 0)
    progress = completed / total if total else 0
    progress_bar.progress(progress)
    summary_box.write(
        f"총 {counts.get('total',0)}개 / 미처리 {counts.get('pending',0)}개 / 이미 처리됨 {counts.get('already_processed',0)}개 / 완료 {completed}개 / 실패 {counts.get('failed',0)}개"
    )
    st.success("분석 완료")

if "status_rows" not in st.session_state:
    st.session_state.status_rows = []
if "counts" not in st.session_state:
    st.session_state.counts = {}

status_rows = st.session_state.status_rows
counts = st.session_state.counts
if counts:
    st.subheader("처리 요약")
    st.write(
        f"총 {counts.get('total',0)}개 / 미처리 {counts.get('pending',0)}개 / 이미 처리됨 {counts.get('already_processed',0)}개 / 완료 {counts.get('completed',0)}개 / 실패 {counts.get('failed',0)}개"
    )
if status_rows:
    st.subheader("처리 상태")
    status_map = {
        "completed": "완료",
        "failed": "실패",
        "already_processed": "이미 처리됨",
    }
    table = [
        {
            "선택": False,
            "파일명": r["filename"],
            "회사명": r.get("company_name", ""),
            "상태": status_map.get(r["status"], r["status"]),
            "에러": r.get("error", ""),
        }
        for r in status_rows
    ]
    edited = st.data_editor(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={"선택": st.column_config.CheckboxColumn(required=False)},
        disabled=["파일명", "회사명", "상태", "에러"],
    )

    reeval_targets = [row["파일명"] for row in edited if row.get("선택")]
    if st.button("선택 재평가") and reeval_targets:
        progress_bar = st.progress(0.0)
        summary_box = st.empty()
        on_progress = _on_progress_factory(progress_bar, summary_box)
        with st.spinner("선택된 파일 재평가 중..."):
            results, result_folder_id, status_rows, counts = run_drive_evaluation(
                folder_id=folder_id,
                model_name=model_name,
                ir_strategy_file_id=ir_strategy_file_id,
                sample_docx_id=report_sample_file_id,
                local_ir_strategy_path=local_ir_path,
                local_sample_docx_path=local_docx_path,
                progress_cb=on_progress,
                difficulty_mode=difficulty_mode,
                reeval_filenames=reeval_targets,
            )
        st.session_state.results = results
        st.session_state.status_rows = status_rows
        st.session_state.counts = counts

results = st.session_state.results

if results:
    st.subheader("평가 결과 목록")
    table = [
        {
            "회사명": r.get("company_name"),
            "총점(100)": r.get("total_score_100"),
            "단계 추정": r.get("stage_estimate"),
            "원본 파일": r.get("source_filename"),
        }
        for r in results
    ]
    st.dataframe(table, use_container_width=True)

    for i, r in enumerate(results):
        if st.button(f"보기: {r.get('company_name')} ({r.get('total_score_100')})", key=f"view_{i}"):
            st.session_state.selected_idx = i

    if st.session_state.selected_idx is not None:
        r = results[st.session_state.selected_idx]
        eval_json = r.get("eval", {})
        st.subheader(f"{r.get('company_name')} 상세")
        st.write(f"총점(100): {r.get('total_score_100')}")
        st.write(f"단계 추정: {r.get('stage_estimate')}")

        st.markdown("### 항목별 점수")
        st.json(eval_json.get("section_scores", {}))

        st.markdown("### 투자자용 요약 및 추천")
        st.json(eval_json.get("investor_report", {}))

        st.markdown("### 상세 피드백")
        st.json(eval_json.get("feedback_report", {}))
else:
    st.info("폴더 ID를 입력하고 분석을 실행하세요.")
