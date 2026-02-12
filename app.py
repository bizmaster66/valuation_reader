import streamlit as st

from src.register_pipeline import run_drive_register


st.set_page_config(page_title="등기사항전부증명서 주식변동 추출", layout="wide")
st.title("등기사항전부증명서 주식변동 추출")

col1, col2 = st.columns([3, 1])
with col1:
    folder_id = st.text_input("Google Drive 폴더 ID", value="")
with col2:
    st.write("")

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

    with st.spinner("Drive에서 PDF를 읽고 스프레드시트를 생성 중입니다..."):
        results, result_folder_id, status_rows, counts, sheet_meta = run_drive_register(
            folder_id=folder_id,
            progress_cb=on_progress,
        )
    st.session_state.results = results
    st.session_state.selected_idx = None
    st.session_state.status_rows = status_rows
    st.session_state.counts = counts
    st.session_state.sheet_meta = sheet_meta
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
            "already_processed": "읽음",
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
            results, result_folder_id, status_rows, counts, sheet_meta = run_drive_register(
                folder_id=folder_id,
                progress_cb=on_progress,
                reeval_filenames=reeval_targets,
            )
        st.session_state.results = results
        st.session_state.status_rows = status_rows
        st.session_state.counts = counts
        st.session_state.sheet_meta = sheet_meta

results = st.session_state.results

if results:
    st.subheader("처리 결과")
    table = [
        {
            "기업명": r.get("company_name"),
            "원본 파일": r.get("source_filename"),
            "행 개수": r.get("row_count"),
        }
        for r in results
    ]
    st.dataframe(table, use_container_width=True)
    sheet_meta = st.session_state.get("sheet_meta") or {}
    if sheet_meta.get("url"):
        st.markdown("### 결과 스프레드시트")
        st.write(sheet_meta.get("url"))
else:
    st.info("폴더 ID를 입력하고 분석을 실행하세요.")
