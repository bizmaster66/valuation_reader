import os
import json
from datetime import datetime
import hashlib

import streamlit as st

from src.evaluator import run_evaluation, compute_weighted_total, summarize_short, run_detail_feedback
from src.gemini_client import get_client
from src.pdf_reader import extract_pages
from src.storage import append_history, load_history
from src.startup_analyzer_adapter import (
    generate_company_profile,
    extract_industry_keywords,
    generate_industry_report,
)
from src.presets import EVAL_ITEMS, merge_presets


# -----------------------------
# Helpers
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


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        return {"error": "JSON parse failed", "raw": text}


def build_packed_text(pages, limit_chars: int = 60000) -> str:
    parts = []
    for p in pages:
        t = (p.get("text") or "").strip()
        if t:
            parts.append(f"[PAGE {p['page']}]\n{t}")
    packed = "\n\n".join(parts)
    return packed[:limit_chars]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI 심사역 (MVP+)", layout="wide")
st.title("AI 심사역 (MVP+) — PDF → 기업명/대표자 → 회사정보/산업리포트 → BM/산업/단계+가중치")

if "eval_cache" not in st.session_state:
    st.session_state.eval_cache = {}
if "detail_cache" not in st.session_state:
    st.session_state.detail_cache = {}

st.sidebar.header("옵션")
use_company_profile = st.sidebar.toggle("회사 정보 정의(홈페이지/뉴스) 실행", value=True)
use_industry_report = st.sidebar.toggle("산업 리포트 생성", value=True)
use_classification = st.sidebar.toggle("BM/산업/단계 추천 + 가중치 UI", value=True)

if st.sidebar.button("전체 리셋"):
    st.session_state.run_pipeline = False
    st.session_state.eval_cache = {}
    st.session_state.detail_cache = {}
    st.rerun()

uploaded_files = st.file_uploader(
    "IR PDF 업로드 (최대 10개)",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("분석 실행", type="primary", disabled=not uploaded_files)

if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False

if run_btn:
    st.session_state.run_pipeline = True



# -----------------------------
# Main
# -----------------------------
if st.session_state.run_pipeline:
    try:
        client = get_client()
    except Exception as e:
        st.error(f"Gemini 클라이언트 생성 실패: {e}")
        st.stop()

    for up in uploaded_files[:10]:
        import hashlib
        file_bytes = up.getvalue()
        file_key = hashlib.md5(file_bytes).hexdigest()[:10]
        key_base = file_key
        cache_key = file_key

        st.markdown("---")
        st.subheader(f"파일: {up.name}")

        pdf_path = save_uploaded_pdf(up)
        pages = extract_pages(pdf_path)
        packed_text = build_packed_text(pages, limit_chars=60000)

        # 1) 기업명/대표자 추출
        extract_prompt = (
            "너는 IR PDF를 읽고 '기업명'과 '대표자 성명'을 찾아야 한다.\n"
            "추정하지 말고, 문서에 명시된 근거 페이지와 짧은 인용(문장 일부)을 함께 제시하라.\n\n"
            "출력은 JSON ONLY이며 필드는 다음과 같다:\n"
            "- company_name\n"
            "- ceo_name\n"
            "- evidence: page, quote\n\n"
            "IR 텍스트:\n"
        ) + packed_text

        resp = client.models.generate_content(model="gemini-2.5-flash", contents=extract_prompt)
        extracted = safe_json_load((resp.text or "").strip())

        company = extracted.get("company_name") or "unknown_company"
        ceo = extracted.get("ceo_name") or "unknown_ceo"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"data/outputs/{company}/{ts}"

        # 00_extract.md 저장
        md_extract = (
            f"# 기업명/대표자 추출 결과\n"
            f"- 파일: {up.name}\n\n"
            f"## 추출\n"
            f"- 기업명: {company}\n"
            f"- 대표자: {ceo}\n\n"
            f"## 근거(evidence)\n"
            f"{json.dumps(extracted.get('evidence', []), ensure_ascii=False, indent=2)}\n"
        )
        write_md(f"{out_dir}/00_extract.md", md_extract)

        tab1, tab2, tab3, tab4 = st.tabs(["00 추출", "01 회사 정보", "02 산업 리포트", "03 분류/가중치"])

        with tab1:
            st.json(extracted)
            st.caption(f"저장됨: {out_dir}/00_extract.md")

        profile = {}
        profile_sources = []
        industry_text = ""
        industry_sources = []

        # 2) 회사 정보 정의
        if use_company_profile and company != "unknown_company":
            with tab2:
                st.info("회사 정보 정의 중(홈페이지/뉴스 검색 기반)...")
            try:
                profile, profile_sources = generate_company_profile(client, company, ceo)

                md_profile = (
                    f"# 회사 정보 정의(홈페이지/뉴스 기반)\n"
                    f"- 회사명: {company}\n"
                    f"- 대표자: {ceo}\n\n"
                    f"## 결과(JSON)\n"
                    f"```json\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n```\n\n"
                    f"## 출처(grounding)\n"
                    f"{json.dumps(profile_sources, ensure_ascii=False, indent=2)}\n"
                )
                write_md(f"{out_dir}/01_company_profile.md", md_profile)

                with tab2:
                    st.success("완료")
                    st.json(profile)
                    if profile_sources:
                        st.caption("출처 일부(상위 10개)")
                        st.write(profile_sources[:10])
                    st.caption(f"저장됨: {out_dir}/01_company_profile.md")
            except Exception as e:
                with tab2:
                    st.error(f"회사 정보 정의 실패: {e}")
        else:
            with tab2:
                st.caption("옵션이 꺼져있거나 회사명이 확인 불가여서 회사 정보 정의를 건너뜁니다.")

        # 3) 산업 리포트
        if use_industry_report and profile:
            with tab3:
                st.info("산업 리포트 생성 중(검색 기반)...")
            try:
                kws = extract_industry_keywords(profile)
                industry_text, industry_sources = generate_industry_report(client, kws)

                md_industry = (
                    f"# 산업 리포트(검색 기반)\n"
                    f"- 키워드: {', '.join(kws)}\n\n"
                    f"## 리포트\n"
                    f"{industry_text}\n\n"
                    f"## 출처(grounding)\n"
                    f"{json.dumps(industry_sources, ensure_ascii=False, indent=2)}\n"
                )
                write_md(f"{out_dir}/02_industry_report.md", md_industry)

                with tab3:
                    st.success("완료")
                    st.markdown(industry_text)
                    if industry_sources:
                        st.caption("출처 일부(상위 10개)")
                        st.write(industry_sources[:10])
                    st.caption(f"저장됨: {out_dir}/02_industry_report.md")
            except Exception as e:
                with tab3:
                    st.error(f"산업 리포트 생성 실패: {e}")
        else:
            with tab3:
                st.caption("옵션이 꺼져있거나 profile이 없어서 산업 리포트를 건너뜁니다.")

        # 4) BM/산업/단계 추천 + 가중치 UI
        bm_final = ""
        industry_final = ""
        stage_final = ""
        weights_final = {}

        if use_classification:
            context = {
                "company_name": company,
                "ceo_name": ceo,
                "ir_text_excerpt": packed_text[:8000],
                "company_profile": profile if profile else {},
            }

            classify_prompt = (
                "너는 스타트업 IR 심사역이다.\n"
                "아래 입력을 바탕으로 비즈니스 모델(BM), 산업 분야(Industry), 투자유치 단계(Stage)를 추천하라.\n"
                "추정이 어렵다면 '확인 불가'라고 쓰고 이유를 짧게 써라.\n"
                "출력은 JSON ONLY이며 필드는 다음과 같다:\n"
                "- business_model (예: SaaS/플랫폼/제조/딥테크/바이오/커머스/콘텐츠/기타)\n"
                "- industry (예: 모빌리티/헬스케어/핀테크/AI/교육/리테일/기타)\n"
                "- stage (예: Pre-seed/Seed/Series A/Series B+/확인 불가)\n"
                "- reason (근거 3~6문장)\n\n"
                "입력:\n"
            ) + json.dumps(context, ensure_ascii=False)

            try:
                r_cls = client.models.generate_content(model="gemini-2.5-flash", contents=classify_prompt)
                cls = safe_json_load((r_cls.text or "").strip())
            except Exception as e:
                cls = {"business_model": "", "industry": "", "stage": "", "reason": f"추천 실패: {e}"}

            bm_s = (cls.get("business_model") or "").strip()
            ind_s = (cls.get("industry") or "").strip()
            stg_s = (cls.get("stage") or "").strip()
            reason = (cls.get("reason") or "").strip()

            bm_options = ["SaaS", "플랫폼", "제조", "딥테크", "바이오", "커머스", "콘텐츠", "기타"]
            ind_options = ["모빌리티", "헬스케어", "핀테크", "AI", "교육", "리테일", "기타"]
            stage_options = ["Pre-seed", "Seed", "Series A", "Series B+", "확인 불가"]

            def pick_default(options, suggested):
                return suggested if suggested in options else options[-1]

            # Streamlit 위젯 키(파일별 충돌 방지)
            key_base = file_key

            with tab4:
                st.subheader("BM / 산업 / 투자단계 추천 및 확정")
                c1, c2, c3 = st.columns(3)
                with c1:
                    bm_final = st.selectbox(
                        "비즈니스 모델(BM)",
                        bm_options,
                        index=bm_options.index(pick_default(bm_options, bm_s)),
                        key=f"{key_base}_bm",
                    )
                with c2:
                    industry_final = st.selectbox(
                        "산업 분야(Industry)",
                        ind_options,
                        index=ind_options.index(pick_default(ind_options, ind_s)),
                        key=f"{key_base}_ind",
                    )
                with c3:
                    stage_final = st.selectbox(
                        "투자유치 단계(Stage)",
                        stage_options,
                        index=stage_options.index(pick_default(stage_options, stg_s)),
                        key=f"{key_base}_stg",
                    )

                if reason:
                    st.caption(f"추천 근거: {reason}")

                st.divider()
                st.subheader("가중치(추천 preset) → 수정 가능")

                w = merge_presets(bm_final, stage_final)
                edited = {}
                for k in EVAL_ITEMS:
                    edited[k] = st.slider(
                        k,
                        min_value=0.0,
                        max_value=0.30,
                        value=float(w.get(k, 0.0)),
                        step=0.01,
                        key=f"{key_base}_w_{k}",
                    )

                ssum = sum(edited.values()) or 1.0
                weights_final = {k: v / ssum for k, v in edited.items()}
                st.caption(f"정규화 합계: {sum(weights_final.values()):.2f}")

            md_cls = (
                f"# BM/산업/투자단계 및 가중치 확정\n"
                f"- 회사명: {company}\n"
                f"- 대표자: {ceo}\n\n"
                f"## 추천\n"
                f"- BM(추천): {bm_s}\n"
                f"- 산업(추천): {ind_s}\n"
                f"- 단계(추천): {stg_s}\n\n"
                f"## 확정\n"
                f"- BM(확정): {bm_final}\n"
                f"- 산업(확정): {industry_final}\n"
                f"- 단계(확정): {stage_final}\n\n"
                f"## 추천 근거\n{reason}\n\n"
                f"## 가중치(정규화)\n"
                f"```json\n{json.dumps(weights_final, ensure_ascii=False, indent=2)}\n```\n"
            )
            write_md(f"{out_dir}/03_classification_and_weights.md", md_cls)

        # 5) 종합 평가(0~5, 페이지 근거, 총점 0~100, 80점 추천서 분기)
        with tab4:
            st.divider()
            st.subheader("AI 심사역 종합 평가")

            do_eval = st.button("종합 평가 생성", key=f"{key_base}_do_eval")

            # 캐시 상태 표시(항상)
            cached = st.session_state.eval_cache.get(cache_key)
            st.caption(f"[디버그] cache_key={cache_key} / cached={'YES' if cached else 'NO'}")
            if cached:
                st.metric("총점(0~100)", cached["total_score"])
                st.caption(f"톤: {cached['tone']}")
                st.caption(f"저장 경로: {cached['out_dir']}/04_evaluation.md")

            # 버튼 눌렀을 때만 평가 수행
            if do_eval:
                try:
                    bm_for_eval = bm_final or "확인 불가"
                    industry_for_eval = industry_final or "확인 불가"
                    stage_for_eval = stage_final or "확인 불가"

                    # 가중치 기본값 보정(확인불가면 안전한 기본 프리셋 사용)
                    if weights_final:
                        weights_for_eval = weights_final
                    else:
                        bm_seed = bm_for_eval if bm_for_eval != "확인 불가" else "SaaS"
                        stg_seed = stage_for_eval if stage_for_eval != "확인 불가" else "Seed"
                        weights_for_eval = merge_presets(bm_seed, stg_seed)

                    evaluation = run_evaluation(
                        client=client,
                        model_name="gemini-2.5-flash",
                        company=company,
                        ceo=ceo,
                        packed_text=packed_text,
                        bm=bm_for_eval,
                        industry=industry_for_eval,
                        stage=stage_for_eval,
                        weights=weights_for_eval,
                    )

                    items = evaluation.get("items", []) or []
                    total_score = float(compute_weighted_total(items, weights_for_eval))
                    tone = "recommend" if total_score >= 80 else "critical"

                    evaluation["total_score_100"] = total_score
                    evaluation["tone"] = tone

                    # 캐시에 저장
                    st.session_state.eval_cache[cache_key] = {
                        "evaluation": evaluation,
                        "total_score": total_score,
                        "tone": tone,
                        "out_dir": out_dir,
                    }

                    # 파일 저장
                    write_md(f"{out_dir}/04_evaluation.json", json.dumps(evaluation, ensure_ascii=False, indent=2))

                    short = summarize_short(items)
                    md_eval = f"""# IR 종합 분석 평가
- 회사명: {company}
- 대표자: {ceo}
- 총점(0~100): {total_score}
- 톤: {tone}

## 한줄 요약
{short}

## 항목별 평가(0~5)
"""
                    for it in items:
                        name = it.get("name", "")
                        score = it.get("score", "")
                        exempt = it.get("exempt", False)
                        pages = it.get("evidence_pages", [])
                        md_eval += f"\n### {name} — {score}점" + (" (면제)" if exempt else "") + "\n"
                        md_eval += f"- 근거 페이지: {pages}\n"
                        md_eval += f"- ✅ 강점: {it.get('strengths','')}\n"
                        md_eval += f"- ❌ 보완: {it.get('weaknesses','')}\n"
                        if (not exempt) and (float(score or 0) <= 3):
                            md_eval += f"- 💡 제안: {it.get('suggestions','')}\n"
                        md_eval += f"- ❓ 질문: {it.get('investor_questions','')}\n"

                    md_eval += "\n## 종합 코멘트\n" + (evaluation.get("overall_commentary", "") or "")
                    if tone == "recommend":
                        rec = evaluation.get("recommendation_note", "") or ""
                        if rec:
                            md_eval += "\n\n## 추천 의견(80점 이상)\n" + rec

                    write_md(f"{out_dir}/04_evaluation.md", md_eval)

                    st.success("종합 평가 생성 완료")
                    st.metric("총점(0~100)", total_score)
                    st.caption(short)
                    st.caption(f"저장됨: {out_dir}/04_evaluation.md")

                except Exception as e:
                    st.error(f"종합 평가 생성 실패: {e}")
                
                append_history({
                    "company_name": company,
                    "ceo_name": ceo,
                    "bm": bm_for_eval,
                    "industry": industry_for_eval,
                    "stage": stage_for_eval,
                    "total_score": total_score,
                    "recommendation": "YES" if total_score >= 80 else "NO",
                    "file_name": up.name,
                    "output_path": out_dir
                })



        # 6) 상세 피드백(요청 시 생성)  ✅캐시 기반으로 안정화
        with tab4:
            st.divider()
            st.subheader("상세 피드백(요청 시 생성)")
            do_detail = st.button("상세 피드백 생성", key=f"{key_base}_do_detail")

            # 캐시된 상세피드백이 있으면 항상 보여주기
            cached_d = st.session_state.detail_cache.get(cache_key)
            if cached_d:
                st.success("상세 피드백이 이미 생성되어 있습니다(리셋되지 않음).")
                st.caption(f"저장 경로: {cached_d['out_dir']}/05_detail_feedback.md")
                st.download_button(
                    "상세 피드백 md 다운로드",
                    data=cached_d["detail_md"].encode("utf-8"),
                    file_name=f"{company}_detail_feedback.md",
                    key=f"{key_base}_dl_detail_cached",
                )

        if do_detail:
            # ✅ 종합평가가 캐시에 있는지 확인
            cached_eval = st.session_state.eval_cache.get(cache_key)
            if not cached_eval or not cached_eval.get("evaluation"):
                with tab4:
                    st.warning("상세 피드백을 만들려면 먼저 '종합 평가 생성'을 실행해야 합니다.")
            else:
                eval_json = cached_eval["evaluation"]

                try:
                    with tab4:
                        with st.spinner("상세 피드백 생성 중..."):
                            detail_md = run_detail_feedback(
                                client=client,
                                model_name="gemini-2.5-flash",
                                company=company,
                                ceo=ceo,
                                bm=bm_final or "확인 불가",
                                industry=industry_final or "확인 불가",
                                stage=stage_final or "확인 불가",
                                evaluation_json=eval_json,
                            )

                    # 파일 저장
                    write_md(f"{out_dir}/05_detail_feedback.md", detail_md)

                    # ✅ 캐시에 저장(리셋 방지)
                    st.session_state.detail_cache[cache_key] = {
                        "detail_md": detail_md,
                        "out_dir": out_dir,
                    }

                    with tab4:
                        st.success("상세 피드백 생성 완료")
                        st.caption(f"저장됨: {out_dir}/05_detail_feedback.md")
                        st.download_button(
                            "상세 피드백 md 다운로드",
                            data=detail_md.encode("utf-8"),
                            file_name=f"{company}_detail_feedback.md",
                            key=f"{key_base}_dl_detail_new",
                        )

                except Exception as e:
                    with tab4:
                        st.error(f"상세 피드백 생성 실패: {e}")


        

# -----------------------------
# History view
# -----------------------------
st.markdown("## 히스토리")
hist = load_history()
st.dataframe(hist, use_container_width=True)

if not hist.empty:
    import io
    buf = io.BytesIO()
    hist.to_excel(buf, index=False)
    st.download_button("히스토리 엑셀 다운로드", data=buf.getvalue(), file_name="history.xlsx")
