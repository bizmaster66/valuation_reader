import io
import json
from datetime import datetime
import time
from typing import Any, Dict, List, Tuple, Optional, Callable
import random

import pandas as pd
from pypdf import PdfReader

from src.config_loader import load_yaml
from src.docx_template import extract_headings_from_sample
from src.drive_client import (
    download_file,
    find_or_create_folder,
    get_drive_service,
    list_files_in_folder,
    load_processed_index,
    save_processed_index,
    upload_bytes,
)
from src.ir_evaluator import build_eval_prompt, run_evaluation
from src.md_parser import (
    build_ir_text,
    extract_ceo_name,
    extract_company_name,
    extract_company_candidates,
    normalize_company_for_filename,
)
from src.report_writer import build_feedback_report_docx, build_investor_report_docx


def _extract_pdf_text(pdf_bytes: bytes, max_chars: int = 120000) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt.strip())
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n\n".join(parts)[:max_chars]


def _load_knowledge_text(service, ir_strategy_file_id: str = "", local_path: str = "") -> str:
    if ir_strategy_file_id:
        meta = (
            service.files()
            .get(fileId=ir_strategy_file_id, fields="id,mimeType", supportsAllDrives=True)
            .execute()
        )
        pdf_bytes = download_file(service, meta["id"], meta.get("mimeType"))
        return _extract_pdf_text(pdf_bytes)
    if local_path:
        with open(local_path, "rb") as f:
            return _extract_pdf_text(f.read())
    return ""


def _load_sample_headings(service, sample_docx_id: str = "", local_path: str = "") -> List[str]:
    if sample_docx_id:
        meta = (
            service.files()
            .get(fileId=sample_docx_id, fields="id,mimeType", supportsAllDrives=True)
            .execute()
        )
        docx_bytes = download_file(
            service,
            meta["id"],
            meta.get("mimeType"),
            export_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        return extract_headings_from_sample(docx_bytes)
    if local_path:
        with open(local_path, "rb") as f:
            return extract_headings_from_sample(f.read())
    return extract_headings_from_sample(b"")


def _filter_md_files(files: List[Dict], suffix: str) -> List[Dict]:
    return [f for f in files if f.get("name", "").endswith(suffix)]


def run_drive_evaluation(
    folder_id: str,
    model_name: str,
    ir_strategy_file_id: str = "",
    sample_docx_id: str = "",
    local_ir_strategy_path: str = "",
    local_sample_docx_path: str = "",
    progress_cb: Optional[Callable[[Dict[str, int]], None]] = None,
    difficulty_mode: str = "critical",
    reeval_filenames: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], Dict[str, int]]:
    rules = load_yaml("config/eval_rules.yaml")
    questions = load_yaml("config/questions.yaml").get("questions", {})
    stage_rules = load_yaml("config/stage_rules.yaml")
    sections = [s["name"] for s in rules.get("sections", [])]

    service = get_drive_service()
    files = list_files_in_folder(service, folder_id)
    target_files = _filter_md_files(files, rules["input"]["md_filename_suffix"])

    result_folder_id = find_or_create_folder(service, folder_id, rules["output"]["result_folder_name"])
    processed = load_processed_index(service, result_folder_id)

    knowledge_cfg = rules.get("knowledge_sources", {})
    ir_strategy_file_id = ir_strategy_file_id or knowledge_cfg.get("ir_strategy_file_id", "")
    sample_docx_id = sample_docx_id or knowledge_cfg.get("investor_report_sample_file_id", "")
    local_ir_strategy_path = local_ir_strategy_path or knowledge_cfg.get("local_ir_strategy_path", "")
    local_sample_docx_path = local_sample_docx_path or knowledge_cfg.get(
        "local_investor_report_sample_path", ""
    )

    knowledge_text = _load_knowledge_text(
        service, ir_strategy_file_id=ir_strategy_file_id, local_path=local_ir_strategy_path
    )
    headings = _load_sample_headings(
        service, sample_docx_id=sample_docx_id, local_path=local_sample_docx_path
    )

    results = []
    status_rows = []
    reeval_filenames = set(reeval_filenames or [])
    total_files = len(target_files)
    already_processed = len([f for f in target_files if f["name"] in processed])
    pending_files = total_files - already_processed
    counts = {
        "total": total_files,
        "already_processed": already_processed,
        "pending": pending_files,
        "completed": 0,
        "failed": 0,
    }
    if progress_cb:
        progress_cb(counts)
    for f in target_files:
        filename = f["name"]
        if filename in processed and filename not in reeval_filenames:
            status_rows.append(
                {
                    "filename": filename,
                    "company_name": "",
                    "status": "already_processed",
                    "error": "",
                }
            )
            continue

        md_bytes = download_file(service, f["id"], f.get("mimeType"))
        md_text = md_bytes.decode("utf-8", errors="ignore")

        company = extract_company_name(md_text, filename)
        ceo = extract_ceo_name(md_text)
        ir_text = build_ir_text(md_text)

        prompt = build_eval_prompt(
            company=company,
            ceo=ceo,
            sections=sections,
            questions_by_section=questions,
            stage_rules=stage_rules,
            knowledge_text=knowledge_text,
            md_text=ir_text,
            headings=headings,
            total_score_max=rules["scoring"]["total_score_max"],
            difficulty_mode=difficulty_mode,
        )

        from src.gemini_client import get_client

        client = get_client()
        eval_json = None
        error_msg = ""
        for attempt in range(3):  # 1 try + 2 retries
            try:
                eval_json = run_evaluation(client, model_name=model_name, prompt=prompt)
                break
            except Exception as e:
                error_msg = str(e)
                if attempt < 2:
                    time.sleep(5)
        if not eval_json or isinstance(eval_json, dict) and eval_json.get("error"):
            status_rows.append(
                {
                    "filename": filename,
                    "company_name": company,
                    "status": "failed",
                    "error": error_msg or eval_json.get("error", "") if isinstance(eval_json, dict) else "",
                }
            )
            counts["failed"] += 1
            counts["pending"] = max(0, counts["pending"] - 1)
            if progress_cb:
                progress_cb(counts)
            continue

        # If company name seems off, try one re-run with better candidate
        candidates = extract_company_candidates(md_text, filename)
        combined_text = json.dumps(
            {
                "investor_report": eval_json.get("investor_report", {}),
                "feedback_report": eval_json.get("feedback_report", {}),
            },
            ensure_ascii=False,
        )
        if company and company not in combined_text:
            for c in candidates:
                if c and c in combined_text:
                    company = c
                    prompt = build_eval_prompt(
                        company=company,
                        ceo=ceo,
                        sections=sections,
                        questions_by_section=questions,
                        stage_rules=stage_rules,
                        knowledge_text=knowledge_text,
                        md_text=ir_text,
                        headings=headings,
                        total_score_max=rules["scoring"]["total_score_max"],
                        difficulty_mode=difficulty_mode,
                    )
                    eval_json = run_evaluation(client, model_name=model_name, prompt=prompt)
                    break

        scores = eval_json.get("section_scores", {})
        logic_score = float(eval_json.get("logic_score_10", 0) or 0)
        total_score = float(eval_json.get("total_score_100", 0) or 0)

        # Difficulty adjustment (deterministic by filename)
        def _rand_range(seed_key: str, lo: int, hi: int) -> int:
            rng = random.Random(seed_key)
            return rng.randint(lo, hi)

        bonus = 0
        if difficulty_mode == "neutral":
            bonus = _rand_range(filename + ":neutral", 3, 5)
        elif difficulty_mode == "positive":
            bonus = _rand_range(filename + ":neutral", 3, 5) + _rand_range(filename + ":positive", 3, 5)

        if bonus:
            total_score += bonus

        # Cap total score by mode and proportionally scale section+logic if needed
        cap_map = {"critical": 88, "neutral": 90, "positive": 93}
        cap = cap_map.get(difficulty_mode, 88)
        if total_score > cap:
            factor = cap / total_score if total_score > 0 else 1.0
            # scale section scores
            for k in list(scores.keys()):
                try:
                    scores[k] = round(float(scores[k]) * factor, 2)
                except Exception:
                    pass
            logic_score = round(logic_score * factor, 2)
            total_score = cap

        # persist adjusted scores back
        eval_json["section_scores"] = scores
        eval_json["logic_score_10"] = logic_score
        eval_json["total_score_100"] = total_score
        eval_json["difficulty_mode"] = difficulty_mode
        stage_estimate = eval_json.get("stage_estimate", "")

        company_safe = normalize_company_for_filename(company)
        date_str = datetime.now().strftime(rules["output"]["date_format"])

        # Excel
        df = pd.DataFrame([{k: scores.get(k, "") for k in sections}])
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="scores")
            meta = pd.DataFrame(
                [
                    {
                        "company_name": company,
                        "ceo_name": ceo,
                        "stage_estimate": stage_estimate,
                        "total_score_100": total_score,
                        "logic_score_10": eval_json.get("logic_score_10", ""),
                        "source_filename": filename,
                    }
                ]
            )
            meta.to_excel(writer, index=False, sheet_name="summary")

        excel_name = rules["output"]["filename_template"].format(date=date_str, company=company_safe)
        investor_name = rules["output"]["reports"]["investor_report"]["filename_template"].format(
            date=date_str, company=company_safe
        )
        feedback_name = rules["output"]["reports"]["detailed_feedback"]["filename_template"].format(
            date=date_str, company=company_safe
        )

        # If re-evaluation, append suffix n based on existing files in result folder
        if filename in reeval_filenames:
            existing_files = list_files_in_folder(service, result_folder_id)
            base = excel_name.rsplit(".", 1)[0]
            max_n = 0
            for ef in existing_files:
                name = ef.get("name", "")
                if name.startswith(base + "_재평가"):
                    try:
                        n_part = name.split("_재평가", 1)[1].split(".", 1)[0]
                        max_n = max(max_n, int(n_part))
                    except Exception:
                        pass
            suffix = f"_재평가{max_n + 1}"
            excel_name = base + suffix + ".xlsx"
            investor_name = investor_name.rsplit(".", 1)[0] + suffix + ".docx"
            feedback_name = feedback_name.rsplit(".", 1)[0] + suffix + ".docx"
        upload_bytes(
            service,
            result_folder_id,
            excel_name,
            excel_buf.getvalue(),
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            overwrite=True,
        )

        # Investor report docx
        investor_report = eval_json.get("investor_report", {})
        highlights = investor_report.get(headings[2], []) if len(headings) > 2 else []
        include_recommendation = total_score >= 80
        investor_docx = build_investor_report_docx(
            company=company,
            sections=investor_report,
            highlights=highlights if isinstance(highlights, list) else [str(highlights)],
            achievement=investor_report.get(headings[3], "") if len(headings) > 3 else "",
            funding_plan=investor_report.get(headings[4], "") if len(headings) > 4 else "",
            recommendation=investor_report.get("Recommendation", ""),
            headings=headings,
            include_recommendation=include_recommendation,
        )
        upload_bytes(
            service,
            result_folder_id,
            investor_name,
            investor_docx,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            overwrite=True,
        )

        # Feedback docx
        feedback_docx = build_feedback_report_docx(company, eval_json.get("feedback_report", {}), total_score)
        upload_bytes(
            service,
            result_folder_id,
            feedback_name,
            feedback_docx,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            overwrite=True,
        )

        if filename not in processed:
            processed.append(filename)
        results.append(
            {
                "company_name": company,
                "total_score_100": total_score,
                "stage_estimate": stage_estimate,
                "source_filename": filename,
                "excel_file": excel_name,
                "investor_report_file": investor_name,
                "feedback_file": feedback_name,
                "eval": eval_json,
            }
        )
        status_rows.append(
            {
                "filename": filename,
                "company_name": company,
                "status": "completed",
                "error": "",
            }
        )
        counts["completed"] += 1
        counts["pending"] = max(0, counts["pending"] - 1)
        if progress_cb:
            progress_cb(counts)

    save_processed_index(service, result_folder_id, processed)
    return results, result_folder_id, status_rows, counts
