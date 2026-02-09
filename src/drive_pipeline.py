import io
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
from src.md_parser import build_ir_text, extract_ceo_name, extract_company_name, normalize_company_for_filename
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
) -> Tuple[List[Dict[str, Any]], str]:
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
    for f in target_files:
        filename = f["name"]
        if filename in processed:
            continue

        md_bytes = download_file(service, f["id"], f.get("mimeType"))
        md_text = md_bytes.decode("utf-8", errors="ignore")

        company = extract_company_name(md_text)
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
        )

        from src.gemini_client import get_client

        client = get_client()
        eval_json = run_evaluation(client, model_name=model_name, prompt=prompt)

        scores = eval_json.get("section_scores", {})
        total_score = eval_json.get("total_score_90", 0)
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
                        "total_score_90": total_score,
                        "source_filename": filename,
                    }
                ]
            )
            meta.to_excel(writer, index=False, sheet_name="summary")

        excel_name = rules["output"]["filename_template"].format(date=date_str, company=company_safe)
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
        investor_docx = build_investor_report_docx(
            company=company,
            sections=investor_report,
            highlights=highlights if isinstance(highlights, list) else [str(highlights)],
            achievement=investor_report.get(headings[3], "") if len(headings) > 3 else "",
            funding_plan=investor_report.get(headings[4], "") if len(headings) > 4 else "",
            recommendation=investor_report.get("Recommendation", ""),
            headings=headings,
        )
        investor_name = rules["output"]["reports"]["investor_report"]["filename_template"].format(
            date=date_str, company=company_safe
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
        feedback_docx = build_feedback_report_docx(company, eval_json.get("feedback_report", {}))
        feedback_name = rules["output"]["reports"]["detailed_feedback"]["filename_template"].format(
            date=date_str, company=company_safe
        )
        upload_bytes(
            service,
            result_folder_id,
            feedback_name,
            feedback_docx,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            overwrite=True,
        )

        processed.append(filename)
        results.append(
            {
                "company_name": company,
                "total_score_90": total_score,
                "stage_estimate": stage_estimate,
                "source_filename": filename,
                "excel_file": excel_name,
                "investor_report_file": investor_name,
                "feedback_file": feedback_name,
                "eval": eval_json,
            }
        )

    save_processed_index(service, result_folder_id, processed)
    return results, result_folder_id
