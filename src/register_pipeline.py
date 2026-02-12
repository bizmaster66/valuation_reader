import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader

from src.drive_client import (
    download_file,
    find_or_create_folder,
    get_drive_service,
    get_sheets_service,
    list_files_in_folder,
    load_json_file,
    load_processed_index,
    save_json_file,
    save_processed_index,
)


CACHE_FILENAME = "_register_cache.json"


def _extract_pdf_pages_text(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def _parse_registration_number(text: str) -> str:
    m = re.search(r"등록번호\s*([0-9\-\s]{13,})", text)
    if not m:
        return ""
    digits = re.sub(r"[^0-9]", "", m.group(1))
    return digits[:14]


def _parse_par_value(text: str) -> Optional[int]:
    m = re.search(r"1\s*주\s*의\s*금액\s*금\s*([0-9,]+)\s*원", text)
    if not m:
        m = re.search(r"1\s*주[^0-9]{0,10}금\s*([0-9,]+)\s*원", text)
        if not m:
            return None
    return _to_int(m.group(1))


def _to_int(num: str) -> Optional[int]:
    s = re.sub(r"[^0-9]", "", num or "")
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _extract_share_count(line: str) -> Optional[int]:
    m = re.search(r"([0-9,]+)\s*주", line)
    if not m:
        return None
    return _to_int(m.group(1))


def _format_num(value: Optional[int]) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def _clean_company_ko(name: str) -> str:
    s = re.sub(r"\(.*?\)", "", name or "")
    s = re.sub(r"주식회사|\(주\)|㈜", "", s)
    s = re.sub(r"[A-Za-z]", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _extract_company_name(text: str) -> str:
    lines = (text or "").splitlines()
    collected = []
    in_section = False
    for line in lines:
        if "상  호" in line or "상호" in line:
            in_section = True
            line = line.replace("상  호", "").replace("상호", "").strip()
            if line:
                collected.append(line)
            continue
        if in_section:
            if "본  점" in line or "본점" in line:
                break
            if line.strip():
                collected.append(line.strip())
    names = []
    for line in collected:
        if not ("주식회사" in line or "㈜" in line or "(주)" in line):
            continue
        name = re.sub(r"\s*\d{4}\.\d{2}\.\d{2}.*$", "", line).strip()
        name = name.replace("    .  .", "").strip()
        if name:
            names.append(name)
    return names[-1] if names else ""


def _section_lines(pages: List[str]) -> List[str]:
    end_markers = ["목          적", "목적", "임원에 관한 사항", "종류주식의 내용"]
    lines: List[str] = []
    in_section = False
    for page_text in pages:
        for line in (page_text or "").splitlines():
            if "발행주식의 총수와" in line:
                in_section = True
            if in_section:
                lines.append(line)
                if any(m in line for m in end_markers):
                    return lines
    return lines


def _parse_share_history(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def _new_block():
        return {
            "total": None,
            "common": None,
            "preferred_items": [],
            "capital": None,
            "change_date": "",
            "reg_date": "",
        }

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "발행할 주식의 총수" in line:
            continue
        if "발행주식의 총수와" in line or "자본금의 액" in line:
            continue

        if "발행주식의 총수" in line:
            total = _extract_share_count(line)
            if total is None:
                continue
            if current:
                blocks.append(current)
            current = _new_block()
            current["total"] = total
            m = re.search(r"(\d{4}\.\d{2}\.\d{2})\s*변경", line)
            if m:
                current["change_date"] = m.group(1)
            cap = _to_int(_extract_capital(line))
            if cap is not None:
                current["capital"] = cap
            continue

        if not current:
            continue

        m_change = re.search(r"(\d{4}\.\d{2}\.\d{2})\s*변경", line)
        if m_change and not current["change_date"]:
            current["change_date"] = m_change.group(1)
        m_reg = re.search(r"(\d{4}\.\d{2}\.\d{2})\s*등기", line)
        if m_reg and not current["reg_date"]:
            current["reg_date"] = m_reg.group(1)

        if "보통주식" in line:
            current["common"] = _extract_share_count(line)
            cap = _to_int(_extract_capital(line))
            if cap is not None:
                current["capital"] = cap
            continue

        if "주식" in line and "발행주식의 총수" not in line:
            label = _extract_label(line)
            if label and "보통주식" not in label:
                value = _extract_share_count(line)
                if value is not None:
                    current["preferred_items"].append({"label": label, "value": value})
                cap = _to_int(_extract_capital(line))
                if cap is not None:
                    current["capital"] = cap
            continue

    if current:
        blocks.append(current)
    return blocks


def _extract_label(line: str) -> str:
    m = re.search(r"([가-힣A-Za-z0-9]+주식)", line)
    return (m.group(1) if m else "").strip()


def _extract_capital(line: str) -> str:
    m = re.search(r"금\s*([0-9,]+)\s*원", line)
    return m.group(1) if m else ""


def _preferred_sum(items: List[Dict[str, Any]]) -> Tuple[int, bool]:
    if not items:
        return 0, False
    labels = [i["label"] for i in items]
    has_detail = any(re.search(r"제\d+종", lb) or "전환" in lb or "상환" in lb for lb in labels)
    totals = [i["value"] for i in items if i["value"] is not None]
    if has_detail:
        totals = [i["value"] for i in items if i["label"] != "종류주식" and i["value"] is not None]
    return sum(totals), has_detail


def _compute_preferred(total: Optional[int], common: Optional[int], items: List[Dict[str, Any]]) -> Tuple[Optional[int], bool]:
    if total is None or common is None:
        return None, False
    preferred, has_detail = _preferred_sum(items)
    expected = total - common
    if preferred == expected:
        return preferred, True
    # try alternative including generic 종류주식 if present
    if has_detail:
        alt = sum([i["value"] for i in items if i["value"] is not None])
        if alt == expected:
            return alt, True
        return alt, False
    return preferred, False


def _build_rows(
    company_name: str,
    reg_no: str,
    par_value: Optional[int],
    blocks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    red_rows: List[int] = []
    row_count = len(blocks)
    prev_total = None
    prev_common = None
    prev_pref = None
    for idx, b in enumerate(blocks):
        total = b.get("total")
        common = b.get("common")
        preferred, ok = _compute_preferred(total, common, b.get("preferred_items", []))
        capital = b.get("capital")
        note = ""
        if not ok and preferred is not None:
            note = "우선주 합산 불일치"
            red_rows.append(idx)
        if par_value and capital and total is not None:
            if par_value * total != capital:
                note = (note + "; " if note else "") + "자본금-주식수 불일치"

        delta_total = None if prev_total is None or total is None else total - prev_total
        delta_common = None if prev_common is None or common is None else common - prev_common
        delta_pref = None if prev_pref is None or preferred is None else preferred - prev_pref

        rows.append(
            {
                "행 개수": row_count,
                "주식회사 제거": _clean_company_ko(company_name),
                "기업명": company_name,
                "등록번호": reg_no,
                "변경연월일": b.get("change_date") or "",
                "등기연월일": b.get("reg_date") or "",
                "발행주식의 총수": _format_num(total),
                "보통주식": _format_num(common),
                "우선주식": _format_num(preferred),
                "자본금의 액": _format_num(capital),
                "비고": note,
                "총 주식수 증감": _format_num(delta_total),
                "보통주 증감": _format_num(delta_common),
                "우선주 증감": _format_num(delta_pref),
                "투자유치금액": "",
                "주당 단가": "",
                "기업가치": "",
                "최대": "",
                "최소": "",
            }
        )
        prev_total, prev_common, prev_pref = total, common, preferred
    return rows, red_rows


def _filter_pdf_files(files: List[Dict]) -> List[Dict]:
    out = []
    for f in files:
        name = f.get("name", "")
        mt = f.get("mimeType", "")
        if mt == "application/pdf" or name.lower().endswith(".pdf"):
            out.append(f)
    return out


def _sheet_values_from_rows(rows: List[Dict[str, Any]], columns: List[str]) -> List[List[Any]]:
    values = [columns]
    for r in rows:
        values.append([r.get(c, "") for c in columns])
    return values


def _create_spreadsheet(sheet_service, title: str) -> str:
    body = {"properties": {"title": title}, "sheets": [{"properties": {"title": "주식변동이력"}}]}
    resp = sheet_service.spreadsheets().create(body=body, fields="spreadsheetId").execute()
    return resp["spreadsheetId"]


def _move_file_to_folder(drive_service, file_id: str, folder_id: str) -> None:
    file = drive_service.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(file.get("parents", []))
    drive_service.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=prev_parents,
        fields="id, parents",
        supportsAllDrives=True,
    ).execute()


def _apply_sheet_formats(
    sheet_service,
    spreadsheet_id: str,
    sheet_id: int,
    red_row_indexes: List[int],
    company_breaks: List[int],
    preferred_col_idx: int,
):
    requests = []
    # Force text format for registration number and date columns
    text_cols = [3, 4, 5]
    for col in text_cols:
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "startColumnIndex": col,
                        "endColumnIndex": col + 1,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": {"type": "TEXT"}}},
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )
    for r in red_row_indexes:
        row = r + 1  # header offset
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row,
                        "endRowIndex": row + 1,
                        "startColumnIndex": preferred_col_idx,
                        "endColumnIndex": preferred_col_idx + 1,
                    },
                    "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": {"red": 0.85}}}},
                    "fields": "userEnteredFormat.textFormat.foregroundColor",
                }
            }
        )
    for r in company_breaks:
        row = r + 1  # header offset
        requests.append(
            {
                "updateBorders": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row,
                        "endRowIndex": row + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": 19,
                    },
                    "bottom": {"style": "SOLID_THICK", "width": 2},
                }
            }
        )
    if requests:
        sheet_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()


def run_drive_register(
    folder_id: str,
    progress_cb: Optional[Callable[[Dict[str, int]], None]] = None,
    reeval_filenames: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    drive_service = get_drive_service()
    sheet_service = get_sheets_service()

    files = list_files_in_folder(drive_service, folder_id)
    target_files = _filter_pdf_files(files)

    result_folder_id = find_or_create_folder(drive_service, folder_id, "result")
    processed = load_processed_index(drive_service, result_folder_id)
    cache = load_json_file(drive_service, result_folder_id, CACHE_FILENAME)
    cache_files = cache.get("files", {})

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

    status_rows = []
    results = []
    for f in target_files:
        filename = f["name"]
        if filename in processed and filename not in reeval_filenames:
            cached = cache_files.get(filename, {})
            status_rows.append(
                {
                    "filename": filename,
                    "company_name": cached.get("company_name", ""),
                    "status": "already_processed",
                    "error": "",
                }
            )
            continue
        try:
            pdf_bytes = download_file(drive_service, f["id"], f.get("mimeType"))
            pages = _extract_pdf_pages_text(pdf_bytes)
            full_text = "\n".join(pages)
            reg_no = _parse_registration_number(full_text)
            company_name = _extract_company_name(full_text)
            par_value = _parse_par_value(full_text)
            lines = _section_lines(pages)
            blocks = _parse_share_history(lines)
            if not blocks:
                raise ValueError("발행주식 섹션 추출 실패")
            rows, red_rows = _build_rows(company_name, reg_no, par_value, blocks)

            cache_files[filename] = {
                "company_name": company_name,
                "rows": rows,
                "red_rows": red_rows,
            }
            if filename not in processed:
                processed.append(filename)
            status_rows.append(
                {"filename": filename, "company_name": company_name, "status": "completed", "error": ""}
            )
            results.append({"company_name": company_name, "source_filename": filename, "row_count": len(rows)})
            counts["completed"] += 1
        except Exception as e:
            status_rows.append(
                {"filename": filename, "company_name": "", "status": "failed", "error": str(e)}
            )
            counts["failed"] += 1
        counts["pending"] = max(0, counts["pending"] - 1)
        if progress_cb:
            progress_cb(counts)

    save_processed_index(drive_service, result_folder_id, processed)
    save_json_file(drive_service, result_folder_id, CACHE_FILENAME, {"files": cache_files})

    # Build merged sheet from cache
    columns = [
        "행 개수",
        "주식회사 제거",
        "기업명",
        "등록번호",
        "변경연월일",
        "등기연월일",
        "발행주식의 총수",
        "보통주식",
        "우선주식",
        "자본금의 액",
        "비고",
        "총 주식수 증감",
        "보통주 증감",
        "우선주 증감",
        "투자유치금액",
        "주당 단가",
        "기업가치",
        "최대",
        "최소",
    ]
    merged_rows: List[Dict[str, Any]] = []
    red_row_indexes: List[int] = []
    company_breaks: List[int] = []
    row_cursor = 0
    for fname in sorted(cache_files.keys()):
        entry = cache_files[fname]
        rows = entry.get("rows", [])
        merged_rows.extend(rows)
        for r in entry.get("red_rows", []):
            red_row_indexes.append(row_cursor + r)
        if rows:
            company_breaks.append(row_cursor + len(rows) - 1)
        row_cursor += len(rows)

    folder_meta = (
        drive_service.files()
        .get(fileId=folder_id, fields="name", supportsAllDrives=True)
        .execute()
    )
    folder_name = folder_meta.get("name", folder_id)
    date_str = datetime.now().strftime("%Y%m%d")
    sheet_title = f"{folder_name}_result_{date_str}"

    spreadsheet_id = _create_spreadsheet(sheet_service, sheet_title)
    _move_file_to_folder(drive_service, spreadsheet_id, result_folder_id)

    values = _sheet_values_from_rows(merged_rows, columns)
    sheet_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="주식변동이력!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()

    spreadsheet = sheet_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = spreadsheet["sheets"][0]["properties"]["sheetId"]
    preferred_col_idx = columns.index("우선주식")
    _apply_sheet_formats(sheet_service, spreadsheet_id, sheet_id, red_row_indexes, company_breaks, preferred_col_idx)

    sheet_meta = {
        "id": spreadsheet_id,
        "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
        "title": sheet_title,
    }
    return results, result_folder_id, status_rows, counts, sheet_meta
