import io
import json
import os
from typing import Dict, List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
COMBINED_SCOPES = DRIVE_SCOPES + SHEETS_SCOPES


def _load_service_account_info() -> Dict:
    # 1) Streamlit secrets (JSON or dict)
    try:
        import streamlit as st  # lazy import for local runs

        if "gcp_service_account" in st.secrets:
            sa = st.secrets["gcp_service_account"]
            if isinstance(sa, str):
                return json.loads(sa)
            if isinstance(sa, dict):
                return sa
            # Streamlit AttrDict support
            try:
                return dict(sa)
            except Exception:
                pass
    except Exception:
        pass

    # 2) Env JSON
    env_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "")
    if env_json:
        return json.loads(env_json)

    # 3) GOOGLE_APPLICATION_CREDENTIALS file
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path and os.path.exists(cred_path):
        with open(cred_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError("Service account info not found. Set Streamlit secrets or env vars.")


def _build_credentials(scopes: List[str]):
    info = _load_service_account_info()
    return service_account.Credentials.from_service_account_info(info, scopes=scopes)


def get_drive_service():
    creds = _build_credentials(DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_sheets_service():
    # Sheets API calls (create/update) can require Drive permissions when moving files.
    creds = _build_credentials(COMBINED_SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_files_in_folder(service, folder_id: str) -> List[Dict]:
    files = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="allDrives",
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def find_file_by_name(service, folder_id: str, name: str) -> Optional[Dict]:
    safe_name = name.replace("'", "\\'")
    q = (
        f"'{folder_id}' in parents and trashed=false "
        f"and name = '{safe_name}'"
    )
    resp = (
        service.files()
        .list(
            q=q,
            fields="files(id,name,mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
        )
        .execute()
    )
    files = resp.get("files", [])
    return files[0] if files else None


def find_or_create_folder(service, parent_id: str, folder_name: str) -> str:
    safe_name = folder_name.replace("'", "\\'")
    q = (
        f"'{parent_id}' in parents and trashed=false "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{safe_name}'"
    )
    resp = (
        service.files()
        .list(
            q=q,
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
        )
        .execute()
    )
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return folder["id"]


def download_file(
    service, file_id: str, mime_type: Optional[str] = None, export_mime: Optional[str] = None
) -> bytes:
    # Export Google Docs if needed
    if mime_type and mime_type.startswith("application/vnd.google-apps."):
        if export_mime:
            request = service.files().export(fileId=file_id, mimeType=export_mime, supportsAllDrives=True)
        elif mime_type == "application/vnd.google-apps.document":
            request = service.files().export(fileId=file_id, mimeType="text/plain", supportsAllDrives=True)
        else:
            request = service.files().export(fileId=file_id, mimeType="application/pdf", supportsAllDrives=True)
    else:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def upload_bytes(
    service,
    parent_id: str,
    filename: str,
    content: bytes,
    mime_type: str,
    overwrite: bool = True,
) -> str:
    existing = find_file_by_name(service, parent_id, filename) if overwrite else None
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=False)

    if existing:
        updated = service.files().update(fileId=existing["id"], media_body=media, supportsAllDrives=True).execute()
        return updated["id"]

    metadata = {"name": filename, "parents": [parent_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def load_processed_index(service, result_folder_id: str) -> List[str]:
    index_name = "_processed.json"
    meta = find_file_by_name(service, result_folder_id, index_name)
    if not meta:
        return []
    content = download_file(service, meta["id"], meta.get("mimeType"))
    try:
        data = json.loads(content.decode("utf-8"))
        return data.get("processed", [])
    except Exception:
        return []


def save_processed_index(service, result_folder_id: str, processed: List[str]) -> None:
    index_name = "_processed.json"
    payload = json.dumps({"processed": sorted(set(processed))}, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(
        service,
        result_folder_id,
        index_name,
        payload,
        mime_type="application/json",
        overwrite=True,
    )


def load_json_file(service, folder_id: str, filename: str) -> Dict:
    meta = find_file_by_name(service, folder_id, filename)
    if not meta:
        return {}
    content = download_file(service, meta["id"], meta.get("mimeType"))
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        return {}


def save_json_file(service, folder_id: str, filename: str, payload: Dict) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(
        service,
        folder_id,
        filename,
        content,
        mime_type="application/json",
        overwrite=True,
    )
