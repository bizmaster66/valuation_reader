import os
import threading
import fitz  # PyMuPDF
from google.genai import types


def pdf_to_page_pngs(pdf_path: str, out_dir: str, dpi: int = 220) -> list[str]:
    """
    PDF를 1페이지=1PNG로 렌더링.
    p001.png, p002.png ... 형태로 저장해서 페이지 순서가 절대 안 꼬이게 함.
    """
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    paths = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        p = os.path.join(out_dir, f"p{i+1:03d}.png")
        pix.save(p)
        paths.append(p)
    doc.close()
    return paths


def ocr_page_image(
    client,
    image_path: str,
    page_no: int,
    model_name: str = "gemini-2.5-flash",
    max_chars: int = 8000,
) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

    prompt = (
        f"너는 IR 슬라이드 한 페이지를 OCR+문서이해로 읽는다. (page={page_no})\n"
        "규칙:\n"
        "- 가능한 원문 텍스트를 최대한 그대로 추출\n"
        "- 표/차트/그래프에서 숫자/단위/기간이 보이면 텍스트에 포함\n"
        "- 과장/추정 금지. 보이지 않으면 '확인 불가'\n"
        "- 불필요한 설명 금지. 결과만 출력\n"
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt, img_part],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    text = (resp.text or "").strip()
    return text[:max_chars]


def ocr_page_image_with_timeout(
    client,
    image_path: str,
    page_no: int,
    model_name: str,
    max_chars: int,
    timeout_sec: int = 90,
) -> str:
    """
    ocr_page_image를 별도 스레드에서 실행하고, timeout을 넘기면 TimeoutError 발생.
    SDK/네트워크 블로킹으로 전체 파이프라인이 멈추는 것을 방지.
    """
    result = {"text": None, "err": None}

    def runner():
        try:
            result["text"] = ocr_page_image(
                client=client,
                image_path=image_path,
                page_no=page_no,
                model_name=model_name,
                max_chars=max_chars,
            )
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout_sec)

    if t.is_alive():
        raise TimeoutError(f"OCR timeout after {timeout_sec}s")

    if result["err"] is not None:
        raise result["err"]

    return result["text"] or ""


def ocr_pdf_all_pages(
    client,
    pdf_path: str,
    cache_dir: str,
    dpi: int = 220,
    model_name: str = "gemini-2.5-flash",
    reocr: bool = False,
    max_chars_per_page: int = 8000,
    timeout_sec: int = 90,
    progress_callback=None,   # (page_no, total_pages, stage, extra) -> None
    keep_images: bool = True,
    min_chars_retry: int = 120,
    retry_model: str = "gemini-2.5-pro",
) -> list[dict]:
    """
    강제 OCR 파이프라인:
    PDF → 페이지 PNG → 각 페이지 OCR → [{"page":n,"text":...}] 반환
    캐시: cache_dir/pages/p001.txt 존재하면 재호출 안 함 (reocr=True면 무시)

    - 페이지별 예외가 나도 계속 진행 (pNNN.txt에 [OCR_ERROR] 기록 + ocr_errors.log 누적)
    - timeout_sec으로 블로킹 방지
    - 짧은 페이지(min_chars_retry 미만)는 retry_model로 1회 재시도(비용 고려)
    - keep_images=False면 OCR 후 PNG 삭제(용량 절감)
    """
    pages_dir = os.path.join(cache_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    image_paths = pdf_to_page_pngs(pdf_path, pages_dir, dpi=dpi)
    total_pages = len(image_paths)

    err_log = os.path.join(cache_dir, "ocr_errors.log")

    out_pages = []
    for idx, img_path in enumerate(image_paths):
        page_no = idx + 1
        txt_path = os.path.join(pages_dir, f"p{page_no:03d}.txt")

        if progress_callback:
            progress_callback(page_no, total_pages, "start", {"path": img_path})

        try:
            if (not reocr) and os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt = f.read()
            else:
                txt = ocr_page_image_with_timeout(
                    client=client,
                    image_path=img_path,
                    page_no=page_no,
                    model_name=model_name,
                    max_chars=max_chars_per_page,
                    timeout_sec=timeout_sec,
                )

                # 짧은 페이지 재시도(Flash → Pro)
                if min_chars_retry and retry_model and len((txt or "").strip()) < min_chars_retry:
                    try:
                        txt2 = ocr_page_image_with_timeout(
                            client=client,
                            image_path=img_path,
                            page_no=page_no,
                            model_name=retry_model,
                            max_chars=max_chars_per_page,
                            timeout_sec=min(timeout_sec * 2, 180),
                        )
                        if len((txt2 or "").strip()) > len((txt or "").strip()):
                            txt = txt2
                    except Exception:
                        pass

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txt)

        except Exception as e:
            msg = f"[OCR_ERROR] page={page_no} file={os.path.basename(img_path)} err={type(e).__name__}: {e}"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(msg)
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            txt = msg

        # 용량 절감 옵션
        if not keep_images:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception:
                pass

        out_pages.append({"page": page_no, "text": txt})

        if progress_callback:
            stage = "error" if (txt or "").startswith("[OCR_ERROR]") else "done"
            progress_callback(page_no, total_pages, stage, {"txt_len": len((txt or ""))})

    return out_pages
