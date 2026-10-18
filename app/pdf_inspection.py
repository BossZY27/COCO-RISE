"""Read an uploaded grant PDF for a human to inspect, without publishing its contents."""

import re
from pathlib import Path

import fitz

from scripts.ingest import page_text

MAX_BYTES = 15 * 1024 * 1024
MAX_PAGES = 50
MAX_PAGE_POINTS = 1800
PREVIEW_CHARS = 8000
OCR_DPI = 200
AMOUNT_RE = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?\s*บาท")


def amount_candidates(text: str) -> list[str]:
    """Spot figures for review only; these are never facts or approved amounts."""
    return list(dict.fromkeys(match.group().strip() for match in AMOUNT_RE.finditer(text)))[:25]


def inspect(data: bytes, filename: str) -> dict:
    name = Path(filename).name[:120]
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise ValueError("กรุณาเลือกไฟล์ PDF ที่ถูกต้อง")
    if len(data) > MAX_BYTES:
        raise ValueError("ไฟล์ PDF ใหญ่เกิน 15 MB")
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("เปิดไฟล์ PDF ไม่สำเร็จ") from exc
    try:
        if document.needs_pass:
            raise ValueError("ไฟล์ PDF มีรหัสผ่าน กรุณาปลดล็อกก่อนตรวจ")
        if not 0 < document.page_count <= MAX_PAGES:
            raise ValueError("รองรับไฟล์ PDF ตั้งแต่ 1 ถึง 50 หน้า")
        pages = []
        for number, page in enumerate(document, start=1):
            if max(page.rect.width, page.rect.height) > MAX_PAGE_POINTS:
                text, method = "", "page_too_large"
            else:
                text, method = page_text(page, OCR_DPI)
            pages.append({"page": number, "method": method, "status": "draft", "verified": False,
                          "text": text[:PREVIEW_CHARS], "truncated": len(text) > PREVIEW_CHARS,
                          "amount_candidates": amount_candidates(text)})
        note = "ผลอ่านข้อความและตัวเลขเป็นเพียงตัวช่วยตรวจ ไม่ได้เพิ่มเข้าคลังคำตอบหรืออนุมัติข้อมูล"
        if any(page["method"] == "needs_ocr" for page in pages):
            note += " มีหน้าภาพสแกนที่ OCR อ่านไม่ได้บนเครื่องนี้ ต้องติดตั้งหรือซ่อมระบบ OCR ภาษาไทยก่อน"
        if any(page["method"] == "text_partial" for page in pages):
            note += " บางหน้าแสดงได้เฉพาะข้อความสั้นจากชั้นข้อความ เพราะ OCR ไม่พร้อม"
        if any(page["method"] == "page_too_large" for page in pages):
            note += " มีหน้ากระดาษขนาดใหญ่เกินขอบเขตที่ระบบอ่านอัตโนมัติ"
        return {"filename": name, "pages_total": document.page_count, "pages": pages,
                "saved": False, "verified": False, "note": note}
    finally:
        document.close()
