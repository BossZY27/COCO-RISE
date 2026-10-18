"""Turn the real announcement PDFs into chunk records.

    python scripts/ingest.py --src "C:/.../Funds_resource" --out data/chunks_ingested.json

Text PDFs are read directly. Scanned PDFs need OCR: install Tesseract with the Thai data
(`winget install UB-Mannheim.TesseractOCR` then `pip install pytesseract`), after which this script
picks it up automatically. Amounts are never written into facts.json by a script. A person reads the
page and fills those in, which is the human review step in the architecture.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import fitz

CLAUSE = re.compile(r"^\s*(ข้อ\s*\d+|\d+\.\d+(?:\.\d+)?)\s")
YEAR = re.compile(r"25\d\d")
MAX_CHARS = 900
MIN_CLAUSE_CHARS = 12      # a numbered clause is meaningful even when it is one short line
MIN_LOOSE_CHARS = 40       # untagged text needs more length before it is worth indexing


_READER = None


def _easyocr_reader():
    """Built once. Loading the Thai and English models takes about seven seconds."""
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["th", "en"], gpu=False, verbose=False)
    return _READER


def _ocr(page, dpi: int) -> tuple[str, str]:
    """Tesseract when it is installed, otherwise easyocr, which needs no system install."""
    pix = page.get_pixmap(dpi=dpi)
    try:
        import pytesseract
        from PIL import Image
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(image, lang="tha+eng").strip(), "ocr"
    except Exception:
        pass
    try:
        import numpy as np
        array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        lines = _easyocr_reader().readtext(array[:, :, :3], detail=0, paragraph=True)
        return "\n".join(lines).strip(), "ocr"
    except Exception as exc:
        print(f"    อ่านภาพไม่สำเร็จ: {type(exc).__name__}", flush=True)
        return "", "needs_ocr"


def page_text(page, dpi: int) -> tuple[str, str]:
    text = page.get_text().strip()
    if len(text) > 120:
        return text, "text"
    ocr_text, method = _ocr(page, dpi)
    if ocr_text:
        return ocr_text, method
    if text:
        return text, "text_partial"
    return "", "needs_ocr"


def split_clauses(text: str) -> list[tuple[str, str]]:
    """Split a page into clause-sized pieces and remember which clause each piece came from."""
    pieces, clause, buffer = [], "", []

    def flush():
        body = " ".join(" ".join(buffer).split())
        # A numbered sub-clause is kept however short it is. In the fee table a whole row reads
        # "1.6.1 ระดับต้น ไม่เกิน 150,000 บาท", which is 33 characters and carries a real ceiling.
        floor = MIN_CLAUSE_CHARS if clause else MIN_LOOSE_CHARS
        if len(body) > floor:
            pieces.append((clause or "ทั้งหน้า", body))

    for line in text.splitlines():
        if not line.strip() or line.strip().isdigit():
            continue
        match = CLAUSE.match(line)
        if match or sum(len(b) for b in buffer) > MAX_CHARS:
            flush()
            buffer = []
            if match:
                clause = match.group(1).replace("  ", " ").strip()
        buffer.append(line.strip())
    flush()
    return pieces


def doc_label(path: Path) -> tuple[str, int | None]:
    name = path.stem
    number = re.search(r"(\d{1,2})[-_/](\d{2,4})", name)
    if number:
        year = number.group(2)
        year = f"25{year}" if len(year) == 2 else year
        label = f"ประกาศ {number.group(1)}/{year}"
    else:
        label = "หลักเกณฑ์กลาง" if "หลักเกณฑ์" in name else name
    years = [int(y) for y in YEAR.findall(name)]
    fiscal = max(years) if years else None
    if label == "หลักเกณฑ์กลาง":
        fiscal = None          # the central criteria is not tied to one fiscal year
    return label, fiscal


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows pipes default to cp1252
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder holding the announcement PDFs")
    parser.add_argument("--out", default="data/chunks_ingested.json")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--only", default="", help="substring of the filename to process alone")
    args = parser.parse_args()

    chunks, report = [], []
    for pdf in sorted(Path(args.src).glob("*.pdf")):
        if args.only and args.only not in pdf.name:
            continue
        label, fiscal = doc_label(pdf)
        doc = fitz.open(pdf)
        made, mode = 0, ""
        for number, page in enumerate(doc, start=1):
            text, mode = page_text(page, args.dpi)
            if not text:
                continue
            # Neither a text layer nor OCR proves the document is an approved source.
            # A person must review the original before either can enter the answer path.
            for clause, body in split_clauses(text):
                chunks.append({"id": f"{pdf.stem[:12]}_p{number}_{made:02d}", "doc": label,
                               "fiscal_year": fiscal, "status": "draft",
                               "verified": False, "page": number,
                               "clause": clause, "text": body})
                made += 1
        report.append((pdf.name, len(doc), mode, made))
        print(f"  อ่านจบ {pdf.name[:46]:46s} {mode:>10s} {made:4d} ชิ้น", flush=True)

    Path(args.out).write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'ไฟล์':52s} {'หน้า':>4s} {'วิธีอ่าน':>10s} {'ชิ้นข้อความ':>10s}")
    for name, pages, mode, made in report:
        print(f"{name[:52]:52s} {pages:4d} {mode:>10s} {made:10d}")
    print(f"\nเขียน {len(chunks)} ชิ้นลง {args.out}")
    print("ทุกชิ้นเริ่มเป็น draft และ verified false จนกว่าคนจะตรวจต้นฉบับ")
    print("ขั้นต่อไปที่คนต้องทำเอง:")
    print("  1 เปิดต้นฉบับเทียบทุกชิ้น แล้วให้ผู้รับผิดชอบกดอนุมัติใน /admin")
    print("  2 คัดตัวเลขลง data/facts.json พร้อมระบุ amount_type ว่า ceiling หรือ allocated")
    print("  3 ห้ามใช้สคริปต์ตั้ง verified true ให้ข้อมูลใด ๆ")
    if any(mode == "needs_ocr" for _, _, mode, _ in report):
        print("\nมีไฟล์ที่เป็นภาพสแกน ต้องติดตั้ง OCR ก่อน:")
        print("  winget install UB-Mannheim.TesseractOCR")
        print("  pip install pytesseract pillow")
        print("  แล้วรันคำสั่งเดิมอีกครั้ง")
    return 0


if __name__ == "__main__":
    sys.exit(main())
