"""Put every unchecked amount next to the text the source document actually contains.

The reviewer still decides. This only removes the part that wastes their time: hunting for the line.
Run it, read the column on the right, then confirm the rows on /admin.

    python scripts/verify_report.py
    python scripts/verify_report.py --doc "ประกาศ 42/2568"
"""
import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.knowledge import corpus as store


# OCR on Thai scans drops and swaps the marks that sit above and below the line, so
# "ชั้นเรียน" comes back as "ชั่นเรียน" and "ผู้เชี่ยวชาญ" as "ผู้เชียวชาญ". Removing those marks
# and the spaces leaves a skeleton that both spellings share, which is enough to say the fund
# name is on the page. The amount is still compared digit for digit.
MARKS = re.compile(r"[ัิ-ฺ็-๎\s]")


def skeleton(text: str) -> str:
    return MARKS.sub("", text)


def baht(value: int) -> list[str]:
    """The same number as the document might print it, with and without separators."""
    return [f"{value:,}", str(value)]


def pages_of(doc: str) -> dict[tuple[int, str], str]:
    """Join each page back together, keeping the two readings apart.

    A scanned table puts the fund name and its amount in different chunks, so searching chunk by chunk
    misses the rows that matter most. Joining fixes that. Keeping OCR separate from the hand-typed text
    matters more: only the OCR is an independent reading of the scan, so only it can confirm what a
    person typed. Merging them would let one source vouch for itself.
    """
    joined: dict[tuple[int, str], list[str]] = {}
    for chunk in store.CHUNKS:
        if chunk["doc"] != doc:
            continue
        origin = "OCR จากไฟล์สแกน" if chunk.get("status") == "draft" else "ข้อความที่ทีมพิมพ์เอง"
        joined.setdefault((chunk["page"], origin), []).append(chunk["text"])
    return {key: " ".join(" ".join(parts).split()) for key, parts in joined.items()}


def evidence_for(fact: dict) -> list[tuple[str, str, str]]:
    """Readings of the cited document that carry this amount with the fund name nearby."""
    doc = fact["source"]["doc"]
    wanted = skeleton(fact["fund"])
    hits = []
    for (page, origin), text in sorted(pages_of(doc).items()):
        if fact["amount"] is not None and not any(f in text for f in baht(fact["amount"])):
            continue
        if wanted and wanted not in skeleton(text):
            continue
        window = text
        if fact["amount"] is not None:
            at = max(text.find(f"{fact['amount']:,}"), text.find(str(fact["amount"])))
            window = text[max(0, at - 100):at + 100]
        hits.append((f"หน้า {page}", origin, window))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="", help="limit to one source document")
    args = parser.parse_args()

    rows = [f for f in store.FACTS if not store.is_verified(f)
            and (not args.doc or f["source"]["doc"] == args.doc)]
    if not rows:
        print("ไม่มีตัวเลขที่รอตรวจแล้ว")
        return 0

    print(f"ตัวเลขที่รอตรวจ {len(rows)} แถว\n")
    matched = confirmed_by_ocr = 0
    for fact in rows:
        tier = f" {fact['tier']}" if fact["tier"] else ""
        amount = f"{fact['amount']:,}" if fact["amount"] is not None else "ไม่ระบุต่อทุน"
        kind = "เพดาน" if fact["amount_type"] == "ceiling" else f"จัดสรรปี {fact['fiscal_year']}"
        print(f"[{fact['id']}] {fact['fund']}{tier}")
        print(f"      {kind}: {amount} บาท   อ้างอิง {fact['source']['doc']} "
              f"{fact['source']['clause']} หน้า {fact['source']['page']}")
        hits = evidence_for(fact)
        if hits:
            matched += 1
            if any(origin.startswith("OCR") for _, origin, _ in hits):
                confirmed_by_ocr += 1
            for where, origin, text in hits:
                print(f"      {where} [{origin}]: ...{text}...")
        else:
            print("      ยังไม่พบตัวเลขนี้ในข้อความที่อ่านมาได้ ต้องเปิดเอกสารดูเอง")
        print()

    print(f"สรุป พบข้อความรองรับ {matched} จาก {len(rows)} แถว "
          f"และในจำนวนนี้ {confirmed_by_ocr} แถวมีหลักฐานจาก OCR ของไฟล์สแกนต้นฉบับ")
    print("OCR อ่านผิดได้ ตัวเลขที่พบจึงเป็นตัวช่วยหา ไม่ใช่คำยืนยัน")
    print("เปิด /admin เพื่อกดยืนยันทีละแถว หรือแก้ตัวเลขก่อนกดถ้าเอกสารเขียนต่างจากนี้")
    return 0


if __name__ == "__main__":
    sys.exit(main())
