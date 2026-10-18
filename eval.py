"""Run the test questions through the pipeline and report accuracy and latency.

Usage: python eval.py
"""
import asyncio
import json
import statistics
import sys
from pathlib import Path

from app import pipeline

sys.stdout.reconfigure(encoding="utf-8")      # Windows pipes default to cp1252 and cannot print Thai

CASES = json.loads((Path(__file__).parent / "data" / "eval.json").read_text(encoding="utf-8"))


WITHHELD_MARKERS = ("รอเจ้าหน้าที่ตรวจ", "ยังแสดงไม่ได้", "ยังสรุปสิทธิ์ไม่ได้")


def amount_matches_expectation(case: dict, result: dict) -> bool:
    """Check either a verified value or a correctly withheld unverified value."""
    expected = case["expect_amount"]
    if case.get("expect_status") == "withheld_unverified":
        explains_withholding = any(marker in result["answer"] for marker in WITHHELD_MARKERS)
        hides_distinct_draft_value = (
            not case.get("expect_amount_hidden")
            or expected is None
            or expected not in result["answer"]
        )
        return explains_withholding and hides_distinct_draft_value
    return expected is None or (expected in result["answer"] and
                                any(case["expect_doc"] in c for c in result["citations"]))


def citation_matches_expectation(case: dict, result: dict) -> bool:
    """Draft sources must stay uncited; verified sources must be cited."""
    cited = any(case["expect_doc"] in citation for citation in result["citations"])
    if case.get("expect_status") == "withheld_unverified":
        return not cited
    return cited


async def main() -> int:
    if not CASES:
        print("ไม่พบกรณีทดสอบใน data/eval.json", file=sys.stderr)
        return 2
    rows, latencies = [], []
    for case in CASES:
        result = await pipeline.run(case["question"], case["persona"])
        amount_ok = amount_matches_expectation(case, result)
        cite_ok = citation_matches_expectation(case, result)
        verdict_ok = case["expect_verdict"] is None or result["ruling"]["verdict"] == case["expect_verdict"]
        latencies.append(result["timings_ms"]["total"])
        rows.append((case["persona"], case["question"], amount_ok, cite_ok, verdict_ok,
                     result["ruling"]["verdict"], result["timings_ms"], result["answer"],
                     result["answer_source"], case.get("expect_status") == "withheld_unverified"))

    print(f"โหมดคำตอบ: {pipeline.ANSWER_MODE} | โมเดลตอบ: ไม่เรียกใช้")
    print()
    print(f"{'persona':8s} {'ข้อมูล':>6s} {'อ้างอิง':>7s} {'สิทธิ์':>6s} {'รวม s':>7s}  คำถาม")
    for persona, question, amount_ok, cite_ok, verdict_ok, verdict, times, answer, source, withheld in rows:
        mark = lambda ok: "ผ่าน" if ok else "ไม่ผ่าน"
        data_mark = "กักไว้" if withheld and amount_ok else mark(amount_ok)
        print(f"{persona:8s} {data_mark:>6s} {mark(cite_ok):>7s} {mark(verdict_ok):>6s} "
              f"{times['total']/1000:7.1f}  {question}")
        if not (amount_ok and cite_ok and verdict_ok):
            print(f"         -> {verdict} | {source} | {answer}")

    total = len(rows)
    print(f"\nข้อมูลถูก {sum(r[2] for r in rows)}/{total}  "
          f"อ้างอิงถูก {sum(r[3] for r in rows)}/{total}  "
          f"สิทธิ์ถูก {sum(r[4] for r in rows)}/{total}")
    draft_rows = [row for row in rows if row[-1]]
    verified_rows = [row for row in rows if not row[-1]]
    print(
        f"ค่าที่ตรวจยืนยันแล้ว {sum(row[2] for row in verified_rows)}/{len(verified_rows)} | "
        f"ข้อมูลร่างที่กักไว้ถูกต้อง {sum(row[2] and row[3] for row in draft_rows)}/{len(draft_rows)} "
        "(ไม่ได้นับค่าร่างเป็นคำตอบตัวเลข)"
    )
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(f"เวลาเฉลี่ย {statistics.mean(latencies)/1000:.1f} s  p95 {p95/1000:.1f} s  สูงสุด {max(latencies)/1000:.1f} s")
    return 0 if all(row[2] and row[3] and row[4] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
