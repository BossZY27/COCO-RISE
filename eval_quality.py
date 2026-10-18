"""Measure answer quality across the kinds of question people actually ask.

eval.py checks 18 figures and verdicts. This file checks understanding and wording on a wider set: did the
system find the right fund, does the answer carry what the document says, and does it keep out what it
must not say, such as a figure nobody has reviewed yet. Each case in data/eval_quality.json may state:

    expect_fund        the fund the question is about, "" when no fund should be picked
    must_include       every string must appear in the answer
    must_include_any   at least one of these strings must appear
    must_not_include   none of these may appear

Usage:
    python eval_quality.py              template answers only: fast, and the same on every run
    python eval_quality.py --llm        rejected: verified_template never calls a free-form answer model
    python eval_quality.py --only A B   run the cases whose id starts with these letters
    python eval_quality.py --cases data/eval_holdout.json
                                        the holdout: questions written after the rules were tuned, used to
                                        check that a gain carries to wording the tuning never saw
"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")      # Windows pipes default to cp1252 and cannot print Thai

CATEGORIES = {
    "A": "ตัวเลขเพดานและวงเงิน", "B": "ข้อมูลรอตรวจต้องไม่รั่ว", "C": "จำนวนทุนและวงเงินรวม",
    "D": "นิยาม", "E": "ใครขอได้", "F": "นอกขอบเขต", "G": "คำถามต่อเนื่อง", "H": "รายการและหมวดทุน",
    "I": "เปรียบเทียบ", "J": "เงื่อนไขและรายละเอียด", "K": "วิธีสมัคร", "L": "ปีงบประมาณ",
}


def resolve_dynamic_expectations(value, current_fiscal_year: int):
    """Keep relative-year cases valid after the Thai fiscal year rolls over."""
    if isinstance(value, dict):
        return {key: resolve_dynamic_expectations(item, current_fiscal_year) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_dynamic_expectations(item, current_fiscal_year) for item in value]
    if isinstance(value, str):
        replacements = {
            "{current_fiscal_year}": str(current_fiscal_year),
            "{next_fiscal_year}": str(current_fiscal_year + 1),
            "{previous_fiscal_year}": str(current_fiscal_year - 1),
        }
        for token, year in replacements.items():
            value = value.replace(token, year)
    return value


def check(case: dict, result: dict) -> list[str]:
    """Every way the answer falls short of the case, empty when it passes."""
    answer, problems = result["answer"], []
    if result["answer_source"] == "blocked":
        problems.append("ด่านตรวจบล็อกคำตอบ")
    if "expect_fund" in case:
        want, got = case["expect_fund"] or None, result["slots"].get("fund")
        if got != want:
            problems.append(f"จับทุนได้ {got} แต่ควรเป็น {want}")
    problems += [f"ขาด '{text}'" for text in case.get("must_include", []) if text not in answer]
    options = case.get("must_include_any")
    if options and not any(text in answer for text in options):
        problems.append(f"ไม่มีคำใดใน {options}")
    problems += [f"มี '{text}' ซึ่งไม่ควรมี" for text in case.get("must_not_include", []) if text in answer]
    return problems


async def main(args) -> int:
    from app import pipeline

    cases = json.loads((Path(__file__).parent / args.cases).read_text(encoding="utf-8"))
    cases = resolve_dynamic_expectations(cases, pipeline.thai_fiscal_year())
    if args.only:
        cases = [c for c in cases if c["id"][0] in args.only]
    if not cases:
        print("ไม่พบกรณีทดสอบที่ตรงกับตัวกรอง --only", file=sys.stderr)
        return 2
    rows = []
    for case in cases:
        result = await pipeline.run(case["question"], case["persona"], case.get("history"))
        rows.append((case, result, check(case, result)))

    print(f"โหมดคำตอบ: {pipeline.ANSWER_MODE} | โมเดลตอบ: ไม่เรียกใช้ | {len(rows)} ข้อ\n")
    for letter, label in CATEGORIES.items():
        group = [r for r in rows if r[0]["id"][0] == letter]
        if group:
            passed = sum(not problems for _, _, problems in group)
            print(f"  {letter} {label:<28s} {passed:>2d}/{len(group)}")
    for case, result, problems in rows:
        if problems:
            print(f"\n[{case['id']}] {case['question']}")
            print(f"   ปัญหา: {'; '.join(problems)}")
            print(f"   ตอบ ({result['answer_source']}): {result['answer'][:260]}")

    passed = sum(not p for _, _, p in rows)
    lengths = [len(r["answer"]) for _, r, _ in rows]
    times = sorted(r["timings_ms"]["total"] for _, r, _ in rows)
    print(f"\nผ่าน {passed}/{len(rows)}  ความยาวคำตอบเฉลี่ย {statistics.mean(lengths):.0f} ตัวอักษร  "
          f"เวลาเฉลี่ย {statistics.mean(times)/1000:.1f} s  สูงสุด {times[-1]/1000:.1f} s")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="disabled while answer mode is verified_template")
    parser.add_argument("--only", nargs="*", default=[], help="case id prefixes, such as A B")
    parser.add_argument("--cases", default="data/eval_quality.json", help="case file, relative to this script")
    options = parser.parse_args()
    if options.llm:
        parser.error("โหมด verified_template ไม่เรียกโมเดลเรียบเรียงคำตอบ จึงใช้ --llm ไม่ได้")
    sys.exit(asyncio.run(main(options)))
