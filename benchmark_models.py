"""Retired answer-model benchmark for the fail-closed IRFA pipeline.

Usage: python benchmark_models.py [model-name ...]

IRFA now uses ``verified_template`` answers. The LLM is not allowed to rephrase
the final answer, so comparing answer models would report a capability that the
running service deliberately does not use.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    print(
        "ปิด benchmark โมเดลตอบ: โหมด verified_template ไม่เรียกโมเดลเรียบเรียงคำตอบ "
        "จึงไม่มีโมเดลคำตอบให้เปรียบเทียบ",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
