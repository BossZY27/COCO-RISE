"""Regression tests for command-line evaluation tools."""

import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_eval_quality_llm_option_fails_fast_in_verified_template_mode():
    result = _run("eval_quality.py", "--llm")

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "verified_template" in output
    assert "ไม่เรียกโมเดล" in output
    assert "ผ่าน " not in output


def test_eval_quality_empty_selection_has_clear_error_without_traceback():
    result = _run("eval_quality.py", "--only", "Z")

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ไม่พบกรณีทดสอบ" in output
    assert "Traceback" not in output


def test_model_benchmark_fails_fast_in_verified_template_mode():
    result = _run("benchmark_models.py", "not-installed-on-purpose")

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "verified_template" in output
    assert "ไม่เรียกโมเดล" in output


def test_eval_main_returns_failure_when_any_required_check_fails(monkeypatch, capsys):
    evaluation = importlib.import_module("eval")
    monkeypatch.setattr(
        evaluation,
        "CASES",
        [{
            "persona": "P1",
            "question": "คำถามทดสอบ",
            "expect_amount": "100 บาท",
            "expect_doc": "เอกสารต้นฉบับ",
            "expect_verdict": None,
        }],
    )

    async def fake_run(*_args, **_kwargs):
        return {
            "answer": "ไม่มีตัวเลขตามที่คาดหวัง",
            "citations": [],
            "ruling": {"verdict": "not_evaluated"},
            "timings_ms": {"total": 1},
            "answer_source": "template",
        }

    monkeypatch.setattr(evaluation.pipeline, "run", fake_run)

    assert asyncio.run(evaluation.main()) == 1
    assert "ข้อมูลถูก 0/1" in capsys.readouterr().out


def test_eval_unverified_expectation_requires_withholding_and_no_draft_citation():
    evaluation = importlib.import_module("eval")
    case = {
        "expect_amount": "100,000",
        "expect_doc": "ประกาศรอตรวจ",
        "expect_status": "withheld_unverified",
        "expect_amount_hidden": True,
    }
    safely_withheld = {
        "answer": "วงเงินจริงยังรอเจ้าหน้าที่ตรวจ จึงยังแสดงไม่ได้",
        "citations": ["หลักเกณฑ์กลาง หน้า 4"],
    }
    leaked = {
        "answer": "วงเงินจริง 100,000 บาท แต่ยังรอเจ้าหน้าที่ตรวจ",
        "citations": ["ประกาศรอตรวจ หน้า 3"],
    }

    assert evaluation.amount_matches_expectation(case, safely_withheld)
    assert evaluation.citation_matches_expectation(case, safely_withheld)
    assert not evaluation.amount_matches_expectation(case, leaked)
    assert not evaluation.citation_matches_expectation(case, leaked)
