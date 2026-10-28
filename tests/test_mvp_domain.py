"""Small pure checks for answer gates, migration safety, and model completion."""
from pathlib import Path
import httpx
import pytest
from app.modules.knowledge import corpus as store
from app.modules.query import engine as pipeline
from app.core.settings import Settings
from app.infrastructure.ollama.client import OllamaClient
from app.modules.query.service import verify_candidate
from scripts.migrate_env import migrate


def test_env_migration_preserves_new_keys_and_secrets(tmp_path):
    path = tmp_path / ".env"
    path.write_text("IRFA_ADMIN_TOKEN=old-secret\nRISE_ADMIN_TOKEN=new-secret\nIRFA_LLM_BASE=http://legacy\n")
    assert migrate(path) == 1
    assert path.read_text() == "RISE_ADMIN_TOKEN=new-secret\nRISE_LLM_BASE=http://legacy\n"
    assert migrate(path) == 0
    if __import__("os").name != "nt":
        assert path.stat().st_mode & 0o077 == 0


def test_candidate_gate_rejects_changed_year_amount_fund_and_qualification():
    fact = next(f for f in store.FACTS if f["id"] == "c02")
    slots = pipeline.analyze("เพดานทุนวิจัยสถาบันเท่าไร")
    citations = [f"{fact['source']['doc']} {fact['source']['clause']} หน้า {fact['source']['page']}"]
    template = "เพดานทุนวิจัยสถาบัน ไม่เกิน 25,000 บาท"
    evidence = {"chunks": [], "facts": [fact]}
    assert verify_candidate(template, template, citations, [fact], evidence, slots, {})["ok"]
    for candidate in ("ทุนวิจัยสถาบัน ได้รับเงิน 25,000 บาทแน่นอน", template.replace("25,000", "26,000"),
                      template + " ปี 2570", template.replace("ทุนวิจัยสถาบัน", "ทุนวิจัยในชั้นเรียน")):
        assert not verify_candidate(candidate, template, citations, [fact], evidence, slots, {})["ok"]


def test_ollama_discards_partial_and_truncated_generation():
    client = OllamaClient(Settings(_env_file=None, answer_model="qwen3:4b"))
    def respond(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models":[{"name":"qwen3:4b","digest":"revision-1"}]})
        return httpx.Response(200, json={"done": True, "done_reason":"length", "message":{"content":"unsafe partial"}})
    client.http.close()
    client.http = httpx.Client(base_url="http://model", transport=httpx.MockTransport(respond))
    with pytest.raises(ValueError, match="truncated"):
        client.candidate("template", [])
    client.close()


def test_request_corpus_is_isolated_and_does_not_require_fixture_files(monkeypatch):
    from app.modules.knowledge.context import use_corpus
    corpus = store.make_corpus([], [], [], {}, {}, 1)
    with use_corpus(corpus):
        assert not store.view().FACTS
        assert store.keyword_search("anything", None) == []
    assert store.view().FACTS
