"""Regression checks for real model attempts, evidence fencing and conversation context."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4
import pytest
from app.core.settings import Settings
from app.modules.knowledge import corpus as store
from app.modules.knowledge.context import use_corpus
from app.modules.query.analysis import analyze
from app.modules.query.service import QueryService, verify_candidate


@pytest.fixture
def query():
    facts=deepcopy(store.FACTS)
    chunks=deepcopy(store.CHUNKS)
    version, document=str(uuid4()),str(uuid4())
    for c in chunks:
        c.update(version_id=version, document_id=document, chunk_id=str(uuid4()))
    corpus=store.make_corpus(facts,chunks,deepcopy(store.RULES),{},{},42)
    corpus.citations={}
    for f in facts:
        source=f.get('source') or {}
        corpus.citations[f['id']]=[{"version_id":version,"document_id":document,"page":source.get('page',1),"chunk_id":None,"quote":"reviewed quote","label":source.get('doc','source'),"clause":source.get('clause',''),"url":"/old"}]
    model=SimpleNamespace(settings=Settings(_env_file=None),calls=[])
    def candidate(template, citations, timeout=None):
        model.calls.append((template,timeout))
        return template,'test-revision'
    model.candidate=candidate
    repo=SimpleNamespace(revision=42,saved=[])
    repo.corpus_revision=lambda:repo.revision
    repo.save_query=lambda *args:repo.saved.append(args)
    retrieved=[]
    def search(q,year,corpus,timeout=None):
        retrieved.append(q)
        return [c for c in chunks if c.get('verified') is True and c.get('status')=='active'],'bm25+bge-m3+chromadb',None
    service=QueryService(SimpleNamespace(load=lambda:corpus),repo,SimpleNamespace(search=search),model)
    return SimpleNamespace(service=service,model=model,repo=repo,corpus=corpus,retrieved=retrieved)


def test_typo_and_followup_use_same_pipeline(query):
    first=query.service.run('เพดานทุนวิจัยในชั้นเรียนเท่าไร',[])
    assert first['status']=='answered' and first['answer_source']=='llm'
    assert first['processing']['answer_attempted']
    assert first['context']['fund']=='ทุนวิจัยในชั้นเรียน'
    result=query.service.run('มีเงื่ิอนไขในการรับยังไง',[],first['context'])
    assert result['context']['intent'] in {'eligibility','details'}
    assert result['status']!='clarification_required'
    assert query.retrieved[-1].startswith('ทุนวิจัยในชั้นเรียน')
    assert 'eligible' not in str(result['processing'])
    assert all(c['url'].startswith('/api/sources/') for c in first['citations'])


def test_model_failure_uses_verified_template(query):
    def fail(*args,**kwargs): raise TimeoutError('offline')
    query.model.candidate=fail
    result=query.service.run('เพดานทุนวิจัยสถาบันเท่าไร',[])
    assert result['status']=='answered' and result['answer_source']=='template'
    assert result['processing']['answer_attempted']
    assert result['processing']['answer_fallback_reason']=='TimeoutError'


def test_slot_fallback_is_bounded_and_same_model(query):
    def extract(question,known_fund=None,timeout=None):
        assert 0 < timeout <= 5
        return SimpleNamespace(fund='ทุนวิจัยในชั้นเรียน',tier=None,fiscal_year=None,intent='eligibility')
    query.model.extract_slots=extract
    result=query.service.run('ทุนวิจัยในชั้นเรียน เอาเกณฑ์มาดูหน่อย',[])
    assert result['processing']['slot_attempted']
    assert result['processing']['analysis_method']=='rules+llm'
    assert result['processing']['slot_model']==query.model.settings.answer_model


def test_revision_change_during_model_blocks(query):
    original=query.model.candidate
    def change(*args,**kwargs):
        query.repo.revision+=1
        return original(*args,**kwargs)
    query.model.candidate=change
    result=query.service.run('เพดานทุนวิจัยสถาบันเท่าไร',[])
    assert result['status']=='blocked' and result['citations']==[] and result['context'] is None


def test_bad_candidate_falls_back(query):
    query.model.candidate=lambda t,c,timeout=None:(t.replace('25,000','30,000'),'bad')
    result=query.service.run('เพดานทุนวิจัยสถาบันเท่าไร',[])
    assert result['answer_source']=='template'
    assert result['processing']['answer_fallback_reason']=='verification_failed'
    assert '25,000' in result['answer']


def test_new_question_overrides_client_context(query):
    with use_corpus(query.corpus):
        slots=analyze('เพดานทุนวิจัยสถาบัน ปี 2570 เท่าไร',context={'fund':'ทุนวิจัยในชั้นเรียน','intent':'eligibility','fiscal_year':2569,'year_assumed':False})
        assert slots['fund']=='ทุนวิจัยสถาบัน' and slots['fiscal_year']==2570


def test_unsupported_added_condition_is_rejected(query):
    query.model.candidate=lambda t,c,timeout=None:(t+' ผู้ขอรับทุนต้องเป็นหัวหน้าภาคเท่านั้น','bad')
    result=query.service.run('เพดานทุนวิจัยสถาบันเท่าไร',[])
    assert result['answer_source']=='template'
    assert result['verification']['candidate_problems']


def test_catalogue_has_real_versioned_citations(query):
    result=query.service.run('มีทุนอะไรบ้าง',[])
    assert result['status']=='answered'
    assert result['answer_source']=='template'
    assert result['citations'] and all(c['url'].startswith('/api/sources/') for c in result['citations'])


def test_new_database_fund_context_does_not_consult_json_fixtures(query,monkeypatch):
    query.corpus.FACTS[0]={**query.corpus.FACTS[0],'fund':'ทุนใหม่เฉพาะฐานข้อมูล'}
    query.corpus.FUNDS.append('ทุนใหม่เฉพาะฐานข้อมูล')
    with use_corpus(query.corpus):
        slots=analyze('แล้วทุนนี้มีเงื่อนไขอะไร',context={'fund':'ทุนใหม่เฉพาะฐานข้อมูล','fiscal_year':2569,'intent':'details'})
        assert slots['fund']=='ทุนใหม่เฉพาะฐานข้อมูล'


@pytest.mark.parametrize("fund,clause", [
    ("ทุนสนับสนุนการได้รับทุนบริการวิชาการจากภายนอก", "3.1"),
    ("ทุนสนับสนุนการได้รับทุนพัฒนานวัตกรรมจากภายนอก", "3.2"),
])
def test_innovation_conditions_use_reviewed_passages_without_fact_rows(query, fund, clause):
    # Model and semantic retrieval may both be unavailable; source selection still works.
    query.model.settings.answer_enabled = False
    query.service.retrieval.search = lambda *a, **k: ([], "bm25", "embedding_cooldown")
    result = query.service.run(f"{fund} เงื่อนไขในการรับทุน", [])
    assert result["status"] == "answered"
    assert result["context"]["intent"] == "details"
    assert result["verification"]["ok"]
    assert fund in result["answer"] and "10%" in result["answer"]
    assert "ไม่ใช่วงเงินจัดสรรรายปี" in result["answer"]
    assert "ตามประกาศของวิทยาลัย" in result["answer"]
    assert any(c["page"] == 8 and c["clause"] == clause for c in result["citations"])
    assert {c["clause"] for c in result["citations"]} == {clause, "ข้อ 8"}
    assert not any(f["fund"] == fund for f in query.corpus.FACTS)


def test_innovation_conditions_do_not_quote_unreviewed_passage(query):
    query.model.settings.answer_enabled = False
    for chunk in query.corpus.CHUNKS:
        if chunk.get("clause") == "3.1":
            chunk["verified"] = False
    result = query.service.run("ทุนสนับสนุนการได้รับทุนบริการวิชาการจากภายนอก เงื่อนไขในการรับทุน", [])
    assert "10%" not in result["answer"]
    assert not any(c["clause"] == "3.1" for c in result["citations"])
