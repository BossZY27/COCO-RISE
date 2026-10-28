"""The canonical UI and API must describe the implemented proposal architecture."""
from pathlib import Path
from app.main import app

STATIC=Path(__file__).resolve().parents[1]/'app/static'


def test_one_api_and_no_retired_aliases():
    paths=app.openapi()['paths']
    assert '/api/query' in paths and '/api/documents' in paths
    assert all(p.startswith('/api/') for p in paths)
    assert '/api/update' not in paths


def test_chat_uses_reviewed_context_citations_and_request_feedback():
    page=(STATIC/'chat.html').read_text()
    assert 'PostgreSQL' in page and 'ChromaDB' in page
    assert 'persona_id' not in page
    assert 'JSON.stringify({question: q, history, context: chat.context})' in page
    assert "request_id: requestId" in page
    assert "turn.requestId" in page and "chat.context = e.context" in page
    assert "info.answer_attempted" in page and "info.retrieval_fallback_reason" in page


def test_admin_separates_pdf_review_publication_and_knowledge_approval():
    page=(STATIC/'admin.html').read_text()
    for suffix in ('/api/documents','/api/versions/','/api/jobs','/review','/publish'):
        assert suffix in page
    assert "['facts',facts]" in page and "['rules',rules]" in page
    assert 'confirmed_pages:checks.map' in page
    for page in STATIC.glob('*.html'):
        text=page.read_text()
        assert '/v1/' not in text and '/v2/' not in text and '/api/update' not in text
