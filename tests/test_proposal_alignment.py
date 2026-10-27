"""The demo surface should match the architecture described in the proposal."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_proposal_api_names_are_available_without_removing_versioned_routes():
    paths = app.openapi()["paths"]

    assert "post" in paths["/api/query"]
    assert "post" in paths["/v1/query"]
    assert "post" in paths["/api/update"]
    assert "post" in paths["/v1/admin/documents"]


def test_update_alias_keeps_admin_authentication():
    response = client.post(
        "/api/update",
        content=b"%PDF-1.4\n%%EOF",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 401


def test_demo_explains_current_architecture_and_profile_scope():
    page = client.get("/").text

    assert "ระบบทำงานอย่างไร" in page
    assert "Hybrid retrieval" in page
    assert "ตรวจคำตอบและอ้างอิง" in page
    assert "โปรไฟล์สมมติ (ใช้ทดสอบกฎเท่านั้น)" in page
    assert "ไม่ใช่ระบบจับคู่หรือแนะนำทุน" in page
    assert "PostgreSQL/pgvector ในเล่มเป็นแผนระยะถัดไป" in page
    assert "textarea{min-width:0" in page
    assert ".persona{white-space:normal;width:100%}" in page
    assert "body{overflow:hidden;" in page


def test_admin_upload_uses_the_proposal_update_endpoint():
    page = client.get("/admin").text

    assert "POST /api/update" in page
    assert "เดโมนี้รับ PDF" in page
    assert "fetch('/api/update'" in page
