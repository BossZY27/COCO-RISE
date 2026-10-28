"""One self-hosted contract describes the real canonical application."""
import json
import os
import subprocess
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

ROOT=Path(__file__).resolve().parents[1]


def test_swagger_is_self_hosted_and_documents_admin_security():
    with TestClient(app) as client:
        page=client.get('/docs')
        assert page.status_code==200 and 'https://' not in page.text
        assert '/_swagger/swagger-ui-bundle.js' in page.text
        for asset in ('swagger-ui-bundle.js','swagger-ui.css','favicon-32x32.png'):
            assert client.get('/_swagger/'+asset).status_code==200
        spec=client.get('/openapi.json').json()
        assert spec['info']['title']=='RISE API'
        assert spec['components']['securitySchemes']['AdminToken']['name']=='X-Admin-Token'
        assert spec['paths']['/api/documents']['post']['security']==[{'AdminToken':[]}]
        assert 'application/pdf' in spec['paths']['/api/documents']['post']['requestBody']['content']
        assert client.get('/design/docs').status_code==404


def test_saved_contract_matches_current_application_schemas():
    from scripts.export_openapi import design_schema
    saved=json.loads((ROOT/'docs/openapi-mvp.json').read_text())
    assert saved==design_schema()
    def refs(value):
        if isinstance(value,dict):
            if '$ref' in value: yield value['$ref']
            for v in value.values(): yield from refs(v)
        elif isinstance(value,list):
            for v in value: yield from refs(v)
    for ref in refs(saved):
        value=saved
        for key in ref.removeprefix('#/').split('/'):
            value=value[key.replace('~1','/').replace('~0','~')]


def test_docs_can_be_disabled():
    env={**os.environ,'RISE_EXPOSE_DOCS':'off'}
    result=subprocess.run([sys.executable,'-c',"from fastapi.testclient import TestClient; from app.main import app; c=TestClient(app); print([c.get(p).status_code for p in ['/docs','/openapi.json']])"],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='[404, 404]'
