"""Regression tests for the public demo's HTTP boundary."""

import asyncio
import base64
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core import security as limits
from app.infrastructure.json import audit
from app.modules.documents import review
from app.modules.legacy import api as main
from app.modules.query import engine as pipeline
from app.main import app


RUN_SCRIPT = Path(__file__).resolve().parent.parent / "run.ps1"


def _run_powershell(script: str) -> dict[str, object]:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("pwsh is required for the run.ps1 lifecycle regression test")

    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    completed = subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("__RESULT__")),
        None,
    )
    assert result_line is not None, completed.stdout
    return json.loads(result_line.removeprefix("__RESULT__"))


def _exercise_launcher_environment_lifecycle(
    tmp_path: Path, *, fail: bool, friend_mode: bool
) -> dict[str, object]:
    launcher = tmp_path / "run.ps1"
    shutil.copy2(RUN_SCRIPT, launcher)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "RISE_LIFECYCLE_IMPORTED=dotenv-only",
                "RISE_LIFECYCLE_EXISTING=dotenv-must-lose",
                "RISE_ADMIN_ENABLED=dotenv-admin",
                "RISE_PROFILE_SOURCE=dotenv-profile",
                "RISE_EMBED_MODEL=alternate-embed:model",
            ]
        ),
        encoding="utf-8",
    )
    launcher_path = str(launcher).replace("'", "''")
    should_fail = "$true" if fail else "$false"
    use_friend_mode = "$true" if friend_mode else "$false"
    return _run_powershell(
        f"""
$ErrorActionPreference = "Stop"
$launcher = '{launcher_path}'
$global:launcherShouldFail = {should_fail}
$global:launcherObserved = @{{}}
$env:RISE_LIFECYCLE_IMPORTED = $null
$env:RISE_LIFECYCLE_EXISTING = "process-value"
$env:RISE_ADMIN_ENABLED = $null
$env:RISE_PROFILE_SOURCE = "process-profile"
$env:RISE_EMBED_MODEL = $null

function global:Invoke-WebRequest {{ throw "offline for launcher lifecycle test" }}
function global:ollama {{}}
function global:python {{
    $global:launcherObserved = @{{
        imported = $env:RISE_LIFECYCLE_IMPORTED
        existing = $env:RISE_LIFECYCLE_EXISTING
        admin = $env:RISE_ADMIN_ENABLED
        profile = $env:RISE_PROFILE_SOURCE
        embed = $env:RISE_EMBED_MODEL
    }}
    $global:LASTEXITCODE = 0
    if ($global:launcherShouldFail) {{ throw "forced check failure" }}
}}

$friendMode = {use_friend_mode}
$friendIp = $null
if ($friendMode) {{
    $friendIp = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
        ForEach-Object {{ $_.GetIPProperties().UnicastAddresses }} |
        ForEach-Object {{ $_.Address.IPAddressToString }} |
        Where-Object {{
            $candidate = [System.Net.IPAddress]::Parse($_)
            if ($candidate.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {{
                return $false
            }}
            $bytes = $candidate.GetAddressBytes()
            return $bytes[0] -eq 10 -or
                ($bytes[0] -eq 172 -and $bytes[1] -ge 16 -and $bytes[1] -le 31) -or
                ($bytes[0] -eq 192 -and $bytes[1] -eq 168)
        }} |
        Select-Object -First 1
    if ($null -eq $friendIp) {{
        Write-Output ('__RESULT__' + (@{{ skipped = $true }} | ConvertTo-Json -Compress))
        exit 0
    }}
}}

$caught = $false
try {{
    if ($friendMode) {{
        & $launcher -Check -FriendDemoIp $friendIp
    }} else {{
        & $launcher -Check
    }}
}} catch {{
    $caught = $true
}}

$result = @{{
    skipped = $false
    caught = $caught
    observed = $global:launcherObserved
    after = @{{
        imported = $env:RISE_LIFECYCLE_IMPORTED
        existing = $env:RISE_LIFECYCLE_EXISTING
        admin = $env:RISE_ADMIN_ENABLED
        profile = $env:RISE_PROFILE_SOURCE
        embed = $env:RISE_EMBED_MODEL
    }}
}}
Write-Output ('__RESULT__' + ($result | ConvertTo-Json -Compress -Depth 4))
"""
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_demo_uses_coco_rise_name_without_changing_chat_storage(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>COCO-RISE demo" in response.text
    assert '<div class="brand">COCO-RISE <span>demo</span></div>' in response.text
    assert "rise-demo-chats-v1" in response.text
    assert app.title == "RISE API"


def test_public_responses_set_browser_security_headers(client: TestClient):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert response.json()["semantic_backend"] in {"bge-m3:latest", "trigram"}


def test_oversized_json_body_is_rejected_before_schema_processing(client: TestClient):
    response = client.post(
        "/v1/query",
        content=b'{' + b'"padding":"' + b'x' * (limits.MAX_REQUEST_BODY_BYTES + 1) + b'"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413


def test_body_limit_middleware_forwards_disconnect_after_replaying_body():
    seen = []
    incoming = [
        {"type": "http.request", "body": b'{}', "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive():
        return incoming.pop(0)

    async def downstream(_scope, replay, _send):
        seen.append(await replay())
        seen.append(await replay())

    async def send(_message):
        pass

    middleware = limits.RequestBodyLimitMiddleware(downstream)
    asyncio.run(middleware({"type": "http", "method": "POST", "path": "/v1/query/stream",
                            "headers": [(b"content-length", b"2")]}, receive, send))

    assert seen[0]["body"] == b'{}'
    assert seen[1]["type"] == "http.disconnect"


def test_admin_and_audit_responses_are_never_cacheable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(review, "TOKEN", "test-admin-token")
    headers = {"X-Admin-Token": "test-admin-token"}

    for path in ("/v1/admin/pending", "/v1/recent"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert "no-store" in response.headers["cache-control"]
        assert "private" in response.headers["cache-control"]


def test_untrusted_host_is_rejected(client: TestClient):
    response = client.get("/healthz", headers={"Host": "untrusted.example"})

    assert response.status_code == 400


def test_trusted_host_configuration_rejects_wildcard_subdomains(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RISE_TRUSTED_HOSTS", "127.0.0.1,*.example.com")

    with pytest.raises(RuntimeError, match="ห้ามใช้ \\*"):
        main._trusted_hosts()


@pytest.mark.parametrize(
    "history",
    [
        ["คำถาม"] * (limits.MAX_HISTORY_TURNS + 1),
        ["ก" * (limits.MAX_QUESTION_CHARS + 1)],
    ],
    ids=["too-many-turns", "oversized-turn"],
)
def test_query_history_is_bounded_before_pipeline_processing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, history: list[str]
):
    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid history reached the answer pipeline")

    monkeypatch.setattr(pipeline, "run", must_not_run)
    monkeypatch.setattr(limits, "take", lambda _user_id: (True, 0.0))

    response = client.post(
        "/v1/query",
        json={"question": "มีทุนอะไรบ้าง", "history": history},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", "ก" * (limits.MAX_FEEDBACK_QUESTION_CHARS + 1)),
        ("answer", "ก" * (limits.MAX_FEEDBACK_ANSWER_CHARS + 1)),
        ("note", "ก" * (limits.MAX_FEEDBACK_NOTE_CHARS + 1)),
    ],
    ids=["question", "answer", "note"],
)
def test_feedback_body_is_bounded_before_audit_write(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    monkeypatch.setattr(
        audit,
        "record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid feedback reached the audit log")
        ),
    )
    body = {"question": "คำถาม", "answer": "คำตอบ", "helpful": False, "note": "หมายเหตุ"}
    body[field] = value

    response = client.post("/v1/feedback", json=body)

    assert response.status_code == 422


def test_feedback_has_a_separate_rate_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(limits, "take_feedback", lambda _user_id: (False, 12.5), raising=False)
    monkeypatch.setattr(
        audit,
        "record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rate-limited feedback reached the audit log")
        ),
    )

    response = client.post(
        "/v1/feedback",
        json={"question": "คำถาม", "answer": "คำตอบ", "helpful": True},
    )

    assert response.status_code == 429
    assert "12.5" in response.json()["detail"]


def test_query_rate_limit_cannot_be_bypassed_with_spoofed_user_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    async def answer(*_args, **_kwargs):
        return {"answer": "คำตอบ", "citations": []}

    limits._client_seen.clear()
    monkeypatch.setattr(limits, "CLIENT_RATE_LIMIT", 1)
    monkeypatch.setattr(limits, "take", lambda _user_id: (True, 0.0))
    monkeypatch.setattr(pipeline, "run", answer)

    first = client.post(
        "/v1/query",
        headers={"X-User-Id": "spoof-a"},
        json={"question": "มีทุนอะไรบ้าง", "persona_id": "P1"},
    )
    second = client.post(
        "/v1/query",
        headers={"X-User-Id": "spoof-b"},
        json={"question": "มีทุนอะไรบ้าง", "persona_id": "P2"},
    )

    assert first.status_code == 200
    assert second.status_code == 429


def test_proxy_identity_header_is_ignored_outside_the_proxy_allowlist(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(main, "TRUST_PROXY_USER", True)
    monkeypatch.setenv("RISE_TRUSTED_PROXY_IPS", "127.0.0.1,::1")

    assert main._proxy_user_id("P2", "203.0.113.10") is None
    assert main._proxy_user_id("P2", "127.0.0.1") == "P2"


def test_proxy_allowlist_rejects_non_ip_entries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RISE_TRUSTED_PROXY_IPS", "127.0.0.1,proxy.example")

    with pytest.raises(RuntimeError, match="IP"):
        main._trusted_proxy_ips()


def test_direct_demo_rejects_unknown_persona_before_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("unknown persona reached the answer pipeline")

    monkeypatch.setattr(pipeline, "run", must_not_run)

    response = client.post(
        "/v1/query",
        json={"question": "มีทุนอะไรบ้าง", "persona_id": "spoofed-persona"},
    )

    assert response.status_code == 400


def test_feedback_reports_when_demo_recording_is_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(audit, "ENABLED", False)
    monkeypatch.setattr(limits, "take_feedback", lambda _user_id: (True, 0.0))
    monkeypatch.setattr(limits, "take_feedback_client", lambda _client: (True, 0.0))

    response = client.post(
        "/v1/feedback",
        json={"question": "คำถาม", "answer": "คำตอบ", "helpful": True},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "recorded": False}
    assert "data.recorded" in main.PAGE


def test_friend_demo_launcher_uses_explicit_safe_defaults():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "$FriendDemoIp" in script
    assert "RISE_ADMIN_TOKEN" in script
    assert "RISE_EXPOSE_DOCS" in script
    assert "RISE_AUDIT" in script
    assert "RISE_TRUST_PROXY_USER" in script
    assert "--workers 1" in script
    assert "--no-proxy-headers" in script
    assert "--host 0.0.0.0" not in script
    assert 'Import-DotEnv ".env"' in script
    assert script.index('Import-DotEnv ".env"') < script.index("if ($friendMode)")


def test_launcher_wraps_dotenv_and_friend_overrides_in_lifo_cleanup():
    script = RUN_SCRIPT.read_text(encoding="utf-8")
    import_call = script.index('Import-DotEnv ".env"')
    lifecycle_try = script.index("try {", script.index("function Import-DotEnv"))
    demo_restore = script.rindex("Restore-DemoEnvironment")
    cleanup_start = script.rindex("} finally {", 0, demo_restore)
    final_cleanup = script[cleanup_start:]

    assert lifecycle_try < import_call
    assert final_cleanup.index("Restore-DemoEnvironment") < final_cleanup.index(
        "Restore-DotEnvEnvironment"
    )


def test_launcher_checks_the_embedding_model_configured_in_dotenv():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'GetEnvironmentVariable("RISE_EMBED_MODEL", "Process")' in script
    assert 'foreach ($m in @("bge-m3"))' not in script
    assert "$embedModelForList -in $modelNames" in script


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("friend_mode", [False, True], ids=["normal", "friend"])
def test_launcher_restores_process_environment_after_exit(
    tmp_path: Path, fail: bool, friend_mode: bool
):
    result = _exercise_launcher_environment_lifecycle(
        tmp_path, fail=fail, friend_mode=friend_mode
    )
    if result["skipped"]:
        pytest.skip("no private IPv4 address is available for friend-mode validation")

    observed = result["observed"]
    after = result["after"]
    assert result["caught"] is fail
    assert observed["imported"] == "dotenv-only"
    assert observed["existing"] == "process-value"
    assert observed["admin"] == ("off" if friend_mode else "dotenv-admin")
    assert observed["profile"] == ("demo" if friend_mode else "process-profile")
    assert observed["embed"] == "alternate-embed:model"
    assert after["imported"] is None
    assert after["existing"] == "process-value"
    assert after["admin"] is None
    assert after["profile"] == "process-profile"
    assert after["embed"] is None


def test_launcher_keeps_optional_ollama_from_becoming_a_startup_requirement():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "Get-Command ollama -ErrorAction SilentlyContinue" in script
    assert "RISE_LLM_BASE" in script
    assert "ระบบยังตอบได้" in script
    assert "template อย่างเดียว" not in script


def test_demo_page_displays_only_the_complete_verified_answer():
    assert "fetch('/v1/query'," in main.PAGE
    assert "fetch('/v1/query/stream'," not in main.PAGE
    assert "const e = await res.json();" in main.PAGE
    assert "turn.answer = e.answer;" in main.PAGE
    assert "res.body.getReader()" not in main.PAGE


def test_demo_chat_keeps_history_inside_each_browser_conversation():
    assert "const history = chat.turns.filter(t => t.done && !t.error)" in main.PAGE
    assert "const personaId = chat.personaId" in main.PAGE
    assert "localStorage.setItem(STORAGE_KEY" in main.PAGE
    assert "chats.filter(c => c.id !== id)" in main.PAGE
