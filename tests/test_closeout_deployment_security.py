"""P0-04/05/06 offline contracts. No production services or real model calls."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from app.api import preparation as prep, interview_ws
from app.config import Settings, settings
from app.main import app
from app.models.preparation import PreparationCreate
from app.models.resume import ResumeParseResult
from app.models.user import UserInDB
from app.security import get_current_user, validate_auth_configuration
from app.services import preparation_limits as limits, database
from app.services.operation_store import OperationStore
from app.services.preparation_service import PreparationService


def user():
    return UserInDB(id="security-u", username="synthetic-user", password_hash="not-real")


class FakeRedis:
    def __init__(self):
        self.counts = {}

    async def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, *_):
        return True

    async def ttl(self, *_):
        return 60


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    redis = FakeRedis()
    monkeypatch.setattr("app.cache.rate_limiter.get_redis", AsyncMock(return_value=redis))
    store, service = OperationStore(), PreparationService()
    monkeypatch.setattr(prep, "get_operation_store", lambda: store)
    monkeypatch.setattr("app.services.operation_store.get_operation_store", lambda: store)
    monkeypatch.setattr(prep, "get_preparation_service", lambda: service)
    monkeypatch.setattr(prep, "save_preparation_case", AsyncMock())
    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", AsyncMock())
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    limits._active.clear()
    app.dependency_overrides[get_current_user] = user
    yield service, store, redis
    app.dependency_overrides.clear()
    limits._active.clear()


def test_host_connection_derived_and_explicit_history_url_preserved():
    config = Settings(_env_file=None, postgres_url="", postgres_user="synthetic", postgres_password="a@b:c", postgres_db="demo", postgres_host_port=55432)
    assert config.postgres_url == "postgresql://synthetic:a%40b%3Ac@localhost:55432/demo"
    legacy = "postgresql://legacy:synthetic@old-host:5432/old_db"
    assert Settings(_env_file=None, postgres_url=legacy).postgres_url == legacy


@pytest.mark.parametrize("secret", ["", "dev-change-me", "change-me-to-a-long-random-secret", "replace-with-a-long-random-secret", "x" * 64, "0123456789abcdef" * 4, "<generate-a-random-secret-with-secrets.token_urlsafe(48)>"])
def test_startup_rejects_weak_secret_without_echo(monkeypatch, secret):
    monkeypatch.setattr(settings, "auth_secret_key", secret)
    with pytest.raises(RuntimeError) as error:
        validate_auth_configuration()
    if secret:
        assert secret not in str(error.value)


def test_startup_accepts_random_synthetic_secret():
    validate_auth_configuration()


@pytest.mark.asyncio
async def test_weak_secret_stops_lifespan_before_database(monkeypatch):
    from app import main
    monkeypatch.setattr(settings, "auth_secret_key", "dev-change-me")
    init = AsyncMock()
    monkeypatch.setattr(main, "init_database", init)
    with pytest.raises(RuntimeError):
        async with main.lifespan(app):
            pass
    init.assert_not_called()


class AccountPool:
    """Isolated SQL protocol double, NOT a PostgreSQL engine acceptance test."""
    def __init__(self, **kwargs):
        self.rows, self.sql = {}, []
        self.waited = False

    async def open(self): pass
    async def wait(self): self.waited = True
    async def close(self): pass

    @asynccontextmanager
    async def connection(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, query, params=()):
        normalized = " ".join(query.split()).lower()
        self.sql.append(normalized)
        row = None
        if normalized.startswith("insert into users"):
            row = dict(id=params[0], username=params[1], password_hash=params[2], display_name=params[3], is_active=True)
            self.rows[row["id"]] = row
        elif normalized.startswith("select * from users"):
            row = next((r for r in self.rows.values() if (r["username"].lower() == params[0].lower() if "username" in normalized else r["id"] == params[0])), None)
        elif normalized.startswith("update users"):
            self.rows[params[1]]["last_login_at"] = params[0]
        return SimpleNamespace(fetchone=AsyncMock(return_value=row), fetchall=AsyncMock(return_value=[]))


def test_init_registration_login_and_authenticated_api_with_isolated_account_store(monkeypatch, isolated):
    from app import main
    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", AccountPool)
    monkeypatch.setattr(main, "init_checkpointer", AsyncMock())
    monkeypatch.setattr(main, "close_checkpointer", AsyncMock())
    monkeypatch.setattr(main, "probe_model_capabilities", AsyncMock())
    monkeypatch.setattr(main, "close_redis", AsyncMock())
    database.reset_database_for_tests()
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        pool = database.get_pool()
        assert pool.waited and any("create table if not exists users" in s for s in pool.sql)
        assert not any(s.startswith(("drop ", "truncate ", "delete ")) for s in pool.sql)
        assert client.get("/health").status_code == 200
        payload = dict(username="synthetic@example.test", password="Synthetic-password-123", display_name="Synthetic")
        registered = client.post("/api/auth/register", json=payload)
        assert registered.status_code == 200
        assert client.post("/api/auth/register", json=payload).status_code == 409
        login = client.post("/api/auth/login", json=payload)
        assert login.status_code == 200
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert client.get("/api/auth/me", headers=auth).json()["username"] == payload["username"]
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={**payload, "password": "wrong-password"}).status_code == 401
        case = client.post("/api/preparation/cases", headers=auth, json={"jd_text": "Synthetic RAG JD"})
        assert case.status_code == 201
        assert client.get(f"/api/preparation/cases/{case.json()['case_id']}", headers=auth).status_code == 200
    assert database.get_pool() is None


@pytest.mark.asyncio
async def test_different_operation_ids_cannot_queue_same_user_and_duplicate_reuses(monkeypatch, isolated):
    service, store, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    gate = asyncio.Event()
    monkeypatch.setattr(prep, "_run_analysis", AsyncMock(side_effect=lambda *_: None))
    async def pending(*_):
        await gate.wait()
    monkeypatch.setattr(prep, "_run_analysis", pending)
    first = await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="one"), user())
    task = prep._analysis_tasks["one"]
    second = await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="one"), user())
    assert first == second and prep._analysis_tasks["one"] is task
    with pytest.raises(HTTPException) as error:
        await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="two"), user())
    assert error.value.status_code == 429 and await store.get("two") is None
    gate.set()
    await task
    assert user().id not in limits._active


@pytest.mark.asyncio
async def test_cancel_before_task_starts_releases_user_slot(monkeypatch, isolated):
    service, _, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="cancel"), user())
    task = prep._analysis_tasks["cancel"]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert user().id not in limits._active


@pytest.mark.asyncio
async def test_simultaneous_duplicate_admission_charges_once(monkeypatch, isolated):
    service, _, redis = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    arrived, release = asyncio.Event(), asyncio.Event()
    original = limits.enforce
    async def delayed(*args):
        arrived.set()
        await release.wait()
        await original(*args)
    monkeypatch.setattr(limits, "enforce", delayed)
    task_gate = asyncio.Event()
    async def pending(*_): await task_gate.wait()
    monkeypatch.setattr(prep, "_run_analysis", pending)
    first = asyncio.create_task(prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="same"), user()))
    await arrived.wait()
    second = asyncio.create_task(prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="same"), user()))
    release.set()
    results = await asyncio.gather(first, second)
    assert results[0] == results[1]
    assert sum(redis.counts.values()) == 1
    running = prep._analysis_tasks["same"]
    task_gate.set()
    await running


@pytest.mark.asyncio
async def test_account_and_checkpoint_share_selected_postgres_url(monkeypatch):
    from app.services import checkpoint
    captured = []
    @asynccontextmanager
    async def saver(url):
        captured.append(url)
        yield SimpleNamespace(setup=AsyncMock())
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string", saver)
    monkeypatch.setattr(settings, "checkpointer_backend", "postgres")
    monkeypatch.setattr(settings, "session_store_backend", "postgres")
    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", AccountPool)
    database.reset_database_for_tests()
    checkpoint.reset_checkpointer_for_tests()
    try:
        await database.init_database()
        await checkpoint.init_checkpointer()
        assert captured == [settings.postgres_url]
        assert database.session_store_enabled() and database.get_pool() is not None
    finally:
        await checkpoint.close_checkpointer()
        await database.close_database()


@pytest.mark.asyncio
async def test_create_interview_and_vision_limits_precede_expensive_work(monkeypatch, isolated):
    service, _, _ = isolated
    monkeypatch.setattr(settings, "rate_limit_start_per_minute", 1)
    await limits.enforce("interview:start", user().id, 1, 60)
    getter = AsyncMock()
    monkeypatch.setattr(prep, "get_session_manager", getter)
    with pytest.raises(HTTPException) as error:
        await prep.create_interview_from_case("anything", prep.CreateInterviewRequest(), user())
    assert error.value.status_code == 429
    getter.assert_not_called()
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    monkeypatch.setattr(settings, "rate_limit_vision_per_day", 1)
    await limits.enforce("resume:vision", user().id, 1, 86400)
    parser = AsyncMock()
    monkeypatch.setattr(prep, "parse_resume_with_metadata", parser)
    upload = UploadFile(io.BytesIO(b"\x89PNG\r\n\x1a\n"), filename="resume.png", headers=Headers({"content-type": "image/png"}))
    with pytest.raises(HTTPException) as error:
        await prep.upload_case_resume(case.case_id, user(), upload)
    assert error.value.status_code == 429
    parser.assert_not_called()


@pytest.mark.asyncio
async def test_ocr_operation_cannot_be_resumed_as_analysis(isolated):
    _, store, _ = isolated
    await store.accept(user_id=user().id, kind="parse_resume", payload={}, operation_id="ocr")
    await store.transition("ocr", "failed_retryable")
    with pytest.raises(HTTPException) as error:
        await prep.continue_operation("ocr", user())
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_restart_then_continue_keeps_consumed_model_budget(monkeypatch, isolated):
    from app.llm.client import _record_model_request_started
    from app.models.preparation import OperationRecord
    service, store, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    saved = {}
    async def save(record):
        saved[record.operation_id] = record.model_dump_json()
    async def load(operation_id):
        return OperationRecord.model_validate_json(saved[operation_id]) if operation_id in saved else None
    monkeypatch.setattr("app.services.operation_store.save_operation", save)
    monkeypatch.setattr("app.services.operation_store.load_operation", load)
    physical = []
    async def material(*_):
        await _record_model_request_started(operation="JD", model="stub", queue_elapsed_seconds=0)
        physical.append("request")
        await asyncio.Event().wait()
    monkeypatch.setattr(service, "analyze", material)
    monkeypatch.setattr(settings, "assessment_model_request_budget", 1)
    monkeypatch.setattr(settings, "preparation_timeout_seconds", .02)
    await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="restart"), user())
    await prep._analysis_tasks["restart"]
    restored = OperationStore()
    monkeypatch.setattr(prep, "get_operation_store", lambda: restored)
    monkeypatch.setattr("app.services.operation_store.get_operation_store", lambda: restored)
    await prep.continue_operation("restart", user())
    await prep._analysis_tasks["restart"]
    result = await restored.get("restart")
    assert result.status == "failed" and result.budget["model_requests"] == 0
    assert result.metrics["physical_requests"] == 1 and physical == ["request"]


def test_endpoint_rate_limit_and_redis_unavailable_are_closed(monkeypatch, isolated):
    monkeypatch.setattr(settings, "rate_limit_start_per_minute", 1)
    client = TestClient(app)
    assert client.post("/api/preparation/cases", json={"jd_text": "first"}).status_code == 201
    assert client.post("/api/preparation/cases", json={"jd_text": "second"}).status_code == 429
    monkeypatch.setattr("app.cache.rate_limiter.get_redis", AsyncMock(return_value=None))
    assert client.post("/api/preparation/cases", json={"jd_text": "unavailable"}).status_code == 429


@pytest.mark.asyncio
async def test_analysis_frontloaded_model_calls_consume_persisted_budget(monkeypatch, isolated):
    from app.llm.client import _record_model_request_started
    service, store, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD", resume_text="Synthetic resume"))
    calls = []
    async def material(*_):
        for stage in ("JD", "Resume"):
            await _record_model_request_started(operation=stage, model="stub", queue_elapsed_seconds=0)
            calls.append(stage)
        return case
    monkeypatch.setattr(service, "analyze", material)
    monkeypatch.setattr(settings, "assessment_model_request_budget", 1)
    await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="budget"), user())
    await prep._analysis_tasks["budget"]
    operation = await store.get("budget")
    assert calls == ["JD"] and operation.budget["model_requests"] == 0
    assert operation.status == "failed" and "budget exhausted" in operation.error
    continued = await prep.continue_operation("budget", user())
    assert continued.budget["model_requests"] == 0 and calls == ["JD"]


@pytest.mark.asyncio
async def test_analysis_timeout_releases_slot_and_continue_keeps_budget(monkeypatch, isolated):
    service, store, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    async def hung(*_):
        await asyncio.Event().wait()
    monkeypatch.setattr(service, "analyze", hung)
    monkeypatch.setattr(settings, "preparation_timeout_seconds", .02)
    await prep.start_analysis(case.case_id, prep.StartOperationRequest(operation_id="timeout"), user())
    await prep._analysis_tasks["timeout"]
    record = await store.get("timeout")
    assert record.status == "failed_retryable" and user().id not in limits._active
    await store.consume_budget("timeout", "model_requests")
    remaining = (await store.get("timeout")).budget["model_requests"]
    await prep.continue_operation("timeout", user())
    await prep._analysis_tasks["timeout"]
    assert (await store.get("timeout")).budget["model_requests"] == remaining


@pytest.mark.parametrize("name,body,mime,status", [
    ("resume.exe", b"fake", "application/octet-stream", 415),
    ("resume.pdf", b"fake", "application/pdf", 415),
    ("resume.pdf", b"%PDF-" + b"x" * 1200, "application/pdf", 413),
    ("resume.png", b"\x89PNG\r\n\x1a\n", "text/html", 415),
    ("resume.pdf", b"", "application/pdf", 400),
])
def test_invalid_upload_rejected_before_parser(monkeypatch, isolated, name, body, mime, status):
    monkeypatch.setattr(settings, "resume_upload_max_bytes", 1024)
    parser = AsyncMock()
    monkeypatch.setattr(prep, "parse_resume_with_metadata", parser)
    client = TestClient(app)
    case = client.post("/api/preparation/cases", json={"jd_text": "Synthetic JD"}).json()
    response = client.post(f"/api/preparation/cases/{case['case_id']}/resume", files={"resume_file": (name, body, mime)})
    assert response.status_code == status
    parser.assert_not_called()


def test_streamed_oversize_upload_without_content_length(monkeypatch, isolated):
    monkeypatch.setattr(settings, "resume_upload_max_bytes", 10)
    response = TestClient(app).post("/api/preparation/cases/synthetic/resume", content=iter([b"x" * 140000]), headers={"Content-Type": "multipart/form-data; boundary=synthetic"})
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_unknown_upload_length_reads_at_most_limit_plus_one(monkeypatch):
    monkeypatch.setattr(settings, "resume_upload_max_bytes", 100)
    upload = UploadFile(io.BytesIO(b"%PDF-" + b"x" * 200), filename="resume.pdf", headers=Headers({"content-type": "application/pdf"}))
    with pytest.raises(HTTPException) as error:
        await limits.read_resume(upload)
    assert error.value.status_code == 413 and upload.file.tell() == 101


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["budget", "timeout", "success"])
async def test_ocr_budget_deadline_and_valid_upload(monkeypatch, isolated, mode):
    from app.llm.client import _record_model_request_started
    service, store, _ = isolated
    case = await service.create(user().id, PreparationCreate(jd_text="Synthetic JD"))
    async def parse(_):
        if mode == "timeout":
            await asyncio.Event().wait()
        await _record_model_request_started(operation="vision", model="stub", queue_elapsed_seconds=0)
        return ResumeParseResult(raw_text="Synthetic", normalized_text="Synthetic resume")
    monkeypatch.setattr(prep, "parse_resume_with_metadata", parse)
    monkeypatch.setattr(settings, "resume_parse_timeout_seconds", .02)
    monkeypatch.setattr(settings, "resume_model_request_budget", 0 if mode == "budget" else 1)
    upload = UploadFile(io.BytesIO(b"\x89PNG\r\n\x1a\nsynthetic"), filename="resume.png", headers=Headers({"content-type": "image/png"}))
    if mode == "success":
        result = await prep.upload_case_resume(case.case_id, user(), upload)
        assert result.material_revision == 2 and result.resume_text == "Synthetic resume"
    else:
        with pytest.raises((HTTPException, RuntimeError)):
            await prep.upload_case_resume(case.case_id, user(), upload)
    assert user().id not in limits._active
    operation = next(iter(store._operations.values()))
    assert operation.status == {"budget": "failed", "timeout": "failed_retryable", "success": "succeeded"}[mode]
    assert operation.budget["model_requests"] == (1 if mode == "timeout" else 0)


@pytest.mark.asyncio
async def test_operation_id_cannot_be_reused_by_other_user(isolated):
    _, store, _ = isolated
    await store.accept(user_id="one", kind="prepare_case", case_id="c", payload={}, operation_id="same")
    with pytest.raises(ValueError):
        await store.accept(user_id="two", kind="prepare_case", case_id="c", payload={}, operation_id="same")


@pytest.mark.parametrize("kind", ["missing", "expired", "invalid", "query", "timeout"])
def test_websocket_rejects_auth_before_loading_session(monkeypatch, kind):
    getter = AsyncMock()
    monkeypatch.setattr(interview_ws, "get_user_by_id", getter)
    manager = AsyncMock()
    monkeypatch.setattr(interview_ws, "get_session_manager", manager)
    monkeypatch.setattr(settings, "ws_auth_timeout_seconds", .02)
    token = "invalid"
    if kind == "expired":
        token = jwt.encode({"sub": user().id, "exp": datetime.now(timezone.utc) - timedelta(seconds=1)}, settings.auth_secret_key, algorithm="HS256")
    url = "/api/ws/interview/synthetic" + ("?token=synthetic-legacy-token" if kind == "query" else "")
    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(app).websocket_connect(url) as ws:
            if kind != "timeout":
                ws.send_json({"type": "auth", "token": token} if kind != "missing" else {"type": "answer", "content": "x"})
            ws.receive_json()
    assert error.value.code == 4401
    getter.assert_not_called()
    manager.assert_not_called()


def test_access_filters_remove_query_tokens_and_suppress_raw_frames():
    from app.logging_security import AccessQueryFilter, QueryStringFilter
    from uvicorn.logging import AccessFormatter
    secret = "synthetic-sensitive-token"
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', ("client", "GET", f"/ws?token={secret}&x=1", "1.1", 200), None)
    assert AccessQueryFilter().filter(record)
    assert secret not in AccessFormatter().format(record)
    handshake = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, 'WebSocket %s', (f"/ws?token={secret}",), None)
    assert QueryStringFilter().filter(handshake) and secret not in handshake.getMessage()
    frame = logging.LogRecord("uvicorn.error", logging.DEBUG, "", 0, '< TEXT %s', (secret,), None)
    assert not QueryStringFilter().filter(frame)
    nginx = (Path(__file__).resolve().parents[1] / "web/nginx.conf").read_text()
    log_format = nginx.split("access_log", 1)[0]
    assert "$uri" in log_format and "$request_uri" not in log_format and "$request " not in log_format
