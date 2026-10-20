from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_preparation_limiter(monkeypatch):
    async def allow(*_args, **_kwargs):
        return SimpleNamespace(allowed=True)
    monkeypatch.setattr("app.services.preparation_limits.check_rate_limit", allow)


from fastapi.testclient import TestClient

from app.api import preparation as preparation_api
from app.main import app
from app.models.evidence import EvidenceAssessment
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import ClarificationRequest, PreparationCase, PreparationCreate
from app.models.resume import ResumeParseResult
from app.models.user import UserInDB
from app.security import get_current_user
from app.services.operation_store import OperationStore
from app.services.preparation_service import PreparationService, build_assessment_targets


def _user(user_id: str = "u1") -> UserInDB:
    return UserInDB(
        id=user_id,
        username=f"{user_id}@example.test",
        password_hash="hash",
        display_name=user_id,
        role="user",
        is_active=True,
    )


def test_preparation_operation_result_does_not_expose_raw_evidence():
    compact = preparation_api._compact_assessment_result(
        {
            "status": "succeeded",
            "complete": True,
            "trajectory": [
                {
                    "target_id": "target-rag",
                    "tool": "retrieve_knowledge",
                    "decision_source": "llm_tool_call",
                    "result": {
                        "chunks": [
                            {"chunk_id": "secret", "text": "raw evidence body"}
                        ]
                    },
                }
            ],
            "targets": [
                {
                    "target_id": "target-rag",
                    "verdict": "sufficient",
                    "complete": True,
                }
            ],
        }
    )
    assert compact is not None
    serialized = str(compact)
    assert "raw evidence body" not in serialized
    assert "chunks" not in serialized
    assert compact["trajectory"][0]["decision_source"] == "llm_tool_call"


def test_preparation_api_contract_and_user_isolation(monkeypatch):
    service = PreparationService()
    operations = OperationStore()

    async def noop(*_args, **_kwargs):
        return None

    async def current_user():
        return _user()

    monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
    monkeypatch.setattr(preparation_api, "get_operation_store", lambda: operations)
    monkeypatch.setattr(preparation_api, "_run_analysis", noop)
    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", noop)
    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    app.dependency_overrides[get_current_user] = current_user
    client = TestClient(app)
    try:
        created = client.post(
            "/api/preparation/cases",
            json={"jd_text": "需要 RAG、LangGraph 和可靠性设计经验"},
        )
        assert created.status_code == 201
        case_id = created.json()["case_id"]

        loaded = client.get(f"/api/preparation/cases/{case_id}")
        assert loaded.status_code == 200
        assert loaded.json()["material_revision"] == 1

        updated = client.patch(
            f"/api/preparation/cases/{case_id}",
            json={"project_notes": "实现过 RAG 检索纠错"},
        )
        assert updated.status_code == 200
        assert updated.json()["material_revision"] == 2

        started = client.post(f"/api/preparation/cases/{case_id}/analysis", json={})
        assert started.status_code == 202
        operation_id = started.json()["operation_id"]
        status = client.get(f"/api/preparation/operations/{operation_id}")
        assert status.status_code == 200
        assert status.json()["input_revision"] == 2
        assert started.json()["status_url"] == f"/api/preparation/operations/{operation_id}"
        events = client.get(f"/api/preparation/operations/{operation_id}/events")
        assert events.status_code == 200
        assert events.json()["events"][0]["type"] == "operation_accepted"

        async def other_user():
            return _user("u2")

        app.dependency_overrides[get_current_user] = other_user
        denied = client.get(f"/api/preparation/cases/{case_id}")
        assert denied.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_create_interview_from_ready_case_passes_targets_and_evidence(monkeypatch):
    service = PreparationService()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", noop)
    case = __import__("asyncio").run(
        service.create("u1", PreparationCreate(jd_text="需要 RAG 可靠性经验"))
    )
    matrix = SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
    )
    case.targets = build_assessment_targets(case, matrix)
    case.evidence_assessments = {
        case.targets[0].target_id: EvidenceAssessment(
            target_id=case.targets[0].target_id,
            relevant_chunk_ids=["rag-1"],
            supported_aspects=["technical_knowledge"],
            verdict="sufficient",
        )
    }
    case.status = "ready"
    service._cases[case.case_id] = case

    captured = {}

    class FakeManager:
        def create_session(self, **kwargs):
            captured.update(kwargs)
            from app.models.interview import InterviewSession

            return InterviewSession(
                session_id=kwargs["session_id"],
                user_id=kwargs["user_id"],
                preparation_case_id=kwargs["preparation_case_id"],
                jd_text=kwargs["jd_text"],
            )

        async def persist_session(self, _session_id):
            return None

    async def current_user():
        return _user()

    monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
    monkeypatch.setattr(preparation_api, "get_session_manager", lambda: FakeManager())
    app.dependency_overrides[get_current_user] = current_user
    client = TestClient(app)
    try:
        response = client.post(
            f"/api/preparation/cases/{case.case_id}/interviews",
            json={"mode": "practice", "max_follow_ups": 1},
        )
        assert response.status_code == 201
        assert captured["preparation_case_id"] == case.case_id
        assert captured["assessment_targets"][0].scope == "RAG"
    finally:
        app.dependency_overrides.clear()


def test_create_interview_rejects_pending_or_unassessed_preparation(monkeypatch):
    service = PreparationService()

    async def noop(*_args, **_kwargs):
        return None

    async def current_user():
        return _user()

    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", noop)
    case = __import__("asyncio").run(
        service.create("u1", PreparationCreate(jd_text="需要 RAG 可靠性经验"))
    )
    matrix = SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
    )
    case.targets = build_assessment_targets(case, matrix)
    case.status = "ready"
    service._cases[case.case_id] = case
    monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
    app.dependency_overrides[get_current_user] = current_user
    client = TestClient(app)
    try:
        unassessed = client.post(
            f"/api/preparation/cases/{case.case_id}/interviews",
            json={"mode": "practice", "max_follow_ups": 1},
        )
        assert unassessed.status_code == 409
        assert "no question-eligible targets" in unassessed.json()["detail"]

        case.status = "waiting_input"
        case.clarifications = [
            ClarificationRequest(
                clarification_id="clarify-pending",
                case_id=case.case_id,
                material_revision=1,
                question="请补充职责",
                missing_information="职责",
            )
        ]
        service._cases[case.case_id] = case
        waiting = client.post(
            f"/api/preparation/cases/{case.case_id}/interviews",
            json={"mode": "practice", "max_follow_ups": 1},
        )
        assert waiting.status_code == 409
        assert "not ready" in waiting.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_preparation_resume_upload_versions_material_and_keeps_warnings(monkeypatch, tmp_path):
    service = PreparationService()

    async def noop(*_args, **_kwargs):
        return None

    async def parse(_path):
        return ResumeParseResult(
            raw_text="RAG 项目",
            normalized_text="负责 RAG 检索纠错",
            warnings=["未识别到明确时间范围"],
        )

    async def current_user():
        return _user()

    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", noop)
    monkeypatch.setattr(preparation_api, "save_preparation_case", noop)
    monkeypatch.setattr(preparation_api, "parse_resume_with_metadata", parse)
    monkeypatch.setattr(type(preparation_api.settings), "upload_root", property(lambda _self: tmp_path))
    monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
    app.dependency_overrides[get_current_user] = current_user
    client = TestClient(app)
    try:
        created = client.post("/api/preparation/cases", json={"jd_text": "需要 RAG 工程经验"})
        case_id = created.json()["case_id"]
        response = client.post(
            f"/api/preparation/cases/{case_id}/resume",
            files={"resume_file": ("resume.pdf", b"%PDF-synthetic", "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json()["material_revision"] == 2
        assert response.json()["resume_text"] == "负责 RAG 检索纠错"
        assert response.json()["resume_parse_warnings"] == ["未识别到明确时间范围"]
    finally:
        app.dependency_overrides.clear()


def test_cancel_operation_stops_the_live_background_task(monkeypatch):
    import asyncio

    async def scenario():
        store = OperationStore()

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr("app.services.operation_store.save_operation", noop)
        monkeypatch.setattr("app.services.operation_store.load_operation", noop)
        monkeypatch.setattr(preparation_api, "get_operation_store", lambda: store)
        operation = await store.accept(
            user_id="u1", kind="prepare_case", payload={"revision": 1}, operation_id="cancel-op"
        )
        await store.transition(operation.operation_id, "running")
        task = asyncio.create_task(asyncio.Event().wait())
        preparation_api._analysis_tasks[operation.operation_id] = task
        result = await preparation_api.cancel_operation(operation.operation_id, _user())
        await asyncio.sleep(0)
        assert result.status == "cancelled"
        assert task.cancelled()
        preparation_api._analysis_tasks.pop(operation.operation_id, None)

    asyncio.run(scenario())


def test_analysis_pauses_before_agent_and_resumes_from_frozen_targets(monkeypatch):
    import asyncio

    async def scenario():
        store = OperationStore()
        target = build_assessment_targets(
            PreparationCase(case_id="case-hitl", user_id="u1", jd_text="需要 RAG"),
            SkillMatrix(
                position_title="AI Engineer",
                experience_level="mid",
                skills=[
                    SkillItem(
                        name="RAG", category="framework", weight=1, is_required=True
                    )
                ],
            ),
        )[0]
        draft = PreparationCase(case_id="case-hitl", user_id="u1", jd_text="需要 RAG")
        waiting = draft.model_copy(
            update={
                "status": "waiting_input",
                "targets": [target],
                "clarifications": [
                    ClarificationRequest(
                        clarification_id="clarify-1",
                        case_id=draft.case_id,
                        material_revision=1,
                        question="请补充项目规模",
                        missing_information="项目规模",
                    )
                ],
            }
        )

        class FakeService:
            def __init__(self):
                self.case = draft
                self.analyze_calls = 0

            async def get(self, case_id, user_id):
                return self.case if (case_id, user_id) == ("case-hitl", "u1") else None

            async def analyze(self, _case_id, _user_id):
                self.analyze_calls += 1
                self.case = waiting
                return self.case

        service = FakeService()
        agent_calls = 0

        class FakeAgent:
            def __init__(self, _operations):
                pass

            async def run(self, _context):
                nonlocal agent_calls
                agent_calls += 1
                return {"status": "succeeded"}

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr("app.services.operation_store.save_operation", noop)
        monkeypatch.setattr("app.services.operation_store.load_operation", noop)
        monkeypatch.setattr(preparation_api, "get_operation_store", lambda: store)
        monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
        monkeypatch.setattr(preparation_api, "JobAssessmentAgent", FakeAgent)
        monkeypatch.setattr(
            preparation_api.settings, "assessment_autonomous_loop_enabled", False
        )
        operation = await store.accept(
            user_id="u1",
            kind="prepare_case",
            case_id="case-hitl",
            input_revision=1,
            payload={"revision": 1},
            operation_id="hitl-op",
        )

        await preparation_api._run_analysis(operation.operation_id, "case-hitl", "u1")
        paused = await store.get(operation.operation_id, "u1")
        assert paused is not None and paused.status == "waiting_input"
        assert paused.result["agent_loop"] == {"enabled": False, "status": "disabled"}
        assert paused.result["evidence_validation"] == {
            "status": "not_performed",
            "mode": "none",
        }
        assert service.analyze_calls == 1
        assert agent_calls == 0

        service.case = waiting.model_copy(
            update={
                "status": "ready",
                "clarifications": [
                    waiting.clarifications[0].model_copy(
                        update={"status": "answered", "answer": "日请求百万级"}
                    )
                ],
            }
        )
        monkeypatch.setattr(
            preparation_api.settings, "assessment_autonomous_loop_enabled", True
        )
        await store.transition(operation.operation_id, "accepted")
        await preparation_api._run_analysis(operation.operation_id, "case-hitl", "u1")
        completed = await store.get(operation.operation_id, "u1")
        assert completed is not None and completed.status == "succeeded"
        assert service.analyze_calls == 1
        assert agent_calls == 1
        assert completed.result["material_analysis"]["status"] == "completed"
        assert completed.result["evidence_validation"]["status"] == "not_performed"
        assert completed.result["evidence_sufficiency"] == "unknown"
        assert completed.result["agent_loop"] == {"enabled": False, "status": "disabled"}
        assert completed.result["assessment_result"] == {
            "status": "succeeded",
            "targets": [],
            "trajectory": [],
        }
        events = await store.events(operation.operation_id, "u1")
        assert events is not None
        assert [
            item["status"]
            for item in events
            if item["type"] in {"operation_accepted", "operation_status"}
        ] == [
            "accepted",
            "running",
            "waiting_input",
            "accepted",
            "running",
            "succeeded",
        ]

    asyncio.run(scenario())


def test_default_analysis_completes_materials_without_running_agent(monkeypatch):
    import asyncio

    async def scenario():
        store = OperationStore()
        matrix = SkillMatrix(
            position_title="AI Engineer",
            experience_level="mid",
            skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
        )
        draft = PreparationCase(case_id="case-ready", user_id="u1", jd_text="需要 RAG")
        ready = draft.model_copy(
            update={
                "status": "ready",
                "skill_matrix": matrix,
                "targets": build_assessment_targets(draft, matrix),
            }
        )

        class FakeService:
            def __init__(self):
                self.case = draft
                self.analyze_calls = 0

            async def get(self, _case_id, _user_id):
                return self.case

            async def analyze(self, _case_id, _user_id):
                self.analyze_calls += 1
                self.case = ready
                return ready

        service = FakeService()

        assessment_calls = 0

        class DeterministicAssessment:
            def __init__(self, _operations):
                pass

            async def run(self, _context):
                nonlocal assessment_calls
                assessment_calls += 1
                return {"status": "succeeded", "complete": False}

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr("app.services.operation_store.save_operation", noop)
        monkeypatch.setattr("app.services.operation_store.load_operation", noop)
        monkeypatch.setattr(preparation_api, "get_operation_store", lambda: store)
        monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: service)
        monkeypatch.setattr(preparation_api, "JobAssessmentAgent", DeterministicAssessment)
        monkeypatch.setattr(
            preparation_api.settings, "assessment_autonomous_loop_enabled", False
        )
        operation = await store.accept(
            user_id="u1",
            kind="prepare_case",
            case_id=draft.case_id,
            input_revision=1,
            payload={"revision": 1},
            operation_id="material-only-op",
        )

        await preparation_api._run_analysis(operation.operation_id, draft.case_id, "u1")
        completed = await store.get(operation.operation_id, "u1")

        assert completed is not None and completed.status == "succeeded"
        assert service.analyze_calls == 1
        assert completed.result["material_analysis"] == {
            "status": "completed",
            "target_count": 1,
        }
        assert completed.result["evidence_validation"] == {
            "status": "not_performed",
            "mode": "none",
        }
        assert completed.result["evidence_sufficiency"] == "unknown"
        assert "agent_result" not in completed.result
        assert completed.result["assessment_result"]["complete"] is False
        assert completed.result["agent_loop"] == {"enabled": False, "status": "disabled"}
        assert assessment_calls == 1

    asyncio.run(scenario())


def test_historical_agent_waiting_resumes_agent_only_after_explicit_continue(monkeypatch):
    import asyncio

    async def scenario():
        store = OperationStore()
        matrix = SkillMatrix(
            position_title="AI Engineer",
            experience_level="mid",
            skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
        )
        case = PreparationCase(
            case_id="case-historical-agent",
            user_id="u1",
            jd_text="需要 RAG",
            status="ready",
            skill_matrix=matrix,
            targets=build_assessment_targets(
                PreparationCase(case_id="case-historical-agent", user_id="u1", jd_text="需要 RAG"),
                matrix,
            ),
        )

        class FakeService:
            async def get(self, _case_id, _user_id):
                return case

            async def analyze(self, _case_id, _user_id):
                raise AssertionError("frozen material analysis should be reused")

        agent_calls = 0

        class FakeAgent:
            def __init__(self, _operations):
                pass

            async def run(self, _context):
                nonlocal agent_calls
                agent_calls += 1
                return {"status": "succeeded"}

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr("app.services.operation_store.save_operation", noop)
        monkeypatch.setattr("app.services.operation_store.load_operation", noop)
        monkeypatch.setattr(preparation_api, "get_operation_store", lambda: store)
        monkeypatch.setattr(preparation_api, "get_preparation_service", lambda: FakeService())
        monkeypatch.setattr(preparation_api, "JobAssessmentAgent", FakeAgent)
        monkeypatch.setattr(
            preparation_api.settings, "assessment_autonomous_loop_enabled", False
        )
        operation = await store.accept(
            user_id="u1",
            kind="prepare_case",
            case_id=case.case_id,
            input_revision=1,
            payload={"revision": 1},
            operation_id="historical-agent-op",
        )
        await store.transition(
            operation.operation_id,
            "waiting_input",
            result={"agent_result": {"status": "waiting_input"}},
        )
        assert agent_calls == 0

        await store.transition(operation.operation_id, "accepted")
        await preparation_api._run_analysis(operation.operation_id, case.case_id, "u1")
        completed = await store.get(operation.operation_id, "u1")

        assert completed is not None and completed.status == "succeeded"
        assert agent_calls == 1
        assert completed.result["agent_loop"] == {"enabled": True, "status": "executed"}
        assert completed.result["evidence_validation"]["status"] == "not_performed"

    asyncio.run(scenario())


def test_evidence_state_is_completed_only_for_persisted_assessments():
    import asyncio

    async def scenario():
        matrix = SkillMatrix(
            position_title="AI Engineer",
            experience_level="mid",
            skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
        )
        base = PreparationCase(case_id="case-evidence", user_id="u1", jd_text="需要 RAG")
        target = build_assessment_targets(base, matrix)[0]
        assessment = EvidenceAssessment(
            target_id=target.target_id,
            relevant_chunk_ids=["chunk-1"],
            supported_aspects=["RAG"],
            verdict="sufficient",
        )
        case = base.model_copy(
            update={
                "targets": [target],
                "evidence_assessments": {target.target_id: assessment},
            }
        )

        class FakeOperations:
            async def tool_calls(self, _operation_id, tool_name):
                assert tool_name == "assess_evidence"
                return [
                    SimpleNamespace(
                        status="succeeded",
                        result={
                            "assessment": assessment.model_dump(mode="json"),
                            "validation_mode": "deterministic",
                        },
                    )
                ]

        validation, sufficiency = await preparation_api._evidence_result(
            FakeOperations(), "evidence-op", case
        )

        assert validation == {
            "status": "completed",
            "mode": "deterministic",
            "validated_target_count": 1,
            "target_count": 1,
        }
        assert sufficiency == "sufficient"

    asyncio.run(scenario())
