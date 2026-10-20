"""Small additive SQL migration runner for Mockview business tables."""

from __future__ import annotations

from typing import Any

MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "20261013_01_preparation_operations",
        [
            """
            CREATE TABLE IF NOT EXISTS preparation_cases (
                case_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                material_revision INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                resume_text TEXT NOT NULL DEFAULT '',
                project_notes TEXT NOT NULL DEFAULT '',
                preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_preparation_cases_user_updated ON preparation_cases (user_id, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS assessment_targets (
                target_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES preparation_cases(case_id) ON DELETE CASCADE,
                material_revision INTEGER NOT NULL,
                requirement_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                citations JSONB NOT NULL DEFAULT '[]'::jsonb,
                verification_goal TEXT NOT NULL,
                constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
                status TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_assessment_targets_case_revision ON assessment_targets (case_id, material_revision)",
            """
            CREATE TABLE IF NOT EXISTS agent_operations (
                operation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                case_id TEXT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                input_revision INTEGER,
                graph_version TEXT NOT NULL,
                state_version TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                result JSONB,
                error TEXT,
                budget JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agent_operations_user_updated ON agent_operations (user_id, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS agent_tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL REFERENCES agent_operations(operation_id) ON DELETE CASCADE,
                tool_name TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                input_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL,
                result JSONB,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                budget_counted BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
        ],
    ),
    (
        "20261013_02_preparation_clarifications",
        [
            """
            CREATE TABLE IF NOT EXISTS preparation_clarifications (
                clarification_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES preparation_cases(case_id) ON DELETE CASCADE,
                material_revision INTEGER NOT NULL,
                question TEXT NOT NULL,
                missing_information TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_preparation_clarifications_case_revision ON preparation_clarifications (case_id, material_revision)",
        ],
    ),
    (
        "20261013_03_knowledge_releases",
        [
            """
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                document_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                owner_user_id TEXT,
                scope TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (document_id, version)
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_documents_hash_scope ON knowledge_documents (content_hash, scope, COALESCE(owner_user_id, ''))",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status ON knowledge_documents (status, updated_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS knowledge_releases (
                release_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                document_versions JSONB NOT NULL,
                chroma_collection TEXT NOT NULL,
                bm25_snapshot TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER,
                tokenization_version TEXT NOT NULL,
                chunking_version TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                activated_at TIMESTAMPTZ
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_release_pointer (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                release_id TEXT REFERENCES knowledge_releases(release_id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
        ],
    ),
    (
        "20261013_04_case_interview_links",
        [
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS parent_case_id TEXT",
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS evidence_chunks JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS evidence_assessments JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS preparation_case_id TEXT",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS assessment_targets JSONB NOT NULL DEFAULT '[]'::jsonb",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS evidence_chunks JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS evidence_assessments JSONB NOT NULL DEFAULT '{}'::jsonb",
        ],
    ),
    (
        "20261013_05_pending_answers",
        [
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS pending_answers JSONB NOT NULL DEFAULT '{}'::jsonb",
        ],
    ),
    (
        "20261013_06_preparation_resume_warnings",
        [
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS resume_parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb",
        ],
    ),
    (
        "20261013_07_interview_session_events",
        [
            """
            CREATE TABLE IF NOT EXISTS interview_session_event_sequences (
                session_id TEXT PRIMARY KEY REFERENCES interview_sessions(session_id) ON DELETE CASCADE,
                sequence BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS interview_session_events (
                session_id TEXT NOT NULL REFERENCES interview_sessions(session_id) ON DELETE CASCADE,
                sequence BIGINT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, sequence)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_interview_session_events_replay ON interview_session_events (session_id, sequence)",
        ],
    ),
    (
        "20261014_08_agent_operation_events",
        [
            """
            CREATE TABLE IF NOT EXISTS agent_operation_events (
                event_id TEXT PRIMARY KEY,
                event_order BIGSERIAL UNIQUE,
                operation_id TEXT NOT NULL REFERENCES agent_operations(operation_id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agent_operation_events_timeline ON agent_operation_events (operation_id, created_at, event_id)",
        ],
    ),
    (
        "20261014_09_case_knowledge_release",
        [
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS knowledge_release_id TEXT REFERENCES knowledge_releases(release_id)",
        ],
    ),
    (
        "20261015_10_progressive_question_generation",
        [
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS skill_matrix JSONB",
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS resume_profile JSONB",
            "ALTER TABLE preparation_cases ADD COLUMN IF NOT EXISTS analysis_strategy_version TEXT NOT NULL DEFAULT 'material-analysis-v1'",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS analysis_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS material_revision INTEGER",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS knowledge_release_id TEXT",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS operation_state JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE agent_operations ADD COLUMN IF NOT EXISTS error_details JSONB",
            """
            CREATE TABLE IF NOT EXISTS interview_question_slots (
                session_id TEXT NOT NULL REFERENCES interview_sessions(session_id) ON DELETE CASCADE,
                blueprint_id TEXT NOT NULL,
                slot_id TEXT NOT NULL,
                input_version TEXT NOT NULL,
                current_attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                question JSONB,
                error TEXT,
                terminated BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, blueprint_id, slot_id)
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_interview_question_slot_attempt ON interview_question_slots (session_id, blueprint_id, slot_id, current_attempt)",
        ],
    ),
    (
        "20261016_11_operation_metrics",
        [
            "ALTER TABLE agent_operations ADD COLUMN IF NOT EXISTS error_details JSONB",
            "ALTER TABLE agent_operations ADD COLUMN IF NOT EXISTS metrics JSONB NOT NULL DEFAULT '{}'::jsonb",
        ],
    ),
    (
        "20261016_12_session_lifecycle",
        [
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS completion_reason TEXT",
            "ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS paused_at TIMESTAMPTZ",
        ],
    ),
]


async def run_migrations(conn: Any) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for version, statements in MIGRATIONS:
        cursor = await conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
        )
        if await cursor.fetchone():
            continue
        async with conn.transaction():
            for statement in statements:
                await conn.execute(statement)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
