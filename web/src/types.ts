export type InterviewMode = "practice" | "professional";

export interface UserPublic {
  id: string;
  username: string;
  display_name: string;
  role: string;
  created_at?: string | null;
  last_login_at?: string | null;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: "bearer";
  user: UserPublic;
}

export interface StartInterviewResponse {
  session_id: string;
  message: string;
  websocket_url: string;
  mode: InterviewMode;
  resume?: ResumeParseResult;
}

export interface InterviewSessionSummary {
  session_id: string;
  preparation_case_id?: string | null;
  title: string;
  mode: InterviewMode;
  status: "pending" | "analyzing" | "interviewing" | "evaluating" | "pausing" | "paused" | "completed" | "failed";
  graph_started: boolean;
  has_report: boolean;
  question_count: number;
  assessment_count: number;
  current_question_index: number;
  max_follow_ups: number;
  jd_preview: string;
  overall_score?: number | null;
  grade?: "A" | "B" | "C" | "D" | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  completion_reason?: "normal" | "early" | "no_assessment" | null;
  pending_action?: "answer_retry_required" | null;
  paused_at?: string | null;
}

export interface ResumeParseResult {
  normalized_text?: string;
  candidate_name?: string;
  contact?: Record<string, unknown>;
  skills?: string[];
  projects?: unknown[];
  education?: unknown[];
  work_experience?: unknown[];
  [key: string]: unknown;
}

export interface SkillScore {
  skill_name: string;
  score: number;
  evidence: string;
}

export interface ReportHighlight {
  point: string;
  evidence: string;
}

export interface RoundScore {
  round_name: string;
  score: number;
  grade: "A" | "B" | "C" | "D";
  summary: string;
}

export interface MissedKnowledge {
  question: string;
  question_id?: string | null;
  score: number | null;
  scorable?: boolean;
  missed_points: string[];
  reference_answer: string;
}

export interface InterviewReport {
  skill_scores?: SkillScore[];
  overall_score?: number;
  grade?: "A" | "B" | "C" | "D";
  strengths?: ReportHighlight[];
  improvements?: ReportHighlight[];
  overall_assessment?: string;
  round_scores?: RoundScore[];
  technical_depth_score?: number;
  technical_breadth_score?: number;
  hiring_recommendation?: string;
  total_questions?: number;
  missed_knowledge?: MissedKnowledge[];
  study_suggestions?: string[];
  coverage?: {
    covered_skills?: string[];
    uncovered_skills?: string[];
    scored_questions?: number;
    unscored_questions?: number;
  };
}

export interface TranscriptItem {
  message_id: string;
  role: "interviewer" | "candidate" | "system";
  content: string;
  question_id?: string | null;
  turn_id?: string | null;
  kind?: string | null;
  answer_id?: string | null;
  created_at?: string | null;
  processing_status?: "accepted" | "succeeded" | "failed_retryable" | null;
}

export interface PublicTranscript {
  session: InterviewSessionSummary;
  items: TranscriptItem[];
  question_reviews: Array<{
    question_id: string;
    display_id: number;
    primary_skill: string;
    score?: number | null;
    scorable: boolean;
  }>;
  has_report: boolean;
  report?: InterviewReport | null;
}

export interface AssessmentTarget {
  target_id: string;
  scope: string;
  importance: number;
  verification_goal: string;
  sources: string[];
  status: "pending" | "ready" | "partial" | "missing_evidence";
}

export interface ClarificationRequest {
  clarification_id: string;
  question: string;
  missing_information: string;
  status: "pending" | "answered" | "cancelled";
  answer?: string | null;
}

export interface PreparationCase {
  case_id: string;
  parent_case_id?: string | null;
  material_revision: number;
  status: "draft" | "analyzing" | "waiting_input" | "ready" | "failed" | "cancelled";
  jd_text: string;
  resume_text: string;
  resume_parse_warnings: string[];
  project_notes: string;
  targets: AssessmentTarget[];
  clarifications: ClarificationRequest[];
}

export interface OperationRecord {
  operation_id: string;
  status: "accepted" | "running" | "waiting_input" | "succeeded" | "failed_retryable" | "failed" | "cancelled";
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface RepracticeOptions {
  source_session_id: string;
  case_id: string;
  material_revision: number;
  weak_skills: Array<{ skill_name: string; score: number }>;
}

export interface KnowledgeDocument {
  document_id: string;
  title: string;
  source_url?: string | null;
  source_type: string;
  status: "candidate" | "approved" | "rejected" | "withdrawn";
  content: string;
}

export interface KnowledgeRelease {
  release_id: string;
  status: "building" | "ready" | "active" | "failed" | "retired";
  document_versions: Record<string, number>;
  error?: string | null;
}

export interface MemoryObservation {
  id: string;
  content: string;
  tags: string[];
  source: string;
  source_id?: string | null;
  structured: Record<string, unknown>;
  created_at: string;
}

export type ServerMessage =
  | {
      type: "status";
      stage: string;
      message: string;
      sequence?: number | null;
    }
  | {
      type: "question";
      question_index: number;
      total_questions: number;
      content: string;
      skill_tags: string[];
      difficulty: string;
      question_id: string;
      turn_id: string;
    }
  | {
      type: "follow_up";
      question_index: number;
      follow_up_number: number;
      content: string;
      question_id: string;
      turn_id: string;
    }
  | {
      type: "interview_end";
      message: string;
    }
  | {
      type: "report";
      data: InterviewReport;
    }
  | {
      type: "error";
      message: string;
      code: string;
      retryable: boolean;
      retry_after_seconds?: number | null;
    }
  | {
      type: "transcript_snapshot";
      data: PublicTranscript;
    }
  | {
      type: "stream_chunk";
      event: "start" | "delta" | "end";
      stream_id: string;
      kind: "question" | "follow_up";
      chunk: string;
      content: string;
      done: boolean;
      question_index: number;
      total_questions: number;
      follow_up_number?: number | null;
      skill_tags: string[];
      difficulty: string;
      sequence?: number | null;
      question_id: string;
      turn_id: string;
    };

export interface ChatItem {
  id: string;
  turnId?: string;
  streamId?: string;
  role: "interviewer" | "candidate" | "system";
  content: string;
  meta?: string;
  streaming?: boolean;
}

export interface CurrentTurn {
  questionIndex?: number;
  totalQuestions?: number;
  skillTags?: string[];
  difficulty?: string;
  followUpNumber?: number;
  questionId?: string;
  turnId?: string;
}
