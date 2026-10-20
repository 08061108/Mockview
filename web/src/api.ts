import type {
  AuthTokenResponse,
  InterviewMode,
  InterviewReport,
  InterviewSessionSummary,
  StartInterviewResponse,
  UserPublic,
  PreparationCase,
  OperationRecord,
  KnowledgeDocument,
  KnowledgeRelease,
  MemoryObservation,
  PublicTranscript,
} from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000/api";

export const HEALTH_URL = API_BASE_URL.replace(/\/api$/, "/health");

function absoluteApiBase(): string {
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    return API_BASE_URL;
  }
  return new URL(API_BASE_URL, window.location.origin).toString().replace(/\/$/, "");
}

export function websocketUrl(sessionId: string, afterSequence = 0): string {
  const wsBase = absoluteApiBase().replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  const params = new URLSearchParams();
  if (afterSequence > 0) params.set("after_sequence", String(afterSequence));
  const suffix = params.size ? `?${params.toString()}` : "";
  return `${wsBase}/ws/interview/${sessionId}${suffix}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.message ?? response.statusText;
    if (detail && typeof detail === "object" && "message" in detail) {
      const value = detail as { code?: string; message: string; retryable?: boolean };
      throw new ApiError(value.code ?? "INTERNAL_ERROR", value.message, value.retryable ?? false);
    }
    throw new Error(formatApiError(detail));
  }
  return payload as T;
}

function formatApiError(detail: unknown): string {
  if (typeof detail !== "string") {
    return JSON.stringify(detail);
  }
  const map: Record<string, string> = {
    "Authentication required.": "请先登录后再继续操作。",
    "Invalid authentication token.": "登录状态无效，请重新登录。",
    "Authentication token has expired.": "登录已过期，请重新登录。",
    "User not found or inactive.": "账号不存在或已被停用。",
    "Invalid username or password.": "用户名或密码错误。",
    "Username already exists.": "用户名已存在。",
    "Session access denied": "当前账号无权访问这个面试会话。",
  };
  return map[detail] ?? detail;
}

function authHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
  };
}

export async function checkHealth(): Promise<{ status: string; model?: string; debug?: boolean }> {
  const response = await fetch(HEALTH_URL);
  return parseJson(response);
}

export async function register(input: {
  username: string;
  password: string;
  displayName: string;
}): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: input.username,
      password: input.password,
      display_name: input.displayName,
    }),
  });
  return parseJson(response);
}

export async function login(input: {
  username: string;
  password: string;
}): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: input.username,
      password: input.password,
    }),
  });
  return parseJson(response);
}

export async function getMe(token: string): Promise<UserPublic> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function listInterviewSessions(token: string): Promise<InterviewSessionSummary[]> {
  const response = await fetch(`${API_BASE_URL}/interview/sessions`, {
    headers: authHeaders(token),
  });
  const payload = await parseJson<{ items: InterviewSessionSummary[] }>(response);
  return payload.items;
}

export async function getInterviewReport(token: string, sessionId: string): Promise<InterviewReport> {
  const response = await fetch(`${API_BASE_URL}/interview/report/${sessionId}`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function exportInterviewReport(
  token: string,
  sessionId: string,
  format: "markdown" | "pdf" | "html",
): Promise<Blob> {
  const response = await fetch(
    `${API_BASE_URL}/interview/report/${sessionId}/export?format=${encodeURIComponent(format)}`,
    {
      headers: authHeaders(token),
    },
  );
  if (!response.ok) {
    await parseJson(response);
  }
  return response.blob();
}

export async function startInterview(input: {
  jdText: string;
  maxFollowUps: number;
  mode: InterviewMode;
  token: string;
  resumeFile?: File | null;
}): Promise<StartInterviewResponse> {
  if (input.resumeFile) {
    const form = new FormData();
    form.append("jd_text", input.jdText);
    form.append("resume_file", input.resumeFile);
    form.append("max_follow_ups", String(input.maxFollowUps));
    form.append("mode", input.mode);
    const response = await fetch(`${API_BASE_URL}/interview/start-with-resume`, {
      method: "POST",
      headers: authHeaders(input.token),
      body: form,
    });
    return parseJson(response);
  }

  const response = await fetch(`${API_BASE_URL}/interview/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(input.token),
    },
    body: JSON.stringify({
      jd_text: input.jdText,
      max_follow_ups: input.maxFollowUps,
      mode: input.mode,
    }),
  });
  return parseJson(response);
}

export function authenticateWebsocket(socket: WebSocket, token: string): void {
  socket.send(JSON.stringify({ type: "auth", token }));
}

export class ApiError extends Error {
  constructor(public code: string, message: string, public retryable: boolean) {
    super(message);
  }
}

export async function createPreparationCase(
  token: string,
  input: { jdText: string; resumeText?: string; projectNotes?: string },
): Promise<PreparationCase> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      jd_text: input.jdText,
      resume_text: input.resumeText ?? "",
      project_notes: input.projectNotes ?? "",
    }),
  });
  return parseJson(response);
}

export async function getInterviewTranscript(token: string, sessionId: string): Promise<PublicTranscript> {
  const response = await fetch(`${API_BASE_URL}/interview/session/${sessionId}/transcript`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

async function postSessionAction(token: string, sessionId: string, action: "pause" | "resume" | "end") {
  const response = await fetch(`${API_BASE_URL}/interview/session/${sessionId}/${action}`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export const pauseInterview = (token: string, sessionId: string) => postSessionAction(token, sessionId, "pause");
export const resumeInterview = (token: string, sessionId: string) => postSessionAction(token, sessionId, "resume");
export const endInterview = (token: string, sessionId: string) => postSessionAction(token, sessionId, "end");

export async function uploadPreparationResume(
  token: string,
  caseId: string,
  resumeFile: File,
): Promise<PreparationCase> {
  const form = new FormData();
  form.append("resume_file", resumeFile);
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}/resume`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
  return parseJson(response);
}

export async function startPreparationAnalysis(
  token: string,
  caseId: string,
): Promise<{ operation_id: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}/analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({}),
  });
  return parseJson(response);
}

export async function getPreparationCase(token: string, caseId: string): Promise<PreparationCase> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function getPreparationProgress(token: string, caseId: string): Promise<{ case: PreparationCase; operation: OperationRecord | null }> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}/progress`, { headers: authHeaders(token) });
  return parseJson(response);
}

export async function getRepracticeOptions(token: string, sessionId: string): Promise<import("./types").RepracticeOptions> {
  const response = await fetch(`${API_BASE_URL}/preparation/sessions/${sessionId}/repractice-options`, { headers: authHeaders(token) });
  return parseJson(response);
}

export async function getPreparationOperation(
  token: string,
  operationId: string,
): Promise<OperationRecord> {
  const response = await fetch(`${API_BASE_URL}/preparation/operations/${operationId}`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function continuePreparationOperation(
  token: string,
  operationId: string,
): Promise<OperationRecord> {
  const response = await fetch(`${API_BASE_URL}/preparation/operations/${operationId}/continue`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function cancelPreparationOperation(
  token: string,
  operationId: string,
): Promise<OperationRecord> {
  const response = await fetch(`${API_BASE_URL}/preparation/operations/${operationId}/cancel`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function answerPreparationClarification(
  token: string,
  caseId: string,
  clarificationId: string,
  answer: string,
): Promise<PreparationCase> {
  const response = await fetch(
    `${API_BASE_URL}/preparation/cases/${caseId}/clarifications/${clarificationId}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify({ answer }),
    },
  );
  return parseJson(response);
}

export async function createInterviewFromPreparation(
  token: string,
  caseId: string,
  input: { mode: InterviewMode; maxFollowUps: number },
): Promise<StartInterviewResponse> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}/interviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ mode: input.mode, max_follow_ups: input.maxFollowUps,
      request_id: `prepared:${input.mode}:${input.maxFollowUps}` }),
  });
  return parseJson(response);
}

export async function createRepracticeCase(
  token: string,
  caseId: string,
  selectedSkills: string[],
  sourceSessionId: string,
): Promise<PreparationCase> {
  const response = await fetch(`${API_BASE_URL}/preparation/cases/${caseId}/repractice`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ source_session_id: sourceSessionId, selected_skills: selectedSkills }),
  });
  return parseJson(response);
}

export async function listMemoryObservations(token: string): Promise<MemoryObservation[]> {
  const response = await fetch(`${API_BASE_URL}/memory/observations`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function disputeMemoryObservation(
  token: string,
  observationId: string,
): Promise<MemoryObservation> {
  const response = await fetch(`${API_BASE_URL}/memory/observations/${observationId}/dispute`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function correctMemoryFact(
  token: string,
  content: string,
  replacesMemoryId?: string,
): Promise<MemoryObservation> {
  const response = await fetch(`${API_BASE_URL}/memory/facts`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ content, replaces_memory_id: replacesMemoryId ?? null }),
  });
  return parseJson(response);
}

export async function listKnowledgeCandidates(token: string): Promise<KnowledgeDocument[]> {
  const response = await fetch(`${API_BASE_URL}/knowledge/candidates`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function reviewKnowledgeCandidate(
  token: string,
  documentId: string,
  status: "approved" | "rejected" | "withdrawn",
): Promise<KnowledgeDocument> {
  const response = await fetch(`${API_BASE_URL}/knowledge/candidates/${documentId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ status }),
  });
  return parseJson(response);
}

export async function publishKnowledgeRelease(token: string): Promise<KnowledgeRelease> {
  const response = await fetch(`${API_BASE_URL}/knowledge/releases`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseJson(response);
}
