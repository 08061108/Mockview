import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Bot,
  CheckCircle2,
  CircleStop,
  ClipboardList,
  FileText,
  History,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquareText,
  Play,
  Pause,
  RotateCcw,
  Send,
  Sparkles,
  Upload,
  User,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, RefObject } from "react";
import {
  API_BASE_URL,
  checkHealth,
  exportInterviewReport,
  getInterviewReport,
  getInterviewTranscript,
  pauseInterview,
  resumeInterview,
  endInterview,
  getMe,
  listInterviewSessions,
  login,
  register,
  startInterview,
  createPreparationCase,
  uploadPreparationResume,
  startPreparationAnalysis,
  getPreparationProgress,
  getRepracticeOptions,
  continuePreparationOperation,
  cancelPreparationOperation,
  answerPreparationClarification,
  createInterviewFromPreparation,
  createRepracticeCase,
  listMemoryObservations,
  disputeMemoryObservation,
  correctMemoryFact,
  listKnowledgeCandidates,
  reviewKnowledgeCandidate,
  publishKnowledgeRelease,
  websocketUrl,
  authenticateWebsocket,
} from "./api";
import type {
  ChatItem,
  CurrentTurn,
  InterviewMode,
  InterviewReport,
  InterviewSessionSummary,
  ResumeParseResult,
  ServerMessage,
  UserPublic,
  PreparationCase,
  OperationRecord,
  KnowledgeDocument,
  KnowledgeRelease,
  MemoryObservation,
  PublicTranscript,
  RepracticeOptions,
} from "./types";

const sampleJd = `岗位：AI 应用开发工程师

岗位职责：
1. 参与 AI Agent、RAG、智能问答和工作流编排等大模型应用的设计与开发；
2. 使用 Python / FastAPI 构建后端服务，支持流式对话、文件上传、异步任务和 WebSocket 实时交互；
3. 负责向量检索、Prompt Engineering、结构化输出、模型调用稳定性和效果评估；
4. 与产品和前端协作，将原型能力落地为可演示、可部署的 AI 应用。

岗位要求：
- 熟悉 Python、FastAPI、LangChain/LangGraph 或同类 Agent 框架；
- 理解 RAG 的切分、召回、重排、评估和工程化优化；
- 熟悉 REST API、WebSocket、数据库和基础部署流程；
- 有 AI 应用项目经验，能解释架构设计、异常处理和效果评估方法。`;

type View = "setup" | "preparation" | "observations" | "knowledge" | "history" | "interview" | "record" | "report";
type ConnectionState = "idle" | "connecting" | "open" | "closed" | "error";
type AuthMode = "login" | "register";

const uid = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const AUTH_TOKEN_KEY = "ai-mock-auth-token";
const RECONNECT_DELAYS_MS = [1000, 2000, 4000] as const;
const RECONNECT_TOTAL_LIMIT_MS = 30_000;
const STABLE_CONNECTION_MS = 15_000;

function streamMessageMeta(payload: Extract<ServerMessage, { type: "stream_chunk" }>) {
  return payload.kind === "follow_up"
    ? `Follow-up for question ${payload.question_index} #${payload.follow_up_number ?? 1}`
    : `Question ${payload.question_index}/${payload.total_questions} · ${difficultyName(
        payload.difficulty,
      )}`;
}

function normalizeMessageContent(content: string) {
  return content.trim().replace(/\s+/g, " ");
}

function appendInterviewerMessage(items: ChatItem[], nextItem: ChatItem) {
  if (nextItem.turnId) {
    const index = items.findIndex((item) => item.turnId === nextItem.turnId);
    if (index >= 0) {
      return items.map((item, current) =>
        current === index ? { ...item, ...nextItem, id: item.id } : item,
      );
    }
  }
  const previous = items[items.length - 1];
  if (previous?.role !== "interviewer") return [...items, nextItem];

  const previousContent = normalizeMessageContent(previous.content);
  const currentContent = normalizeMessageContent(nextItem.content);
  const sameContent = previousContent === currentContent;
  const completesStream =
    previous.streaming && currentContent.length > 0 && currentContent.startsWith(previousContent);
  if (!sameContent && !completesStream) return [...items, nextItem];

  return [...items.slice(0, -1), { ...nextItem, id: previous.id }];
}

const modeName = (mode: InterviewMode) =>
  mode === "professional" ? "专业面试模式" : "练习模式";

const transcriptMessages = (data: PublicTranscript): ChatItem[] =>
  data.items.map((item) => {
    const review = item.question_id
      ? data.question_reviews.find((value) => value.question_id === item.question_id)
      : undefined;
    return {
      id: item.message_id,
      turnId: item.turn_id ?? undefined,
      role: item.role,
      content: item.content,
      meta:
        item.processing_status === "failed_retryable"
          ? "回答待显式重试"
          : item.role === "interviewer" && review
            ? `${review.primary_skill} · ${review.scorable && review.score ? `最终评分 ${review.score}/10` : "未评分"}`
            : undefined,
    };
  });

const difficultyName = (difficulty?: string) => {
  const map: Record<string, string> = {
    easy: "基础",
    medium: "中等",
    hard: "进阶",
  };
  return difficulty ? (map[difficulty] ?? difficulty) : "";
};

const statusText = (stage: string, fallback: string) => {
  const map: Record<string, string> = {
    analyzing_jd: "正在分析 JD、简历和历史记忆，准备个性化问题。",
    analyzing_resume: "正在分析简历中的项目、技能和经历。",
    resume_analyzed: "简历分析完成。",
    preparing_profile: "正在并行分析 JD 和简历。",
    profile_ready: "JD 和简历分析完成，开始准备个性化题目。",
    jd_analyzed: "JD 分析完成。",
    retrieving_questions: "正在从题库检索相关参考问题。",
    retrieval_complete: "题库检索完成。",
    planning_questions: "正在规划个性化面试问题。",
    planning_blueprint: "正在冻结本轮考察目标和题目槽位。",
    blueprint_ready: "考察蓝图已就绪，正在生成首题。",
    generating_question: "正在生成并校验当前题目。",
    question_ready: "当前题目已通过校验。",
    question_failed: "当前题目生成失败，请选择重试或阶段性结束。",
    questions_planned: "个性化问题规划完成。",
    questions_ready: "问题已准备好，面试即将开始。",
    resumed: "已恢复到当前面试进度。",
    waiting: "正在等待下一轮问题。",
    processing: "正在处理面试流程，请稍候。",
    preparing_interview: "正在准备材料分析、考察蓝图和首题。",
    evaluating_answer: "正在评估回答并准备下一步。",
    retrying: "正在重试失败的面试步骤。",
    evaluating: "正在结束面试并生成评估报告。",
    pausing: "暂停请求已接受，正在完成当前操作。",
    paused: "面试已暂停。",
    pong: "连接正常。",
  };
  return map[stage] ?? fallback;
};

export function App() {
  const [view, setView] = useState<View>("setup");
  const [jdText, setJdText] = useState(sampleJd);
  const [mode, setMode] = useState<InterviewMode>("professional");
  const [maxFollowUps, setMaxFollowUps] = useState(2);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(AUTH_TOKEN_KEY) ?? "");
  const [currentUser, setCurrentUser] = useState<UserPublic | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [authError, setAuthError] = useState("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [projectNotes, setProjectNotes] = useState("");
  const [preparationCase, setPreparationCase] = useState<PreparationCase | null>(null);
  const [preparationOperation, setPreparationOperation] = useState<OperationRecord | null>(null);
  const [activePreparationId, setActivePreparationId] = useState("");
  const [preparationRefresh, setPreparationRefresh] = useState(0);
  const [repracticeOptions, setRepracticeOptions] = useState<RepracticeOptions | null>(null);
  const [repracticeNotice, setRepracticeNotice] = useState("");
  const [isPreparing, setIsPreparing] = useState(false);
  const [knowledgeCandidates, setKnowledgeCandidates] = useState<KnowledgeDocument[]>([]);
  const [knowledgeRelease, setKnowledgeRelease] = useState<KnowledgeRelease | null>(null);
  const [observations, setObservations] = useState<MemoryObservation[]>([]);
  const [resume, setResume] = useState<ResumeParseResult | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [historyItems, setHistoryItems] = useState<InterviewSessionSummary[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [currentTurn, setCurrentTurn] = useState<CurrentTurn | null>(null);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("填写 JD，可选上传简历后开始模拟面试。");
  const [stage, setStage] = useState("ready");
  const [error, setError] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isAwaitingAnswer, setIsAwaitingAnswer] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [report, setReport] = useState<InterviewReport | null>(null);
  const [health, setHealth] = useState<{ status: string; model?: string; debug?: boolean } | null>(
    null,
  );

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectEnabledRef = useRef(false);
  const reconnectStartedAtRef = useRef<number | null>(null);
  const stableConnectionTimerRef = useRef<number | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const lastEventSequenceRef = useRef(0);
  const pendingAnswerRef = useRef<{
    answerId: string;
    turnId: string;
    content: string;
    socket: WebSocket | null;
  } | null>(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    let active = true;
    if (!authToken) {
      setCurrentUser(null);
      setIsCheckingAuth(false);
      return;
    }

    setIsCheckingAuth(true);
    getMe(authToken)
      .then((user) => {
        if (active) {
          setCurrentUser(user);
        }
      })
      .catch(() => {
        if (active) {
          localStorage.removeItem(AUTH_TOKEN_KEY);
          setAuthToken("");
          setCurrentUser(null);
        }
      })
      .finally(() => {
        if (active) {
          setIsCheckingAuth(false);
        }
      });

    return () => {
      active = false;
    };
  }, [authToken]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, status]);

  useEffect(() => {
    return () => closeSocket();
  }, []);

  const canStart = Boolean(currentUser && authToken && jdText.trim().length >= 10 && !isStarting);
  const canPrepare = Boolean(currentUser && authToken && jdText.trim().length >= 10 && !isPreparing);

  function selectPreparation(next: PreparationCase) {
    if (!currentUser) return;
    localStorage.setItem(`mockview:preparation:${currentUser.id}`, JSON.stringify({caseId: next.case_id, mode, maxFollowUps}));
    setActivePreparationId(next.case_id);
    setPreparationCase(next);
    setPreparationOperation(null);
    setPreparationRefresh((value) => value + 1);
    setView("preparation");
  }

  useEffect(() => {
    setPreparationCase(null);
    setPreparationOperation(null);
    const saved = currentUser ? localStorage.getItem(`mockview:preparation:${currentUser.id}`) ?? "" : "";
    let caseId = saved;
    if (saved.startsWith("{")) {
      try {
        const pointer = JSON.parse(saved);
        caseId = typeof pointer.caseId === "string" ? pointer.caseId : "";
        if (["practice", "professional"].includes(pointer.mode)) setMode(pointer.mode);
        if (Number.isInteger(pointer.maxFollowUps) && pointer.maxFollowUps >= 0 && pointer.maxFollowUps <= 5) setMaxFollowUps(pointer.maxFollowUps);
      } catch { caseId = ""; }
    }
    setActivePreparationId(caseId);
    if (caseId) setView("preparation");
  }, [currentUser?.id]);

  useEffect(() => {
    if (!activePreparationId || !authToken || !currentUser) return;
    let active = true;
    let timer: number | undefined;
    async function refresh() {
      try {
        const progress = await getPreparationProgress(authToken, activePreparationId);
        if (!active) return;
        setPreparationCase(progress.case);
        setPreparationOperation(progress.operation);
        if (progress.operation && ["accepted", "running"].includes(progress.operation.status)) {
          timer = window.setTimeout(refresh, 2000);
        }
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "恢复准备任务失败，请刷新重试。");
      }
    }
    void refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, [activePreparationId, authToken, currentUser?.id, preparationRefresh]);

  useEffect(() => {
    let active = true;
    setRepracticeOptions(null);
    setRepracticeNotice("");
    if (report && sessionId && authToken) {
      getRepracticeOptions(authToken, sessionId).then((options) => {
        if (active) {
          setRepracticeOptions(options);
          if (!options.weak_skills.length) setRepracticeNotice("本次没有可用于复练的有效薄弱技能（最终技能分低于 8 分）。");
        }
      }).catch((err) => {
        if (active) setRepracticeNotice(err instanceof Error ? err.message : "无法加载复练来源。");
      });
    }
    return () => { active = false; };
  }, [report, sessionId, authToken]);

  function refreshPreparation() {
    setPreparationRefresh((value) => value + 1);
  }

  async function handlePrepare() {
    if (!canPrepare) return;
    setIsPreparing(true);
    setError("");
    try {
      const nextCase = await createPreparationCase(authToken, {
        jdText: jdText.trim(),
        projectNotes: projectNotes.trim(),
      });
      selectPreparation(nextCase);
      const materialCase = resumeFile
        ? await uploadPreparationResume(authToken, nextCase.case_id, resumeFile)
        : nextCase;
      setPreparationCase(materialCase);
      await startPreparationAnalysis(authToken, materialCase.case_id);
      refreshPreparation();
    } catch (err) {
      setError(err instanceof Error ? err.message : "准备任务创建失败。");
    } finally {
      setIsPreparing(false);
    }
  }

  async function handleStartPrepared() {
    if (!preparationCase || isStarting) return;
    setIsStarting(true);
    setError("");
    try {
      if (currentUser) localStorage.setItem(`mockview:preparation:${currentUser.id}`,
        JSON.stringify({caseId: preparationCase.case_id, mode, maxFollowUps}));
      const response = await createInterviewFromPreparation(authToken, preparationCase.case_id, {
        mode,
        maxFollowUps,
      });
      lastEventSequenceRef.current = 0;
      pendingAnswerRef.current = null;
      setReport(null);
      setCurrentTurn(null);
      setAnswer("");
      setIsAwaitingAnswer(false);
      setMode(response.mode);
      setSessionId(response.session_id);
      setMessages([{ id: uid(), role: "system", content: "已按考察目标创建面试，正在准备第一题。" }]);
      setView("interview");
      connectSocket(response.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建目标面试失败。");
    } finally {
      setIsStarting(false);
    }
  }

  async function handleClarification(clarificationId: string, answer: string) {
    if (!preparationCase || !answer.trim()) return;
    try {
    const updatedCase = await answerPreparationClarification(
      authToken,
      preparationCase.case_id,
      clarificationId,
      answer.trim(),
    );
    setPreparationCase(updatedCase);
    const allAnswered = updatedCase.clarifications.every(
      (item) => item.status !== "pending",
    );
    if (allAnswered && preparationOperation?.status === "waiting_input") {
      const continued = await continuePreparationOperation(
        authToken,
        preparationOperation.operation_id,
      );
      setPreparationOperation(continued);
      refreshPreparation();
    }
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交补充失败，请重试。");
    }
  }

  async function handleContinuePreparation() {
    if (!preparationCase) return;
    try {
      if (preparationOperation) await continuePreparationOperation(authToken, preparationOperation.operation_id);
      else await startPreparationAnalysis(authToken, preparationCase.case_id);
      refreshPreparation();
    } catch (err) { setError(err instanceof Error ? err.message : "继续分析失败。"); }
  }

  async function handleCancelPreparation() {
    if (!preparationOperation) return;
    try {
      await cancelPreparationOperation(authToken, preparationOperation.operation_id);
      refreshPreparation();
    } catch (err) { setError(err instanceof Error ? err.message : "取消失败。"); }
  }

  async function openObservationsView() {
    setView("observations");
    setError("");
    try {
      setObservations(await listMemoryObservations(authToken));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载能力观察失败。");
    }
  }

  async function handleDisputeObservation(observationId: string) {
    await disputeMemoryObservation(authToken, observationId);
    setObservations(await listMemoryObservations(authToken));
  }

  async function handleCorrectObservation(observationId: string, content: string) {
    await correctMemoryFact(authToken, content, observationId);
    setObservations(await listMemoryObservations(authToken));
  }

  async function handleRepractice(selectedSkills: string[]) {
    if (!repracticeOptions || !selectedSkills.length || isPreparing) return;
    setIsPreparing(true);
    setError("");
    try {
      const next = await createRepracticeCase(authToken, repracticeOptions.case_id,
        selectedSkills, repracticeOptions.source_session_id);
      selectPreparation(next);
      await startPreparationAnalysis(authToken, next.case_id);
      refreshPreparation();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建复练任务失败。");
    } finally { setIsPreparing(false); }
  }

  async function openKnowledgeView() {
    setView("knowledge");
    setError("");
    try {
      setKnowledgeCandidates(await listKnowledgeCandidates(authToken));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载知识候选失败。");
    }
  }

  async function handleKnowledgeReview(documentId: string, nextStatus: "approved" | "rejected") {
    await reviewKnowledgeCandidate(authToken, documentId, nextStatus);
    setKnowledgeCandidates(await listKnowledgeCandidates(authToken));
  }

  async function handleKnowledgePublish() {
    setKnowledgeRelease(await publishKnowledgeRelease(authToken));
  }

  async function loadHistory() {
    if (!authToken) return;
    setIsHistoryLoading(true);
    setHistoryError("");
    try {
      setHistoryItems(await listInterviewSessions(authToken));
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "加载历史记录失败。");
    } finally {
      setIsHistoryLoading(false);
    }
  }

  async function handleAuthSubmit(event: FormEvent) {
    event.preventDefault();
    setAuthError("");
    setIsAuthenticating(true);
    try {
      const response =
        authMode === "login"
          ? await login({ username: authUsername.trim(), password: authPassword })
          : await register({
              username: authUsername.trim(),
              password: authPassword,
              displayName: displayName.trim(),
            });
      localStorage.setItem(AUTH_TOKEN_KEY, response.access_token);
      setAuthToken(response.access_token);
      setCurrentUser(response.user);
      setAuthPassword("");
      setView("setup");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "认证失败，请稍后重试。");
    } finally {
      setIsAuthenticating(false);
    }
  }

  function clearReconnectTimer() {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }

  function clearStableConnectionTimer() {
    if (stableConnectionTimerRef.current !== null) {
      window.clearTimeout(stableConnectionTimerRef.current);
      stableConnectionTimerRef.current = null;
    }
  }

  function markBusinessProgress() {
    reconnectAttemptsRef.current = 0;
    reconnectStartedAtRef.current = null;
    clearStableConnectionTimer();
  }

  function disableAutoReconnect() {
    reconnectEnabledRef.current = false;
    reconnectAttemptsRef.current = 0;
    reconnectStartedAtRef.current = null;
    clearReconnectTimer();
    clearStableConnectionTimer();
  }

  function closeSocket() {
    disableAutoReconnect();
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close(1000, "client navigation");
    }
  }

  function handleLogout() {
    closeSocket();
    localStorage.removeItem(AUTH_TOKEN_KEY);
    setAuthToken("");
    setCurrentUser(null);
    setAuthPassword("");
    setAuthError("");
    resetAll();
  }

  async function handleStart() {
    if (!canStart) return;
    setIsStarting(true);
    setError("");
    setStatus(resumeFile ? "正在上传并解析简历，请稍候。" : "正在创建面试会话。");
    try {
      const response = await startInterview({
        jdText: jdText.trim(),
        maxFollowUps,
        mode,
        token: authToken,
        resumeFile,
      });
      lastEventSequenceRef.current = 0;
      pendingAnswerRef.current = null;
      setSessionId(response.session_id);
      lastEventSequenceRef.current = 0;
      setResume(response.resume ?? null);
      setMessages([
        {
          id: uid(),
          role: "system",
          content:
            response.mode === "professional"
              ? "专业面试会话已创建，正在准备第一题。"
              : "练习面试会话已创建，正在准备第一题。",
        },
      ]);
      setView("interview");
      connectSocket(response.session_id);
      void loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建面试失败");
      setStatus("创建会话失败，请检查登录状态、后端服务或模型配置。");
    } finally {
      setIsStarting(false);
    }
  }

  function scheduleReconnect(id: string) {
    if (
      !reconnectEnabledRef.current ||
      reconnectTimerRef.current !== null ||
      reconnectAttemptsRef.current >= RECONNECT_DELAYS_MS.length ||
      (reconnectStartedAtRef.current !== null &&
        Date.now() - reconnectStartedAtRef.current >= RECONNECT_TOTAL_LIMIT_MS)
    ) {
      if (
        reconnectEnabledRef.current &&
        (reconnectAttemptsRef.current >= RECONNECT_DELAYS_MS.length ||
          (reconnectStartedAtRef.current !== null &&
            Date.now() - reconnectStartedAtRef.current >= RECONNECT_TOTAL_LIMIT_MS))
      ) {
        setConnection("error");
        setStatus("自动重连失败，请点击“重连”继续当前面试。");
      }
      return;
    }

    const attempt = reconnectAttemptsRef.current;
    if (reconnectStartedAtRef.current === null) {
      reconnectStartedAtRef.current = Date.now();
    }
    const delay = RECONNECT_DELAYS_MS[attempt];
    reconnectAttemptsRef.current += 1;

    setConnection("connecting");
    setStatus(`连接中断，将在 ${delay / 1000}s 后自动重连（第 ${attempt + 1} 次）。`);

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connectSocket(id, { resetAttempts: false });
    }, delay);
  }

  function connectSocket(
    id = sessionId,
    options: { resetAttempts?: boolean } = { resetAttempts: true },
  ) {
    if (!id || !authToken) return;

    if (options.resetAttempts) {
      reconnectAttemptsRef.current = 0;
      reconnectStartedAtRef.current = null;
    }

    reconnectEnabledRef.current = true;
    clearReconnectTimer();

    const previousSocket = socketRef.current;
    if (previousSocket && previousSocket.readyState < WebSocket.CLOSING) {
      previousSocket.close(1000, "reconnecting");
    }
    setConnection("connecting");
    setStatus("正在连接实时面试通道。");
    const ws = new WebSocket(websocketUrl(id, lastEventSequenceRef.current));
    socketRef.current = ws;

    ws.onopen = () => {
      if (socketRef.current !== ws) return;
      authenticateWebsocket(ws, authToken);
      setError("");
      setConnection("open");
      setStatus("连接成功，面试官正在准备第一轮问题。");
      clearStableConnectionTimer();
      stableConnectionTimerRef.current = window.setTimeout(() => {
        if (socketRef.current === ws && ws.readyState === WebSocket.OPEN) {
          markBusinessProgress();
        }
      }, STABLE_CONNECTION_MS);
    };

    ws.onmessage = (event) => {
      if (socketRef.current !== ws) return;
      const payload = JSON.parse(event.data) as ServerMessage;
      handleServerMessage(payload);
    };

    ws.onerror = () => {
      if (socketRef.current !== ws) return;
      setConnection("error");
      setError("WebSocket 连接异常，请稍后重连或检查后端服务。");
    };

    ws.onclose = (event) => {
      if (socketRef.current !== ws) return;

      socketRef.current = null;
      clearStableConnectionTimer();
      setConnection((prev) => (prev === "error" ? "error" : "closed"));
      if (event.code === 4401 || event.code === 4403) {
        reconnectEnabledRef.current = false;
        setError("登录已失效或无权访问此会话，请重新登录后连接。");
        return;
      }
      scheduleReconnect(id);
    };
  }

  async function openHistoryView() {
    closeSocket();
    setView("history");
    await loadHistory();
  }

  async function openHistoricalReport(item: InterviewSessionSummary) {
    setError("");
    try {
      const [data, transcript] = await Promise.all([
        getInterviewReport(authToken, item.session_id),
        getInterviewTranscript(authToken, item.session_id),
      ]);
      closeSocket();
      setSessionId(item.session_id);
      setMode(item.mode);
      setMessages(transcriptMessages(transcript));
      setReport(data);
      setConnection("idle");
      setView("report");
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "加载报告失败。");
    }
  }

  async function openHistoricalRecord(item: InterviewSessionSummary) {
    setHistoryError("");
    try {
      const transcript = await getInterviewTranscript(authToken, item.session_id);
      closeSocket();
      setSessionId(item.session_id);
      setMode(item.mode);
      setMessages(transcriptMessages(transcript));
      setReport(transcript.report ?? null);
      setView("record");
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "加载面试记录失败。");
    }
  }

  async function downloadReport(sessionId: string, format: "markdown" | "pdf") {
    setError("");
    setHistoryError("");
    try {
      const blob = await exportInterviewReport(authToken, sessionId, format);
      if (format === "pdf") {
        openPrintableReport(blob, sessionId);
        return;
      }
      downloadBlob(blob, `interview-report-${sessionId.slice(0, 8)}.md`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "导出报告失败。";
      setError(message);
      setHistoryError(message);
    }
  }

  async function resumeHistoricalInterview(item: InterviewSessionSummary) {
    closeSocket();
    setSessionId(item.session_id);
    lastEventSequenceRef.current = 0;
    setMode(item.mode);
    setResume(null);
    setMessages([]);
    setCurrentTurn(null);
    setAnswer("");
    setReport(null);
    setError("");
    setStatus("正在恢复历史面试。");
    setStage("resuming");
    setView("interview");
    try {
      if (item.status === "paused" || item.status === "pausing") {
        await resumeInterview(authToken, item.session_id);
      }
      connectSocket(item.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "恢复面试失败。");
    }
  }

  function handleServerMessage(payload: ServerMessage) {
    const sequence = "sequence" in payload ? payload.sequence : null;
    if (sequence != null) {
      if (sequence <= lastEventSequenceRef.current) {
        return;
      }
      lastEventSequenceRef.current = sequence;
    }

    if (payload.type === "transcript_snapshot") {
      setMessages(transcriptMessages(payload.data));
      if (payload.data.report) setReport(payload.data.report);
      if (payload.data.session.pending_action === "answer_retry_required") {
        setError("上次提交的回答未能确认完成，请点击“重试失败步骤”显式重试。");
        setStage("retry_required");
        setIsAwaitingAnswer(false);
      }
      if (payload.data.session.status === "paused" || payload.data.session.status === "pausing") {
        setStage(payload.data.session.status);
        setIsAwaitingAnswer(false);
      }
      return;
    }

    if (payload.type === "status") {
      if (payload.sequence != null) markBusinessProgress();
      setStage(payload.stage);
      setStatus(statusText(payload.stage, payload.message));
      return;
    }

    const reconcilePendingAnswer = (turnId: string) => {
      const pending = pendingAnswerRef.current;
      if (!pending) return;
      if (pending.turnId !== turnId) {
        pendingAnswerRef.current = null;
        return;
      }
      const socket = socketRef.current;
      if (socket && socket.readyState === WebSocket.OPEN && pending.socket !== socket) {
        socket.send(JSON.stringify({
          type: "answer",
          content: pending.content,
          answer_id: pending.answerId,
          turn_id: pending.turnId,
        }));
        pending.socket = socket;
        setIsAwaitingAnswer(false);
        setStatus("正在确认断线前提交的回答。");
      }
    };

    if (payload.type === "stream_chunk") {
      markBusinessProgress();
      const streamMeta = streamMessageMeta(payload);

      if (payload.event === "start") {
        setIsAwaitingAnswer(false);
        setStage("generating_question");
        setCurrentTurn({
          questionIndex: payload.question_index,
          totalQuestions: payload.total_questions,
          skillTags: payload.skill_tags,
          difficulty: payload.difficulty,
          followUpNumber: payload.follow_up_number ?? undefined,
          questionId: payload.question_id,
          turnId: payload.turn_id,
        });
        setStatus(
          payload.kind === "follow_up"
            ? "正在生成追问，请稍候。"
            : "正在生成面试问题，请稍候。",
        );
        setMessages((items) => {
          const existing = items.some((item) => item.turnId === payload.turn_id);
          if (existing) {
            return items.map((item) =>
              item.turnId === payload.turn_id
                ? { ...item, streamId: payload.stream_id, streaming: true, meta: streamMeta }
                : item,
            );
          }
          return [
            ...items,
            {
              id: payload.stream_id,
              turnId: payload.turn_id,
              streamId: payload.stream_id,
              role: "interviewer",
              content: "",
              streaming: true,
              meta: streamMeta,
            },
          ];
        });
        return;
      }

      if (payload.event === "delta") {
        setMessages((items) => {
          const existing = items.some((item) => item.turnId === payload.turn_id);
          if (existing) {
            return items.map((item) =>
              item.turnId === payload.turn_id
                ? {
                    ...item,
                    content:
                      item.streamId === payload.stream_id
                        ? item.content + payload.chunk
                        : payload.chunk,
                    streamId: payload.stream_id,
                  }
                : item,
            );
          }
          return [
            ...items,
            {
              id: payload.stream_id,
              turnId: payload.turn_id,
              streamId: payload.stream_id,
              role: "interviewer",
              content: payload.chunk,
              streaming: true,
              meta: streamMeta,
            },
          ];
        });
        return;
      }

      setIsAwaitingAnswer(true);
      setStage("awaiting_answer");
      setStatus(
        payload.kind === "follow_up"
          ? "追问已生成，请继续作答。"
          : "问题已生成，请作答。",
      );
      setMessages((items) => {
        const existing = items.some((item) => item.turnId === payload.turn_id);
        if (existing) {
          return items.map((item) =>
            item.turnId === payload.turn_id
              ? {
                  ...item,
                  content: payload.content || item.content,
                  streamId: payload.stream_id,
                  streaming: false,
                  meta: streamMeta,
                }
              : item,
          );
        }
        return [
          ...items,
          {
            id: payload.stream_id,
            turnId: payload.turn_id,
            streamId: payload.stream_id,
            role: "interviewer",
            content: payload.content,
            streaming: false,
            meta: streamMeta,
          },
        ];
      });
      reconcilePendingAnswer(payload.turn_id);
      return;
    }

    if (payload.type === "question") {
      markBusinessProgress();
      setIsAwaitingAnswer(true);
      setStage("awaiting_answer");
      setCurrentTurn({
        questionIndex: payload.question_index,
        totalQuestions: payload.total_questions,
        skillTags: payload.skill_tags,
        difficulty: payload.difficulty,
        questionId: payload.question_id,
        turnId: payload.turn_id,
      });
      setStatus("请作答当前问题。");
      setMessages((items) =>
        appendInterviewerMessage(items, {
          id: payload.turn_id || uid(),
          turnId: payload.turn_id,
          role: "interviewer",
          content: payload.content,
          streaming: false,
          meta: `第 ${payload.question_index}/${payload.total_questions} 题 · ${difficultyName(
            payload.difficulty,
          )}`,
        }),
      );
      reconcilePendingAnswer(payload.turn_id);
      return;
    }

    if (payload.type === "follow_up") {
      markBusinessProgress();
      setIsAwaitingAnswer(true);
      setStage("awaiting_answer");
      setCurrentTurn({
        questionIndex: payload.question_index,
        followUpNumber: payload.follow_up_number,
        questionId: payload.question_id,
        turnId: payload.turn_id,
      });
      setStatus("面试官正在追问，请继续补充回答。");
      setMessages((items) =>
        appendInterviewerMessage(items, {
          id: payload.turn_id || uid(),
          turnId: payload.turn_id,
          role: "interviewer",
          content: payload.content,
          streaming: false,
          meta: `第 ${payload.question_index} 题追问 · 第 ${payload.follow_up_number} 次`,
        }),
      );
      reconcilePendingAnswer(payload.turn_id);
      return;
    }

    if (payload.type === "interview_end") {
      markBusinessProgress();
      setIsAwaitingAnswer(false);
      setStatus("面试已结束，正在整理评估报告。");
      setStage("evaluating");
      return;
    }

    if (payload.type === "report") {
      pendingAnswerRef.current = null;
      disableAutoReconnect();
      setReport(payload.data);
      setStatus("报告已生成。");
      setStage("completed");
      setView("report");
      void loadHistory();
      return;
    }

    if (payload.type === "error") {
      setError(payload.message);
      setStatus("处理过程中出现错误，请查看提示后重试。");
    }
  }

  function submitAnswer(event: FormEvent) {
    event.preventDefault();
    const content = answer.trim();
    if (!content || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
    const turnId = currentTurn?.turnId;
    if (!turnId) {
      setError("当前回合标识尚未就绪，请等待问题完整显示后再提交。");
      return;
    }
    const answerId = uid();
    socketRef.current.send(
      JSON.stringify({
        type: "answer",
        content,
        answer_id: answerId,
        turn_id: turnId,
      }),
    );
    pendingAnswerRef.current = {
      answerId,
      turnId,
      content,
      socket: socketRef.current,
    };
    setMessages((items) => [...items, { id: uid(), role: "candidate", content }]);
    setAnswer("");
    setIsAwaitingAnswer(false);
    setStatus("已提交回答，正在评估。");
    setStage("processing");
  }

  async function pauseCurrentInterview() {
    try {
      await pauseInterview(authToken, sessionId);
      setStage("pausing");
      setStatus("暂停请求已接受；当前评分完成后会进入暂停状态。");
      setIsAwaitingAnswer(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "暂停面试失败。");
    }
  }

  async function resumeCurrentInterview() {
    try {
      await resumeInterview(authToken, sessionId);
      setStage("resuming");
      setStatus("正在恢复面试。");
      connectSocket(sessionId, { resetAttempts: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "恢复面试失败。");
    }
  }

  async function stopInterview() {
    if (!window.confirm("确定提前结束本次面试吗？已接受的回答会保留。")) return;
    setStatus("正在提前结束面试并生成阶段性报告。");
    setIsAwaitingAnswer(false);
    try {
      await endInterview(authToken, sessionId);
      const transcript = await getInterviewTranscript(authToken, sessionId);
      setMessages(transcriptMessages(transcript));
      setReport(transcript.report ?? null);
      setStage("completed");
      setView(transcript.has_report ? "report" : "record");
      closeSocket();
      void loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "结束面试失败。");
    }
  }

  function retryCurrentOperation() {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "retry", retry_id: uid() }));
    setError("");
    setIsAwaitingAnswer(false);
    setStage("retrying");
    setStatus("已提交显式重试，正在恢复失败的出题步骤。");
  }

  function resetAll() {
    closeSocket();
    setView("setup");
    setSessionId("");
    setMessages([]);
    setCurrentTurn(null);
    setAnswer("");
    setStatus("填写 JD，可选上传简历后开始模拟面试。");
    setStage("ready");
    setError("");
    setReport(null);
    setResume(null);
    setConnection("idle");
    pendingAnswerRef.current = null;
  }

  const progress = useMemo(() => {
    if (!currentTurn?.questionIndex || !currentTurn.totalQuestions) return 0;
    return Math.round((currentTurn.questionIndex / currentTurn.totalQuestions) * 100);
  }, [currentTurn]);

  return (
    <main className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <header className="topbar">
        <button className="brand" onClick={resetAll} title="回到开始页">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>
            <strong>Mockview</strong>
            <small>JD 与简历驱动的智能面试练习平台</small>
          </span>
        </button>
        <div className="topbar-actions">
          {currentUser && (
            <span className="api-pill" title={currentUser.username}>
              <User size={14} />
              {currentUser.display_name || currentUser.username}
            </span>
          )}
          <StatusPill connection={connection} />
          <span className="api-pill" title={API_BASE_URL}>
            <Activity size={14} />
            {health?.status === "ok" ? health.model ?? "后端在线" : "检查后端"}
          </span>
          {currentUser && (
            <button className="icon-text-button" onClick={handleLogout} type="button">
              <LogOut size={14} />
              退出
            </button>
          )}
        </div>
      </header>

      {currentUser && (
        <nav className="view-tabs" aria-label="工作台视图">
          <button
            className={view === "setup" ? "active" : ""}
            onClick={() => {
              closeSocket();
              setView("setup");
            }}
            type="button"
          >
            <Play size={15} />
            开始面试
          </button>
          <button className={view === "history" ? "active" : ""} onClick={openHistoryView} type="button">
            <History size={15} />
            历史记录
          </button>
          <button className={view === "observations" ? "active" : ""} onClick={openObservationsView} type="button">
            <ClipboardList size={15} />
            能力观察
          </button>
          {preparationCase && (
            <button className={view === "preparation" ? "active" : ""} onClick={() => setView("preparation")} type="button">
              <ClipboardList size={15} />
              准备任务
            </button>
          )}
          {currentUser.role === "admin" && (
            <button className={view === "knowledge" ? "active" : ""} onClick={openKnowledgeView} type="button">
              <FileText size={15} />
              知识审核
            </button>
          )}
        </nav>
      )}

      {!currentUser && (
        <AuthPage
          mode={authMode}
          setMode={setAuthMode}
          username={authUsername}
          setUsername={setAuthUsername}
          password={authPassword}
          setPassword={setAuthPassword}
          displayName={displayName}
          setDisplayName={setDisplayName}
          isSubmitting={isAuthenticating}
          isChecking={isCheckingAuth}
          error={authError}
          onSubmit={handleAuthSubmit}
        />
      )}

      {currentUser && view === "setup" && (
        <SetupPage
          jdText={jdText}
          setJdText={setJdText}
          mode={mode}
          setMode={setMode}
          maxFollowUps={maxFollowUps}
          setMaxFollowUps={setMaxFollowUps}
          currentUser={currentUser}
          resumeFile={resumeFile}
          setResumeFile={setResumeFile}
          isStarting={isStarting}
          isPreparing={isPreparing}
          canStart={canStart}
          canPrepare={canPrepare}
          projectNotes={projectNotes}
          setProjectNotes={setProjectNotes}
          error={error}
          onStart={handleStart}
          onPrepare={handlePrepare}
        />
      )}

      {currentUser && view === "preparation" && preparationCase && (
        <PreparationPage
          preparationCase={preparationCase}
          operation={preparationOperation}
          error={error}
          isStarting={isStarting}
          onRefresh={() => void refreshPreparation()}
          onStart={handleStartPrepared}
          onClarify={handleClarification}
          onContinue={handleContinuePreparation}
          onCancel={handleCancelPreparation}
          onBack={() => setView("setup")}
        />
      )}

      {currentUser && view === "observations" && (
        <ObservationsPage
          items={observations}
          error={error}
          onRefresh={openObservationsView}
          onDispute={handleDisputeObservation}
          onCorrect={handleCorrectObservation}
        />
      )}

      {currentUser && view === "knowledge" && (
        <KnowledgePage
          documents={knowledgeCandidates}
          release={knowledgeRelease}
          error={error}
          onReview={handleKnowledgeReview}
          onPublish={handleKnowledgePublish}
        />
      )}

      {currentUser && view === "interview" && (
        <InterviewPage
          mode={mode}
          sessionId={sessionId}
          resume={resume}
          status={status}
          stage={stage}
          error={error}
          messages={messages}
          currentTurn={currentTurn}
          progress={progress}
          answer={answer}
          setAnswer={setAnswer}
          isAwaitingAnswer={isAwaitingAnswer}
          connection={connection}
          transcriptRef={transcriptRef}
          onSubmitAnswer={submitAnswer}
          onStop={stopInterview}
          onPause={pauseCurrentInterview}
          onResume={resumeCurrentInterview}
          onRetry={retryCurrentOperation}
          onReconnect={() => connectSocket(sessionId, { resetAttempts: true })}
          onBack={() => {
            closeSocket();
            setView("setup");
          }}
        />
      )}

      {currentUser && view === "history" && (
        <HistoryPage
          items={historyItems}
          isLoading={isHistoryLoading}
          error={historyError}
          onRefresh={loadHistory}
          onViewReport={openHistoricalReport}
          onViewRecord={openHistoricalRecord}
          onResume={resumeHistoricalInterview}
          onExport={downloadReport}
        />
      )}

      {currentUser && view === "record" && (
        <TranscriptRecordPage
          sessionId={sessionId}
          messages={messages}
          report={report}
          onBack={() => setView("history")}
          onViewReport={() => setView("report")}
        />
      )}

      {currentUser && view === "report" && report && (
        <ReportPage
          report={report}
          mode={mode}
          sessionId={sessionId}
          messages={messages}
          onRestart={resetAll}
          onBackToInterview={() => setView("interview")}
          onExport={downloadReport}
          onRepractice={repracticeOptions?.weak_skills.length ? handleRepractice : undefined}
          weakSkills={repracticeOptions?.weak_skills ?? []}
          repracticeNotice={repracticeNotice}
          isPreparing={isPreparing}
          error={error}
        />
      )}
    </main>
  );
}

function AuthPage(props: {
  mode: AuthMode;
  setMode: (value: AuthMode) => void;
  username: string;
  setUsername: (value: string) => void;
  password: string;
  setPassword: (value: string) => void;
  displayName: string;
  setDisplayName: (value: string) => void;
  isSubmitting: boolean;
  isChecking: boolean;
  error: string;
  onSubmit: (event: FormEvent) => void;
}) {
  const isRegister = props.mode === "register";
  return (
    <section className="auth-layout">
      <div className="auth-copy">
        <span className="eyebrow">用户工作区</span>
        <h1>先登录，再开始你的专属模拟面试</h1>
        <p>
          每个账号会拥有独立的面试会话、短期检查点和长期记忆空间，后续可以继续扩展历史记录与报告管理。
        </p>
        <div className="signal-strip">
          <span>JWT 身份认证</span>
          <span>账号隔离会话</span>
          <span>长期记忆绑定用户</span>
        </div>
      </div>

      <form className="auth-panel" onSubmit={props.onSubmit}>
        <div className="panel-heading">
          <KeyRound size={20} />
          <div>
            <h2>{isRegister ? "创建账号" : "登录账号"}</h2>
            <p>{isRegister ? "注册后会自动进入面试工作台。" : "使用你的账号进入个人面试空间。"}</p>
          </div>
        </div>

        <div className="mode-switch" aria-label="认证模式">
          <button
            className={!isRegister ? "active" : ""}
            onClick={() => props.setMode("login")}
            type="button"
          >
            登录
          </button>
          <button
            className={isRegister ? "active" : ""}
            onClick={() => props.setMode("register")}
            type="button"
          >
            注册
          </button>
        </div>

        <label className="field">
          <span>用户名</span>
          <input
            autoComplete="username"
            minLength={3}
            maxLength={64}
            required
            value={props.username}
            onChange={(event) => props.setUsername(event.target.value)}
            placeholder="例如 guguohan"
          />
        </label>

        {isRegister && (
          <label className="field">
            <span>显示名称</span>
            <input
              autoComplete="name"
              maxLength={64}
              value={props.displayName}
              onChange={(event) => props.setDisplayName(event.target.value)}
              placeholder="可选，用于页面右上角展示"
            />
          </label>
        )}

        <label className="field">
          <span>密码</span>
          <input
            autoComplete={isRegister ? "new-password" : "current-password"}
            minLength={isRegister ? 6 : 1}
            maxLength={128}
            required
            type="password"
            value={props.password}
            onChange={(event) => props.setPassword(event.target.value)}
            placeholder={isRegister ? "至少 6 位" : "输入密码"}
          />
        </label>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            {props.error}
          </div>
        )}

        <button className="primary-action" disabled={props.isSubmitting || props.isChecking} type="submit">
          {props.isSubmitting || props.isChecking ? <Loader2 className="spin" size={18} /> : <KeyRound size={18} />}
          {props.isChecking ? "正在检查登录状态" : isRegister ? "注册并进入" : "登录"}
        </button>
      </form>
    </section>
  );
}

function SetupPage(props: {
  jdText: string;
  setJdText: (value: string) => void;
  mode: InterviewMode;
  setMode: (value: InterviewMode) => void;
  maxFollowUps: number;
  setMaxFollowUps: (value: number) => void;
  currentUser: UserPublic;
  resumeFile: File | null;
  setResumeFile: (file: File | null) => void;
  isStarting: boolean;
  isPreparing: boolean;
  canStart: boolean;
  canPrepare: boolean;
  projectNotes: string;
  setProjectNotes: (value: string) => void;
  error: string;
  onStart: () => void;
  onPrepare: () => void;
}) {
  return (
    <section className="setup-layout">
      <div className="setup-copy">
        <span className="eyebrow">AI 模拟面试工作台</span>
        <h1>把 JD 和简历变成一场可追问的技术面试</h1>
        <p>
          系统会分析岗位要求和候选人经历，结合 RAG 题库、长期记忆和多 Agent
          工作流生成问题，并在实时对话后输出结构化评估报告。
        </p>
        <div className="signal-strip">
          <span>JD / 简历驱动</span>
          <span>RAG 个性化出题</span>
          <span>WebSocket 实时面试</span>
          <span>结构化评估报告</span>
        </div>
      </div>

      <div className="setup-panel">
        <div className="panel-heading">
          <ClipboardList size={20} />
          <div>
            <h2>创建面试会话</h2>
            <p>
              当前账号：{props.currentUser.display_name || props.currentUser.username}。粘贴目标岗位 JD，专业模式下建议上传简历。
            </p>
          </div>
        </div>

        <label className="field">
          <span>岗位 JD</span>
          <textarea
            value={props.jdText}
            onChange={(event) => props.setJdText(event.target.value)}
            rows={9}
            placeholder="粘贴目标岗位描述，包含职责、技能要求和经验要求。"
          />
        </label>

        <div className="grid-2">
          <label className="field">
            <span>每题最多追问</span>
            <input
              type="number"
              min={0}
              max={5}
              value={props.maxFollowUps}
              onChange={(event) => props.setMaxFollowUps(Number(event.target.value))}
            />
          </label>
          <div className="identity-note">
            <User size={16} />
            <span>面试记录会自动绑定当前登录账号。</span>
          </div>
        </div>

        <div className="mode-switch" aria-label="面试模式">
          <button
            className={props.mode === "professional" ? "active" : ""}
            onClick={() => props.setMode("professional")}
            type="button"
          >
            <Sparkles size={16} />
            专业面试
          </button>
          <button
            className={props.mode === "practice" ? "active" : ""}
            onClick={() => props.setMode("practice")}
            type="button"
          >
            <MessageSquareText size={16} />
            练习模式
          </button>
        </div>

        <label className="upload-box">
          <Upload size={20} />
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={(event) => props.setResumeFile(event.target.files?.[0] ?? null)}
          />
          <span>{props.resumeFile ? props.resumeFile.name : "上传简历，可选 PDF / PNG / JPG"}</span>
        </label>

        <label className="field">
          <span>项目补充说明（可选）</span>
          <textarea
            value={props.projectNotes}
            onChange={(event) => props.setProjectNotes(event.target.value)}
            rows={4}
            placeholder="补充你在项目中的具体职责、设计取舍或希望重点验证的能力。"
          />
        </label>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            {props.error}
          </div>
        )}

        <button className="primary-action" disabled={!props.canPrepare} onClick={props.onPrepare}>
          {props.isPreparing ? <Loader2 className="spin" size={18} /> : <ClipboardList size={18} />}
          {props.isPreparing ? "正在分析材料与证据" : "分析证据并准备面试"}
        </button>

        <button className="secondary-button" disabled={!props.canStart} onClick={props.onStart}>
          {props.isStarting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {props.isStarting ? "正在启动" : "兼容快速模式（跳过证据准备）"}
        </button>
      </div>
    </section>
  );
}

function PreparationPage(props: {
  preparationCase: PreparationCase;
  operation: OperationRecord | null;
  error: string;
  isStarting: boolean;
  onRefresh: () => void;
  onStart: () => void;
  onClarify: (clarificationId: string, answer: string) => void;
  onContinue: () => void;
  onCancel: () => void;
  onBack: () => void;
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const pendingClarifications = props.preparationCase.clarifications.filter(
    (item) => item.status === "pending",
  );
  const rawCoverage = props.operation?.result?.evidence_coverage;
  const evidenceCoverage = rawCoverage && typeof rawCoverage === "object"
    ? rawCoverage as {
        eligible_target_count?: number;
        unresolved_target_count?: number;
      }
    : null;
  const ready = props.preparationCase.status === "ready" &&
    props.operation?.status === "succeeded" &&
    pendingClarifications.length === 0 &&
    (evidenceCoverage?.eligible_target_count ?? 0) > 0;
  return (
    <section className="report-layout">
      <div className="report-hero">
        <button className="ghost-button" onClick={props.onBack}>
          <ArrowLeft size={16} /> 返回设置
        </button>
        <div>
          <span className="eyebrow">岗位准备任务</span>
          <h1>考察目标与材料状态</h1>
          <p>
            材料版本 {props.preparationCase.material_revision} · 任务状态 {props.preparationCase.status}
            {props.operation ? ` · 执行 ${props.operation.status}` : ""}
          </p>
        </div>
        <div className="report-actions">
          <button className="secondary-button" onClick={props.onRefresh}>
            <RotateCcw size={16} /> 刷新进度
          </button>
          {!props.operation || ["failed_retryable", "waiting_input"].includes(props.operation.status) ? (
            <button className="secondary-button" disabled={pendingClarifications.length > 0} onClick={props.onContinue}>
              <Play size={16} /> 继续分析
            </button>
          ) : null}
          {props.operation && ["accepted", "running", "waiting_input", "failed_retryable"].includes(props.operation.status) ? (
            <button className="secondary-button" onClick={props.onCancel}>
              <CircleStop size={16} /> 取消任务
            </button>
          ) : null}
          <button className="primary-small" disabled={!ready || props.isStarting} onClick={props.onStart}>
            {props.isStarting ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
            按目标开始面试
          </button>
        </div>
      </div>

      {props.error && <div className="error-callout"><AlertCircle size={16} />{props.error}</div>}
      {props.operation && ["failed", "cancelled"].includes(props.operation.status) && <p>本次分析已结束，预算不会自动重置。请返回准备页修订材料后创建新的准备任务。</p>}
      {props.operation?.error && <div className="error-callout"><AlertCircle size={16} />{props.operation.error}</div>}
      {evidenceCoverage ? (
        <div className="signal-strip">
          可出题目标 {evidenceCoverage.eligible_target_count ?? 0} 个 ·
          证据未解决目标 {evidenceCoverage.unresolved_target_count ?? 0} 个
        </div>
      ) : null}
      {props.preparationCase.resume_parse_warnings.length ? (
        <div className="error-callout">
          <AlertCircle size={16} />
          简历解析提示：{props.preparationCase.resume_parse_warnings.join("；")}
        </div>
      ) : null}

      <div className="report-grid">
        <section className="report-section span-2">
          <h2>具体考察目标</h2>
          <div className="highlight-list">
            {props.preparationCase.targets.map((target) => (
              <article key={target.target_id}>
                <strong>{target.scope} · 权重 {(target.importance * 100).toFixed(0)}%</strong>
                <p>{target.verification_goal}</p>
                <small>来源：{target.sources.join(" / ")} · {target.status}</small>
              </article>
            ))}
            {!props.preparationCase.targets.length && <p>正在分析材料并形成考察目标。</p>}
          </div>
        </section>

        <section className="report-section span-2">
          <h2>待补充材料</h2>
          {pendingClarifications.length ? (
            <div className="highlight-list">
              {pendingClarifications.map((item) => (
                <article key={item.clarification_id}>
                  <strong>{item.question}</strong>
                  <textarea
                    rows={3}
                    value={answers[item.clarification_id] ?? ""}
                    onChange={(event) => setAnswers((current) => ({ ...current, [item.clarification_id]: event.target.value }))}
                    placeholder="补充可核验的项目事实或说明"
                  />
                  <button
                    className="primary-small"
                    disabled={!(answers[item.clarification_id] ?? "").trim()}
                    onClick={() => props.onClarify(item.clarification_id, answers[item.clarification_id] ?? "")}
                  >
                    提交补充
                  </button>
                </article>
              ))}
            </div>
          ) : <p>当前没有必须补充的材料；系统会对未确认经历保持保守判断。</p>}
        </section>
      </div>
    </section>
  );
}

function ObservationsPage(props: {
  items: MemoryObservation[];
  error: string;
  onRefresh: () => void;
  onDispute: (observationId: string) => void;
  onCorrect: (observationId: string, content: string) => Promise<void>;
}) {
  const [corrections, setCorrections] = useState<Record<string, string>>({});
  return (
    <section className="history-layout">
      <div className="history-header">
        <div>
          <span className="eyebrow">长期记忆</span>
          <h1>能力观察与评价依据</h1>
          <p>这里只展示最终单题观察；提出异议后，该观察会从自动复练依据中排除并重算画像。</p>
        </div>
        <button className="secondary-button" onClick={props.onRefresh} type="button">
          <RotateCcw size={16} /> 刷新
        </button>
      </div>
      {props.error && <div className="error-callout"><AlertCircle size={16} />{props.error}</div>}
      {!props.items.length ? (
        <div className="empty-state"><ClipboardList size={18} />完成面试后会在这里形成可追溯的能力观察。</div>
      ) : (
        <div className="history-list">
          {props.items.map((item) => {
            const validity = String(item.structured?.validity ?? "valid");
            return (
              <article className="history-card" key={item.id}>
                <div>
                  <strong>{item.tags.join(" · ") || "面试观察"}</strong>
                  <p>{item.content}</p>
                  <small>来源：{item.source} · {new Date(item.created_at).toLocaleString()}</small>
                </div>
                <button
                  className="secondary-button"
                  disabled={validity !== "valid"}
                  onClick={() => props.onDispute(item.id)}
                  type="button"
                >
                  {validity === "disputed"
                    ? "已提出异议"
                    : validity === "replaced"
                      ? "已被事实修正替代"
                      : "对评价提出异议"}
                </button>
                {validity === "valid" && <div className="field">
                  <label htmlFor={`correction-${item.id}`}>修正相关事实</label>
                  <textarea
                    id={`correction-${item.id}`}
                    value={corrections[item.id] ?? ""}
                    onChange={(event) => setCorrections((current) => ({
                      ...current,
                      [item.id]: event.target.value,
                    }))}
                    placeholder="填写经你确认的事实；原记录会保留替代关系"
                  />
                  <button
                    className="primary-small"
                    disabled={!(corrections[item.id] ?? "").trim()}
                    onClick={async () => {
                      await props.onCorrect(item.id, corrections[item.id] ?? "");
                      setCorrections((current) => ({ ...current, [item.id]: "" }));
                    }}
                    type="button"
                  >
                    保存修正
                  </button>
                </div>}
              </article>
            );
          })}
        </div>
      )}

    </section>
  );
}

function HistoryPage(props: {
  items: InterviewSessionSummary[];
  isLoading: boolean;
  error: string;
  onRefresh: () => void;
  onViewReport: (item: InterviewSessionSummary) => void;
  onViewRecord: (item: InterviewSessionSummary) => void;
  onResume: (item: InterviewSessionSummary) => void;
  onExport: (sessionId: string, format: "markdown" | "pdf") => void;
}) {
  return (
    <section className="history-layout">
      <div className="history-header">
        <div>
          <span className="eyebrow">面试记录</span>
          <h1>历史面试与报告管理</h1>
          <p>查看当前账号下的面试进度、报告结果，或继续未完成的模拟面试。</p>
        </div>
        <button className="secondary-button" onClick={props.onRefresh} type="button">
          {props.isLoading ? <Loader2 className="spin" size={16} /> : <RotateCcw size={16} />}
          刷新
        </button>
      </div>

      {props.error && (
        <div className="error-callout">
          <AlertCircle size={16} />
          {props.error}
        </div>
      )}

      {props.isLoading && !props.items.length ? (
        <div className="empty-state">
          <Loader2 className="spin" size={18} />
          正在加载历史记录
        </div>
      ) : null}

      {!props.isLoading && !props.items.length ? (
        <div className="empty-state">
          <History size={18} />
          还没有历史面试，先创建一场新的模拟面试。
        </div>
      ) : null}

      {props.items.length ? (
        <div className="history-list">
          {props.items.map((item) => (
            <article className="history-item" key={item.session_id}>
              <div className="history-main">
                <div className="history-title-row">
                  <h2>{item.title || "未命名面试"}</h2>
                  <span className={`session-status status-${item.status}`}>
                    {sessionStatusName(item.status)}
                  </span>
                </div>
                <p>{item.jd_preview}</p>
                <div className="history-meta">
                  <span>{modeName(item.mode)}</span>
                  <span>{item.assessment_count} 条评价</span>
                  {item.question_count ? <span>{item.question_count} 道题</span> : null}
                  <span>{formatSessionTime(item.updated_at || item.created_at)}</span>
                </div>
              </div>

              <div className="history-score">
                {item.overall_score ? (
                  <>
                    <strong>{item.overall_score.toFixed(1)}</strong>
                    <span>{item.grade ?? "报告"}</span>
                  </>
                ) : (
                  <>
                    <strong>{item.current_question_index || "-"}</strong>
                    <span>当前题</span>
                  </>
                )}
              </div>

              <div className="history-actions">
                <button className="secondary-button" onClick={() => props.onViewRecord(item)} type="button">
                  <MessageSquareText size={16} />
                  查看记录
                </button>
                {item.has_report ? (
                  <>
                    <button className="primary-small" onClick={() => props.onViewReport(item)} type="button">
                      <FileText size={16} />
                      查看报告
                    </button>
                    <button
                      className="secondary-button"
                      onClick={() => props.onExport(item.session_id, "pdf")}
                      type="button"
                    >
                      <FileText size={16} />
                      打印 PDF
                    </button>
                  </>
                ) : null}
                {item.status !== "completed" && item.status !== "failed" ? (
                  <button className="secondary-button" onClick={() => props.onResume(item)} type="button">
                    <Play size={16} />
                    继续面试
                  </button>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function KnowledgePage(props: {
  documents: KnowledgeDocument[];
  release: KnowledgeRelease | null;
  error: string;
  onReview: (documentId: string, status: "approved" | "rejected") => void;
  onPublish: () => void;
}) {
  return (
    <section className="report-layout">
      <div className="report-hero">
        <div>
          <span className="eyebrow">管理员知识审核</span>
          <h1>候选资料与不可变发布</h1>
          <p>只有审核通过的公共资料会进入新发布版本；构建失败不会切换当前版本。</p>
        </div>
        <button className="primary-small" onClick={props.onPublish}>
          <Play size={16} /> 构建并发布
        </button>
      </div>
      {props.error && <div className="error-callout"><AlertCircle size={16} />{props.error}</div>}
      {props.release && (
        <div className={props.release.status === "failed" ? "error-callout" : "signal-strip"}>
          发布 {props.release.release_id.slice(0, 8)}：{props.release.status}
          {props.release.error ? ` · ${props.release.error}` : ""}
        </div>
      )}
      <div className="report-grid">
        <section className="report-section span-2">
          <h2>待审核候选</h2>
          <div className="highlight-list">
            {props.documents.map((document) => (
              <article key={document.document_id}>
                <strong>{document.title}</strong>
                <p>{document.content.slice(0, 360)}</p>
                {document.source_url && <small>{document.source_url}</small>}
                <div className="report-actions">
                  <button className="primary-small" onClick={() => props.onReview(document.document_id, "approved")}>批准</button>
                  <button className="danger-button" onClick={() => props.onReview(document.document_id, "rejected")}>拒绝</button>
                </div>
              </article>
            ))}
            {!props.documents.length && <p>当前没有待审核候选资料。</p>}
          </div>
        </section>
      </div>
    </section>
  );
}

function sessionStatusName(status: InterviewSessionSummary["status"]) {
  const map: Record<InterviewSessionSummary["status"], string> = {
    pending: "待开始",
    analyzing: "分析中",
    interviewing: "面试中",
    evaluating: "评估中",
    pausing: "暂停中",
    paused: "已暂停",
    completed: "已完成",
    failed: "失败",
  };
  return map[status] ?? status;
}

function formatSessionTime(value?: string | null) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function openPrintableReport(blob: Blob, sessionId: string) {
  const url = URL.createObjectURL(blob);
  const win = window.open(url, `interview-report-${sessionId.slice(0, 8)}`);
  if (!win) {
    downloadBlob(blob, `interview-report-${sessionId.slice(0, 8)}.html`);
    return;
  }
  win.addEventListener("load", () => {
    win.focus();
    win.print();
  });
}

function InterviewPage(props: {
  mode: InterviewMode;
  sessionId: string;
  resume: ResumeParseResult | null;
  status: string;
  stage: string;
  error: string;
  messages: ChatItem[];
  currentTurn: CurrentTurn | null;
  progress: number;
  answer: string;
  setAnswer: (value: string) => void;
  isAwaitingAnswer: boolean;
  connection: ConnectionState;
  transcriptRef: RefObject<HTMLDivElement | null>;
  onSubmitAnswer: (event: FormEvent) => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onRetry: () => void;
  onReconnect: () => void;
  onBack: () => void;
}) {
  return (
    <section className="workspace">
      <aside className="sidebar">
        <button className="ghost-button" onClick={props.onBack}>
          <ArrowLeft size={16} />
          返回设置
        </button>

        <div className="side-section">
          <span className="side-label">会话</span>
          <code>{props.sessionId.slice(0, 8) || "待创建"}</code>
        </div>

        <div className="side-section">
          <span className="side-label">模式</span>
          <strong>{modeName(props.mode)}</strong>
        </div>

        <div className="progress-box">
          <div className="progress-head">
            <span>面试进度</span>
            <strong>{props.progress}%</strong>
          </div>
          <div className="progress-track">
            <div style={{ width: `${props.progress}%` }} />
          </div>
          {props.currentTurn?.questionIndex && (
            <p>
              当前第 {props.currentTurn.questionIndex}
              {props.currentTurn.totalQuestions ? ` / ${props.currentTurn.totalQuestions}` : ""} 题
            </p>
          )}
        </div>

        <div className="side-section">
          <span className="side-label">当前状态</span>
          <p>{props.status}</p>
          <small>{props.stage}</small>
        </div>

        {props.currentTurn?.skillTags?.length ? (
          <div className="side-section">
            <span className="side-label">考察技能</span>
            <div className="tag-list">
              {props.currentTurn.skillTags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
          </div>
        ) : null}

        {props.resume && (
          <div className="resume-mini">
            <FileText size={16} />
            <span>简历已解析，将用于项目深挖和岗位匹配提问。</span>
          </div>
        )}
      </aside>

      <div className="interview-panel">
        <div className="interview-header">
          <div>
            <h2>实时面试</h2>
            <p>
              请像真实面试一样作答。系统会根据你的回答判断是否追问、进入下一题或生成报告。
            </p>
          </div>
          <div className="interview-actions">
            {props.stage === "paused" || props.stage === "pausing" ? (
              <button className="secondary-button" onClick={props.onResume} disabled={props.stage === "pausing"}>
                <Play size={16} />
                恢复
              </button>
            ) : (
              <button className="secondary-button" onClick={props.onPause}>
                <Pause size={16} />
                暂停
              </button>
            )}
            {props.connection !== "open" && (
              <button className="secondary-button" onClick={props.onReconnect}>
                <RotateCcw size={16} />
                重连
              </button>
            )}
            <button className="danger-button" onClick={props.onStop}>
              <CircleStop size={16} />
              结束面试
            </button>
          </div>
        </div>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            <span>{props.error}</span>
            {props.connection === "open" && (
              <button className="secondary-button" onClick={props.onRetry}>
                <RotateCcw size={15} />
                重试失败步骤
              </button>
            )}
          </div>
        )}

        <div className="transcript" ref={props.transcriptRef}>
          {props.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar">
                {message.role === "candidate" ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className="bubble">
                {message.meta && <span className="message-meta">{message.meta}</span>}
                <p>{message.content}</p>
              </div>
            </article>
          ))}
          {!props.isAwaitingAnswer && props.connection === "open" && (
            <div className="thinking-line">
              <Loader2 className="spin" size={16} />
              {props.status}
            </div>
          )}
        </div>

        <form className="composer" onSubmit={props.onSubmitAnswer}>
          <textarea
            value={props.answer}
            onChange={(event) => props.setAnswer(event.target.value)}
            placeholder={props.isAwaitingAnswer ? "输入你的回答..." : "等待面试官提问..."}
            rows={3}
            disabled={!props.isAwaitingAnswer || props.connection !== "open"}
          />
          <button
            className="send-button"
            disabled={!props.answer.trim() || !props.isAwaitingAnswer || props.connection !== "open"}
            type="submit"
            title="发送回答"
          >
            <Send size={18} />
          </button>
        </form>
      </div>
    </section>
  );
}

function TranscriptRecordPage(props: {
  sessionId: string;
  messages: ChatItem[];
  report: InterviewReport | null;
  onBack: () => void;
  onViewReport: () => void;
}) {
  return (
    <section className="report-layout">
      <div className="report-hero">
        <button className="ghost-button" onClick={props.onBack}>
          <ArrowLeft size={16} /> 返回历史
        </button>
        <div>
          <span className="eyebrow">完整面试记录</span>
          <h1>会话 {props.sessionId.slice(0, 8)}</h1>
          <p>包含 AI 主问题、追问和你的回答；待重试回答也会保留。</p>
        </div>
        {props.report ? (
          <button className="primary-small" onClick={props.onViewReport}>
            <FileText size={16} /> 查看报告
          </button>
        ) : null}
      </div>
      <div className="transcript">
        {props.messages.map((message) => (
          <article className={`message ${message.role}`} key={message.id}>
            <div className="avatar">{message.role === "candidate" ? <User size={16} /> : <Bot size={16} />}</div>
            <div className="bubble">
              {message.meta ? <span className="message-meta">{message.meta}</span> : null}
              <p>{message.content}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ReportPage(props: {
  report: InterviewReport;
  mode: InterviewMode;
  sessionId: string;
  messages: ChatItem[];
  onRestart: () => void;
  onBackToInterview: () => void;
  onExport: (sessionId: string, format: "markdown" | "pdf") => void;
  onRepractice?: (skills: string[]) => void;
  weakSkills: Array<{ skill_name: string; score: number }>;
  repracticeNotice: string;
  isPreparing: boolean;
  error: string;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const weakSkillNames = props.weakSkills.map((item) => item.skill_name).join("\n");
  useEffect(() => { setSelected(weakSkillNames ? weakSkillNames.split("\n") : []); }, [props.sessionId, weakSkillNames]);
  const skillScores = props.report.skill_scores ?? [];
  const grade = props.report.grade ?? null;
  const overall = props.report.overall_score ?? null;

  return (
    <section className="report-layout">
      <div className="report-hero">
        <button className="ghost-button" onClick={props.onBackToInterview}>
          <ArrowLeft size={16} />
          返回面试记录
        </button>
        <div className="grade-lockup">
          <span className={`grade ${grade ? `grade-${grade}` : ""}`}>{grade ?? "—"}</span>
          <div>
            <span className="eyebrow">面试报告</span>
            <h1>{overall === null ? "暂无可汇总分数" : `${overall.toFixed(1)} / 10`}</h1>
            <p>{props.report.overall_assessment}</p>
          </div>
        </div>
        <div className="report-actions">
          <button
            className="secondary-button"
            onClick={() => props.onExport(props.sessionId, "markdown")}
          >
            <FileText size={16} />
            导出 Markdown
          </button>
          <button className="secondary-button" onClick={() => props.onExport(props.sessionId, "pdf")}>
            <FileText size={16} />
            打印 / 保存 PDF
          </button>
          <button className="primary-small" onClick={props.onRestart}>
            <RotateCcw size={16} />
            重新开始
          </button>
          {props.onRepractice ? (
            <button className="primary-small" disabled={props.isPreparing || !selected.length} onClick={() => props.onRepractice?.(selected)}>
              <Sparkles size={16} />
              针对薄弱项复练
            </button>
          ) : null}
        </div>
      </div>

      {props.error && <div className="error-callout">{props.error}</div>}
      {props.repracticeNotice && <p>{props.repracticeNotice}</p>}
      {props.weakSkills.length > 0 && <fieldset>
        <legend>选择复练技能（来自有效最终评价）</legend>
        {props.weakSkills.map((item) => <label key={item.skill_name}>
          <input type="checkbox" checked={selected.includes(item.skill_name)} onChange={(event) =>
            setSelected((values) => event.target.checked ? [...values, item.skill_name] : values.filter((value) => value !== item.skill_name))} />
          {item.skill_name} · {item.score}/10
        </label>)}
      </fieldset>}
      <div className="report-grid">
        {props.report.hiring_recommendation && (
          <section className="report-section span-2">
            <h2>面试结论</h2>
            <p className="recommendation">{props.report.hiring_recommendation}</p>
          </section>
        )}

        {props.report.round_scores?.length ? (
          <section className="report-section">
            <h2>轮次表现</h2>
            <div className="round-list">
              {props.report.round_scores.map((round) => (
                <div className="round-item" key={round.round_name}>
                  <div>
                    <strong>{round.round_name}</strong>
                    <p>{round.summary}</p>
                  </div>
                  <span>{round.score.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        <section className="report-section">
          <h2>技能评分</h2>
          <div className="skill-list">
            {skillScores.map((skill) => (
              <div className="skill-item" key={skill.skill_name}>
                <div className="skill-head">
                  <strong>{skill.skill_name}</strong>
                  <span>{skill.score}/10</span>
                </div>
                <div className="score-track">
                  <div style={{ width: `${Math.max(8, skill.score * 10)}%` }} />
                </div>
                <p>{skill.evidence}</p>
              </div>
            ))}
          </div>
        </section>

        <ReportList title="优势表现" items={props.report.strengths ?? []} positive />
        <ReportList title="改进建议" items={props.report.improvements ?? []} />

        {typeof props.report.coverage?.scored_questions === "number" ? (
          <section className="report-section span-2">
            <h2>考察覆盖</h2>
            {props.report.total_questions != null && <p>最终评价题数：{props.report.total_questions}</p>}
            <p>
              已评分 {props.report.coverage.scored_questions ?? 0} 题，未评分 {props.report.coverage.unscored_questions ?? 0} 题。
            </p>
            {(props.report.coverage.uncovered_skills ?? []).length ? (
              <p>未覆盖能力：{props.report.coverage.uncovered_skills?.join("、")}</p>
            ) : null}
          </section>
        ) : null}

        {props.report.missed_knowledge?.length ? (
          <section className="report-section span-2">
            <h2>遗漏知识点</h2>
            <div className="missed-list">
              {props.report.missed_knowledge.map((item) => (
                <details key={item.question}>
                  <summary>
                    <span>{item.question}</span>
                    <strong>{item.score == null ? "未评分" : `${item.score}/10`}</strong>
                  </summary>
                  <ul>
                    {item.missed_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                  <p>{item.reference_answer}</p>
                </details>
              ))}
            </div>
          </section>
        ) : null}

        {props.report.study_suggestions?.length ? (
          <section className="report-section span-2">
            <h2>学习建议</h2>
            <div className="suggestion-list">
              {props.report.study_suggestions.map((item) => (
                <span key={item}>
                  <CheckCircle2 size={16} />
                  {item}
                </span>
              ))}
            </div>
          </section>
        ) : null}

        {props.messages.length ? (
          <section className="report-section span-2">
            <h2>完整面试记录</h2>
            <div className="transcript">
              {props.messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="bubble">
                    <strong>{message.role === "candidate" ? "我的回答" : message.role === "interviewer" ? "AI 面试官" : "系统"}</strong>
                    <p>{message.content}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </section>
  );
}

function ReportList(props: {
  title: string;
  items: { point: string; evidence: string }[];
  positive?: boolean;
}) {
  return (
    <section className="report-section">
      <h2>{props.title}</h2>
      <div className="highlight-list">
        {props.items.map((item) => (
          <article className={props.positive ? "positive" : ""} key={`${item.point}-${item.evidence}`}>
            <strong>{item.point}</strong>
            <p>{item.evidence}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function StatusPill({ connection }: { connection: ConnectionState }) {
  const label: Record<ConnectionState, string> = {
    idle: "未连接",
    connecting: "连接中",
    open: "实时连接",
    closed: "已断开",
    error: "连接异常",
  };
  return (
    <span className={`status-pill ${connection}`}>
      <span />
      {label[connection]}
    </span>
  );
}
