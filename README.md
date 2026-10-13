# Mockview

Mockview 是一个由岗位 JD 与候选人简历驱动的智能面试练习平台。它通过 LangGraph 编排分析、出题、追问和评估流程，结合 RAG 题库检索、长期记忆与实时 WebSocket 对话，帮助开发者完成可复盘的技术面试练习。

> 当前仓库处于个人 Vibe Coding 开发阶段。项目保留了已经验证通过的原始业务能力，并从 Mockview 的独立仓库历史开始继续演进。

## 产品定位

- 面向技术岗位的个人面试练习与复盘工具。
- 根据 JD、简历项目与历史薄弱项生成个性化问题。
- 支持实时追问、提前结束、结构化评分和学习建议。
- 提供 Practice 练习模式与 Professional 双轮面试模式。

Mockview 不替代真实招聘判断；模型生成的问题、评分与建议应作为练习参考。

## 已有功能

- 用户注册、登录与 JWT 鉴权。
- 创建 Practice / Professional 面试会话。
- PDF、PNG、JPG、JPEG 简历解析与 JD 匹配。
- LangGraph 多 Agent 工作流与 PostgreSQL checkpoint。
- ChromaDB 向量检索、BM25、RRF、多查询和元数据重排。
- WebSocket 实时问题、流式内容、动态追问与状态同步。
- Redis 限流、回答锁、会话缓存、报告缓存和在线状态。
- PostgreSQL 会话、报告与长期记忆持久化，pgvector 语义记忆。
- 阶段性或最终面试报告，以及 Markdown 导出。
- REST API、React Web、CLI 与 Gradio 辅助界面。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| Web | React 19、TypeScript、Vite 6、Nginx |
| API | FastAPI、REST、WebSocket、Pydantic v2 |
| Agent | LangGraph StateGraph、条件路由、interrupt、checkpoint |
| 模型 | OpenAI-compatible Chat、Structured Output、Streaming、Vision、Embedding |
| RAG | ChromaDB、BM25、RRF、parent-child chunking、rerank |
| 数据 | PostgreSQL、pgvector、Redis；本地 fallback 使用 SQLite / ChromaDB |
| 简历 | pdfplumber、Vision OCR fallback、链接抽取 |
| 测试 | pytest、pytest-asyncio、TypeScript、Vite build |
| 部署 | Docker Compose、PostgreSQL/pgvector、Redis、FastAPI、Nginx |

## 运行架构

```text
React / Vite / Nginx
        │ REST + WebSocket
        ▼
FastAPI ── JWT / Redis runtime layer
        │
        ▼
SessionManager ── LangGraph interview workflow
        │             ├─ JD / resume analysis
        │             ├─ RAG question planning
        │             ├─ interviewer / follow-up
        │             └─ evaluator / report
        │
        ├─ PostgreSQL + pgvector
        ├─ ChromaDB question index
        ├─ Redis
        └─ OpenAI-compatible model APIs
```

详细设计见 [架构文档](docs/architecture.md) 与 [部署指南](DEPLOYMENT.md)。

## 快速开始

### 1. 配置环境变量

```bash
git clone https://github.com/08061108/Mockview.git mockview
cd mockview
cp .env.example .env
```

至少配置：

```env
LLM_API_KEY=your-llm-api-key
EMBEDDING_API_KEY=your-embedding-api-key
AUTH_SECRET_KEY=replace-with-a-long-random-secret
```

不要提交 `.env`。如果继续使用重命名前已经初始化的数据库，可保留现有 `.env` 中的数据库名、用户和连接 URL；Mockview 的默认新部署标识不会自动迁移或删除旧数据。

### 2. Docker Compose 完整启动

```bash
docker compose build
docker compose run --rm api python -m scripts.init_vector_store --reset
docker compose up -d
docker compose ps
```

访问：

- Web：<http://127.0.0.1>
- API 文档：<http://127.0.0.1:8000/docs>
- API 健康检查：<http://127.0.0.1:8000/health>

`init_vector_store --reset` 会调用配置的 Embedding 服务，并重建本地 Chroma 索引。已有可用索引时无需重复执行。

### 3. Windows + Conda 本地开发

本项目当前验证环境为 Conda `agentic`、Python 3.11。PostgreSQL 与 Redis 可由 Docker 提供：

```powershell
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d postgres redis
```

启动 API：

```powershell
conda activate agentic
python -m scripts.run_api
```

`scripts.run_api` 使用 Windows Selector event loop，避免 psycopg 异步连接与默认 Proactor event loop 的兼容问题。

如果系统级代理不能访问模型服务，可仅对当前终端清除代理变量后再启动，不需要修改系统设置：

```powershell
$env:HTTP_PROXY=$null
$env:HTTPS_PROXY=$null
$env:ALL_PROXY=$null
python -m scripts.run_api
```

启动 Web：

```powershell
cd web
npm install
npm run dev
```

访问 <http://127.0.0.1:5173>。

## 测试与构建

```powershell
conda activate agentic
pytest -q

cd web
npm run build
```

后端测试默认使用离线替身，不要求在测试阶段调用真实模型。`app/rag/embeddings.py` 延迟创建 SDK 客户端，因此未配置密钥时仍可加载离线测试与存储组件，真正调用 Embedding 时会给出明确错误。

## 数据与兼容性

- `.env`、`postgres_data/`、`redis_data/`、`chroma_data/`、`memory_data/`、`uploads/` 和 `exports/` 均不进入 Git。
- Compose 使用工作区内的 bind mount 保存 PostgreSQL、Redis、Chroma、上传与导出数据。
- 项目重命名不会修改业务 API 路径、数据表结构、会话 ID 或已有数据库内容。
- 新 Compose 项目名为 `mockview`；首次切换前应停止旧 Compose 容器，但不要删除数据目录或执行 `down -v`。

## 项目结构

```text
mockview/
├── app/
│   ├── agents/          # LangGraph 节点与状态图
│   ├── api/             # REST / WebSocket 路由
│   ├── cache/           # Redis 限流、锁、缓存与在线状态
│   ├── llm/             # OpenAI-compatible 模型客户端
│   ├── memory/          # 长期记忆与 pgvector / Chroma fallback
│   ├── rag/             # 检索、切块、重排与评估辅助
│   ├── resume/          # 简历解析与匹配
│   └── services/        # 会话、持久化与报告导出
├── web/                 # React / Vite Web 客户端
├── frontend/            # Gradio 辅助调试界面
├── data/                # 题库、知识、示例 JD 与评估集
├── docs/                # 架构和专项设计文档
├── scripts/             # 启动、初始化、评估与调试脚本
├── tests/               # 后端测试
├── docker-compose.yml
└── requirements.txt
```

## 开发计划

1. **基线稳定化**：修复 WebSocket 断线边界、完善配置校验和部署诊断。
2. **练习体验**：改进问题进度、历史报告检索和中断恢复反馈。
3. **评估可信度**：扩充检索 golden set，校准评分并记录可核验的评估证据。
4. **个人发布**：完善安全配置、CI、镜像发布与可观测性说明。

上述计划不改变当前 API 或数据结构，新增功能应在独立变更中实现。

## 相关文档

- [部署指南](DEPLOYMENT.md)
- [架构文档](docs/architecture.md)
- [FastAPI / WebSocket Contract](docs/fastapi_websocket_contract.md)
- [Redis 设计](docs/redis_design.md)
- [RAG 设计](docs/rag_design.md)
- [RAG 评估报告](docs/rag_eval_report.md)
- [评分规则](docs/scoring_rubric.md)

## 授权

当前 Mockview 仓库暂未声明项目级开源许可证。第三方依赖继续遵循其各自许可证；依赖锁文件中的许可证元数据保持不变。
