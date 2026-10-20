# 部署与本地开发

官方推荐根目录 `docker-compose.yml` 的 PostgreSQL、Redis、API、Web 四服务。完整应用保持**单 API 进程、单 worker**。账户始终需要 PostgreSQL，准备任务 admission 依赖 Redis；不能把 Redis 当成完整应用的可选增强。

## 版本与依赖

| 组件 | 入口 |
| --- | --- |
| Python | 3.11，与 Dockerfile 一致 |
| 默认运行依赖 | `requirements.txt` 引用 `requirements.lock` 的精确直接依赖 |
| 测试 | `requirements-dev.txt` |
| 可选历史 RAGAS | `requirements-eval.txt`，不进入默认镜像 |
| 历史 Gradio | `requirements-legacy.txt`，不保证旧协议兼容 |
| Node | 22.x，`npm ci` 使用 package-lock.json |
| 数据库 | PostgreSQL 16 + pgvector；Redis 7 |
| Compose | 2.24+，支持可选 env_file |

pyproject 从同一 requirements.lock 读取运行依赖。该文件是**直接依赖锁**，不是完整传递依赖哈希锁；基础镜像、apt 与传递依赖仍会变化。实际干净构建结果见 [验证](docs/evaluation.md)。

## 新目录启动

将 `.env.example` 复制为 `.env`。配置 Chat、Embedding 的 key/base/model，以及可选 Vision；服务须支持所用流式和结构化输出接口。实验自动循环还需 tool calling。示例中的具体模型名只是一份配置样例，不保证服务可用。

分别生成独立 `AUTH_SECRET_KEY` 和 URL-safe `POSTGRES_PASSWORD`：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

真实值只保存在本地。JWT 空值、弱值和常见占位符会阻止 API 启动；数据库密码不要使用模板值。前端 VITE 变量是公开构建时变量，不得放密钥。

```bash
docker compose build
docker compose up -d postgres redis
docker compose run --rm api python -m scripts.init_vector_store
docker compose up -d api web
docker compose ps
```

初始化读取 `data/question_bank/*.json` 和 `data/knowledge/*.json`，会调用 Embedding。拟公开包仅包含新写的 `mockview_demo.json`，历史开发目录中仍有未确认材料，不能直接打包。`--reset` 是破坏性重建选项，不用于正常启动。改变 Embedding 模型/维度后应使用独立新索引，不能混用向量。

API 自动建立账户、业务、迁移和 checkpoint 表。`/health` 只是健康/能力状态检查，不代表真实模型请求和面试效果通过。默认 Web 为 localhost:80，宿主 API 仅绑定 127.0.0.1:8000；端口可由 `WEB_PORT`、`API_HOST_PORT` 修改。`VITE_API_BASE_URL=/api` 通过 Nginx 同源代理，前后端需一起更新。

## 运行数据和镜像

| 目录 | 用途 |
| --- | --- |
| postgres_data/ | 账户、准备任务、Session、checkpoint、长期记忆/pgvector |
| redis_data/ | 限流、锁、在线状态与缓存持久化 |
| chroma_data/ | 题库索引 |
| data/question_bank/ | 只读题库输入 |
| data/knowledge/ | 额外知识及可选发布快照，不默认公开 |
| uploads/、exports/ | 用户上传与导出，不进入 Git 或镜像 |

Dockerfile 只复制 app、两个运行脚本、依赖和合成示例；.dockerignore 采用允许清单。Compose 仅挂载题库、知识与各运行数据目录，不挂载整个 data。忽略规则不能过滤 bind mount，开发者显式挂载的内容仍会进入运行容器。

已有数据不覆盖。更改 POSTGRES_PASSWORD 环境变量不会更改已初始化数据库密码，更改 Compose 项目名不会迁移 bind mount。备份需覆盖数据库、模型维度相容的索引和自有语料；不要通过删除数据或 `down -v` 排查配置问题。

## 宿主开发

Python 3.11 安装 `requirements-dev.txt`，Node 22 安装 Web。仍在独立目录运行 PostgreSQL/Redis：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d postgres redis
python -m scripts.run_api
```

Windows 的 run_api 使用 Selector event loop，避免 psycopg 与默认 Proactor 不兼容。宿主 POSTGRES_URL 留空时由 POSTGRES_* 字段派生，显式 URL 优先。宿主用 localhost 与映射端口；容器用 postgres:5432、redis:6379。改 REDIS_HOST_PORT 时同时改宿主 REDIS_URL。模板统一使用 PostgreSQL Session/Checkpoint/Memory 与 pgvector；SQLite/Chroma 记忆仍作为离线实验选项保留。

```bash
cd web
npm ci
npm run dev
```

本地 Web 可将 `.env.example` 复制为 `.env.local`，使用 `VITE_API_BASE_URL=http://127.0.0.1:8000/api`；访问 localhost:5173，设置对应 CORS origins。

模型默认不继承系统代理。宿主通过 MODEL_PROXY_URL 显式配置；Compose 通过 DOCKER_MODEL_PROXY_URL 映射，Docker Desktop 可用 host.docker.internal。不要在公开文档或日志中保存个人代理凭据。

## 安全与排障

REST 使用 Bearer JWT；WebSocket 在连接后 5 秒内用 auth 首帧发送 JWT，URL 不接受 token。公网使用 HTTPS/WSS，还需自行设置注册/访问控制和模型额度。本项目未承诺商业多租户治理能力。

上传默认允许 PDF/PNG/JPG/JPEG，10 MiB；Nginx 20 MiB 为 multipart 开销留余量。准备任务默认 240 秒、解析 60 秒，每用户同时一个分析或解析；Redis 不可用时拒绝新准备。实际续跑可消费剩余预算，单纯刷新不会补充或扣减预算。

Nginx access log 使用不含 query 的 URI，不记录鉴权头；原生 error log 关闭以避免不可脱敏的 URI。上游反代也需保持该边界。检查顺序是容器状态、health、合成登录、模型与索引配置、任务公开错误；不要开启含真实正文或鉴权帧的日志。
