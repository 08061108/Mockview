# 验证方法与证据边界

验证分为三层：离线契约测试、真实基础设施配合 Mock 的集成验收、真实模型质量评测。三者不能互相冒充。

## 精选公开测试

使用 Python 3.11 干净环境，并从经发布清单筛选的源码副本运行，目录中不要有真实 `.env`。`app.config` 会加载当前工作目录的 `.env`；不要对含个人配置的开发目录盲目执行测试。最强隔离方式是在容器中安装依赖后用 `--network none` 运行。

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q tests/test_phase3_closure.py tests/test_preparation_api.py tests/test_public_interview_security.py tests/test_closeout_correctness.py
```

| 文件 | 验证对象 |
| --- | --- |
| test_phase3_closure.py | 复练来源、恢复、幂等、评分和公共报告契约 |
| test_preparation_api.py | 准备 API、澄清、用户/版本边界 |
| test_public_interview_security.py | 公共字段隔离、会话/报告投影 |
| test_closeout_correctness.py | 技能聚合、提前结束与逐点评分证据 |
| fixtures/query_rewrite.json | 独立合成引用边界，不读取历史题库或模型完整响应 |

2026-10-20，从无真实 env、无私人数据、无本机安装包的干净源码上下文构建后，在断网容器中执行以上命令：**53 passed，1 项上游弃用警告**。pip check 通过。只替换测试资料依赖，未修改业务图、RAG 或评分实现。

其他测试保留相应代码价值，但不把这次结果称为全部 tests 通过。`test_evidence_regression_frozen_cases.py` 依赖本地历史审计结果和旧语料，暂不进入公开集合；其本地源文件保留。不要为了测试收集通过而上传原始私有 eval/results。

## 干净构建与启动

本轮在筛选后的独立目录执行 API/Web 的 `docker build --no-cache`：API 从 Python 3.11-slim 重新安装运行依赖，Web 从 Node 22-alpine 执行 npm ci、TypeScript 检查和 Vite 构建，均成功。默认镜像不安装 Gradio/RAGAS，未复用旧应用镜像依赖层或本机 node_modules。

另以全新 PostgreSQL/Redis/Chroma 目录启动实际服务，内部 Mock 只支持合成 Embedding 和模型列表。使用公开 demo JSON 初始化了 10 个 chunk；不发起真实 Chat、Vision 或付费模型请求，不重复完整面试验收。个人详细命令、包清单、镜像 digest 和检查结果保留在本地收尾报告；公开声明仅覆盖实际完成的安装与冒烟范围。

Python requirements.lock 锁定直接依赖，不锁全部传递依赖、apt 与镜像 digest；npm package-lock 提供前端解析锁，但不等于无漏洞证明。本次构建成功不代表未来任何平台或网络都可复现；可选 RAGAS 宽版本依赖未作为默认镜像的一部分验证。

2026-10-20 的 npm audit 对原前端锁文件报告 7 项公告命中。仅在现有版本范围内更新锁文件（Vite 6.4.2 → 6.4.4 及相关传递依赖），未改变 package.json 的依赖范围或业务源码；更新后同次 npm 审计为 0 项。该结果仅覆盖当时 npm 公告库与锁文件，不代表 Python、系统包、基础镜像或应用整体无漏洞。

## 已有业务集成证据

此前隔离验收使用真实前端、Nginx、FastAPI、PostgreSQL、Redis、WebSocket 与 Mock LLM。最终增量范围是一条“合成历史 → 弱项 → 复练 case → 证据 → 面试 → 两次回答 → 报告/历史/导出”链路和一个 HITL 案例。验证了首帧鉴权、刷新后原 operation/预算、revision 拒绝、用户隔离、重复启动以及最终分/null 的一致性。

Mock 故意返回与最终单题评价不一致的报告数字，用于证明服务器以最终评价为数值权威。合成历史由夹具植入，不能称为真实用户使用记录；19 个 checkpoint 是该次持久化证据，不是性能指标。

## 可选检索与模型评测

可选 evaluate_retrieval 工具按父题 ID 计算 hit@5、recall@5 和 MRR@10：分别表示是否命中、相关父题召回比例和首个相关结果的倒数排名。需要配套语料、索引、expected IDs 与统一模型配置；更换语料后不能沿用旧 golden ID 或旧分数。

RAGAS、查询策略、排名策略与 Agentic RAG 对照应分别报告样本构成、服务商/模型版本、消耗、失败率和实验开关。历史真实模型原始输出未纳入公开包，不在 README 搬运无法由公开材料复现的提升百分比。

目前未充分验证真实模型的个性化程度、题目质量、评分校准、公平性、学习收益、成本和延迟。默认功能的契约通过不能证明实验自动补查更好；默认关闭开关维持不变。
