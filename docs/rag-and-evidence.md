# RAG 与证据约束

核心文件：`app/rag/{chunking,retriever,postprocess,query_builder,reranker}.py`、`app/agents/evidence_pipeline.py`、`job_assessment.py`、`question_planner.py`。

## 默认检索

初始化脚本将每个结构化题目拆为题干、摘要、回答点、追问方向和生成查询等 child chunk，保存 parent ID、原题及 chunk_text。Embedding 入 Chroma；运行时读取同一集合构建 BM25。中文分词包含连续汉字二元组合，保留 C++/C#/.NET 等标识。

查询分别进入向量与 BM25 分支，以 RRF 的 `Σ 1/(k+rank)` 合并，默认 k=60。rank 是排名，融合分不是模型置信度。随后按父题标识去重、补全父题上下文并进行元数据重排。`legacy` 查询和 ranking 是默认策略；target_expansion、bounded/normalized 是可选对照策略，不因名称较新而默认启用。

RetrievalOutcome 区分 hybrid/vector_only/bm25_only、分支失败、空结果和缓存。外部检索故障与“确实没有资料”不能混成同一个结论；失败策略和公开错误由检索/运行层管理。父题补全用于理解上下文，并不让每个 child 自动获得整个父题所有事实的引用权。

## 证据充分性与出题门禁

AssessmentTarget 区分技术知识与个人经历缺口。EvidenceChunk 保存 document/chunk ID、来源类型、作用域、发布状态和定位。确定性评估检查目标身份与实质支持，结果可为 sufficient/partial/insufficient/conflicting；未解决目标保留在 coverage 中。

技术资料不能证明候选人实际做过某事。个人经历缺口需要用户澄清；JD 技能出现、简历关键词或某篇公开文章不自动构成已验证履历。默认准备入口只把证据充分目标传给题目规划。

题目有稳定 question_id、target_id、评分点和 rubric_citations。每个评分点只可引用检索到且具备合法可定位正文的 chunk；不存在的 source_id、仅靠标题包装或父题附加元数据的支持不能算通过。父题原文只能按真实 parent ID 引用。确定性规则不等于事实核查完备性，真实模型仍可能输出语义不当题目。

## 可选 Agentic RAG

`ASSESSMENT_AUTONOMOUS_LOOP_ENABLED=false` 为默认。开启后，JobAssessmentAgent 对证据缺口通过受限工具选择读取材料、观察、检索知识、公开搜索、请求澄清或结束评估。工具执行绑定 user/case/revision/operation，工具调用输入摘要与结果入账；同一调用可复用，且受 decisions、model_requests、searches、extracts、semantic_checks 和超时上限约束。

公开搜索还需 `PUBLIC_SEARCH_ENABLED=true` 与对应服务凭据。候选公开知识经过审核/发布状态，不能把私人材料直接转成公共语料；发布有独立 release 和索引切换流程。`ASSESSMENT_SEMANTIC_ENABLED`、`QUESTION_SEMANTIC_VALIDATION_ENABLED` 与 `QUESTION_SEMANTIC_SHADOW_ENABLED` 独立配置；shadow 只观察，不能改变主流程判定。

默认关闭原因是尚无充分、可公开复现的真实模型闭环效果与成本证据支持默认启用。保留实现便于学习和有界实验，不宣称它稳定优于确定性主流程。历史实验原始输出保留本地；公开验证以合成契约测试为主。
