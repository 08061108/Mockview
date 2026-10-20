# 脚本入口与支持范围

主应用只推荐 React + FastAPI；脚本不应绕过账户归属或业务 API 作为第二套产品入口。

| 分类 | 文件/模式 | 当前处理 |
| --- | --- | --- |
| 受支持运行 | `run_api.py`、`init_vector_store.py` | 保留；宿主启动与索引初始化，初始化会调用 Embedding，reset 会重建集合 |
| 可选检索评测 | `evaluate_retrieval.py`、`evaluate_rag_stabilization.py`、`evaluate_bm25_tokenization.py` | 保留源码；运行需要与案例匹配的获授权语料/索引。quality_compare 被测试导入，不能误删 |
| 可选 RAGAS | `evaluate_ragas.py`、`evaluate_rag_variants.py`、`compare_ragas_results.py`、`run_ragas_experiments.py` | 保留工具；额外 eval 依赖，真实评测可能付费；旧 golden/原始结果不随默认公开集合提供 |
| 历史 CLI/Gradio | `run_interview_cli.py`、`../frontend/` | 标记不受支持，原文件保留本地，建议后续经确认归档；不宣传可用 |
| 个人验收/诊断 | `accept_*`、`probe_*`、`diagnose_model_transport.py`、`validate_real_agent_tool_calling.py`、`verify_*`、`run_frozen_p0.py`、`p0_sitecustomize.py`、`inspect_memory.py`、`test_vision.py`、`test_ws_client.py`、`test_retrieval.py` | 保留本地并列在私人清单；可能读取个人配置/记忆、打印模型正文或依赖历史数据，不能直接当公开测试命令 |
| 历史汇总或实验 | 其余 evaluate_*、`evaluate.py`、`eval_usage.py` | 逐文件分类见个人发布报告；不因实验默认关闭删除实现，也不整体保证脚本可在缺少历史产物时运行 |

本轮没有删除、移动任何脚本。公开可保留的脚本也须读其参数和数据入口，不要对现有个人目录直接跑诊断。完整项目测试与演示以 [验证文档](../docs/evaluation.md) 为准。
