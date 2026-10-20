# 来源与第三方组件

## 项目与数据

所有者在发布准备时说明：“自建，Codex 写的，无（已有许可证）”。这里记录自建与 AI 辅助开发事实，不据此声明所有代码完全原创或已经取得所有可能的第三方授权。

项目 LICENSE 尚未选定，本轮不自动添加。所有者需确认拟采用许可证、版权主体与年份，以及是否曾输入/复制第三方代码、提示词、教程或模板；如有，应按具体来源保留署名与条款。不通过修改 Git 历史隐去来源。

历史题库 162 条、知识 40 条在此前审查中没有逐条来源/许可字段。此次没有复制它们进入公开验证目录；需要明确它们是自行编写、AI 生成、改编还是外部导入，以及输入素材来源。`data/question_bank/mockview_demo.json` 与 `tests/fixtures/query_rewrite.json` 是本轮新写的独立合成材料，没有从历史模型完整响应抽取。历史 eval 原始输出、外部网页快照和私人学习材料不属于公开清单。

## 已识别的组件

| 组件 | 核查入口 | 处理 |
| --- | --- | --- |
| FastAPI | [官方许可证](https://github.com/fastapi/fastapi/blob/master/LICENSE) | MIT；使用包依赖，保留包内许可 |
| LangGraph | [官方许可证](https://github.com/langchain-ai/langgraph/blob/main/LICENSE) | MIT；使用包依赖，不宣称自行实现框架 |
| Chroma | [官方许可证](https://github.com/chroma-core/chroma/blob/main/LICENSE) | Apache-2.0；分发时检查对应安装版本的 LICENSE/NOTICE |
| React / React DOM / Scheduler | npm 锁定包的 LICENSE | 运行时前端组件，包内 MIT 文本已汇入静态 notices |
| Lucide React 0.468.0 | 安装包 LICENSE 与 [上游说明](https://github.com/lucide-icons/lucide/blob/main/LICENSE) | 包为 ISC，包含 Feather 来源说明；不得仅标“自制图标” |
| PostgreSQL、pgvector、Redis、Nginx、Python/Node 镜像 | 对应镜像/精确版本的许可证与分发内容 | 独立部署依赖；镜像再分发前需连同系统包核查，不用项目许可证覆盖它们 |
| 其他 Python/npm 传递依赖 | 实际安装清单、包元数据与许可证文件 | 直接依赖声明不等于完整 SBOM 或授权审计；发布二进制/镜像前继续确认 |

`web/public/third-party-notices.txt` 保存当前锁定前端运行时依赖的包内许可原文，Vite 构建后提供 `/third-party-notices.txt`。这是第三方署名，不是 Mockview 的项目 LICENSE。更新包版本时同步核对 notices。

没有附带第三方截图、头像、商标或录屏；README 演示说明不冒充真实效果素材。链接上游主分支用于查找，具体再分发条款以实际锁定版本为准。本表是工程清单，不代表覆盖全部权利链的法律结论。
