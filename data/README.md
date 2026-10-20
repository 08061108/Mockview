# 数据范围

拟公开数据只有 `question_bank/mockview_demo.json`：本轮新写的合成 RAG 示例，用于索引初始化与机制演示，不是质量评测集。只有一个主题，不能据此宣称覆盖任意 JD。

历史 `question_bank/*.json`、`knowledge/`、`eval/` 与压缩包保留本地，来源与再分发范围待确认。已有 Git 跟踪不因忽略规则自动解除；发布前必须逐项处理清单。不要整目录暂存或上传。

当前初始化脚本读取 `data/question_bank/*.json` 和 `data/knowledge/*.json`。在现有开发工作区执行会读到本地历史材料；公开源码验证必须使用经清单选取的干净目录。新用户可在自己的目录加入自有或获授权材料，沿用示例 schema，然后在独立 Chroma 目录初始化。可选知识发布机制产生的 `knowledge/releases/` 是运行数据，不属于源码。
