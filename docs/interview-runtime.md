# 面试运行时

## 图与渐进出题

`graph.py` 定义 practice/professional 两张图。准备入口将已分析的技能矩阵、简历和证据快照传入图；兼容快速入口可能重新分析，快速 practice 不具备同等简历处理行为。

practice 主干为规划 → generate_question → ask_question → interrupt_before assess_answer → assess_answer → 追问或 finalize_question → advance_question / evaluate_practice。professional 在第一轮总结后规划第二轮，再综合评价。蓝图与具体题目分离，当前题按需生成；预取最多占用受限并发，不一次生成整场所有题。

## 答案、追问与评分

稳定的 question_id、turn_id、answer_id 标识题、回合、提交。SessionManager 记录 pending answer 和处理结果，重复答案可恢复或复用；Redis 锁配合进程内任务控制，不应据此宣称支持任意多 worker。

过程评价决定是否追问、追问类别和方向；最终单题评价读取同题主答与追问全轨迹。提示依赖、替代方案、可评分性与证据单独记录。综合报告的数值来自最终题评价，经确定性技能聚合覆盖模型草稿；professional 完整两轮按既有 55%/45% 权重组合。

不可评分用 null，不是零分；题数按最终评价数计，含已评分与未评分，未回答且未结算的规划题不计入。MissedKnowledge 的分数/遗漏点由最终评价重建，模型参考答案只有稳定 ID 或唯一精确题面匹配时才可使用。自然语言叙述仍受模型质量限制。

## 持久化、暂停与恢复

LangGraph PostgreSQL checkpoint 保存图状态；业务 Session 表保存归属、模式、材料版本、当前状态和报告。会话事件有 sequence，WebSocket 重连可用 after_sequence 重放；重连重新发送 auth 首帧，不通过 URL 传 JWT。

暂停/提前结束由 SessionManager 协调正在运行的任务；恢复使用已有 checkpoint、稳定 turn 与 pending answer 记录。提前结束需要等待相关回答结算，不能只拼接过程分。API 进程退出时可能将未完成工作标为可恢复，不保证每种外部故障自动无感继续。

准备 HITL 与面试 interrupt 分属两个流程：前者等待材料补充，后者等待当前回答。GET progress 只读当前 revision 的原 operation；回答澄清后继续同一 operation，避免重复材料分析和预算重置。浏览器只恢复当前账号最近准备任务；跨设备任务发现未实现。

## 幂等与预算

| 操作 | 当前幂等边界 |
| --- | --- |
| 复练 case | user + 来源 Session + revision + 排序技能集合 |
| 准备分析 | 相同 case/revision 返回既有 operation |
| 从 case 启动面试 | 有 request_id 时形成稳定 Session ID；同 ID 不同参数拒绝；省略仍允许随机新会话 |
| 澄清答案 | 相同已提交答案不再次追加材料；不同答案须通过材料更新 |
| 回答 | answer_id 与 turn_id、pending 状态及处理任务协作 |

LLM transport retries、业务 request attempts 与结构化修复有各自上限，不能简单相乘后宣称总预算无限。实际物理模型请求计入 operation 预算；超时包括排队与调用边界。准备默认 240 秒，简历解析 60 秒；启动/出题/回答另有配置。刷新不扣预算、不重做分析；真实续跑可能消费剩余预算；失败继续不会自动补回预算。

没有引入独立任务队列或多实例协调器。当前锁、内存后台任务、会话管理共同依赖单 worker 边界；多 worker 属未来架构工作，不是本阶段功能。
