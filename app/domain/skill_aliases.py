"""Deterministic technical-skill aliases shared by matching and retrieval."""

from __future__ import annotations

import re


_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rag", "retrieval augmented generation", "检索增强生成", "检索增强"),
    ("查询改写", "query rewriting", "query rewrite", "rewrite query"),
    ("bm25", "关键词检索", "keyword retrieval"),
    ("embedding", "向量", "向量检索", "vector search", "语义检索"),
    ("redis", "缓存", "cache"),
    ("分布式锁", "distributed lock", "redis lock"),
    ("消息队列", "message queue", "mq", "kafka", "rocketmq", "rabbitmq"),
    ("幂等", "idempotency", "idempotent"),
    ("顺序消息", "message ordering", "ordered message"),
    ("langgraph", "state graph", "状态图"),
    ("agent", "智能体", "tool calling", "工具调用"),
    ("websocket", "长连接", "双向通信"),
    ("mysql", "关系型数据库", "relational database"),
    ("mvcc", "多版本并发控制", "multi version concurrency control"),
    ("python", "py"),
    ("asyncio", "异步io", "异步 i/o", "协程", "coroutine"),
    ("gil", "全局解释器锁", "global interpreter lock"),
    ("docker", "容器", "container"),
    ("kubernetes", "k8s", "容器编排"),
    ("微服务", "microservice", "microservices"),
    ("系统设计", "system design", "架构设计"),
    ("可观测性", "observability", "监控", "metrics", "tracing"),
    ("多租户", "multi tenant", "multi-tenant", "tenant isolation", "租户隔离"),
    ("断线重连", "reconnect", "connection recovery"),
    ("成本治理", "cost governance", "token budget", "模型预算"),
    ("数据质量", "data quality"),
    ("流处理", "stream processing", "event time", "事件时间"),
    ("容错", "fault tolerance", "failure recovery"),
)


def normalize_skill_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff+#.]", "", value.casefold())


def aliases_for_skill(skill: str) -> tuple[str, ...]:
    normalized = normalize_skill_text(skill)
    matched: list[str] = []
    for group in _ALIAS_GROUPS:
        normalized_group = {normalize_skill_text(item) for item in group}
        if normalized in normalized_group or any(
            item and (item in normalized or normalized in item)
            for item in normalized_group
        ):
            matched.extend(group)
    values = [skill, *matched]
    return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))


def find_skill_mention(text: str, skill: str) -> str | None:
    normalized_text = normalize_skill_text(text)
    for alias in aliases_for_skill(skill):
        normalized_alias = normalize_skill_text(alias)
        if normalized_alias and normalized_alias in normalized_text:
            return alias
    return None


def retrieval_aliases(skills: list[str], *, limit: int = 8) -> list[str]:
    values: list[str] = []
    for skill in skills:
        for alias in aliases_for_skill(skill):
            normalized = normalize_skill_text(alias)
            if normalized and normalized not in {
                normalize_skill_text(item) for item in values
            }:
                values.append(alias)
            if len(values) >= limit:
                return values
    return values
