"""叫停机制：让机器人别再提某件事。

问题：记忆里有「进行中的话题」和「想说但还没说」，提示词又鼓励"追问上次没聊完的事"。
一旦某个话题被反复记起，就会变成执念——用户说"别管这个了"，下一轮它又拎出来。

做法：记一份"别再提"的关键词表（按会话），在**三处**同时生效：

1. **注入记忆时过滤**：命中关键词的话题/念头不写进提示词（记录仍在，解除后自动回来）。
2. **提示词里明说**：给一段"不要提的话题"，并规定不许主动带回来、不许解释"我被要求别提"。
3. **主动开话题时避开**：同一段提示词对 proactive 也生效，因为它是每轮都拼的。

为什么用关键词而不是"删除记忆"：话题一旦被删，模型下次照样能从上下文里重新学回来；
而关键词表是**独立于记忆**的约束，删不掉、也不会被沉淀覆盖。解除后记忆自动恢复，
不会造成不可逆的数据丢失。

生命周期：默认 7 天自动解除；`ttl_hours=0` 表示永久，直到手动清除。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .store import JsonStore, now_ms
from .text import clean, truncate

__all__ = ["SuppressBank", "SuppressStore", "normalize_keyword", "parse_suppress_request"]

# 常见动词：说/提/聊/讲/谈/扯/念叨/琢磨/纠结/惦记/管
_VERBS = r"(?:说|提|聊|讲|谈|扯|念叨|琢磨|纠结|惦记|管|问)"
_FILLER_LEAD = ("这个", "那个", "关于", "有关", "什么", "一下")
_FILLER_TAIL = ("的事情", "的事", "事情", "的话题", "话题", "这件事", "那次", "了", "啦", "吧", "啊", "哈", "呢")

# 指代词解析出来的"关键词"没有信息量，不能当叫停对象。
_GENERIC = frozenset(
    {
        "话题", "事情", "事", "东西", "这个", "那个", "这个事", "那个事",
        "这些", "那些", "这类", "那类", "它", "什么", "梗",
    }
)

_PATTERNS = (
    # 别再说世界杯了 / 别再提那个事了 / 别老是聊这个了
    re.compile(rf"别(?:再|老是|总|一直|又)?{_VERBS}(?:这个|那个|一下)?(?P<x>[^，。！？,.!?、\s]{{2,24}}?)(?:了|啦|吧|啊|行不行|好不好)"),
    # 不要再提世界杯了 / 不要聊这个话题了
    re.compile(rf"不要(?:再|老是|总|一直)?{_VERBS}(?:这个|那个|一下)?(?P<x>[^，。！？,.!?、\s]{{2,24}}?)(?:了|啦|吧|啊)"),
    # 别再管世界杯了
    re.compile(rf"别管(?P<x>[^，。！？,.!?、\s]{{2,24}}?)(?:了|啦|吧)"),
    # 以后别提世界杯
    re.compile(rf"(?:以后|之后)(?:别|不要)(?:再)?{_VERBS}(?P<x>[^，。！？,.!?、\s]{{2,24}})"),
)


def _clean_keyword(raw: str) -> str:
    text = clean(raw)
    # 去掉 @ 残留、引号、标点
    text = re.sub(r"^[@\s\"'“”‘’「」『』]+|[@\s\"'“”‘’「」『』]+$", "", text)
    # 反复剥掉"这个/那个…的事/话题"，直到稳定
    changed = True
    while changed:
        changed = False
        for lead in _FILLER_LEAD:
            if text.startswith(lead) and len(text) > len(lead):
                text = text[len(lead) :]
                changed = True
        for tail in sorted(_FILLER_TAIL, key=len, reverse=True):
            if text.endswith(tail) and len(text) > len(tail):
                text = text[: -len(tail)]
                changed = True
    text = text.strip()
    if normalize_keyword(text) in _GENERIC:
        return ""
    return text


def parse_suppress_request(text: object) -> str | None:
    """从一句话里解析出"别再提什么"。

    Returns:
        需要叫停的关键词；不是叫停请求时返回 ``None``。

    Examples:
        >>> parse_suppress_request("别再说世界杯了")
        '世界杯'
        >>> parse_suppress_request("别再提那个抽卡的事了")
        '抽卡'
        >>> parse_suppress_request("今天吃什么")
    """
    body = clean("" if text is None else str(text))
    if not body:
        return None
    for pattern in _PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        keyword = _clean_keyword(match.group("x"))
        if 2 <= len(keyword) <= 24:
            return keyword
    return None


def normalize_keyword(text: object) -> str:
    """归一化：去空白与标点，小写。用于匹配。"""
    raw = clean("" if text is None else str(text)).lower()
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】]+", "", raw)


def _default_state() -> dict[str, Any]:
    return {"items": []}


class SuppressStore:
    """单个会话的"别再提"列表。"""

    def __init__(
        self,
        path: Path,
        *,
        ttl_hours: float = 168.0,
        max_items: int = 30,
        on_warning=None,
    ) -> None:
        self.store = JsonStore(path, _default_state, on_warning=on_warning)
        self.ttl_ms = int(max(0.0, float(ttl_hours)) * 3600 * 1000)
        self.max_items = max(1, int(max_items))
        self._items: list[dict[str, Any]] = []
        raw = self.store.load()
        rows = raw.get("items") if isinstance(raw, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                keyword = str(row.get("text") or "").strip()
                if not keyword:
                    continue
                self._items.append(
                    {
                        "text": truncate(keyword, 24),
                        "by": truncate(str(row.get("by") or ""), 40),
                        "createdAt": int(row.get("createdAt") or now_ms()),
                        "expiresAt": int(row.get("expiresAt") or 0),
                    }
                )
        self._items = self._items[-self.max_items :]
        self.dirty = False

    # ---------------------------------------------------------------- 存取
    def save(self) -> None:
        self._prune()
        self.store.save({"items": self._items})
        self.dirty = False

    def save_if_dirty(self) -> None:
        if self.dirty:
            self.save()

    def _prune(self) -> None:
        now = now_ms()
        kept = [item for item in self._items if not item["expiresAt"] or now < item["expiresAt"]]
        if len(kept) != len(self._items):
            self._items = kept
            self.dirty = True

    # ---------------------------------------------------------------- 写入
    def add(self, keyword: object, by: str = "") -> bool:
        text = _clean_keyword(str(keyword or ""))
        if len(text) < 2:
            return False
        self._prune()
        norm = normalize_keyword(text)
        for item in self._items:
            if normalize_keyword(item["text"]) == norm:
                # 已存在：刷新有效期
                item["expiresAt"] = now_ms() + self.ttl_ms if self.ttl_ms > 0 else 0
                item["by"] = truncate(clean(by), 40) or item.get("by", "")
                self.dirty = True
                return True
        self._items.append(
            {
                "text": truncate(text, 24),
                "by": truncate(clean(by), 40),
                "createdAt": now_ms(),
                "expiresAt": now_ms() + self.ttl_ms if self.ttl_ms > 0 else 0,
            }
        )
        self._items = self._items[-self.max_items :]
        self.dirty = True
        return True

    def remove(self, keyword: object = "") -> int:
        if not keyword:
            removed = len(self._items)
            self._items = []
        else:
            norm = normalize_keyword(keyword)
            before = len(self._items)
            self._items = [
                item for item in self._items if normalize_keyword(item["text"]) != norm
            ]
            removed = before - len(self._items)
        if removed:
            self.dirty = True
        return removed

    # ---------------------------------------------------------------- 匹配
    def active(self) -> list[dict[str, Any]]:
        self._prune()
        return list(self._items)

    def match(self, text: object) -> str | None:
        """这段文本命中了哪条叫停关键词。"""
        body = normalize_keyword(text)
        if not body:
            return None
        for item in self.active():
            keyword = normalize_keyword(item["text"])
            if keyword and keyword in body:
                return str(item["text"])
        return None

    def blocks(self, text: object) -> bool:
        return self.match(text) is not None

    def filter(self, texts: Iterable[object]) -> list[object]:
        return [t for t in texts if not self.blocks(t)]

    def hidden_count(self, texts: Iterable[object]) -> int:
        return sum(1 for t in texts if self.blocks(t))

    # ---------------------------------------------------------------- 展示
    def format_for_prompt(self) -> str:
        items = self.active()
        if not items:
            return ""
        lines = ["【不要提的话题（有人明确叫停过）】"]
        for item in items:
            by = item.get("by") or "群友"
            lines.append(f"- 「{item['text']}」（{by} 说别再提了）")
        lines.append(
            "（这些话题不要再主动提、不要追问、不要拿它们举例；群友自己聊到时可以接一两句，"
            "但不要主动带回来。也不要解释「我被要求别提这个」，就当自己没想到。）"
        )
        return "\n".join(lines)

    def summary(self) -> str:
        items = self.active()
        if not items:
            return "没有叫停的话题。"
        now = now_ms()
        lines = ["当前叫停的话题："]
        for item in items:
            left = ""
            if item["expiresAt"]:
                hours = max(0.0, (item["expiresAt"] - now) / 3_600_000)
                left = f"，{hours:.0f} 小时后自动解除" if hours >= 1 else "，即将解除"
            lines.append(f"- 「{item['text']}」{('（' + item['by'] + ' 叫停）') if item.get('by') else ''}{left}")
        return "\n".join(lines)


class SuppressBank:
    """按会话管理叫停列表。"""

    def __init__(self, directory: Path, options: dict[str, Any] | None = None, on_warning=None) -> None:
        self.directory = Path(directory)
        self.options = dict(options or {})
        self.on_warning = on_warning
        self._cache: dict[str, SuppressStore] = {}

    def get(self, session_key: str) -> SuppressStore:
        key = _safe(session_key)
        store = self._cache.get(key)
        if store is None:
            store = SuppressStore(
                self.directory / f"{key}.json",
                ttl_hours=self.options.get("ttl_hours", 168.0),
                max_items=self.options.get("max_items", 30),
                on_warning=self.on_warning,
            )
            self._cache[key] = store
        return store

    def flush(self) -> None:
        for store in self._cache.values():
            store.save_if_dirty()


def _safe(key: str) -> str:
    out = [ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(key or "default")]
    name = "".join(out).strip("_") or "default"
    return name[:96]
