"""轻量记忆：进行中的话题 / 想说未说的话 / 对群友的印象。

这是 qq-bridge `formatMemoryV2` + `appendMemoryV2` 的等价移植。
三类记忆的设计意图值得保留：

- ``activeTopic``：**还值得捡回来**的话题。带 ``lastMentionAt``，过 24 小时自动过期，
  超过 2 小时没提起会标「已搁置」——避免模型把三天前的旧话题当成刚聊的。
- ``pendingThought``：**想说但没说**的话。带 ``expiresAt``（默认 2 小时），
  过期就淡出，防止攒一堆“我本来想吐槽”的陈年包袱。
- ``memberImpression``：对**具体群友**的具体印象（是“爱在群里报菜名”而不是“人很好”）。
  带互动次数与最后出现时间，超过 50 人按最久未见淘汰。

每类都有硬上限，因为记忆是每轮都要拼进 prompt 的——不设上限的“记忆”最终会变成
token 黑洞，并把真正重要的近期信息挤出去。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .store import JsonStore, now_ms
from .text import clean, truncate

__all__ = ["CATEGORIES", "MemoryBank", "SessionMemory"]

CATEGORIES = ("activeTopic", "pendingThought", "memberImpression")
RESERVED_NAMES = frozenset({"__proto__", "constructor", "prototype"})
MINUTE_MS = 60 * 1000


def _default_state() -> dict[str, Any]:
    return {"activeTopics": [], "pendingThoughts": [], "memberImpressions": {}}


def safe_filename(key: str) -> str:
    """把 ``群聊:12345`` 这类会话键变成安全的文件名。"""
    out = []
    for ch in str(key or "default"):
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    name = "".join(out).strip("_") or "default"
    return name[:96]


class SessionMemory:
    """单个会话（一个群 / 一个私聊）的记忆。"""

    def __init__(
        self,
        path: Path,
        *,
        max_topics: int = 20,
        max_thoughts: int = 20,
        max_impressions: int = 50,
        topic_ttl_hours: float = 24.0,
        thought_ttl_hours: float = 2.0,
        on_warning=None,
    ) -> None:
        self.store = JsonStore(path, _default_state, on_warning=on_warning)
        self.max_topics = max(1, int(max_topics))
        self.max_thoughts = max(1, int(max_thoughts))
        self.max_impressions = max(1, int(max_impressions))
        self.topic_ttl_ms = int(max(0.1, float(topic_ttl_hours)) * 3600 * 1000)
        self.thought_ttl_ms = int(max(0.1, float(thought_ttl_hours)) * 3600 * 1000)
        self._data = self._load()
        self.dirty = False

    # ---------------------------------------------------------------- 存取
    def _load(self) -> dict[str, Any]:
        raw = self.store.load()
        if not isinstance(raw, dict):
            return _default_state()
        data = _default_state()
        topics = raw.get("activeTopics")
        if isinstance(topics, list):
            data["activeTopics"] = [t for t in topics if isinstance(t, dict) and t.get("text")][
                -self.max_topics :
            ]
        thoughts = raw.get("pendingThoughts")
        if isinstance(thoughts, list):
            data["pendingThoughts"] = [
                t for t in thoughts if isinstance(t, dict) and t.get("text")
            ][-self.max_thoughts :]
        impressions = raw.get("memberImpressions")
        if isinstance(impressions, dict):
            normalized: dict[str, dict[str, Any]] = {}
            for key, value in impressions.items():
                if not isinstance(value, dict):
                    continue
                # 新格式：key 是 QQ 号，value 里带 name（显示用）。
                # 旧格式：key 是昵称。先原样保留，等这个人在群里说话时再迁移（见 rename_impression）。
                normalized[str(key)] = {
                    "name": truncate(clean(value.get("name") or key), 40),
                    "traits": [truncate(clean(t), 50) for t in (value.get("traits") or [])][:10],
                    "interactionCount": int(value.get("interactionCount") or 0),
                    "lastSeenAt": int(value.get("lastSeenAt") or 0),
                }
            data["memberImpressions"] = normalized
        return data

    def save(self) -> None:
        self._data["activeTopics"] = self._data["activeTopics"][-self.max_topics :]
        self._data["pendingThoughts"] = self._data["pendingThoughts"][-self.max_thoughts :]
        self.store.save(self._data)
        self.dirty = False

    def save_if_dirty(self) -> None:
        if self.dirty:
            self.save()

    # ---------------------------------------------------------------- 清理
    def prune(self) -> None:
        """按 TTL 淘汰过期内容（写库前调用即可，读时也会过滤）。"""
        now = now_ms()
        topics = self._data.get("activeTopics") or []
        kept_topics = [
            t for t in topics if not t.get("lastMentionAt") or now - int(t["lastMentionAt"]) < self.topic_ttl_ms
        ]
        thoughts = self._data.get("pendingThoughts") or []
        kept_thoughts = [
            t
            for t in thoughts
            if not t.get("expiresAt") or now < int(t["expiresAt"])
        ]
        changed = len(kept_topics) != len(topics) or len(kept_thoughts) != len(thoughts)
        self._data["activeTopics"] = kept_topics
        self._data["pendingThoughts"] = kept_thoughts
        if changed:
            self.dirty = True

    # ---------------------------------------------------------------- 写入
    def add_topic(
        self,
        text: object,
        participants: Iterable[object] = (),
        pending_question: object = "",
    ) -> bool:
        body = truncate(clean(text), 200).strip()
        if not body:
            return False
        self.prune()
        now = now_ms()
        existing = next(
            (t for t in self._data["activeTopics"] if str(t.get("text")) == body), None
        )
        people = [truncate(clean(p), 40) for p in participants if clean(p)][:10]
        if existing is not None:
            existing["lastMentionAt"] = now
            if people:
                merged = list(dict.fromkeys([*(existing.get("participants") or []), *people]))
                existing["participants"] = merged[:10]
            if clean(pending_question):
                existing["pendingQuestion"] = truncate(clean(pending_question), 200)
        else:
            self._data["activeTopics"].append(
                {
                    "text": body,
                    "lastMentionAt": now,
                    "participants": people,
                    "pendingQuestion": truncate(clean(pending_question), 200),
                }
            )
        self._data["activeTopics"] = self._data["activeTopics"][-self.max_topics :]
        self.dirty = True
        return True

    def add_thought(
        self,
        text: object,
        motivation: object = "curiosity",
        ttl_ms: int | None = None,
    ) -> bool:
        body = truncate(clean(text), 200).strip()
        if not body:
            return False
        self.prune()
        now = now_ms()
        ttl = int(ttl_ms) if ttl_ms is not None else self.thought_ttl_ms
        existing = next(
            (t for t in self._data["pendingThoughts"] if str(t.get("text")) == body), None
        )
        if existing is not None:
            existing["createdAt"] = now
            existing["expiresAt"] = now + ttl
            if clean(motivation):
                existing["motivation"] = truncate(clean(motivation), 50)
        else:
            self._data["pendingThoughts"].append(
                {
                    "text": body,
                    "createdAt": now,
                    "expiresAt": now + ttl,
                    "motivation": truncate(clean(motivation), 50) or "curiosity",
                }
            )
        self._data["pendingThoughts"] = self._data["pendingThoughts"][-self.max_thoughts :]
        self.dirty = True
        return True

    def add_impression(self, uid: object, trait: object, name: str = "") -> bool:
        """记录对某个群友的印象。

        ``uid`` 必须是 **QQ 号**，不是昵称。理由：
        - 同一个人改名后不会再"变成另一个人"；
        - 同一个群里两个人重名也不会互相覆盖；
        - 私聊里同一个人天然是同一份记录。

        昵称只用于显示（``name``），不参与主键。
        """
        key = truncate(clean(uid), 40).strip()
        if not key or key in RESERVED_NAMES:
            return False
        display = truncate(clean(name), 40).strip() or key
        body = truncate(clean(trait), 50).strip()
        impressions = self._data.setdefault("memberImpressions", {})
        old = impressions.get(key) or {}
        traits = [str(t) for t in (old.get("traits") or [])][:10]
        if body and body not in traits:
            traits.append(body)
        impressions[key] = {
            "name": display,
            "traits": traits[:10],
            "interactionCount": int(old.get("interactionCount") or 0) + 1,
            "lastSeenAt": now_ms(),
        }
        if len(impressions) > self.max_impressions:
            ordered = sorted(
                impressions.items(), key=lambda kv: int(kv[1].get("lastSeenAt") or 0)
            )
            for key_to_drop, _ in ordered[: len(impressions) - self.max_impressions]:
                impressions.pop(key_to_drop, None)
        self.dirty = True
        return True

    def rename_impression(self, uid: str, name: str) -> bool:
        """群友改名时刷新显示名；顺便把旧版按昵称存的记录迁移到 QQ 号。

        Returns:
            是否发生了修改（调用方据此决定要不要落盘）。
        """
        key = truncate(clean(uid), 40).strip()
        display = truncate(clean(name), 40).strip()
        if not key or not display:
            return False
        impressions = self._data.setdefault("memberImpressions", {})
        changed = False
        entry = impressions.get(key)
        if entry is not None and entry.get("name") != display:
            entry["name"] = display
            changed = True
        # 旧格式迁移：key 是昵称的记录，遇到同一个昵称的真人发言就搬到 QQ 号下。
        legacy = impressions.get(display)
        if legacy is not None and display != key:
            if entry is None:
                impressions[key] = {
                    "name": display,
                    "traits": list(legacy.get("traits") or []),
                    "interactionCount": int(legacy.get("interactionCount") or 0),
                    "lastSeenAt": int(legacy.get("lastSeenAt") or 0),
                }
            else:
                merged = list(entry.get("traits") or [])
                for trait in legacy.get("traits") or []:
                    if trait not in merged:
                        merged.append(trait)
                entry["traits"] = merged[:10]
                entry["interactionCount"] = int(entry.get("interactionCount") or 0) + int(
                    legacy.get("interactionCount") or 0
                )
            impressions.pop(display, None)
            changed = True
        if changed:
            self.dirty = True
        return changed

    def remove(self, category: str, needle: str = "", target: str = "") -> int:
        """删除一条记忆；返回删除条数。``memberImpression`` 按人名整体删除。"""
        removed = 0
        if category == "activeTopic":
            before = len(self._data["activeTopics"])
            self._data["activeTopics"] = [
                t for t in self._data["activeTopics"] if str(t.get("text")) != needle
            ]
            removed = before - len(self._data["activeTopics"])
        elif category == "pendingThought":
            before = len(self._data["pendingThoughts"])
            self._data["pendingThoughts"] = [
                t for t in self._data["pendingThoughts"] if str(t.get("text")) != needle
            ]
            removed = before - len(self._data["pendingThoughts"])
        elif category == "memberImpression":
            if not target:
                return 0
            impressions = self._data["memberImpressions"]
            if target in impressions:
                del impressions[target]
                removed = 1
            else:
                # 也允许按昵称删（命令行里人只会输名字）
                for key, value in list(impressions.items()):
                    if str(value.get("name") or "") == target:
                        del impressions[key]
                        removed = 1
                        break
        if removed:
            self.dirty = True
        return removed

    def clear(self, category: str | None = None) -> None:
        if not category or category == "activeTopic":
            self._data["activeTopics"] = []
        if not category or category == "pendingThought":
            self._data["pendingThoughts"] = []
        if not category or category == "memberImpression":
            self._data["memberImpressions"] = {}
        self.dirty = True

    # ---------------------------------------------------------------- 读取
    def _visible_topics(self) -> list[dict]:
        now = now_ms()
        return [
            t
            for t in self._data["activeTopics"]
            if not t.get("lastMentionAt") or now - int(t["lastMentionAt"]) < self.topic_ttl_ms
        ]

    def _visible_thoughts(self) -> list[dict]:
        now = now_ms()
        return [t for t in self._data["pendingThoughts"] if not t.get("expiresAt") or now < int(t["expiresAt"])]

    def format_for_prompt(
        self,
        max_topics: int = 10,
        max_impressions: int = 10,
        affinity_visible: Callable[[str], bool] | None = None,
        affinity_score: Callable[[str], float] | None = None,
        skip_text: Callable[[str], bool] | None = None,
        max_topic_age_min: float | None = None,
    ) -> str:
        """生成注入 prompt 的记忆段（格式与 qq-bridge 一致，模型最容易理解）。

        Args:
            affinity_visible: 某个群友的印象还值不值得出现在提示词里；
                返回 False 的人整条不展示（关系太差的群友不再占记忆位）。
            affinity_score: 取关系分数，用于排序与标注——分数低的人排在前面，
                条数超限时**先被丢掉**。
            skip_text: 这条内容要不要跳过（"别再提"命中时返回 True）。
                话题和念头会被过滤掉——**记录仍在**，叫停解除后自动回来，
                所以这是可逆的，不是删数据。
            max_topic_age_min: 话题超过这么久没被提起就不再注入。
                注意这是**注入层**的年龄线，和存储层的 TTL 是两回事：
                TTL 24 小时是"什么时候清掉记录"，这里是"什么时候不再当作可聊的话题"。
                旧事重提多半就是这两者混在一起导致的。
        """
        lines: list[str] = []
        topics = self._visible_topics()
        if max_topic_age_min is not None and max_topic_age_min > 0:
            cutoff = now_ms() - int(float(max_topic_age_min) * MINUTE_MS)
            topics = [
                t
                for t in topics
                if not t.get("lastMentionAt") or int(t["lastMentionAt"]) >= cutoff
            ]
        if skip_text is not None:
            topics = [
                t
                for t in topics
                if not skip_text(str(t.get("text") or ""))
                and not skip_text(str(t.get("pendingQuestion") or ""))
            ]
        if topics:
            lines.append("【进行中的话题】")
            for t in topics[-max(1, max_topics) :]:
                last = int(t.get("lastMentionAt") or 0)
                ago_min = int((now_ms() - last) / MINUTE_MS) if last else 0
                stale = "（已搁置）" if last and now_ms() - last > 2 * 3600 * 1000 else ""
                when = f"{ago_min}分钟前" if ago_min > 0 else "刚刚"
                pending = t.get("pendingQuestion")
                extra = f"；待追问：{pending}" if pending else ""
                lines.append(f"- {t.get('text')}{stale}（{when}）{extra}")
        thoughts = self._visible_thoughts()
        if skip_text is not None:
            thoughts = [t for t in thoughts if not skip_text(str(t.get("text") or ""))]
        if thoughts:
            lines.append("【你想说但还没说的】")
            for t in thoughts[-max(1, max_topics) :]:
                mot = t.get("motivation")
                lines.append(f"- {t.get('text')}" + (f"（{mot}）" if mot else ""))
        impressions = self._data.get("memberImpressions") or {}
        if impressions:
            lines.append("【对群友的印象】")
            items = list(impressions.items())
            if affinity_visible is not None:
                items = [item for item in items if affinity_visible(item[0])]

            def _rank(item: tuple[str, dict]) -> tuple[float, int]:
                uid, im = item
                score = affinity_score(uid) if affinity_score is not None else 50.0
                return (float(score), int(im.get("lastSeenAt") or 0))

            items.sort(key=_rank)
            # 展示用昵称，绝不在提示词里出现 QQ 号（模型会把数字当内容念出来）。
            for uid, im in items[-max(1, max_impressions) :]:
                label = str(im.get("name") or uid)
                traits = [str(x) for x in (im.get("traits") or [])]
                body = "、".join(traits) if traits else "暂无记录"
                cold = ""
                if affinity_score is not None and affinity_score(uid) < 30:
                    cold = "（最近不太对付）"
                lines.append(
                    f"- {label}：{body}（互动 {int(im.get('interactionCount') or 0)} 次）{cold}"
                )
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        return {
            "activeTopics": len(self._visible_topics()),
            "pendingThoughts": len(self._visible_thoughts()),
            "memberImpressions": len(self._data.get("memberImpressions") or {}),
        }

    def is_empty(self) -> bool:
        return not (
            self._visible_topics()
            or self._visible_thoughts()
            or (self._data.get("memberImpressions") or {})
        )


class MemoryBank:
    """按会话管理的记忆集合（每个会话一个 JSON 文件）。"""

    def __init__(self, directory: Path, options: dict[str, Any] | None = None, on_warning=None) -> None:
        self.directory = Path(directory)
        self.options = dict(options or {})
        self.on_warning = on_warning
        self._cache: dict[str, SessionMemory] = {}

    def get(self, session_key: str) -> SessionMemory:
        key = safe_filename(session_key)
        mem = self._cache.get(key)
        if mem is None:
            mem = SessionMemory(
                self.directory / f"{key}.json",
                max_topics=self.options.get("max_topics", 20),
                max_thoughts=self.options.get("max_thoughts", 20),
                max_impressions=self.options.get("max_impressions", 50),
                topic_ttl_hours=self.options.get("topic_ttl_hours", 24.0),
                thought_ttl_hours=self.options.get("thought_ttl_hours", 2.0),
                on_warning=self.on_warning,
            )
            self._cache[key] = mem
        return mem

    def flush(self) -> None:
        for mem in self._cache.values():
            mem.save_if_dirty()
