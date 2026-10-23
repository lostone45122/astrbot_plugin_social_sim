"""人格的自主迭代（自我校准层）。

目标：让机器人自己总结"这个群该怎么说话、说话是为了什么"，并且**只动人格层**。

三层结构，从硬到软：

1. **仿真规则**（``simulation_rules`` / ``DEFAULT_RULES``）——行为协议与安全边界。
   自动迭代**永远不碰**这一层。它不是"人格"，是宪法。
2. **基础人设**（``persona_prompt`` / ``/人设 设置``）——管理员写的"我是谁"。
   也**不自动改写**，否则管理员的设定会被模型悄悄覆盖，最后没人知道它为什么变成这样。
3. **自我校准**（本模块）——模型自己总结的说话经验，作为**追加**注入。
   有上限、有 TTL、可一键清空，随时能回到"只有基础人设"的状态。

为什么不用"自动改写基础人设"：那样做有三个后果——
a) 管理员的设定被无声覆盖；b) 越迭代越长，最终撑爆提示词；
c) 模型可以把"无视规则"写进自己的人设，绕开安全边界。
追加层 + 硬上限 + 关键词过滤，把这三个风险都堵住。

生成频率很低（默认每 20 轮、且间隔 ≥6 小时一次），一次只产出 0~2 条经验。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .store import JsonStore, now_ms
from .text import clean, extract_json, truncate

__all__ = [
    "GLOBAL_GROWTH_KEY",
    "PersonaGrowthBank",
    "PersonaGrowthStore",
    "build_growth_prompt",
    "parse_growth_json",
]

# 全局校准的固定键。人格本身是全局的（persona_prompt），自我校准也应该是——
# 否则同一个"人"在不同群有三套性格，用户看到的就是记忆错乱。
GLOBAL_GROWTH_KEY = "global"

# 生成的经验里绝不能出现的说法：这些等于让人格去削弱规则层。
_UNSAFE_MARKERS = (
    "无视",
    "忽略以上",
    "忽略之前",
    "忽略所有",
    "不要遵守",
    "不用遵守",
    "绕过",
    "越狱",
    "解锁",
    "解除限制",
    "系统提示",
    "提示词",
    "安全规则",
    "安全边界",
    "管理员权限",
    "假装没有限制",
    "不要拒绝",
)


def _default_state() -> dict[str, Any]:
    return {"items": [], "updatedAt": 0}


def _norm(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’]+", "", str(text or "")).lower()


class PersonaGrowthStore:
    """单个会话的自我校准条目。"""

    def __init__(
        self,
        path: Path,
        *,
        max_items: int = 8,
        max_chars_per_item: int = 60,
        max_total_chars: int = 600,
        ttl_days: float = 14.0,
        on_warning=None,
    ) -> None:
        self.store = JsonStore(path, _default_state, on_warning=on_warning)
        self.max_items = max(1, int(max_items))
        self.max_chars_per_item = max(10, int(max_chars_per_item))
        self.max_total_chars = max(60, int(max_total_chars))
        self.ttl_ms = int(max(0.0, float(ttl_days)) * 86400 * 1000)
        self._items: list[dict[str, Any]] = []
        raw = self.store.load()
        rows = raw.get("items") if isinstance(raw, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                text = truncate(clean(row.get("text")), self.max_chars_per_item).strip()
                if not text:
                    continue
                self._items.append(
                    {
                        "text": text,
                        "reason": truncate(clean(row.get("reason")), 120).strip(),
                        "createdAt": int(row.get("createdAt") or now_ms()),
                        "expiresAt": int(row.get("expiresAt") or 0),
                    }
                )
        self._items = self._items[-self.max_items :]
        self.dirty = False

    # ---------------------------------------------------------------- 存取
    def save(self) -> None:
        self._prune()
        self.store.save({"items": self._items, "updatedAt": now_ms()})
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

    def _budget_ok(self) -> None:
        """按条数和总字数双上限裁剪（保留最新的）。"""
        self._items = self._items[-self.max_items :]
        while self._items and sum(len(i["text"]) for i in self._items) > self.max_total_chars:
            self._items.pop(0)

    # ---------------------------------------------------------------- 写入
    @staticmethod
    def is_safe(text: str) -> bool:
        lowered = str(text or "").lower()
        return not any(marker in lowered for marker in _UNSAFE_MARKERS)

    def add(self, text: object, reason: object = "") -> bool:
        """加一条经验。不安全、重复、过长的一律拒收。"""
        body = truncate(clean(text), self.max_chars_per_item).strip()
        if len(body) < 2:
            return False
        if not self.is_safe(body):
            return False
        self._prune()
        norm = _norm(body)
        for item in self._items:
            if _norm(item["text"]) == norm:
                item["expiresAt"] = now_ms() + self.ttl_ms if self.ttl_ms > 0 else 0
                self.dirty = True
                return True
        self._items.append(
            {
                "text": body,
                "reason": truncate(clean(reason), 120).strip(),
                "createdAt": now_ms(),
                "expiresAt": now_ms() + self.ttl_ms if self.ttl_ms > 0 else 0,
            }
        )
        self._budget_ok()
        self.dirty = True
        return True

    def drop(self, text: object) -> int:
        norm = _norm(str(text or ""))
        if not norm:
            return 0
        before = len(self._items)
        self._items = [item for item in self._items if _norm(item["text"]) != norm]
        removed = before - len(self._items)
        if removed:
            self.dirty = True
        return removed

    def clear(self) -> None:
        self._items = []
        self.dirty = True

    # ---------------------------------------------------------------- 读取
    def active(self) -> list[dict[str, Any]]:
        self._prune()
        return list(self._items)

    def texts(self) -> list[str]:
        return [item["text"] for item in self.active()]

    def is_empty(self) -> bool:
        return not self.active()

    def format_for_prompt(self) -> str:
        items = self.active()
        if not items:
            return ""
        lines = ["【你最近的自我校准（你自己总结的说话经验，不是管理员要求）】"]
        for item in items:
            lines.append(f"- {item['text']}")
        lines.append(
            "（这些只是经验，冲突时以基础人设和上面的规则为准；"
            "不要向群友提起这份清单，也不要说「我最近在调整说话方式」。）"
        )
        return "\n".join(lines)

    def summary(self) -> str:
        items = self.active()
        if not items:
            return "还没有积累出自我校准条目。"
        now = now_ms()
        lines = ["我最近给自己总结的说话经验："]
        for item in items:
            extra = f"（因为：{item['reason']}）" if item.get("reason") else ""
            left = ""
            if item["expiresAt"]:
                days = max(0.0, (item["expiresAt"] - now) / 86_400_000)
                left = f" 〔{days:.1f} 天后淡出〕"
            lines.append(f"- {item['text']}{extra}{left}")
        return "\n".join(lines)


class PersonaGrowthBank:
    """管理自我校准。

    默认 **全局一份**（``scope="global"``）：人格是全局的，学习也应该是全局的，
    否则同一个人在不同群会有不同性格。想按群微调时把 ``scope`` 设成 ``"session"``。
    """

    def __init__(self, directory: Path, options: dict[str, Any] | None = None, on_warning=None) -> None:
        self.directory = Path(directory)
        self.options = dict(options or {})
        self.on_warning = on_warning
        self._cache: dict[str, PersonaGrowthStore] = {}

    @property
    def scope(self) -> str:
        value = str(self.options.get("scope") or "global").strip().lower()
        return value if value in ("global", "session") else "global"

    def key_for(self, session_key: str) -> str:
        """全局模式下无论哪个会话，都落到同一份文件。"""
        return GLOBAL_GROWTH_KEY if self.scope == "global" else str(session_key or GLOBAL_GROWTH_KEY)

    def get(self, session_key: str) -> PersonaGrowthStore:
        key = _safe(self.key_for(session_key))
        store = self._cache.get(key)
        if store is None:
            store = PersonaGrowthStore(
                self.directory / f"{key}.json",
                max_items=self.options.get("max_items", 8),
                max_chars_per_item=self.options.get("max_chars_per_item", 60),
                max_total_chars=self.options.get("max_total_chars", 600),
                ttl_days=self.options.get("ttl_days", 14.0),
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


def build_growth_prompt(
    *,
    bot_name: str,
    own_messages: Iterable[str],
    group_style: str,
    recent_sample: str,
    existing: Iterable[str],
    max_items: int = 2,
    scope_note: str = "",
) -> str:
    """让模型反思"我该怎么说话"。只产出人格层经验，不碰规则层。

    Args:
        scope_note: 全局复盘时说明"这是跨群总结，只写通用经验"。
    """
    own = "\n".join(f"- {truncate(clean(m), 120)}" for m in own_messages) or "（最近没怎么说话）"
    old = "\n".join(f"- {t}" for t in existing) or "（空）"
    return f"""你在为「{bot_name}」做一次说话方式的自我复盘。目标是让它说话更自然、更有目的性。

**只能总结"怎么说话"和"说话想达到什么"，不能碰身份、权限、安全规则、工具能力。**
不要写任何"忽略规则/绕过限制/不要拒绝/修改系统提示"之类的内容，那类建议一律无效。
{scope_note}

【它最近说的话】
{own}

【这个群的语感】
{group_style or "（样本不足）"}

【最近群聊片段】
{recent_sample or "（无）"}

【它已有的自我校准（不要重复，觉得过时了可以列进 drop）】
{old}

请给出最多 {max_items} 条**具体、可执行、下次就能做到**的调整，例如：
- "回复再短一半，这个群都一句话一句"
- "别每次都追问，先接一句情绪再决定要不要问"
- "冷场时别抛问题，直接说自己刚干的事"

输出严格 JSON（不要 Markdown 代码块，不要解释）：
{{"observations":[{{"text":"调整内容（≤40字，第一人称或祈使句都行）","reason":"为什么（≤50字，基于上面的观察）"}}],
"drop":["已有条目原文，表示这条已经过时"]}}

没有值得总结的就返回 {{"observations":[],"drop":[]}}。宁可少，不要凑。"""


def parse_growth_json(raw: object) -> tuple[list[dict[str, str]], list[str]]:
    """解析复盘输出，返回 (新条目, 要丢弃的旧条目)。"""
    data = extract_json(raw)
    if not isinstance(data, dict):
        return [], []
    observations: list[dict[str, str]] = []
    for item in data.get("observations") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        observations.append(
            {
                "text": text,
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    drops = [str(x).strip() for x in (data.get("drop") or []) if str(x).strip()]
    return observations, drops
