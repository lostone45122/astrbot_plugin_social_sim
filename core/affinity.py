"""群友关系权重（好感度）。

解决的问题：群里有人经常冒犯 AI 或别的群友，但记忆、群风、回复概率原本都一视同仁——
模型对这种人依然"热情"，记忆里还给他留着位置。这里给每个群友维护一个 0~100 的分数，
分数低到一定程度，**记忆、画像、回复概率**会一起降权。

三条设计原则，都是为了避免把插件变成"记仇机器"：

1. **只惩罚"指向人"的话。** "这游戏真垃圾"没有对象，不算；"你真垃圾""@阿伟 傻逼"算。
   实现上要求侮辱词与"人指代"（你/他/她/@/群昵称/bot 别名）同句出现。
2. **要"经常"才算。** 默认 `min_offenses_to_apply: 3`，一两次玩笑不会触发任何降权；
   单次扣分也很小（默认 6 分，从 50 分要连着 3~4 次才进入"冷淡"区）。
3. **会自己恢复。** 每过一小时 +3 分，每说一句正常的话 +0.2 分，回到 50（中性）为止。
   所以它是"最近让人不舒服"的度量，不是永久黑名单。

同时，降权**不等于**可以让 AI 骂回去：提示词里明确写了"冷淡 ≠ 攻击，禁止报复和公开嘲讽"。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .store import JsonStore, now_ms
from .text import clean, truncate

__all__ = [
    "NEUTRAL_SCORE",
    "AffinityBank",
    "AffinityStore",
    "detect_offense",
    "detect_positive",
]

NEUTRAL_SCORE = 50.0
MAX_SCORE = 100.0
MIN_SCORE = 0.0

# 明确针对人的侮辱词。中文群聊里真正伤人的高频词就这些，宁可漏判也不要误判。
DEFAULT_INSULT_WORDS: tuple[str, ...] = (
    "傻逼", "傻b", "沙比", "煞笔", "sb", "脑残", "弱智", "智障", "白痴", "蠢货",
    "废物", "垃圾", "狗东西", "贱", "去死", "死开", "滚蛋", "滚出", "滚", "爬",
    "闭嘴", "有病",
    "神经病", "你妈", "尼玛", "你算什么东西", "什么东西", "恶心", "别bb", "别叫",
    "装什么", "你懂个屁", "傻狗", "蠢狗", "没用的东西", "废物点心",
)

# 表达善意/赞美的词。用来让分数慢慢回中性，避免"一次说错话背一辈子"。
DEFAULT_POSITIVE_WORDS: tuple[str, ...] = (
    "谢谢", "感谢", "多谢", "辛苦了", "大佬", "牛逼", "厉害", "666", "强啊",
    "好用", "靠谱", "帮大忙", "爱了", "喜欢你", "可爱", "稳", "牛", "赞",
)

# 侮辱词后面紧跟这些词时，说的其实是"东西"而不是"人"（这游戏真垃圾 / 这代码真废物）。
_THING_SUFFIXES = (
    "游戏", "电影", "片子", "剧", "番", "歌", "画", "手机", "电脑", "东西", "玩意",
    "系统", "网络", "代码", "服务器", "驱动", "版本", "更新", "活动", "策划", "运营",
    "机制", "设计", "体验", "功能", "插件", "项目", "公司", "学校", "天气", "滚动",
    "滚轮", "爬虫", "爬取",
)

# 人指代：出现这些词才认为"在说人"。
_PERSON_REFS = ("你", "您", "他", "她", "这人", "那位", "楼上", "楼下", "群友", "管理")
_AT_RE = re.compile(r"\[at[:：][^\]]*\]|\[CQ:at[^\]]*\]")

_OFFENSE = "offense"
_POSITIVE = "positive"


def _default_state() -> dict[str, Any]:
    return {"members": {}, "updatedAt": 0}


def safe_name(key: str) -> str:
    out = [ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(key or "default")]
    name = "".join(out).strip("_") or "default"
    return name[:96]


def _normalize(text: str) -> str:
    """把 @ 标记与 CQ 码替换成统一占位，便于判断"有没有对着人说话"。"""
    body = _AT_RE.sub(" @人 ", clean(text))
    return body.replace("@", " @人 ")


def _is_thing_mention(body: str, index: int, word: str) -> bool:
    """侮辱词后面紧跟"东西类"名词时，判定不是在说人。"""
    tail = body[index + len(word) : index + len(word) + 4]
    return any(tail.startswith(suffix) for suffix in _THING_SUFFIXES)


def _has_person_ref(body: str, names: tuple[str, ...]) -> bool:
    if " @人 " in body:
        return True
    if any(ref in body for ref in _PERSON_REFS):
        return True
    lowered = body.lower()
    return any(name and name.lower() in lowered for name in names)


def detect_offense(
    text: object,
    *,
    names: tuple[str, ...] = (),
    extra_words: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """判断这句话是不是在攻击某个人（AI 或群友）。

    Args:
        text: 群消息文本。
        names: bot 别名 + 群昵称，用来识别"点名"。
        extra_words: 配置里追加的侮辱词。

    Returns:
        ``{"word": 命中词, "targeted": True}`` 或 ``None``。
        只有 ``targeted`` 为真才应该扣分——误伤比漏判更难挽回。
    """
    body = _normalize("" if text is None else str(text))
    if not body.strip():
        return None
    lowered = body.lower()
    words = sorted({*(w for w in DEFAULT_INSULT_WORDS), *(w.lower() for w in extra_words if w)}, key=len, reverse=True)
    for word in words:
        idx = lowered.find(word)
        if idx < 0:
            continue
        if _is_thing_mention(body, idx, word):
            continue
        # 指向人：@ 了谁、出现"你/他/她"、或者点了 bot/群友的名字。
        if _has_person_ref(body, names):
            return {"word": word, "targeted": True}
        # 没有指代，但整句就是个祈使句（"滚"全句只有这一个词）→ 仍然算，但语义最弱。
        if body.strip() == word:
            return {"word": word, "targeted": True}
        return None
    return None


def detect_positive(text: object, extra_words: tuple[str, ...] = ()) -> str | None:
    """判断这句话是否在表达善意/赞美；返回命中的词或 None。"""
    body = _normalize("" if text is None else str(text)).lower()
    if not body:
        return None
    for word in sorted({*(w for w in DEFAULT_POSITIVE_WORDS), *(w.lower() for w in extra_words if w)}, key=len, reverse=True):
        if word and word in body:
            return word
    return None


class AffinityStore:
    """单个会话的关系权重表。"""

    def __init__(
        self,
        path: Path,
        *,
        offense_penalty: float = 6.0,
        min_offenses_to_apply: int = 3,
        recover_per_hour: float = 3.0,
        recover_per_message: float = 0.2,
        hide_memory_below: float = 20.0,
        mute_reply_below: float = 25.0,
        max_evidence: int = 5,
        on_warning=None,
    ) -> None:
        self.store = JsonStore(path, _default_state, on_warning=on_warning)
        self.offense_penalty = max(0.5, float(offense_penalty))
        self.min_offenses_to_apply = max(1, int(min_offenses_to_apply))
        self.recover_per_hour = max(0.0, float(recover_per_hour))
        self.recover_per_message = max(0.0, float(recover_per_message))
        self.hide_memory_below = float(hide_memory_below)
        self.mute_reply_below = float(mute_reply_below)
        self.max_evidence = max(1, int(max_evidence))
        raw = self.store.load()
        members = raw.get("members") if isinstance(raw, dict) else None
        self._members: dict[str, dict[str, Any]] = {}
        if isinstance(members, dict):
            for uid, entry in members.items():
                if not isinstance(entry, dict):
                    continue
                self._members[str(uid)] = {
                    "name": str(entry.get("name") or uid)[:40],
                    "score": float(entry.get("score", NEUTRAL_SCORE)),
                    "offenses": int(entry.get("offenses") or 0),
                    "positives": int(entry.get("positives") or 0),
                    "lastOffenseAt": int(entry.get("lastOffenseAt") or 0),
                    "lastPositiveAt": int(entry.get("lastPositiveAt") or 0),
                    "updatedAt": int(entry.get("updatedAt") or 0),
                    "evidence": [
                        truncate(clean(item), 80)
                        for item in (entry.get("evidence") or [])
                        if clean(item)
                    ][-self.max_evidence :],
                }
        self._last_recover = int(raw.get("updatedAt") or now_ms()) if isinstance(raw, dict) else now_ms()
        self.dirty = False

    # ---------------------------------------------------------------- 存取
    def save(self) -> None:
        self._recover()
        self.store.save({"members": self._members, "updatedAt": now_ms()})
        self.dirty = False

    def save_if_dirty(self) -> None:
        if self.dirty:
            self.save()

    def _entry(self, uid: str, name: str = "") -> dict[str, Any]:
        key = str(uid or "unknown")
        entry = self._members.get(key)
        if entry is None:
            entry = {
                "name": truncate(clean(name), 40) or key,
                "score": NEUTRAL_SCORE,
                "offenses": 0,
                "positives": 0,
                "lastOffenseAt": 0,
                "lastPositiveAt": 0,
                "updatedAt": now_ms(),
                "evidence": [],
            }
            self._members[key] = entry
        elif name:
            entry["name"] = truncate(clean(name), 40)
        return entry

    # ---------------------------------------------------------------- 计分
    def _recover(self) -> None:
        """按时间把分数拉回中性（只做一次，惰性计算）。"""
        now = now_ms()
        elapsed_h = max(0.0, (now - self._last_recover) / 3_600_000)
        self._last_recover = now
        if elapsed_h <= 0 or self.recover_per_hour <= 0:
            return
        gain = elapsed_h * self.recover_per_hour
        for entry in self._members.values():
            score = float(entry.get("score", NEUTRAL_SCORE))
            if score < NEUTRAL_SCORE:
                entry["score"] = min(NEUTRAL_SCORE, score + gain)
            elif score > NEUTRAL_SCORE:
                entry["score"] = max(NEUTRAL_SCORE, score - gain)

    def score(self, uid: str) -> float:
        """当前分数（读的时候顺带做时间恢复，所以随时读到的是"现在"的分）。"""
        self._recover()
        entry = self._members.get(str(uid or "unknown"))
        if entry is None:
            return NEUTRAL_SCORE
        return float(entry.get("score", NEUTRAL_SCORE))

    def record_offense(self, uid: str, name: str, text: object, word: str = "") -> float:
        self._recover()
        entry = self._entry(uid, name)
        entry["score"] = max(MIN_SCORE, float(entry["score"]) - self.offense_penalty)
        entry["offenses"] = int(entry["offenses"]) + 1
        entry["lastOffenseAt"] = now_ms()
        entry["updatedAt"] = now_ms()
        sample = truncate(clean(text), 80)
        if sample:
            entry["evidence"] = ([*(entry.get("evidence") or []), sample])[-self.max_evidence :]
        self.dirty = True
        return float(entry["score"])

    def record_positive(self, uid: str, name: str, amount: float = 1.0) -> float:
        self._recover()
        entry = self._entry(uid, name)
        entry["score"] = min(MAX_SCORE, float(entry["score"]) + max(0.0, amount))
        entry["positives"] = int(entry["positives"]) + 1
        entry["lastPositiveAt"] = now_ms()
        entry["updatedAt"] = now_ms()
        # 说正常话也慢慢回血：让"以前不愉快"能自然淡出。
        entry["score"] = min(MAX_SCORE, float(entry["score"]) + self.recover_per_message)
        self.dirty = True
        return float(entry["score"])

    def touch_neutral(self, uid: str, name: str) -> None:
        """普通发言：只做轻微回血，不新建记录。"""
        entry = self._members.get(str(uid or "unknown"))
        if entry is None or self.recover_per_message <= 0:
            return
        score = float(entry.get("score", NEUTRAL_SCORE))
        if score < NEUTRAL_SCORE:
            entry["score"] = min(NEUTRAL_SCORE, score + self.recover_per_message)
            self.dirty = True

    # ---------------------------------------------------------------- 降权
    def multiplier(self, uid: str) -> float:
        """回复概率的乘数（0~1）。"""
        entry = self._members.get(str(uid or "unknown"))
        if entry is None or int(entry.get("offenses") or 0) < self.min_offenses_to_apply:
            return 1.0
        score = self.score(uid)
        if score >= NEUTRAL_SCORE:
            return 1.0
        if score >= self.mute_reply_below + 10:
            return 0.75
        if score >= self.mute_reply_below:
            return 0.45
        if score >= self.hide_memory_below:
            return 0.2
        return 0.05

    def style_weight(self, uid: str) -> float:
        """群风统计的权重（0~1）：关系差的人不太能塑造"这个群的语感"。"""
        entry = self._members.get(str(uid or "unknown"))
        if entry is None or int(entry.get("offenses") or 0) < self.min_offenses_to_apply:
            return 1.0
        score = self.score(uid)
        if score >= NEUTRAL_SCORE:
            return 1.0
        if score >= self.mute_reply_below:
            return 0.5
        if score >= self.hide_memory_below:
            return 0.2
        return 0.0

    def memory_visible(self, uid_or_name: str) -> bool:
        """这个人的印象还值不值得写进提示词。

        记忆按 **QQ 号** 存主键，所以优先按号查；查不到再按昵称回退（兼容旧数据）。
        找不到就按中性处理。
        """
        key = str(uid_or_name or "")
        entry = self._members.get(key) or self._find_by_name(key)
        if entry is None:
            return True
        if int(entry.get("offenses") or 0) < self.min_offenses_to_apply:
            return True
        uid = next((k for k, v in self._members.items() if v is entry), "")
        return self.score(uid) >= self.hide_memory_below

    def _find_by_name(self, name: str) -> dict[str, Any] | None:
        target = clean(name)
        if not target:
            return None
        for entry in self._members.values():
            if entry.get("name") == target:
                return entry
        return None

    # ---------------------------------------------------------------- 展示
    def summary(self, threshold: float | None = None) -> list[dict[str, Any]]:
        """给 ``/关系`` 用的列表（按分数升序，最冷的在前）。"""
        limit = self.hide_memory_below if threshold is None else threshold
        rows = [
            {
                "uid": uid,
                "name": entry.get("name") or uid,
                "score": round(float(entry.get("score", NEUTRAL_SCORE)), 1),
                "offenses": int(entry.get("offenses") or 0),
                "positives": int(entry.get("positives") or 0),
                "lastOffenseAt": int(entry.get("lastOffenseAt") or 0),
                "evidence": list(entry.get("evidence") or []),
            }
            for uid, entry in self._members.items()
        ]
        rows.sort(key=lambda row: (row["score"], -row["offenses"]))
        return [row for row in rows if row["score"] <= 60 or row["offenses"] or row["positives"]]

    def cold_members(self) -> list[dict[str, Any]]:
        return [row for row in self.summary() if row["score"] < NEUTRAL_SCORE and row["offenses"] >= self.min_offenses_to_apply]

    def reset(self) -> None:
        self._members = {}
        self._last_recover = now_ms()
        self.save()

    def is_empty(self) -> bool:
        return not self._members


class AffinityBank:
    """按会话管理的关系权重表（每个会话一个 JSON 文件）。"""

    def __init__(self, directory: Path, options: dict[str, Any] | None = None, on_warning=None) -> None:
        self.directory = Path(directory)
        self.options = dict(options or {})
        self.on_warning = on_warning
        self._cache: dict[str, AffinityStore] = {}

    def get(self, session_key: str) -> AffinityStore:
        key = safe_name(session_key)
        store = self._cache.get(key)
        if store is None:
            store = AffinityStore(
                self.directory / f"{key}.json",
                offense_penalty=self.options.get("offense_penalty", 6.0),
                min_offenses_to_apply=self.options.get("min_offenses_to_apply", 3),
                recover_per_hour=self.options.get("recover_per_hour", 3.0),
                recover_per_message=self.options.get("recover_per_message", 0.2),
                hide_memory_below=self.options.get("hide_memory_below", 20.0),
                mute_reply_below=self.options.get("mute_reply_below", 25.0),
                max_evidence=self.options.get("max_evidence", 5),
                on_warning=self.on_warning,
            )
            self._cache[key] = store
        return store

    def flush(self) -> None:
        for store in self._cache.values():
            store.save_if_dirty()
