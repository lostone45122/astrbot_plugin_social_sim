"""学习群友的说话策略（群风画像）。

qq-bridge 把「学习群友策略」交给模型自己观察（见其 qq-chat-v2 预设的
【学习群友策略】段落）。在 AstrBot 里我们可以做得更实：**先统计，再交给模型参考**。

这里不花任何 token，纯靠本地统计，回答四个问题：

1. 这个群的人一条消息大概多长？（决定 AI 该说 8 个字还是 40 个字）
2. 他们爱不爱连发？（决定 AI 该分条还是憋成一条）
3. 口癖/高频短语是什么？（决定 AI 用不用“草”“难绷”“笑死”这类词）
4. 谁话多、什么时候热闹？（决定 AI 该不该现在接话）

统计结果只当**参考**，不照抄——这一点在注入的提示词里会明确写出来，
否则模型很容易变成复读机。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .store import JsonStore, now_ms
from .text import clean, truncate

__all__ = ["GroupStyleLearner", "StyleBank"]

_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f000-\U0001f2ff" "]+"
)
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_AT_RE = re.compile(r"\[at[:：][^\]]*\]|\[CQ:at[^\]]*\]|@\S+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 常见虚词/高频词：作为“口癖”没有信息量，反而会挤掉真正有群味的短语。
_STOP_PHRASES = frozenset(
    {
        "什么","怎么","这个","那个","我们","你们","他们","自己","就是","不是","可以","没有","知道",
        "觉得","现在","时候","一个","这样","那样","真的","还是","因为","所以","但是","如果","然后",
        "已经","应该","可能","不要","不是吧","哈哈","哈哈哈","呵呵","嘻嘻","嗯嗯","好的","好吧",
        "有没有","是不是","为什么","什么样","怎么办","怎么样","一下","一点","有点","一点都",
        "今天","明天","昨天","晚上","早上","下午","刚才","等下","这个月","那个月",
        "东西","事情","问题","感觉","意思","情况","时间","地方","朋友","老师","同学",
        "我就","我就说","你说","我说","他说","她说","这就","那就","这不","那不是",
    }
)

_MIN_CHARS = 40


def _default_profile() -> dict[str, Any]:
    return {
        "messages": 0,
        "chars": 0,
        "senders": {},
        "phrases": {},
        "questions": 0,
        "exclaims": 0,
        "emojis": 0,
        "bursts": 0,
        "hours": [0] * 24,
        "createdAt": now_ms(),
        "updatedAt": 0,
        "lastSender": "",
        "lastTs": 0,
    }


class GroupStyleLearner:
    """单个会话的群风统计。"""

    def __init__(
        self,
        path: Path,
        *,
        phrase_min_len: int = 2,
        phrase_max_len: int = 6,
        min_messages: int = 30,
        max_sample_messages: int = 2000,
        on_warning=None,
    ) -> None:
        self.store = JsonStore(path, _default_profile, on_warning=on_warning)
        self.phrase_min_len = max(2, min(4, int(phrase_min_len)))
        self.phrase_max_len = max(self.phrase_min_len, min(8, int(phrase_max_len)))
        self.min_messages = max(1, int(min_messages))
        self.max_sample_messages = max(50, int(max_sample_messages))
        self._data = self._load()
        self._dirty_since_save = 0

    # ---------------------------------------------------------------- 存取
    def _load(self) -> dict[str, Any]:
        raw = self.store.load()
        data = _default_profile()
        if not isinstance(raw, dict):
            return data
        for key in ("messages", "chars", "questions", "exclaims", "emojis", "bursts"):
            try:
                data[key] = max(0.0, float(raw.get(key) or 0))
            except (TypeError, ValueError):
                data[key] = 0.0
        senders = raw.get("senders")
        if isinstance(senders, dict):
            data["senders"] = {
                str(k): {
                    "count": max(0.0, float((v or {}).get("count") or 0)),
                    "chars": max(0.0, float((v or {}).get("chars") or 0)),
                }
                for k, v in senders.items()
                if isinstance(v, dict)
            }
        phrases = raw.get("phrases")
        if isinstance(phrases, dict):
            data["phrases"] = {
                str(k): max(0.0, float(v or 0)) for k, v in phrases.items() if str(k)
            }
        hours = raw.get("hours")
        if isinstance(hours, list) and len(hours) == 24:
            data["hours"] = [max(0.0, float(h or 0)) for h in hours]
        data["lastSender"] = str(raw.get("lastSender") or "")
        data["lastTs"] = int(raw.get("lastTs") or 0)
        data["createdAt"] = int(raw.get("createdAt") or now_ms())
        data["updatedAt"] = int(raw.get("updatedAt") or 0)
        return data

    def save(self) -> None:
        self._data["updatedAt"] = now_ms()
        self.store.save(self._data)
        self._dirty_since_save = 0

    # ---------------------------------------------------------------- 观察
    def observe(
        self,
        sender: str,
        text: str,
        ts: int | None = None,
        *,
        save_every: int = 20,
        weight: float = 1.0,
    ) -> None:
        """记录一条群友消息。``text`` 请传已经清洗过的纯文本。

        ``weight`` 是这个人的"话语权重"（0~1）：关系差的群友权重低，他的消息
        对"这个群爱怎么说话"的贡献就小，权重为 0 时完全不参与统计——否则一个
        天天骂人的人也会把群风带偏。
        """
        body = clean(text)
        if not body or weight <= 0:
            return
        ts = int(ts) if ts else now_ms()
        data = self._data
        counted = data["messages"] < self.max_sample_messages
        if counted:
            data["messages"] += weight
            data["chars"] += len(body) * weight
        sender = truncate(clean(sender), 40) or "未知"
        entry = data["senders"].setdefault(sender, {"count": 0.0, "chars": 0.0})
        if counted:
            entry["count"] = float(entry.get("count") or 0) + weight
            entry["chars"] = float(entry.get("chars") or 0) + len(body) * weight
        if "？" in body or "?" in body:
            data["questions"] += weight
        if "！" in body or "!" in body:
            data["exclaims"] += weight
        if _EMOJI_RE.search(body):
            data["emojis"] += weight
        # 30 秒内同一个人接着说 = 一次“连发”，这是最直观的真人节奏信号。
        if data["lastSender"] == sender and data["lastTs"] and 0 < ts - data["lastTs"] <= 30_000:
            data["bursts"] += weight
        data["lastSender"] = sender
        data["lastTs"] = ts
        try:
            data["hours"][(ts // 3_600_000) % 24] += weight
        except (IndexError, TypeError):
            pass
        self._learn_phrases(body, weight)
        self._dirty_since_save += 1
        if save_every > 0 and self._dirty_since_save >= save_every:
            self.save()

    def _candidate_ngrams(self, body: str) -> set[str]:
        """从一条消息里取出**本条内不重复**的 n-gram 候选。"""
        text = _URL_RE.sub(" ", body)
        text = _AT_RE.sub(" ", text)
        text = _EMOJI_RE.sub(" ", text)
        text = _PUNCT_RE.sub(" ", text)
        if len(text) < self.phrase_min_len:
            return set()
        out: set[str] = set()
        for seg in text.split():
            if len(seg) < self.phrase_min_len:
                continue
            for n in range(self.phrase_min_len, self.phrase_max_len + 1):
                if len(seg) < n:
                    break
                for i in range(len(seg) - n + 1):
                    gram = seg[i : i + n]
                    if gram in _STOP_PHRASES:
                        continue
                    # 至少含一个汉字，或本身就是纯英短词（如 yyds/xswl）。
                    if not _CJK_RE.search(gram) and not gram.isalpha():
                        continue
                    if gram.isdigit():
                        continue
                    out.add(gram)
        return out

    def _learn_phrases(self, body: str, weight: float = 1.0) -> None:
        phrases = self._data["phrases"]
        for gram in self._candidate_ngrams(body):
            phrases[gram] = float(phrases.get(gram) or 0) + weight
        if len(phrases) > 600:
            top = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)[:400]
            self._data["phrases"] = dict(top)

    # ---------------------------------------------------------------- 读取
    def is_ready(self) -> bool:
        return float(self._data.get("messages") or 0) >= self.min_messages

    def metrics(self) -> dict[str, Any]:
        messages = float(self._data.get("messages") or 0)
        chars = float(self._data.get("chars") or 0)
        avg_len = round(chars / messages, 1) if messages else 0.0
        senders = self._data.get("senders") or {}
        top_senders = sorted(
            senders.items(), key=lambda kv: float(kv[1].get("count") or 0), reverse=True
        )[:5]
        hours = self._data.get("hours") or [0] * 24
        peak_hours = sorted(range(24), key=lambda h: hours[h], reverse=True)[:3]
        return {
            "messages": round(messages, 1),
            "avg_len": avg_len,
            "burst_ratio": round(float(self._data.get("bursts") or 0) / messages, 2) if messages else 0.0,
            "question_ratio": round(float(self._data.get("questions") or 0) / messages, 2) if messages else 0.0,
            "exclaim_ratio": round(float(self._data.get("exclaims") or 0) / messages, 2) if messages else 0.0,
            "emoji_ratio": round(float(self._data.get("emojis") or 0) / messages, 2) if messages else 0.0,
            "top_senders": [(name, round(float(info.get("count") or 0), 1)) for name, info in top_senders],
            "peak_hours": [h for h in peak_hours if hours[h] > 0],
        }

    def top_phrases(self, limit: int = 8, min_count: int = 3) -> list[tuple[str, int]]:
        phrases = self._data.get("phrases") or {}
        ranked = sorted(
            ((p, float(c)) for p, c in phrases.items() if float(c) >= min_count),
            key=lambda kv: (kv[1], len(kv[0])),
            reverse=True,
        )
        # 去掉被更长短语完全覆盖的短条目（“笑死”与“笑死了”只留一个）。
        picked: list[tuple[str, int]] = []
        for phrase, count in ranked:
            if any(phrase in longer for longer, _ in picked):
                continue
            picked.append((phrase, int(round(count))))
            if len(picked) >= max(1, int(limit)):
                break
        return picked

    def format_for_prompt(self, limit_phrases: int = 8) -> str:
        """生成注入 prompt 的【本群语感画像】；样本太少时返回空串。"""
        if not self.is_ready():
            return ""
        m = self.metrics()
        lines = [
            f"【本群语感画像（统计自最近 {int(round(m['messages']))} 条群消息）】",
            f"- 一条消息平均 {m['avg_len']} 个字（一句话说清就好，别写小作文）",
        ]
        if m["burst_ratio"] >= 0.35:
            lines.append(
                f"- 群友爱连发（{int(m['burst_ratio'] * 100)}% 的消息是 30 秒内接着说的）：可以分 2~3 条短句，而不是憋一大段"
            )
        elif m["burst_ratio"] <= 0.12:
            lines.append(
                f"- 群友很少连发（连发率 {int(m['burst_ratio'] * 100)}%）：尽量一条说完，别刷屏"
            )
        else:
            lines.append(f"- 分条节奏中等（连发率 {int(m['burst_ratio'] * 100)}%）：想说两句就分两条")
        if m["question_ratio"] >= 0.25:
            lines.append(f"- 群里问句多（{int(m['question_ratio'] * 100)}%）：接话时用问句/反问不算突兀")
        if m["emoji_ratio"] >= 0.2:
            lines.append(f"- 表情/emoji 用得不少（{int(m['emoji_ratio'] * 100)}% 带表情）：可以自然带一点")
        phrases = self.top_phrases(limit_phrases)
        if phrases:
            body = "、".join(f"{p}（{c}次）" for p, c in phrases)
            lines.append(f"- 群里的高频口头语：{body}")
        if m["top_senders"]:
            names = "、".join(f"{n}（{int(round(c))}条）" for n, c in m["top_senders"][:3])
            lines.append(f"- 最近话最多的人：{names}")
        if m["peak_hours"]:
            hours = "、".join(f"{h}点" for h in m["peak_hours"])
            lines.append(f"- 这个群最热闹的时间：{hours}")
        lines.append(
            "（以上只是统计参考，不是要照抄的模板：学它们的**长度、分条节奏、接话时机**，"
            "别把口头语堆成复读机；每个群的语感不同，先观察再融入。）"
        )
        return "\n".join(lines)

    def summary_text(self) -> str:
        """给 ``/群风`` 指令看的人类可读报告。"""
        m = self.metrics()
        if not m["messages"]:
            return "还没有观察到任何群消息。"
        head = f"样本 {int(round(m['messages']))} 条 | 平均 {m['avg_len']} 字 | 连发率 {int(m['burst_ratio'] * 100)}%"
        extra = (
            f"问句 {int(m['question_ratio'] * 100)}% / 感叹 {int(m['exclaim_ratio'] * 100)}% / 表情 {int(m['emoji_ratio'] * 100)}%"
        )
        phrases = self.top_phrases(12, min_count=2)
        phrase_line = "口癖：" + ("、".join(f"{p}×{c}" for p, c in phrases) if phrases else "（样本不足）")
        senders = "、".join(f"{n}×{int(round(c))}" for n, c in m["top_senders"]) or "（无）"
        ready = "可注入" if self.is_ready() else f"样本不足（需 {self.min_messages} 条）"
        return "\n".join([head, extra, phrase_line, f"话多的人：{senders}", f"状态：{ready}"])

    def reset(self) -> None:
        self._data = _default_profile()
        self.save()


class StyleBank:
    """按会话管理的群风统计集合。"""

    def __init__(
        self,
        directory: Path,
        options: dict[str, Any] | None = None,
        on_warning=None,
    ) -> None:
        self.directory = Path(directory)
        self.options = dict(options or {})
        self.on_warning = on_warning
        self._cache: dict[str, GroupStyleLearner] = {}

    def get(self, session_key: str) -> GroupStyleLearner:
        key = _safe(session_key)
        learner = self._cache.get(key)
        if learner is None:
            learner = GroupStyleLearner(
                self.directory / f"{key}.json",
                phrase_min_len=self.options.get("phrase_min_len", 2),
                phrase_max_len=self.options.get("phrase_max_len", 6),
                min_messages=self.options.get("learn_min_messages", 30),
                max_sample_messages=self.options.get("max_sample_messages", 2000),
                on_warning=self.on_warning,
            )
            self._cache[key] = learner
        return learner

    def flush(self) -> None:
        for learner in self._cache.values():
            learner.save()


def _safe(key: str) -> str:
    out = [ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(key or "default")]
    name = "".join(out).strip("_") or "default"
    return name[:96]


def _iter_names(senders: Iterable[dict]) -> list[str]:
    return [str(s.get("name") or "") for s in senders]
