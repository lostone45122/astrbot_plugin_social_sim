"""仿真群友状态机。

这是 qq-bridge 一代仿真（``reserved``）的核心：`观望 → 活跃 → 试探 → 退场` 四态机。
它解决的问题是「怎样才像个真人群友」，而不是「怎样尽快回复」：

- **观望（idle）**：默认状态。只有被点名、命中必回关键词、或小概率被普通闲聊带动，
  才进入活跃。大部分消息只是“看到”，不进上下文之外的任何处理。
- **活跃（active）**：参与对话中。每 10~30 秒集中看一次新消息（不是每条都回），
  由模型自己决定接不接；进入时会随机一个 15~30 分钟的最长活跃时间，
  到点主动说一句收尾话退场——否则在热闹的群里它会一直说下去。
- **试探（probing）**：活跃期冷场时，小概率说一句试试还有没有人在；
  说完 2 分钟没人接就 100% 回观望。
- **退场（exiting）**：正在说收尾话。这期间的新消息只进摘要，不接话；
  收尾发言完成（或选择沉默）后回观望。

关键设计取舍：**活跃期由后台 tick 驱动，而不是每条消息驱动**。
消息驱动会导致“群里每来一条就生成一次”，既费钱又像机器人刷屏。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .prompts import build_context_block, build_reply_prompt, split_by_freshness
from .text import clean, is_directed_at, is_silent, looks_like_unfinished, truncate

__all__ = ["SocialSimulator", "SessionState", "ChatMessage"]

IDLE = "idle"
ACTIVE = "active"
PROBING = "probing"
EXITING = "exiting"

PHASE_LABEL = {
    IDLE: "观望",
    ACTIVE: "活跃",
    PROBING: "试探",
    EXITING: "退场",
}


@dataclass
class ChatMessage:
    """一条群聊消息（仿真器视角的最小信息）。"""

    sender: str
    sender_id: str
    text: str
    ts: float
    is_self: bool = False
    message_id: str = ""
    images: list[str] = field(default_factory=list)
    """图片的本地路径（已经由插件下载/转换好），只保留仍然存在的文件。"""
    vision: bool = False
    """这些图是否已获准交给视觉模型（按需识图模式下由触发词/引用放行）。"""
    vision_used: bool = False
    """已经用掉，不再重复上传（避免每轮重发同一张图）。"""
    is_system: bool = False
    """系统通知（戳一戳/禁言/头衔…）。不参与群风统计与关系权重。"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "sender_id": self.sender_id,
            "text": self.text,
            "ts": self.ts,
            "is_self": self.is_self,
            "is_system": self.is_system,
            "images": list(self.images),
        }


@dataclass
class SessionState:
    """一个会话的仿真状态。"""

    key: str
    umo: str
    is_group: bool
    phase: str = IDLE
    recent: deque = field(default_factory=lambda: deque(maxlen=60))
    idle_summary: list[str] = field(default_factory=list)

    last_message_at: float = 0.0
    last_other_message_at: float = 0.0
    last_check_at: float = 0.0
    last_ai_reply_at: float = 0.0
    active_since: float = 0.0
    active_deadline: float = 0.0
    probe_deadline: float = 0.0
    next_proactive_at: float = 0.0
    exiting_since: float = 0.0

    busy: bool = False
    """正在生成/发送，避免重入。"""
    pending: bool = False
    """活跃期有尚未处理的新消息。"""
    send_times: deque = field(default_factory=lambda: deque(maxlen=200))
    turns: int = 0
    """本会话 AI 成功发言的轮数（用于记忆沉淀节流）。"""

    def phase_label(self) -> str:
        return PHASE_LABEL.get(self.phase, self.phase)


class SocialSimulator:
    """仿真群友的调度器。

    它不做任何 IO，所有能力都通过 ``hooks``（插件实例）注入：

    - ``hooks.build_system_prompt(key, participation=...)``
    - ``hooks.think(key, system_prompt, user_prompt, tag)``
    - ``hooks.send_parts(key, parts)``
    - ``hooks.on_ai_spoke(key, parts)``
    - ``hooks.after_turn(key)``  （记忆沉淀节流入口）
    - ``hooks.log(msg)``
    """

    def __init__(self, cfg: dict[str, Any], hooks: Any) -> None:
        self.cfg = cfg
        self.hooks = hooks
        self.states: dict[str, SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._stopped = False
        self.rng = random.Random()

    # ---------------------------------------------------------------- 配置
    def _c(self, key: str, default: Any) -> Any:
        value = self.cfg.get(key, default)
        if value is None:
            return default
        return value

    def _f(self, key: str, default: float) -> float:
        try:
            return float(self._c(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _i(self, key: str, default: int) -> int:
        try:
            return int(self._c(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _b(self, key: str, default: bool) -> bool:
        value = self._c(key, default)
        return bool(value)

    # ---------------------------------------------------------------- 生命周期
    def state(self, key: str, umo: str = "", is_group: bool = True) -> SessionState:
        st = self.states.get(key)
        if st is None:
            st = SessionState(key=key, umo=umo or key, is_group=is_group)
            st.idle_summary = []
            self.states[key] = st
        elif umo:
            st.umo = umo
        return st

    def lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def start(self, interval: float = 5.0) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._loop(interval), name="social-sim-tick")

    async def stop(self) -> None:
        self._stopped = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for bg in list(self._bg):
            bg.cancel()
        self._bg.clear()

    async def _loop(self, interval: float) -> None:
        interval = max(1.0, float(interval))
        while not self._stopped:
            try:
                await asyncio.sleep(interval)
                if self._stopped:
                    break
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 后台循环绝不能因单次异常退出
                self.hooks.log(f"仿真 tick 异常：{exc!r}")

    # ---------------------------------------------------------------- 状态迁移
    def _enter_active(self, st: SessionState, now: float) -> None:
        st.phase = ACTIVE
        st.active_since = now
        st.last_check_at = now
        st.probe_deadline = 0.0
        if self._b("active_duration_enabled", True) and st.is_group:
            lo = max(0.05, self._f("active_duration_min_min", 15.0))
            hi = max(lo, self._f("active_duration_max_min", 30.0))
            st.active_deadline = now + self.rng.uniform(lo, hi) * 60
        else:
            st.active_deadline = 0.0
        self.hooks.log(f"{st.key} 进入活跃（{st.phase_label()}）")

    def _enter_idle(self, st: SessionState, now: float) -> None:
        st.phase = IDLE
        st.active_deadline = 0.0
        st.probe_deadline = 0.0
        st.pending = False
        st.exiting_since = 0.0
        st.idle_summary.clear()
        self._schedule_proactive(st, now)
        self.hooks.log(f"{st.key} 回到观望")

    def _schedule_proactive(self, st: SessionState, now: float) -> None:
        if not self._b("proactive_enabled", True) or not st.is_group:
            st.next_proactive_at = 0.0
            return
        lo = max(1.0, self._f("proactive_check_min_min", 45.0))
        hi = max(lo, self._f("proactive_check_max_min", 90.0))
        st.next_proactive_at = now + self.rng.uniform(lo, hi) * 60

    # ---------------------------------------------------------------- 消息入口
    async def on_message(self, msg: ChatMessage, key: str, umo: str, is_group: bool) -> bool:
        """记录一条消息；返回「是否立刻触发一次发言尝试」。"""
        st = self.state(key, umo, is_group)
        now = msg.ts
        st.recent.append(msg)
        st.last_message_at = now

        if msg.is_self:
            st.last_ai_reply_at = now
            # 自己说的话也算参与：活跃期不该因为“没人回”立刻冷场。
            st.last_other_message_at = max(st.last_other_message_at, now)
            return False

        st.last_other_message_at = now

        if st.phase == EXITING:
            # 退场期间不接话，只把内容压成摘要。
            self._summarize(st, msg)
            return False

        if st.phase in (ACTIVE, PROBING):
            if st.phase == PROBING:
                # 试探期有人接话 → 确认群友还在，回到活跃。
                self._enter_active(st, now)
            st.pending = True
            return False

        # 观望：只有“明确被找”或“小概率被带动”才开口。
        direct = is_directed_at(
            msg.text,
            names=self.hooks.bot_aliases(),
            keywords=self._c("must_reply_keywords", []),
        )
        # 关系权重：经常冒犯 bot/群友的人，被接话的概率会逐步降到很低。
        mult = self._affinity(msg, key)
        if direct:
            if mult >= 0.6 or self.rng.random() <= max(0.15, mult):
                self._enter_active(st, now)
                return True
            # 被点名也冷淡：不是拉黑，只是"这次不想理"，仍会回到观望后正常参与别人。
            self.hooks.log(f"{st.key} 被点名但关系权重低（{mult:.2f}），本轮不接")
            self._summarize(st, msg)
            return False
        probability = max(0.0, min(1.0, self._f("trigger_probability", 0.12))) * mult
        if self.rng.random() < probability:
            self._enter_active(st, now)
            return True
        self._summarize(st, msg)
        return False

    def _affinity(self, msg: ChatMessage | None, key: str = "") -> float:
        """取这个人的关系权重乘数；没有该能力时按 1.0（不降权）。"""
        if msg is None:
            return 1.0
        getter = getattr(self.hooks, "affinity_multiplier", None)
        if not callable(getter):
            return 1.0
        try:
            value = float(getter(key or msg.sender_id, msg.sender_id, msg.sender))
        except Exception:  # noqa: BLE001 - 权重计算失败绝不影响正常聊天
            return 1.0
        return max(0.0, min(1.0, value))

    def _summarize(self, st: SessionState, msg: ChatMessage) -> None:
        """观望期把“路过看到的东西”压成摘要，避免上下文无限膨胀。"""
        limit = self._i("summary_max_lines", 12)
        line = f"{msg.sender}：{truncate(clean(msg.text), 80)}"
        st.idle_summary.append(line)
        if len(st.idle_summary) > limit:
            del st.idle_summary[: len(st.idle_summary) - limit]

    # ---------------------------------------------------------------- 生成
    def build_blocks(self, st: SessionState, reason: str, trigger: str = "") -> tuple[str, str]:
        """生成 (system_prompt, user_prompt)。

        上下文按"新鲜度"切成两段：还热着的进【刚刚到现在】，超过 ``fresh_window_sec``
        的进【更早的消息】并明确标注。这是"旧事重提"的关键修复——以前所有消息长得
        一样，模型分不清哪句是三小时前的，冷场时就把旧话当新话接。
        """
        ctx_limit = self._i("context_window", 20)
        older_limit = self._i("older_context_limit", 6)
        fresh_sec = self._f("fresh_window_sec", 600.0)
        now = time.time()
        fresh, older = split_by_freshness(
            [m.as_dict() for m in st.recent], now=now, fresh_sec=fresh_sec
        )
        context_block = build_context_block(
            fresh, limit=ctx_limit, now=now, with_time=True
        )
        older_block = build_context_block(
            older, limit=older_limit, max_chars=200, now=now, with_time=True
        )
        if not context_block and st.idle_summary and not older_block:
            # 观望摘要本身就是"更早看到的东西"，只能当背景，不能当现在。
            older_block = "\n".join(st.idle_summary)
        ai_count, other_count, burst_recent, senders, direct_hits = self._activity(st)
        participation = ""
        try:
            from .prompts import _participation_hint  # 延迟导入，避免循环

            participation = _participation_hint(
                ai_count=ai_count,
                other_count=other_count,
                last_ai_gap_s=time.time() - (st.last_ai_reply_at or 0) if st.last_ai_reply_at else 9999,
                recent_ai_2min=sum(
                    1
                    for m in st.recent
                    if m.is_self and time.time() - m.ts < 120
                ),
                burst_recent=burst_recent,
                active_senders=senders,
                direct_hits=direct_hits,
            )
        except Exception:  # noqa: BLE001
            participation = ""
        system_prompt = self.hooks.build_system_prompt(
            st.key, participation=participation, is_group=st.is_group
        )
        hints: list[str] = []
        if reason == "proactive":
            hints.append(
                "群里已经安静了一段时间，这是一次你可以主动冒泡的机会：开个新话题、"
                "或者分享一个刚想到的念头。**不要捞很久以前的旧话题**（那是旧事重提），"
                "想不到就别硬说。"
            )
        elif reason == "probe":
            hints.append("你刚说了一句没人接，可以再轻轻试一句（别连发，别问“有人吗”）；不想再试就直接潜水。")
        elif reason == "exit":
            hints.append("你已经参与有一阵子了，现在该自然地收尾/潜水了：说一句简短的退场话（例如“我先潜水了”“去忙了”），之后就安静下来。")
        elif reason == "poke":
            hints.append(
                "有人刚拍了拍你——这是明确冲你来的。可以自然回一句（“干嘛”“拍我干嘛”），"
                "也可以回戳/装作没注意；不要长篇大论。"
            )
        elif reason == "idle_trigger":
            hints.append("这条消息像是冲你来的，或者你确实有话想接。")
        user_prompt = build_reply_prompt(
            context_block=context_block,
            trigger=trigger or ("（群里暂时没有新消息）" if reason in ("proactive", "probe") else ""),
            reason=self._reason_text(reason),
            is_group=st.is_group,
            silent_token=self._c("silent_token", "[SILENT]"),
            hints=hints,
            older_block=older_block,
            time_aware=True,
        )
        return system_prompt, user_prompt

    @staticmethod
    def _reason_text(reason: str) -> str:
        return {
            "idle_trigger": "群里有人在说话，而且这一轮轮到你决定要不要参与",
            "active": "群里来了新消息，你在活跃期，可以决定要不要接",
            "proactive": "群安静了很久，你获得一次主动开口的机会",
            "probe": "你刚才说的话没人接，冷场了",
            "exit": "你已经聊了一阵子，到了该收尾的时候",
            "poke": "有人拍了拍你",
        }.get(reason, reason)

    def _activity(self, st: SessionState) -> tuple[int, int, int, int, int]:
        now = time.time()
        hour_msgs = [m for m in st.recent if 0 <= now - m.ts < 3600]
        ai_count = sum(1 for m in hour_msgs if m.is_self)
        others = [m for m in hour_msgs if not m.is_self]
        other_count = len(others)
        five = [m for m in st.recent if 0 <= now - m.ts < 300 and not m.is_self]
        burst_recent = len(five)
        senders = len({m.sender_id or m.sender for m in five})
        direct_hits = sum(
            1
            for m in five
            if is_directed_at(
                m.text,
                names=self.hooks.bot_aliases(),
                keywords=self._c("must_reply_keywords", []),
            )
        )
        return ai_count, other_count, burst_recent, senders, direct_hits

    async def speak(self, key: str, reason: str, trigger: str = "") -> bool:
        """生成并发送一轮发言；返回是否真的发出了消息。"""
        st = self.states.get(key)
        if st is None:
            return False
        lock = self.lock(key)
        if lock.locked():
            # 已经有一轮在跑，别叠加。
            return False
        async with lock:
            if st.busy:
                return False
            st.busy = True
            # 先把“待处理”标记清掉：即使这一轮模型选择潜水，也不该在下一个 tick 立刻重试，
            # 否则会变成每 5 秒问一次模型“要不要说话”，既费钱又像机器人。
            st.pending = False
            st.last_check_at = time.time()
            try:
                system_prompt, user_prompt = self.build_blocks(st, reason, trigger)
                images: list[str] = []
                getter = getattr(self.hooks, "images_for_turn", None)
                if callable(getter):
                    try:
                        images = [str(p) for p in (getter(key) or []) if p]
                    except Exception:  # noqa: BLE001 - 取图失败不该阻断发言
                        images = []
                try:
                    raw = await self.hooks.think(
                        key, system_prompt, user_prompt, reason, image_urls=images
                    )
                except Exception as exc:  # noqa: BLE001
                    # 模型调用失败必须按"空回合"处理：否则退场/试探这些依赖返回值
                    # 收尾的分支会被跳过，会话钉在 EXITING 直到 60 分钟兜底。
                    self.hooks.log(f"{key} 生成本轮失败（{reason}）：{exc!r}")
                    raw = ""
                silent_token = str(self._c("silent_token", "[SILENT]"))
                if not raw or not str(raw).strip():
                    self.hooks.log(f"{key} 空回合（{reason}）")
                    if reason == "exit" and st.phase == EXITING:
                        # 退场回合失败就安静回去，别在收尾时突然冒一句"网卡了"。
                        self._enter_idle(st, time.time())
                        return False
                    # 该说话却一个字都没生成（模型报错/被审核拒绝/空回复）：
                    # 与其让群友对着空气等，不如用一句人话把场子圆过去。
                    fallback = getattr(self.hooks, "error_line", None)
                    line = ""
                    if callable(fallback):
                        try:
                            line = str(fallback(key) or "").strip()
                        except Exception:  # noqa: BLE001
                            line = ""
                    if line:
                        sent_fallback = await self.hooks.send_parts(key, line)
                        if sent_fallback:
                            st.last_ai_reply_at = time.time()
                            st.last_check_at = time.time()
                    return False
                if is_silent(raw, silent_token):
                    self.hooks.log(f"{key} 选择潜水（{reason}）")
                    if reason == "exit" and st.phase == EXITING:
                        self._enter_idle(st, time.time())
                    return False
                sent = await self.hooks.send_parts(key, raw)
                if not sent:
                    if reason == "exit" and st.phase == EXITING:
                        self._enter_idle(st, time.time())
                    return False
                now = time.time()
                st.last_ai_reply_at = now
                st.last_check_at = now
                st.pending = False
                st.turns += 1
                for part in sent:
                    st.recent.append(
                        ChatMessage(
                            sender=self.hooks.bot_name(),
                            sender_id="self",
                            text=part,
                            ts=now,
                            is_self=True,
                        )
                    )
                self.hooks.on_ai_spoke(key, sent)
                if reason == "exit" and st.phase == EXITING:
                    self._enter_idle(st, now)
                try:
                    await self.hooks.after_turn(key)
                except Exception as exc:  # noqa: BLE001
                    self.hooks.log(f"{key} 记忆沉淀失败（不影响聊天）：{exc!r}")
                return True
            finally:
                st.busy = False

    # ---------------------------------------------------------------- 定时驱动
    async def tick(self, now: float | None = None) -> None:
        """驱动所有会话的状态机。

        每个会话单独起一个 task：一次 LLM 调用可能几十秒，串行 tick 会让
        “活跃群 A 在生成”把“冷场群 B 该收尾”一起拖住。
        """
        now = time.time() if now is None else now
        for st in list(self.states.values()):
            task = asyncio.create_task(self._tick_one_safe(st, now))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    async def _tick_one_safe(self, st: SessionState, now: float) -> None:
        try:
            await self._tick_one(st, now)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.hooks.log(f"{st.key} 状态机异常：{exc!r}")

    async def _tick_one(self, st: SessionState, now: float) -> None:
        self._forget_if_idle(st, now)
        if st.busy:
            return
        if st.phase == IDLE:
            await self._tick_idle(st, now)
        elif st.phase == ACTIVE:
            await self._tick_active(st, now)
        elif st.phase == PROBING:
            if st.probe_deadline and now > st.probe_deadline:
                self.hooks.log(f"{st.key} 试探无回应，回到观望")
                self._enter_idle(st, now)
        elif st.phase == EXITING:
            # 极端兜底：60 分钟还没收尾完成（例如模型一直超时），强制回观望。
            if st.exiting_since and now - st.exiting_since > 3600:
                self.hooks.log(f"{st.key} 退场等待超时，强制回到观望")
                self._enter_idle(st, now)

    def _forget_if_idle(self, st: SessionState, now: float) -> None:
        """安静太久就自然遗忘：丢弃旧上下文，只留最近几条。

        这正是"旧事重提"的触发场景——群里聊了很多，bot 插不上嘴，消息全留在缓冲里；
        等活跃人数一降，模型看着几小时前的对话，就当成"正在聊的事"接着说起来。
        与其让模型自己克制，不如在数据层把旧上下文扔掉。
        """
        forget_after = self._f("forget_after_idle_sec", 1800.0)
        if forget_after <= 0:
            return
        last = max(
            st.last_message_at or 0.0,
            st.last_other_message_at or 0.0,
            st.last_ai_reply_at or 0.0,
        )
        if not last or now - last <= forget_after:
            return
        keep = max(1, self._i("keep_after_forget", 5))
        dropped = 0
        while len(st.recent) > keep:
            st.recent.popleft()
            dropped += 1
        if st.idle_summary:
            st.idle_summary.clear()
            dropped += 1
        if dropped:
            self.hooks.log(
                f"{st.key} 安静了 {int((now - last) / 60)} 分钟，丢弃旧上下文（自然遗忘，{dropped} 条）"
            )

    async def _tick_idle(self, st: SessionState, now: float) -> None:
        if not self._b("proactive_enabled", True) or not st.is_group:
            return
        if not st.next_proactive_at:
            self._schedule_proactive(st, now)
            return
        if now < st.next_proactive_at:
            return
        # 安静够久才算“冷场机会”，否则重新排下一次。
        idle_threshold = max(60.0, self._f("proactive_idle_min_min", 15.0) * 60)
        quiet = now - (st.last_other_message_at or 0)
        self._schedule_proactive(st, now)
        if quiet < idle_threshold:
            return
        probability = max(0.0, min(1.0, self._f("proactive_probability", 0.2)))
        if self.rng.random() >= probability:
            return
        self._enter_active(st, now)
        await self.speak(st.key, "proactive")

    async def _tick_active(self, st: SessionState, now: float) -> None:
        # 1) 活跃超时：到点主动收尾退场（仅群聊）。
        if st.active_deadline and now >= st.active_deadline:
            st.phase = EXITING
            st.exiting_since = now
            st.active_deadline = 0.0
            self.hooks.log(f"{st.key} 活跃超时，提示收尾退场")
            await self.speak(st.key, "exit")
            return

        # 2) 有新消息 → 按 10~30 秒的节奏集中处理一次。
        gap_min = max(3.0, self._f("active_check_min_sec", 10.0))
        gap_max = max(gap_min, self._f("active_check_max_sec", 30.0))
        since_check = now - (st.last_check_at or 0)
        if st.pending and since_check >= self.rng.uniform(gap_min, gap_max):
            skip = max(0.0, min(0.9, self._f("skip_probability", 0.15)))
            if self.rng.random() < skip:
                st.pending = False
                st.last_check_at = now
                return
            # 抢话保护：最后一条是别人、而且话没说完/刚发不久，就再等等。
            last_other = next(
                (m for m in reversed(st.recent) if not m.is_self), None
            )
            if last_other is not None:
                quiet = now - last_other.ts
                min_quiet = max(1.0, self._f("min_quiet_after_sec", 6.0))
                if looks_like_unfinished(last_other.text):
                    min_quiet = max(min_quiet, min_quiet * 1.5)
                if quiet < min_quiet:
                    return
                # 关系权重低的人带来的消息，不值得每次都认真回。
                mult = self._affinity(last_other, st.key)
                if mult < 1.0 and self.rng.random() > mult:
                    st.pending = False
                    st.last_check_at = now
                    self.hooks.log(f"{st.key} 活跃期跳过（对方关系权重 {mult:.2f}）")
                    return
            await self.speak(st.key, "active")
            return

        # 3) 冷场：一段时间没人说话。
        idle_window = max(30.0, self._f("idle_window_sec", 360.0))
        quiet = now - (st.last_other_message_at or st.last_message_at or now)
        if quiet < idle_window:
            return
        probability = max(0.0, min(1.0, self._f("idle_retry_probability", 0.25)))
        if self.rng.random() < probability:
            st.phase = PROBING
            st.probe_deadline = now + max(30.0, self._f("idle_retry_wait_sec", 120.0))
            self.hooks.log(f"{st.key} 冷场，试探性说一句")
            await self.speak(st.key, "probe")
        else:
            self.hooks.log(f"{st.key} 冷场，回到观望")
            self._enter_idle(st, now)

    # ---------------------------------------------------------------- 外部控制
    def forget(self, key: str) -> bool:
        """让这个会话“失忆”：清对话缓冲、摘要、轮数与所有计时，回到观望。

        为什么必须清**对话缓冲**：记忆库清空后，提示词里仍然会带上最近
        ``context_window`` 条群聊原文，模型完全可以从这些原文重新推断出
        “谁爱报菜名”“这个群爱连发”——用户看到的就是“我明明清了，它又把画像
        读回来了”。更糟的是记忆沉淀也拿这段缓冲当素材，不清就会在下一轮
        自动把“印象”写回去。

        不清的：``send_times``（发送频率限制是安全阀，不属于记忆）。
        """
        st = self.states.get(key)
        if st is None:
            return False
        st.recent.clear()
        st.idle_summary.clear()
        st.turns = 0
        st.pending = False
        st.phase = IDLE
        st.last_message_at = 0.0
        st.last_other_message_at = 0.0
        st.last_ai_reply_at = 0.0
        st.last_check_at = 0.0
        st.active_since = 0.0
        st.active_deadline = 0.0
        st.probe_deadline = 0.0
        st.exiting_since = 0.0
        st.next_proactive_at = 0.0
        self._schedule_proactive(st, time.time())
        return True

    def force_phase(self, key: str, phase: str, umo: str = "", is_group: bool = True) -> str:
        st = self.state(key, umo, is_group)
        now = time.time()
        if phase == IDLE:
            self._enter_idle(st, now)
        elif phase == ACTIVE:
            self._enter_active(st, now)
        else:
            st.phase = phase
        return st.phase_label()

    def snapshot(self, key: str) -> dict[str, Any]:
        st = self.states.get(key)
        if st is None:
            return {"phase": IDLE, "phase_label": PHASE_LABEL[IDLE], "recent": 0}
        return {
            "phase": st.phase,
            "phase_label": st.phase_label(),
            "recent": len(st.recent),
            "turns": st.turns,
            "last_ai_reply_at": st.last_ai_reply_at,
            "active_deadline": st.active_deadline,
            "probe_deadline": st.probe_deadline,
            "next_proactive_at": st.next_proactive_at,
            "summary": list(st.idle_summary),
        }
