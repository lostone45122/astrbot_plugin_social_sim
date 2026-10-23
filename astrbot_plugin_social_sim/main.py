"""AstrBot 群友仿真插件：仿真模式 / 学习群友策略 / 记忆。

核心玩法移植自 `Derpyu520/qq-bridge`（QQ ↔ DeepSeek Harness 桥接器）的三块能力：

1. **仿真模式**：``reserved`` 一代仿真的「观望 / 活跃 / 试探 / 退场」状态机。
   只有当被点名、命中必回词、或小概率被带动时才开口；活跃一段时间后主动收尾退场；
   冷场时小概率试探；模型可以输出 ``[SILENT]`` 选择潜水。
2. **学习群友策略**：本地统计群里的说话节奏（平均长度、连发率、问句比例、口癖、
   活跃时段），把结果作为「参考」注入提示词，让模型学群友的语感而不是照抄。
3. **记忆**：进行中的话题 / 想说未说的话 / 对群友的印象。带 TTL 与硬上限，
   并用一次低频的“记忆沉淀”调用自动维护（也可用指令手动查看/清空）。

对 qq-bridge 的必要改造（AstrBot 没有 agent 工具协议，只有一次性生成）：

- 输出协议从“调用 qq_send_message 工具”改成“**一行 = 一条消息**”+ ``[SILENT]``。
- 记忆维护从“模型自己调 qq_memory_append”改成“每 N 轮跑一次 JSON 沉淀”。
- 群风从“让模型自己观察”升级为“先统计，再交给模型参考”。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, StarTools

try:  # 图片组件：不同版本导出位置稳定，缺失时多模态功能自动降级
    from astrbot.api.message_components import Image as _ImageComponent
except Exception:  # noqa: BLE001 - pragma: no cover
    _ImageComponent = None  # type: ignore[assignment]

from .core.affinity import AffinityBank, detect_offense, detect_positive
from .core.events import describe_notice
from .core.growth import PersonaGrowthBank, build_growth_prompt, parse_growth_json
from .core.memory import MemoryBank
from .core.prompts import (
    build_attitude_block,
    build_context_block,
    build_extraction_prompt,
    build_system_prompt,
)
from .core.simulator import ACTIVE, IDLE, ChatMessage, SocialSimulator
from .core.store import JsonStore
from .core.style import StyleBank
from .core.suppress import SuppressBank, parse_suppress_request
from .core.text import clean, extract_json, split_bursts, truncate, wants_vision

__all__ = ["SocialSimPlugin"]

PLUGIN_NAME = "astrbot_plugin_social_sim"
MODE_SIMULATION = "simulation"
MODE_ASSIST = "assist"


def _default_sessions() -> dict[str, Any]:
    return {}


class SocialSimPlugin(Star):
    """群友仿真插件（仿真 / 学群风 / 记忆）。"""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config: dict[str, Any] = dict(config or {})
        self._closed = False
        self._data_dir = self._resolve_data_dir()
        self.sessions_store = JsonStore(
            self._data_dir / "sessions.json", _default_sessions, on_warning=self._warn
        )
        self.sessions: dict[str, dict[str, Any]] = self._load_sessions()

        sim_cfg = dict(self.config.get("simulation") or {})
        self.sim = SocialSimulator(sim_cfg, self)

        mem_cfg = dict(self.config.get("memory") or {})
        self.memory = MemoryBank(
            self._data_dir / "memory", mem_cfg, on_warning=self._warn
        )
        style_cfg = dict(self.config.get("style") or {})
        self.style = StyleBank(self._data_dir / "style", style_cfg, on_warning=self._warn)
        affinity_cfg = dict(self.config.get("affinity") or {})
        self.affinity = AffinityBank(
            self._data_dir / "affinity", affinity_cfg, on_warning=self._warn
        )
        suppress_cfg = dict(self.config.get("suppress") or {})
        self.suppress = SuppressBank(
            self._data_dir / "suppress", suppress_cfg, on_warning=self._warn
        )
        growth_cfg = dict(self.config.get("growth") or {})
        self.growth = PersonaGrowthBank(
            self._data_dir / "growth", growth_cfg, on_warning=self._warn
        )

        self._last_extract: dict[str, float] = {}
        self._rng = random.Random()
        self._wake_prefixes = ["/"]
        self._background: set[asyncio.Task] = set()
        self._mute_until: dict[str, float] = {}
        self._last_error_line: dict[str, float] = {}
        self._line_cursor: dict[str, int] = {}
        self._last_growth: dict[str, float] = {}

    # ================================================================ 基础设施
    @staticmethod
    def _resolve_data_dir() -> Path:
        try:
            return StarTools.get_data_dir()
        except Exception:  # noqa: BLE001 - 解析失败时退回显式插件名
            try:
                return StarTools.get_data_dir("astrbot_plugin_social_sim")
            except Exception:  # noqa: BLE001
                path = Path.cwd() / "data" / "plugin_data" / "astrbot_plugin_social_sim"
                path.mkdir(parents=True, exist_ok=True)
                return path

    def _warn(self, message: str) -> None:
        try:
            self.logger.warning(f"[群友仿真] {message}")
        except Exception:  # noqa: BLE001
            logger.warning(f"[群友仿真] {message}")

    def log(self, message: str) -> None:
        if self.config.get("debug", False):
            self._warn(message)

    def _load_sessions(self) -> dict[str, dict[str, Any]]:
        raw = self.sessions_store.load()
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                out[str(key)] = {
                    "enabled": value.get("enabled"),
                    "persona": str(value.get("persona") or ""),
                    "env": str(value.get("env") or ""),
                }
        return out

    def _save_sessions(self) -> None:
        self.sessions_store.save(self.sessions)

    def _session_cfg(self, key: str) -> dict[str, Any]:
        return self.sessions.setdefault(
            key, {"enabled": None, "persona": "", "env": ""}
        )

    def _forget_context(self, key: str) -> None:
        """丢弃该会话的对话缓冲与记忆沉淀计时。

        清记忆/清画像之后**必须**跟着调用它，否则会出现“清了又被读回来”：

        1. 提示词每轮都会带上最近 ``context_window`` 条群聊原文，模型能从里面
           重新推断出“谁爱报菜名”“这个群爱连发”；
        2. 记忆沉淀也是拿这段缓冲当素材，不清就会在下一次沉淀时把印象写回去。
        """
        with contextlib.suppress(Exception):
            self.sim.forget(key)
        self._last_extract.pop(key, None)

    # ================================================================ 生命周期
    async def initialize(self) -> None:
        try:
            prefixes = self.context.get_config().get("wake_prefix")
            if isinstance(prefixes, list) and prefixes:
                self._wake_prefixes = [str(p) for p in prefixes if str(p)]
        except Exception:  # noqa: BLE001
            pass
        self._closed = False
        await self.sim.start(float(self.config.get("tick_interval_sec", 5) or 5))
        self.log("插件已启动，仿真状态机开始运行")

    async def terminate(self) -> None:
        self._closed = True
        await self.sim.stop()
        with contextlib.suppress(Exception):
            self.memory.flush()
        with contextlib.suppress(Exception):
            self.style.flush()
        with contextlib.suppress(Exception):
            self.affinity.flush()
        with contextlib.suppress(Exception):
            self.suppress.flush()
        with contextlib.suppress(Exception):
            self.growth.flush()
        with contextlib.suppress(Exception):
            self._save_sessions()
        self.log("插件已停止")

    # ================================================================ 配置读取
    def _mode(self) -> str:
        mode = str(self.config.get("mode") or MODE_SIMULATION).strip().lower()
        return mode if mode in (MODE_SIMULATION, MODE_ASSIST) else MODE_SIMULATION

    def bot_name(self) -> str:
        return str(self.config.get("bot_name") or "群友").strip() or "群友"

    def bot_aliases(self) -> list[str]:
        names = [self.bot_name()]
        aliases = self.config.get("bot_aliases") or []
        if isinstance(aliases, str):
            aliases = [a for a in aliases.replace("，", ",").split(",")]
        if isinstance(aliases, list):
            names.extend(str(a).strip() for a in aliases if str(a).strip())
        return list(dict.fromkeys(names))

    def _session_enabled(self, key: str, is_group: bool) -> bool:
        """某个会话要不要被本插件接管。

        优先级：会话级开关 > 全局总开关 > 群/私聊白名单。
        空白名单 = 全部放行（与 qq-bridge 的 fail-closed 相反，因为 AstrBot 的
        插件默认就对所有会话生效，默认不开反而让人以为插件坏了）。
        """
        cfg = self.sessions.get(key) or {}
        override = cfg.get("enabled")
        if override is not None:
            return bool(override)
        if not self.config.get("enabled", True):
            return False
        if is_group:
            groups = self._as_str_list(self.config.get("group_whitelist"))
            return (not groups) or self._group_in(groups, key)
        if not self.config.get("private_enabled", False):
            return False
        allowed = self._as_str_list(self.config.get("private_whitelist"))
        return (not allowed) or any(str(x) in key for x in allowed)

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            items = value.replace("，", ",").replace("\n", ",").split(",")
        elif isinstance(value, list):
            items = value
        else:
            return []
        return [str(item).strip() for item in items if str(item).strip()]

    @staticmethod
    def _group_in(whitelist: list[str], key: str) -> bool:
        group_id = key.split(":", 1)[-1]
        return any(item in (group_id, key, f"group:{group_id}") for item in whitelist)

    # ================================================================ 出错时的"人话"
    def _errors_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("errors") or {})

    def _is_llm_error_text(self, text: str) -> bool:
        """判断这段文本是不是 AstrBot 的 LLM 报错。

        只匹配确切的几种，宁可漏也不要把正常回复改掉。AstrBot 在
        ``tool_loop_agent_runner`` 里把失败写成 ``LLM 响应错误: …``；
        provider 层的异常串也可能直接透出来。
        """
        if not text:
            return False
        markers = self._as_str_list(self._errors_cfg().get("error_markers")) or [
            "LLM 响应错误",
            "All chat models failed",
            "chat models failed",
            "Gemini request failed",
        ]
        return any(marker and marker in text for marker in markers)

    def error_line(self, key: str) -> str:
        """挑一句"网卡了/没看清"的人话；节流后返回空串表示这次不说。"""
        cfg = self._errors_cfg()
        if not cfg.get("enabled", True) or not cfg.get("send_on_failure", True):
            return ""
        interval = float(cfg.get("min_interval_sec", 90) or 0)
        now = time.time()
        if interval > 0 and now - self._last_error_line.get(key, 0) < interval:
            return ""
        lines = self._as_str_list(cfg.get("messages"))
        if not lines:
            lines = [
                "刚网卡了一下，你再说一遍？",
                "啊？没看清，重发一下",
                "我这边转圈圈了，等我一会儿",
                "眼睛花了，你刚才说啥",
            ]
        self._last_error_line[key] = now
        return str(self._rng.choice(lines))

    def _pick_busy_line(self, key: str) -> str:
        """给"平台层报错"用的替换句：同样按次数轮换，不按时间节流。

        平台报错是用户可见的失败，必须换掉；但连续失败时也不要反复同一句。
        """
        cfg = self._errors_cfg()
        lines = self._as_str_list(cfg.get("messages"))
        if not lines:
            lines = ["刚网卡了一下，你再说一遍？"]
        self._line_cursor[key] = (self._line_cursor.get(key, 0) + 1) % max(1, len(lines))
        return str(lines[self._line_cursor[key]])

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """把 AstrBot 默认链路的 LLM 报错换成一句人话。

        AstrBot 原生也支持这件事（人格里的 ``custom_error_message``），
        那个优先级更高：设了的话错误文本已经是自定义内容，这里匹配不上就不会动它。
        这里是"没配也能看"的兜底，覆盖所有会话（包括本插件没接管的）。
        """
        cfg = self._errors_cfg()
        if not cfg.get("enabled", True) or not cfg.get("replace_platform_errors", True):
            return
        try:
            result = event.get_result()
            if result is None or not getattr(result, "chain", None):
                return
            if getattr(result, "async_stream", None) is not None:
                return  # 流式结果不在这里改
            text = "".join(
                str(getattr(comp, "text", "") or "")
                for comp in result.chain
                if getattr(comp, "text", None) is not None
            )
            if not self._is_llm_error_text(text):
                return
            is_group = not event.is_private_chat()
            key = self._session_key(event, is_group) or "unknown"
            line = self._pick_busy_line(key)
            event.set_result(event.plain_result(line))
            self._warn(f"{key} LLM 报错已替换为：{line}")
        except Exception as exc:  # noqa: BLE001 - 兜底链路绝不能自己炸
            self.log(f"替换报错消息失败：{exc!r}")

    # ================================================================ 多模态 / 系统消息
    def _multimodal_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("multimodal") or {})

    def _system_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("system_events") or {})

    def _name_of(self, key: str, uid: str) -> str:
        """QQ 号 → 昵称。系统通知里只有号，用最近消息里见过的昵称补上。"""
        target = str(uid or "")
        if not target:
            return ""
        st = self.sim.states.get(key)
        if st is not None:
            for msg in reversed(st.recent):
                if msg.sender_id == target and not msg.is_system and msg.sender:
                    return msg.sender
        return ""

    async def _collect_images(self, event: AstrMessageEvent) -> list[str]:
        """把这条消息里的图片转成本地路径（AstrBot 的标准多模态入口）。

        用 ``Image.convert_to_file_path()`` 而不是自己下 URL：它统一处理
        http/base64/file:// 三种来源，并带下载缓存。
        """
        cfg = self._multimodal_cfg()
        if not cfg.get("enabled", True) or _ImageComponent is None:
            return []
        limit = int(cfg.get("max_images", 3) or 3)
        if limit <= 0:
            return []
        out: list[str] = []
        with contextlib.suppress(Exception):
            for comp in event.get_messages():
                if not isinstance(comp, _ImageComponent):
                    continue
                try:
                    path = await comp.convert_to_file_path()
                except Exception as exc:  # noqa: BLE001 - 单张图失败不影响这条消息
                    self.log(f"图片获取失败：{exc!r}")
                    continue
                if path and os.path.isfile(str(path)):
                    out.append(str(path))
                if len(out) >= limit:
                    break
        return out

    def images_for_turn(self, key: str) -> list[str]:
        """本轮要交给视觉模型的图片。

        两种模式：

        - ``auto``：最近几条消息里的图都带上（旧行为，费 token、容易污染印象）。
        - ``on_demand``（默认）：**只带已获准的图**——被引用过的、或被点名要求看的。
          取走一次就标记 ``vision_used``，不会每轮重复上传同一张图。
        """
        cfg = self._multimodal_cfg()
        if not cfg.get("enabled", True):
            return []
        if self._vision_mode() == "on_demand":
            return self._take_approved_images(key)
        limit = int(cfg.get("max_images", 3) or 3)
        if limit <= 0:
            return []
        st = self.sim.states.get(key)
        if st is None:
            return []
        include_recent = bool(cfg.get("include_recent_images", True))
        source = list(st.recent)
        if not include_recent:
            source = source[-1:] if source else []
        out: list[str] = []
        for msg in reversed(source):
            for path in reversed(msg.images):
                if path and path not in out and os.path.isfile(path):
                    out.append(path)
                if len(out) >= limit:
                    return out
        return out

    # ------------------------------------------------ 按需识图的判定与放行
    def _vision_enabled(self) -> bool:
        return bool(self._multimodal_cfg().get("enabled", True)) and _ImageComponent is not None

    def _vision_mode(self) -> str:
        mode = str(self._multimodal_cfg().get("mode") or "on_demand").strip().lower()
        return mode if mode in ("auto", "on_demand") else "on_demand"

    @staticmethod
    def _quoted_ids(event: AstrMessageEvent) -> list[str]:
        """取出这条消息引用（回复）了哪些消息 id。

        不 import ``Reply`` 类：按类名 + ``id`` 属性识别，跨版本更稳。
        """
        out: list[str] = []
        with contextlib.suppress(Exception):
            for comp in event.get_messages():
                # 不 import Reply 类：按类名识别，跨版本/子类都能命中。
                if "reply" not in type(comp).__name__.lower():
                    continue
                rid = getattr(comp, "id", None)
                if rid is None:
                    continue
                out.append(str(rid))
        return out

    def _approve_quoted(self, key: str, quoted_ids: list[str]) -> bool:
        """引用的是含图消息 → 放行那张图。引用是最明确的"我在说这张图"。"""
        if not quoted_ids:
            return False
        st = self.sim.states.get(key)
        if st is None:
            return False
        target = set(quoted_ids)
        approved = False
        for msg in reversed(st.recent):
            if msg.vision_used or not msg.images:
                continue
            if msg.message_id and msg.message_id in target:
                msg.vision = True
                approved = True
        return approved

    def _is_vision_invite(self, text: str) -> bool:
        cfg = self._multimodal_cfg()
        return wants_vision(
            text,
            names=self.bot_aliases(),
            extra_words=self._as_str_list(cfg.get("invite_words")),
            require_name=bool(cfg.get("invite_requires_name", True)),
        )

    def _approve_latest_pending(self, key: str) -> bool:
        """没引用，但明确说了"你看这张" → 放行最近那张还没看过的图。"""
        st = self.sim.states.get(key)
        if st is None:
            return False
        ttl = float(self._multimodal_cfg().get("pending_ttl_sec", 300) or 300)
        now = time.time()
        for msg in reversed(st.recent):
            if not msg.images or msg.vision_used or msg.vision:
                continue
            if ttl > 0 and now - msg.ts > ttl:
                return False  # 太久了，别把半小时前的图当"这张"
            msg.vision = True
            return True
        return False

    def _take_approved_images(self, key: str) -> list[str]:
        """取走已获准的图；默认用一次就标记，避免每轮重复上传。"""
        cfg = self._multimodal_cfg()
        limit = int(cfg.get("vision_max_images", 1) or 1)
        if limit <= 0:
            return []
        consume = bool(cfg.get("consume_after_view", True))
        st = self.sim.states.get(key)
        if st is None:
            return []
        out: list[str] = []
        for msg in reversed(st.recent):
            if not msg.vision or msg.vision_used:
                continue
            added = False
            for path in msg.images:
                if len(out) >= limit:
                    break
                if path and path not in out and os.path.isfile(path):
                    out.append(path)
                    added = True
            if added and consume:
                msg.vision_used = True
            if len(out) >= limit:
                break
        return out

    async def _handle_notice(self, key: str, umo: str, is_group: bool, notice: dict[str, Any]) -> None:
        """处理戳一戳 / 禁言 / 头衔等系统通知。"""
        text = str(notice.get("text") or "").strip()
        if not text:
            return
        if notice.get("muted_self"):
            seconds = int(notice.get("mute_seconds") or 0)
            if seconds > 0:
                self._mute_until[key] = time.time() + seconds
                self._warn(f"{key} 机器人被禁言 {seconds} 秒，期间暂停发送")
            return
        if self._mode() == MODE_ASSIST:
            return
        sender_name = str(notice.get("sender_name") or "系统")
        sender_id = str(notice.get("sender_id") or "system")
        st = self.sim.state(key, umo, is_group)
        st.recent.append(
            ChatMessage(
                sender=sender_name,
                sender_id=sender_id or "system",
                text=text,
                ts=time.time(),
                is_self=False,
                is_system=True,
            )
        )
        st.last_message_at = time.time()
        cfg = self._system_cfg()
        if not notice.get("poke_me") or not cfg.get("poke_trigger", True):
            self.log(f"{key} 系统通知：{text}")
            return
        # 被戳：这是明确冲我来的，按配置的概率决定要不要回
        prob = float(cfg.get("poke_trigger_probability", 1.0) or 0)
        self.sim.force_phase(key, ACTIVE, umo, is_group)
        if self.sim.rng.random() > max(0.0, min(1.0, prob)):
            self.log(f"{key} 被戳，但这轮选择不接（概率 {prob:.2f}）")
            return
        task = asyncio.create_task(
            self.sim.speak(key, "poke", trigger=text), name=f"social-sim-poke-{key}"
        )
        self._track(task)

    # ================================================================ 叫停机制
    def _suppress_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("suppress") or {})

    def suppress_enabled(self) -> bool:
        return bool(self._suppress_cfg().get("enabled", True))

    def _detect_suppress(self, key: str, text: str, sender_name: str) -> str | None:
        """识别"别再说了"并记下来；返回被叫停的关键词。"""
        if not self.suppress_enabled():
            return None
        cfg = self._suppress_cfg()
        keyword = parse_suppress_request(text)
        if not keyword:
            return None
        # 可选：要求先点名，避免把对别人说的话当成对自己的要求
        if bool(cfg.get("require_name", False)) and not wants_vision(
            text, names=self.bot_aliases(), require_name=True
        ):
            return None
        store = self.suppress.get(key)
        if not store.add(keyword, sender_name):
            return None
        store.save()
        self._warn(f"{key} 已叫停话题「{keyword}」（{sender_name}）")
        return keyword

    # ================================================================ 关系权重
    def _affinity_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("affinity") or {})

    def affinity_enabled(self) -> bool:
        return bool(self._affinity_cfg().get("enabled", True))

    def affinity_multiplier(self, key: str, uid: str, name: str = "") -> float:
        """给仿真器用的"回复概率乘数"。"""
        if not self.affinity_enabled():
            return 1.0
        try:
            return self.affinity.get(key).multiplier(uid)
        except Exception:  # noqa: BLE001
            return 1.0

    def _track_affinity(self, key: str, sender_id: str, sender_name: str, text: str) -> None:
        """记录一次冒犯/善意。

        只认``targeted``（指向人）的冒犯：群里说"这游戏真垃圾"不该被记仇。
        """
        if not self.affinity_enabled():
            return
        cfg = self._affinity_cfg()
        store = self.affinity.get(key)
        try:
            verdict = detect_offense(
                text,
                names=tuple([*self.bot_aliases(), sender_name]),
                extra_words=tuple(self._as_str_list(cfg.get("insult_words"))),
            )
            if verdict and verdict.get("targeted"):
                store.record_offense(sender_id, sender_name, text, str(verdict.get("word") or ""))
                self.log(
                    f"{key} 记录冒犯：{sender_name} 命中「{verdict.get('word')}」，"
                    f"当前关系分 {store.score(sender_id):.1f}"
                )
                return
            positive = detect_positive(
                text, extra_words=tuple(self._as_str_list(cfg.get("positive_words")))
            )
            if positive:
                store.record_positive(sender_id, sender_name)
                return
            store.touch_neutral(sender_id, sender_name)
        except Exception as exc:  # noqa: BLE001 - 关系权重出问题不能影响聊天
            self.log(f"{key} 关系权重更新失败：{exc!r}")

    # ================================================================ 人格成长
    def _growth_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("growth") or {})

    def _growth_material(self, key: str) -> tuple[list[str], str]:
        """收集复盘素材。

        全局模式：把所有会话（群 + 私聊）最近的消息按时间合并，给每行标上场合；
        这样它总结的是"我这个人怎么说话"，而不是"我在这个群怎么说话"。
        """
        scope = self.growth.scope
        if scope == "global":
            sources = list(self.sim.states.items())
        else:
            st = self.sim.states.get(key)
            sources = [(key, st)] if st is not None else []
        rows: list[tuple[float, bool, str]] = []
        for skey, st in sources:
            if st is None:
                continue
            label = "私聊" if skey.startswith("private:") else "群聊"
            for msg in st.recent:
                if getattr(msg, "is_system", False):
                    continue
                text = f"{msg.text}"
                if not text:
                    continue
                rows.append((float(msg.ts or 0), bool(msg.is_self), (label, msg.sender, text)))
        rows.sort(key=lambda item: item[0])
        own = [row[2][2] for row in rows if row[1]][-12:]
        sample = "\n".join(
            f"[{row[2][0]}] {row[2][1]}：{row[2][2]}" for row in rows if not row[1]
        )
        # 只取最后 30 行，避免提示词过长
        lines = sample.splitlines()[-30:]
        return own, "\n".join(lines)

    # ================================================================ 提示词装配
    def build_system_prompt(
        self, key: str, participation: str = "", is_group: bool = True
    ) -> str:
        cfg = self.sessions.get(key) or {}
        persona = str(cfg.get("persona") or "").strip() or str(
            self.config.get("persona_prompt") or ""
        ).strip()
        env = str(cfg.get("env") or "").strip() or str(
            self.config.get("environment_note") or ""
        ).strip()
        mem_cfg = dict(self.config.get("memory") or {})
        style_block = ""
        if dict(self.config.get("style") or {}).get("enabled", True):
            with contextlib.suppress(Exception):
                style_block = self.style.get(key).format_for_prompt(
                    int(dict(self.config.get("style") or {}).get("inject_top_phrases", 8) or 8)
                )
        memory_block = ""
        attitude_block = ""
        suppress_block = ""
        skip_text = None
        growth_block = ""
        if self._growth_cfg().get("enabled", True):
            with contextlib.suppress(Exception):
                growth_block = self.growth.get(key).format_for_prompt()
        if self.suppress_enabled():
            with contextlib.suppress(Exception):
                sup = self.suppress.get(key)
                suppress_block = sup.format_for_prompt()
                skip_text = sup.blocks
        if mem_cfg.get("enabled", True):
            with contextlib.suppress(Exception):
                mem = self.memory.get(key)
                if self.affinity_enabled():
                    aff = self.affinity.get(key)
                    memory_block = mem.format_for_prompt(
                        int(mem_cfg.get("inject_max_topics", 10) or 10),
                        int(mem_cfg.get("inject_max_impressions", 10) or 10),
                        affinity_visible=aff.memory_visible,
                        affinity_score=aff.score,
                        skip_text=skip_text,
                        max_topic_age_min=float(
                            mem_cfg.get("inject_max_topic_age_min", 120) or 0
                        ),
                    )
                else:
                    memory_block = mem.format_for_prompt(
                        int(mem_cfg.get("inject_max_topics", 10) or 10),
                        int(mem_cfg.get("inject_max_impressions", 10) or 10),
                        skip_text=skip_text,
                        max_topic_age_min=float(
                            mem_cfg.get("inject_max_topic_age_min", 120) or 0
                        ),
                    )
        if self.affinity_enabled():
            with contextlib.suppress(Exception):
                attitude_block = build_attitude_block(
                    self.affinity.get(key).cold_members(),
                    int(self._affinity_cfg().get("attitude_max_members", 5) or 5),
                )
        return build_system_prompt(
            bot_name=self.bot_name(),
            persona=persona,
            style_block=style_block,
            memory_block=memory_block,
            participation=participation,
            initial_context=env,
            is_group=is_group,
            silent_token=str(
                dict(self.config.get("simulation") or {}).get("silent_token") or "[SILENT]"
            ),
            extra_rules=str(self.config.get("extra_rules") or ""),
            # 管理员可在 WebUI 里整段替换“仿真层”提示词；留空则用内置默认。
            rules_text=str(self.config.get("simulation_rules") or ""),
            attitude_block=attitude_block,
            vision=bool(self._multimodal_cfg().get("enabled", True)) and _ImageComponent is not None,
            vision_mode=self._vision_mode(),
            suppress_block=suppress_block,
            growth_block=growth_block,
        )

    # ================================================================ LLM 调用
    async def _call_provider(
        self,
        provider: Any,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
        key: str,
    ) -> str:
        try:
            resp = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                contexts=[],
                image_urls=images,
                session_id=f"{PLUGIN_NAME}:{key}",
            )
        except TypeError:
            # 兼容老版本 provider / 不认 image_urls 的实现。
            if images:
                self.log(f"{key} provider 不接受 image_urls，本轮降级为纯文本")
            resp = await provider.text_chat(
                prompt=user_prompt, system_prompt=system_prompt, contexts=[]
            )
        text = getattr(resp, "completion_text", None)
        return str(text) if text is not None else ""

    async def think(
        self,
        key: str,
        system_prompt: str,
        user_prompt: str,
        tag: str = "",
        image_urls: list[str] | None = None,
    ) -> str:
        """调用当前会话的对话模型，返回纯文本。

        ``image_urls`` 走 AstrBot 的标准多模态入口：本地文件路径列表。
        Gemini / OpenAI / Claude 等视觉模型会直接看到图；不支持图像的 provider
        收到这个参数可能报 TypeError，此时降级成纯文本重试。

        **调用失败一律返回空串，不往上抛异常**，原因有两条：

        1. 状态机把"空回合"当成"这轮没话说"，会正常收尾（退场/试探不会卡死）；
           而异常会直接跳过这些收尾分支，把会话钉在 EXITING 直到 60 分钟兜底。
        2. 图片被安全策略拒绝时（Gemini 常见：``block_reason=PROHIBITED_CONTENT``），
           一次失败就放弃太浪费——按配置去掉图片重试一次，通常就能继续说话。
        """
        umo = self._umo_for(key)
        provider = await self._resolve_provider(umo)
        if provider is None:
            self._warn("没有可用的对话模型，请在 AstrBot 里配置一个 LLM Provider")
            return ""
        images = [str(p) for p in (image_urls or []) if str(p).strip()]
        try:
            text = await self._call_provider(provider, system_prompt, user_prompt, images, key)
            self.log(f"{key} 模型返回（{tag}）{len(text)} 字")
            return text
        except Exception as exc:  # noqa: BLE001 - 失败要降级，不能打断状态机
            self._warn(f"{key} 模型调用失败（{tag or 'chat'}）：{exc!r}")
            retry = bool(self._multimodal_cfg().get("retry_without_images", True))
            if images and retry:
                try:
                    text = await self._call_provider(
                        provider, system_prompt, user_prompt, [], key
                    )
                    if text:
                        self._warn(f"{key} 去掉图片后重试成功（图片可能触发了平台审核）")
                    return text
                except Exception as exc2:  # noqa: BLE001
                    self._warn(f"{key} 去掉图片重试仍失败：{exc2!r}")
            return ""

    async def _resolve_provider(self, umo: str):
        """取当前会话的对话模型。

        ``get_using_provider_async`` 是 4.14+ 的推荐接口，``get_using_provider``
        是它取代的同步旧接口（会发 DeprecationWarning）。这里按可用性降级，
        这样插件在老一点的 4.x 上也能跑，而不是直接抛 AttributeError。
        """
        getter = getattr(self.context, "get_using_provider_async", None)
        if callable(getter):
            try:
                provider = await getter(umo=umo or None)
                if provider is not None:
                    return provider
            except TypeError:
                with contextlib.suppress(Exception):
                    return await getter()
            except Exception as exc:  # noqa: BLE001
                self.log(f"get_using_provider_async 失败：{exc!r}")
        legacy = getattr(self.context, "get_using_provider", None)
        if callable(legacy):
            try:
                return legacy(umo or None)
            except TypeError:
                with contextlib.suppress(Exception):
                    return legacy()
            except Exception as exc:  # noqa: BLE001
                self._warn(f"取对话模型失败：{exc!r}")
        return None

    def _umo_for(self, key: str) -> str:
        st = self.sim.states.get(key)
        if st is not None and st.umo:
            return st.umo
        return str(key)

    # ================================================================ 发送
    async def send_parts(self, key: str, raw: str) -> list[str]:
        """按输出协议切条并逐条发送；返回实际发出的文本。"""
        sim_cfg = dict(self.config.get("simulation") or {})
        parts = split_bursts(
            raw,
            max_chars=int(sim_cfg.get("max_reply_chars", 500) or 500),
            split_mode=str(sim_cfg.get("burst_split", "newline") or "newline"),
            max_parts=int(sim_cfg.get("burst_max_messages", 8) or 8),
        )
        if not parts:
            return []
        if not sim_cfg.get("burst_enabled", True):
            parts = [" ".join(parts)] if len(parts) > 1 else parts
        umo = self._umo_for(key)
        lo = int(sim_cfg.get("burst_interval_min_ms", 900) or 900)
        hi = int(sim_cfg.get("burst_interval_max_ms", 2600) or 2600)
        sent: list[str] = []
        for index, part in enumerate(parts):
            if self._closed:
                break
            if index:
                await asyncio.sleep(self._rng.uniform(min(lo, hi), max(lo, hi)) / 1000)
            if not self._rate_ok(key):
                self._warn(f"{key} 触发发送频率上限，本轮剩余消息已丢弃")
                break
            if await self._send_text(key, umo, part):
                sent.append(part)
        return sent

    def _rate_ok(self, key: str) -> bool:
        sim_cfg = dict(self.config.get("simulation") or {})
        per_min = int(sim_cfg.get("max_send_per_minute", 8) or 8)
        per_hour = int(sim_cfg.get("max_send_per_hour", 60) or 60)
        st = self.sim.states.get(key)
        if st is None:
            return True
        now = time.time()
        recent_min = [t for t in st.send_times if now - t <= 60]
        recent_hour = [t for t in st.send_times if now - t <= 3600]
        if len(recent_min) >= per_min or len(recent_hour) >= per_hour:
            return False
        st.send_times.append(now)
        return True

    async def _send_text(self, key: str, umo: str, text: str) -> bool:
        body = clean(text)
        if not body:
            return False
        until = self._mute_until.get(key, 0.0)
        if until > time.time():
            # 被禁言期间发出去必然失败，安静跳过比刷一堆错误日志好。
            self.log(f"{key} 处于禁言中，本轮不发送")
            return False
        try:
            await self.context.send_message(umo, MessageChain(chain=[Plain(body)]))
            self.log(f"{key} 发送：{body[:40]}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._warn(f"{key} 发送失败：{exc!r}")
            return False

    # ================================================================ 回调（仿真器用）
    def on_ai_spoke(self, key: str, parts: list[str]) -> None:
        """AI 发言后的副作用：记录到群风学习里需要区分“自己”和“群友”，这里不需要。"""
        return

    async def after_turn(self, key: str) -> None:
        """每轮发言后的后台维护：人格自我复盘 + 记忆沉淀（各自节流）。"""
        st = self.sim.states.get(key)
        if st is None:
            return
        now = time.time()

        # A) 人格自主迭代（全局层：跨群/私聊共用一套校准，默认每 20 轮、间隔 ≥6 小时）
        gcfg = self._growth_cfg()
        if gcfg.get("enabled", True):
            every_g = int(gcfg.get("every_n_turns", 20) or 0)
            interval_g = float(gcfg.get("min_interval_sec", 21600) or 0)
            throttle = self.growth.key_for(key)
            due = (every_g <= 0 or st.turns % every_g == 0) and (
                now - self._last_growth.get(throttle, 0) >= interval_g
            )
            if due:
                self._last_growth[throttle] = now
                self._track(
                    asyncio.create_task(self.reflect_growth(key), name=f"social-sim-growth-{key}")
                )

        # B) 记忆沉淀
        mem_cfg = dict(self.config.get("memory") or {})
        if not mem_cfg.get("enabled", True) or not mem_cfg.get("auto_extract", True):
            return
        every = int(mem_cfg.get("extract_every_n_turns", 4) or 4)
        if every > 0 and st.turns % every != 0:
            return
        min_interval = float(mem_cfg.get("extract_min_interval_sec", 600) or 600)
        if now - self._last_extract.get(key, 0) < min_interval:
            return
        self._last_extract[key] = now
        # 后台跑：记忆沉淀是一次完整的 LLM 调用，放在发言链路里会把下一句拖慢十几秒。
        self._track(asyncio.create_task(self.consolidate(key), name=f"social-sim-mem-{key}"))

    async def reflect_growth(self, key: str) -> None:
        """让模型复盘"我该怎么说话"，把结论写进自我校准层。

        全局模式下素材来自**所有会话**（群 + 私聊），产出写进同一份校准；
        这样人格是连续的，不会出现"在 A 群一个性格、在 B 群另一个"。

        只读上下文，不写仿真规则、不写基础人设——这是这个功能的安全边界。
        """
        cfg = self._growth_cfg()
        if not cfg.get("enabled", True):
            return
        store = self.growth.get(key)
        own, sample = self._growth_material(key)
        min_messages = int(cfg.get("min_messages", 8) or 8)
        if len(sample.splitlines()) < min_messages:
            self.log(f"{key} 自我校准素材不足（{len(sample.splitlines())}/{min_messages}）")
            return
        style_text = ""
        with contextlib.suppress(Exception):
            style_text = self.style.get(key).summary_text()
        global_scope = self.growth.scope == "global"
        prompt = build_growth_prompt(
            bot_name=self.bot_name(),
            own_messages=own,
            group_style=style_text,
            recent_sample=sample,
            existing=store.texts(),
            max_items=int(cfg.get("max_items_per_round", 2) or 2),
            scope_note=(
                "这是**跨群 / 跨私聊的全局复盘**：只总结在多个场合都成立的经验。"
                "只在某一个群成立的特殊习惯不要写进来——它会应用到所有会话。"
                if global_scope
                else "这是**单个会话的复盘**：经验只对当前会话生效。"
            ),
        )
        raw = await self.think(
            key,
            "你是一个只输出 JSON 的复盘助手，不聊天、不解释。",
            prompt,
            tag="growth",
        )
        observations, drops = parse_growth_json(raw)
        if not observations and not drops:
            self.log(f"{key} 自我校准：这轮没有可总结的")
            return
        applied = 0
        rejected = 0
        for item in observations[: max(1, int(cfg.get("max_items_per_round", 2) or 2))]:
            if store.add(item.get("text"), item.get("reason")):
                applied += 1
            else:
                rejected += 1
        for text in drops:
            store.drop(text)
        store.save()
        if applied or drops:
            self._warn(
                f"{key} 自我校准更新 {applied} 条"
                + (f"（{rejected} 条被安全过滤拒绝）" if rejected else "")
            )

    async def consolidate(self, key: str) -> None:
        """把最近的对话沉淀成记忆（一次便宜的 JSON 调用）。"""
        st = self.sim.states.get(key)
        if st is None:
            return
        mem = self.memory.get(key)
        context_block = build_context_block(
            [m.as_dict() for m in st.recent],
            limit=int(dict(self.config.get("memory") or {}).get("extract_context_window", 30) or 30),
        )
        if not context_block.strip():
            return
        prompt = build_extraction_prompt(
            context_block=context_block,
            bot_name=self.bot_name(),
            existing_memory=mem.format_for_prompt(20, 20),
        )
        raw = await self.think(
            key,
            "你是一个只输出 JSON 的记忆整理器，不聊天。",
            prompt,
            tag="memory",
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            self.log(f"{key} 记忆沉淀：未解析出 JSON，跳过")
            return
        applied = 0
        for item in self._as_list(data.get("activeTopics")):
            if isinstance(item, dict) and item.get("text"):
                if mem.add_topic(
                    item.get("text"),
                    participants=self._as_list(item.get("participants")),
                    pending_question=item.get("pendingQuestion") or "",
                ):
                    applied += 1
        for item in self._as_list(data.get("pendingThoughts")):
            if isinstance(item, dict) and item.get("text"):
                if mem.add_thought(item.get("text"), item.get("motivation") or "curiosity"):
                    applied += 1
        # 印象按 QQ 号存：模型给的是昵称，这里映射回号；映射不到就跳过（宁缺勿错）
        name_to_uid: dict[str, str] = {}
        for msg in st.recent:
            if getattr(msg, "is_system", False):
                continue
            if msg.sender and msg.sender_id and msg.sender not in name_to_uid:
                name_to_uid[msg.sender] = msg.sender_id
        for item in self._as_list(data.get("memberImpressions")):
            if not (isinstance(item, dict) and item.get("target") and item.get("trait")):
                continue
            target = clean(item.get("target"))
            uid = name_to_uid.get(target, "")
            if not uid and target.isdigit():
                uid = target
            if not uid:
                self.log(f"{key} 沉淀跳过一条印象（找不到对应 QQ 号）：{target}")
                continue
            if mem.add_impression(uid, item.get("trait"), target):
                applied += 1
        for text in self._as_list(data.get("dropTopics")):
            mem.remove("activeTopic", needle=str(text))
        for text in self._as_list(data.get("dropThoughts")):
            mem.remove("pendingThought", needle=str(text))
        mem.save()
        self.log(f"{key} 记忆沉淀完成，新增/更新 {applied} 条")

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    # ================================================================ 事件入口
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent) -> None:
        """观察所有消息：学习群风、驱动仿真状态机。"""
        try:
            await self._handle(event)
        except Exception as exc:  # noqa: BLE001 - 单个会话异常不能影响全局
            self._warn(f"处理消息异常：{exc!r}")

    async def _handle(self, event: AstrMessageEvent) -> None:
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        if not key:
            return
        self_id = str(event.get_self_id() or "")
        sender_id = str(event.get_sender_id() or "")
        if not self._session_enabled(key, is_group):
            return
        umo = str(event.unified_msg_origin)

        # 0) 系统通知（戳一戳 / 禁言 / 头衔…）。
        #    这类事件的 message_str 是空的，而且"被禁言的人"可能就是机器人自己
        #    （此时 sender_id == self_id），所以必须放在"忽略自己消息"之前。
        if self._system_cfg().get("enabled", True):
            notice = describe_notice(
                getattr(getattr(event, "message_obj", None), "raw_message", None),
                self_id,
                lambda uid: (
                    self.bot_name()
                    if self_id and str(uid) == self_id
                    else self._name_of(key, uid)
                ),
            )
            if notice:
                await self._handle_notice(key, umo, is_group, notice)
                return

        # 普通消息：自己发的不算群聊语料。
        if sender_id and self_id and sender_id == self_id:
            return

        text = clean(event.get_message_str())
        # 协作扩展点：别的插件可以把补充说明挂在这个 extra 上（例如"图里是一张表情包"）
        with contextlib.suppress(Exception):
            extra_text = str(event.get_extra("social_sim_extra_text") or "").strip()
            if extra_text:
                text = f"{text} {extra_text}".strip()
        images = await self._collect_images(event)
        quoted_ids = self._quoted_ids(event)
        # 按需识图：默认不加载，只有被引用或被点名要求看图时才放行
        current_approved = False
        if images:
            if not self._vision_enabled():
                current_approved = False
            elif self._vision_mode() == "auto":
                current_approved = True
            else:
                current_approved = self._approve_quoted(key, quoted_ids) or self._is_vision_invite(text)
            base = "[图片]" if len(images) == 1 else f"[图片×{len(images)}]"
            marker = base if current_approved else base[:-1] + "·未查看]"
            text = f"{text} {marker}".strip() if text else marker
        elif self._vision_mode() == "on_demand" and self._vision_enabled():
            # 这一条没有图，但可能是"引用上一张图 + 让我看"，或者直接说"你刚才那张图看看"
            if self._approve_quoted(key, quoted_ids) or (
                self._is_vision_invite(text) and self._approve_latest_pending(key)
            ):
                self.log(f"{key} 按需识图：已放行上一张图片")
        if not text:
            return
        if self._is_command_text(text):
            # 指令交给指令处理器；指令不参与仿真，也不该被当作群聊语料。
            return

        sender_name = str(event.get_sender_name() or sender_id or "群友")
        # 记忆里的印象按 QQ 号存；昵称只是显示名，每次发言顺便刷新一次，
        # 同时把旧版按昵称存的记录迁移到号下。
        if sender_id and dict(self.config.get("memory") or {}).get("enabled", True):
            with contextlib.suppress(Exception):
                if self.memory.get(key).rename_impression(sender_id, sender_name):
                    self.memory.get(key).save()
        # 0.5) "别再说 xxx 了" → 记进叫停表，之后不注入、不主动提
        stopped = self._detect_suppress(key, text, sender_name)
        if stopped:
            self.log(f"{key} 叫停：{stopped}")
        # 1) 关系权重：先更新分数，后面的群风统计与仿真概率都用它。
        if self.affinity_enabled():
            self._track_affinity(key, sender_id, sender_name, text)
        # 2) 学习群风（纯本地统计，不花 token）。关系差的人权重低，带不动群风。
        if is_group and dict(self.config.get("style") or {}).get("enabled", True):
            weight = 1.0
            if self.affinity_enabled():
                with contextlib.suppress(Exception):
                    weight = self.affinity.get(key).style_weight(sender_id)
            with contextlib.suppress(Exception):
                self.style.get(key).observe(sender_name, text, weight=weight)

        mode = self._mode()
        if mode == MODE_ASSIST:
            return

        # 3) 仿真接管：先关掉 AstrBot 默认的 LLM 回复，再由状态机决定要不要开口。
        event.should_call_llm(False)
        message_id = ""
        with contextlib.suppress(Exception):
            message_id = str(getattr(event.message_obj, "message_id", "") or "")
        msg = ChatMessage(
            sender=sender_name,
            sender_id=sender_id,
            text=text,
            ts=time.time(),
            is_self=False,
            message_id=message_id,
            images=images,
            vision=bool(images) and current_approved,
        )
        trigger = await self.sim.on_message(msg, key, umo, is_group)
        if trigger:
            task = asyncio.create_task(
                self.sim.speak(
                    key, "idle_trigger", trigger=f"{sender_name}：{truncate(text, 200)}"
                ),
                name=f"social-sim-reply-{key}",
            )
            self._track(task)

    def _track(self, task: asyncio.Task) -> None:
        """持有后台任务引用，避免被 GC；异常由仿真器内部记录。"""
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(self._log_task_error)

    def _log_task_error(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._warn(f"后台任务异常：{exc!r}")

    @staticmethod
    def _session_key(event: AstrMessageEvent, is_group: bool) -> str:
        if is_group:
            gid = str(event.get_group_id() or "")
            if not gid:
                return ""
            return f"group:{gid}"
        sid = str(event.get_sender_id() or "")
        return f"private:{sid}" if sid else ""

    def _is_command_text(self, text: str) -> bool:
        head = text.lstrip()
        return any(head.startswith(prefix) for prefix in self._wake_prefixes)

    # ================================================================ 辅助模式
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req) -> None:
        """``assist`` 模式：不接管回复，只把群风与记忆注入 AstrBot 自己的 LLM 请求。"""
        if self._mode() != MODE_ASSIST:
            return
        try:
            is_group = not event.is_private_chat()
            key = self._session_key(event, is_group)
            if not key or not self._session_enabled(key, is_group):
                return
            blocks: list[str] = []
            if dict(self.config.get("style") or {}).get("enabled", True):
                block = self.style.get(key).format_for_prompt(
                    int(dict(self.config.get("style") or {}).get("inject_top_phrases", 8) or 8)
                )
                if block:
                    blocks.append(block)
            mem_cfg = dict(self.config.get("memory") or {})
            if mem_cfg.get("enabled", True):
                block = self.memory.get(key).format_for_prompt(
                    int(mem_cfg.get("inject_max_topics", 10) or 10),
                    int(mem_cfg.get("inject_max_impressions", 10) or 10),
                )
                if block:
                    blocks.append("【你对这个群的记忆】\n" + block)
            if blocks:
                req.system_prompt = (req.system_prompt or "") + "\n\n" + "\n\n".join(blocks)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"注入提示失败：{exc!r}")

    # ================================================================ 指令
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        if not self.config.get("admin_only_commands", True):
            return True
        try:
            return bool(event.is_admin())
        except Exception:  # noqa: BLE001
            return False

    @filter.command("仿真")
    async def cmd_sim(self, event: AstrMessageEvent):
        """查看/控制仿真状态。用法：/仿真 [开|关|接手|潜水|重置]"""
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        if not key:
            yield event.plain_result("无法识别当前会话。")
            return
        arg = self._command_arg(event, "仿真").strip()
        cfg = self._session_cfg(key)
        if arg in ("开", "on", "启用"):
            cfg["enabled"] = True
            self._save_sessions()
            yield event.plain_result(f"已开启「{key}」的仿真模式。")
            return
        if arg in ("关", "off", "关闭"):
            cfg["enabled"] = False
            self._save_sessions()
            with contextlib.suppress(Exception):
                self.sim.force_phase(key, IDLE, str(event.unified_msg_origin), is_group)
            yield event.plain_result(f"已关闭「{key}」的仿真模式（群聊不再由本插件接管）。")
            return
        if arg in ("接手", "活跃", "active"):
            label = self.sim.force_phase(key, ACTIVE, str(event.unified_msg_origin), is_group)
            yield event.plain_result(f"已把「{key}」切到 {label}，下一次有人说话就会参与。")
            return
        if arg in ("潜水", "观望", "idle"):
            label = self.sim.force_phase(key, IDLE, str(event.unified_msg_origin), is_group)
            yield event.plain_result(f"已把「{key}」切到 {label}。")
            return
        if arg in ("重置", "reset"):
            self.sim.states.pop(key, None)
            yield event.plain_result(f"已重置「{key}」的仿真状态。")
            return
        snap = self.sim.snapshot(key)
        mem_stats = self.memory.get(key).stats()
        style = self.style.get(key)
        lines = [
            f"会话：{key}",
            f"模式：{self._mode()}　状态：{snap['phase_label']}　累计发言轮数：{snap['turns']}",
            f"记忆：话题 {mem_stats['activeTopics']} / 念头 {mem_stats['pendingThoughts']} / 群友印象 {mem_stats['memberImpressions']}",
            f"群风样本：{style.metrics()['messages']} 条（{'已可注入' if style.is_ready() else '样本不足'}）",
        ]
        if snap.get("active_deadline"):
            left = int(max(0, snap["active_deadline"] - time.time()) / 60)
            lines.append(f"活跃剩余：约 {left} 分钟后自动收尾")
        yield event.plain_result("\n".join(lines))

    @filter.command("记忆")
    async def cmd_memory(self, event: AstrMessageEvent):
        """查看/清空记忆。用法：/记忆 [清空] [话题|念头|印象]"""
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        mem = self.memory.get(key)
        arg = self._command_arg(event, "记忆").strip()
        if arg.startswith("清空") or arg.startswith("清除"):
            rest = arg[2:].strip()
            category = {"话题": "activeTopic", "念头": "pendingThought", "印象": "memberImpression"}.get(rest)
            mem.store.backup()
            mem.clear(category)
            mem.save()
            if category is None:
                # 全清时必须连对话缓冲一起清：否则提示词里还带着最近 N 条群聊原文，
                # 模型会从里面把“谁爱报菜名”重新推断出来，看起来像没清干净。
                self._forget_context(key)
                yield event.plain_result(
                    f"已清空「{key}」的记忆（全部），并丢弃了对话缓冲——"
                    "模型不会再从旧上下文里重新推断画像。"
                )
            else:
                yield event.plain_result(f"已清空「{key}」的记忆（{rest}）。")
            return
        text = mem.format_for_prompt(20, 20)
        yield event.plain_result(text or f"「{key}」还没有任何记忆。")

    @filter.command("重置")
    async def cmd_reset(self, event: AstrMessageEvent):
        """彻底重置本会话：记忆 + 群风画像 + 对话缓冲 + 仿真状态。

        注意与 AstrBot 内置的 ``/reset`` 不是一回事：那个重置的是 AstrBot 自己的
        对话上下文，不动本插件的记忆与画像。
        """
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        if not key:
            yield event.plain_result("无法识别当前会话。")
            return
        mem = self.memory.get(key)
        mem.store.backup()
        mem.clear()
        mem.save()
        self.style.get(key).reset()
        with contextlib.suppress(Exception):
            self.affinity.get(key).reset()
        self._forget_context(key)
        yield event.plain_result(
            f"已彻底重置「{key}」：\n"
            "· 记忆（话题 / 念头 / 群友印象）已清空，并已备份\n"
            "· 群风画像已归零（样本需重新攒到设定条数才会再次注入）\n"
            "· 群友关系权重已全部回到中性（50 分）\n"
            "· 对话缓冲与仿真状态已丢弃，不会再从旧上下文推断画像\n"
            "· 叫停列表、人设与环境说明保留（要清用 /别提 清除、/人设 清除、/环境 清除）"
        )

    @filter.command("群风")
    async def cmd_style(self, event: AstrMessageEvent):
        """查看本群学到的说话策略。用法：/群风 [重置]"""
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        learner = self.style.get(key)
        arg = self._command_arg(event, "群风").strip()
        if arg in ("重置", "reset", "清空") and self._is_admin(event):
            learner.reset()
            yield event.plain_result(f"已重置「{key}」的群风统计。")
            return
        yield event.plain_result(f"「{key}」群风画像：\n{learner.summary_text()}")

    @filter.command("别提")
    async def cmd_suppress(self, event: AstrMessageEvent):
        """叫停某个话题。用法：/别提 <关键词> 或 /别提 清除 [关键词]

        加：任何群友都能加（这是"我不想聊这个"的个人表达，且只影响本群、会过期）。
        清：只有管理员能清，避免有人把别人叫停的话题偷偷放回来。
        """
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        if not key:
            yield event.plain_result("无法识别当前会话。")
            return
        if not self.suppress_enabled():
            yield event.plain_result("叫停功能未启用（配置项 suppress.enabled）。")
            return
        store = self.suppress.get(key)
        arg = self._command_arg(event, "别提").strip()
        if not arg:
            yield event.plain_result(f"「{key}」{store.summary()}")
            return
        if arg.startswith("清除") or arg.startswith("解除"):
            if not self._is_admin(event):
                yield event.plain_result("解除叫停需要管理员。")
                return
            rest = arg[2:].strip()
            removed = store.remove(rest)
            store.save()
            if removed:
                yield event.plain_result(
                    f"已解除 {removed} 条叫停{'（' + rest + '）' if rest else '（全部）'}。"
                    "被隐藏的记忆会自动恢复。"
                )
            else:
                yield event.plain_result(f"没有找到匹配的叫停项（{rest or '全部'}）。")
            return
        if not store.add(arg, event.get_sender_name() or ""):
            yield event.plain_result("没听懂要叫停什么，换个说法试试，例如：/别提 世界杯")
            return
        store.save()
        yield event.plain_result(
            f"好，之后不再主动提「{arg}」了。\n"
            "（相关记忆只是暂时不注入，不会删；叫停到期或解除后会自动恢复。）"
        )

    @filter.command("关系")
    async def cmd_affinity(self, event: AstrMessageEvent):
        """查看/重置本群的群友关系权重。用法：/关系 [重置]"""
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        arg = self._command_arg(event, "关系").strip()
        if not self.affinity_enabled():
            yield event.plain_result(
                "关系权重未启用（配置项 affinity.enabled）。此时所有群友都按中性处理。"
            )
            return
        store = self.affinity.get(key)
        if arg in ("重置", "reset", "清空"):
            store.reset()
            yield event.plain_result(f"已把「{key}」的关系权重全部重置为中性（50 分）。")
            return
        rows = store.summary()
        if not rows:
            yield event.plain_result(
                f"「{key}」还没有关系记录（所有群友默认中性 50 分）。"
            )
            return
        cfg = self._affinity_cfg()
        lines = [
            f"「{key}」群友关系权重（中性 50，越低越冷淡）",
            f"降权门槛：冒犯满 {cfg.get('min_offenses_to_apply', 3)} 次才生效；"
            f"低于 {cfg.get('mute_reply_below', 25)} 分明显少接话；"
            f"低于 {cfg.get('hide_memory_below', 20)} 分不再进记忆提示",
        ]
        for row in rows:
            score = row["score"]
            if score >= 60:
                tag = "亲近"
            elif score >= 50:
                tag = "正常"
            elif score >= 30:
                tag = "偏冷"
            else:
                tag = "很冷"
            line = (
                f"- {row['name']}：{score:.0f} 分（{tag}）"
                f"　冒犯 {row['offenses']} 次 / 善意 {row['positives']} 次"
            )
            lines.append(line)
            if row["evidence"] and score < 50:
                lines.append(f"　最近一次：「{row['evidence'][-1]}」")
        yield event.plain_result("\n".join(lines))

    @filter.command("人设")
    async def cmd_persona(self, event: AstrMessageEvent):
        """查看/设置本会话的人设。用法：/人设 [设置 <文本>|清除|成长 [清除]]"""
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        cfg = self._session_cfg(key)
        arg = self._command_arg(event, "人设").strip()

        # 子命令：成长（查看 / 清除模型自己总结的说话经验）
        if arg.startswith("成长") or arg.startswith("自省"):
            store = self.growth.get(key)
            scope_label = (
                "全局 · 所有群和私聊共用"
                if self.growth.scope == "global"
                else f"仅本会话 {key}"
            )
            rest = arg[2:].strip()
            if rest.startswith("清除") or rest.startswith("清空") or rest.startswith("重置"):
                store.clear()
                store.save()
                yield event.plain_result(
                    f"已清空自我校准（{scope_label}），回到纯基础人设。"
                    "基础人设和仿真规则没动过。"
                )
                return
            yield event.plain_result(
                f"自我校准〔{scope_label}〕\n{store.summary()}\n"
                "（这是模型自己总结的说话经验，只追加在人格层；"
                "仿真规则与基础人设不受影响，随时可以 /人设 成长 清除）"
            )
            return

        if arg in ("清除", "清空", "reset"):
            cfg["persona"] = ""
            self._save_sessions()
            yield event.plain_result(f"已清除「{key}」的专属人设，回到全局人设。")
            return
        if arg.startswith("设置"):
            body = arg[2:].strip()
            if not body:
                yield event.plain_result("用法：/人设 设置 你是一个爱吐槽的群友……")
                return
            cfg["persona"] = truncate(body, 4000)
            self._save_sessions()
            yield event.plain_result(f"已为「{key}」设置专属人设（{len(cfg['persona'])} 字）。")
            return
        current = cfg.get("persona") or str(self.config.get("persona_prompt") or "")
        yield event.plain_result(
            f"「{key}」当前人设{'（会话专属）' if cfg.get('persona') else '（全局）'}：\n"
            f"{current or '（未设置，模型会按默认群友语气说话）'}"
        )

    @filter.command("环境")
    async def cmd_env(self, event: AstrMessageEvent):
        """查看/设置本群的背景说明。用法：/环境 [设置 <文本>|清除]"""
        if not self._is_admin(event):
            yield event.plain_result("只有管理员可以使用该指令。")
            return
        is_group = not event.is_private_chat()
        key = self._session_key(event, is_group)
        cfg = self._session_cfg(key)
        arg = self._command_arg(event, "环境").strip()
        if arg in ("清除", "清空", "reset"):
            cfg["env"] = ""
            self._save_sessions()
            yield event.plain_result(f"已清除「{key}」的环境说明。")
            return
        if arg.startswith("设置"):
            body = arg[2:].strip()
            if not body:
                yield event.plain_result("用法：/环境 设置 这是一个游戏交流群，大家习惯深夜水群……")
                return
            cfg["env"] = truncate(body, 2000)
            self._save_sessions()
            yield event.plain_result(f"已为「{key}」设置环境说明（{len(cfg['env'])} 字）。")
            return
        yield event.plain_result(
            f"「{key}」当前环境说明：\n{cfg.get('env') or '（未设置）'}"
        )

    def _command_arg(self, event: AstrMessageEvent, name: str) -> str:
        text = clean(event.get_message_str()).lstrip()
        for prefix in self._wake_prefixes or ["/"]:
            if not prefix:
                continue
            head = prefix + name
            if text.startswith(head):
                return text[len(head) :]
        # 兜底：按名字第一次出现的位置截断。
        idx = text.find(name)
        return text[idx + len(name) :] if idx >= 0 else ""
