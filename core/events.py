"""QQ 系统消息（OneBot v11 通知事件）的中文描述。

AstrBot 的 aiocqhttp 适配器**会把通知事件转成消息事件**提交进来
（`_convert_handle_notice_event`），但它把 ``message_str`` 留空、只给戳一戳塞了一个
``Poke`` 组件。好处是插件能收到这些事件；坏处是文本为空，模型什么也看不到。

这里把原始的 ``raw_message``（就是那个 OneBot 事件 dict）翻译成一行人话，塞进对话缓冲，
模型就能知道"刚才有人拍了拍我""群友被禁言了""某某拿了龙王头衔"。

支持的通知类型（其余类型给一条通用描述，不会被丢掉）：

| notice_type | 说明 |
| --- | --- |
| ``group_ban`` | 禁言 / 解除禁言（自己也被禁言时用于暂停发送） |
| ``notify`` + ``poke`` | 戳一戳（戳的是 bot 时算"直接找我"） |
| ``notify`` + ``honor`` | 群头衔：龙王 / 群聊之火 / 快乐源泉 |
| ``notify`` + ``lucky_king`` | 运气王 |
| ``group_admin`` | 设置 / 取消管理员 |
| ``group_increase`` / ``group_decrease`` | 入群 / 退群 / 被踢 |
| ``group_recall`` / ``friend_recall`` | 撤回消息 |
| ``group_msg_emoji_like`` | 表情回应 |

**这不是要替代别的插件**：如果已经有插件把图片/系统消息转成文字注入了 prompt，
把本模块的开关（``system_events.enabled``）关掉即可，两者不冲突。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["HONOR_NAMES", "describe_notice", "notice_summary_kind"]

HONOR_NAMES = {
    "talkative": "龙王",
    "performer": "群聊之火",
    "emotion": "快乐源泉",
}

_NOTICE_LABELS = {
    "group_ban": "禁言",
    "group_admin": "管理员变更",
    "group_increase": "入群",
    "group_decrease": "退群",
    "group_recall": "撤回",
    "friend_recall": "撤回",
    "group_msg_emoji_like": "表情回应",
}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _fmt_duration(seconds: int) -> str:
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        minutes = seconds // 60
        rest = seconds % 60
        return f"{minutes} 分钟" + (f" {rest} 秒" if rest else "")
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours} 小时" + (f" {minutes} 分钟" if minutes else "")


def notice_summary_kind(raw: Any) -> str:
    """取出通知类型（用于统计与开关），非通知事件返回空串。"""
    if not isinstance(raw, dict) or raw.get("post_type") != "notice":
        return ""
    notice_type = str(raw.get("notice_type") or "")
    sub_type = str(raw.get("sub_type") or "")
    return f"{notice_type}.{sub_type}" if sub_type else notice_type


def describe_notice(
    raw: Any,
    self_id: str = "",
    name_of: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    """把 OneBot 通知事件翻译成一行中文描述。

    Args:
        raw: ``event.message_obj.raw_message``（OneBot v11 事件 dict）。
        self_id: 机器人自己的 QQ 号，用于判断"戳的是不是我""被禁言的是不是我"。
        name_of: QQ 号 → 昵称。取不到就退回显示 QQ 号。

    Returns:
        ``{"text", "sender_id", "sender_name", "kind", "poke_me", "muted_self", "mute_seconds"}``
        或 ``None``（不是通知事件）。
    """
    if not isinstance(raw, dict) or raw.get("post_type") != "notice":
        return None
    notice_type = str(raw.get("notice_type") or "")
    sub_type = str(raw.get("sub_type") or "")
    if not notice_type:
        return None

    self_id = str(self_id or "")

    def name(uid: Any) -> str:
        key = str(uid or "")
        if not key:
            return "某人"
        if name_of is not None:
            try:
                resolved = str(name_of(key) or "").strip()
            except Exception:  # noqa: BLE001
                resolved = ""
            if resolved:
                return resolved
        return key

    user_id = raw.get("user_id") or raw.get("target_id") or 0
    operator_id = raw.get("operator_id") or 0
    target_id = raw.get("target_id") or 0
    sender_name = name(user_id)

    result: dict[str, Any] = {
        "text": "",
        "sender_id": str(user_id or ""),
        "sender_name": sender_name,
        "kind": f"{notice_type}.{sub_type}" if sub_type else notice_type,
        "poke_me": False,
        "muted_self": False,
        "mute_seconds": 0,
    }

    if notice_type == "group_ban":
        duration = _as_int(raw.get("duration"))
        who = name(user_id)
        by = name(operator_id) if operator_id else "管理员"
        if sub_type == "lift_ban":
            result["text"] = f"【系统】{by} 解除了 {who} 的禁言"
        else:
            span = _fmt_duration(duration) or "一段时间"
            result["text"] = f"【系统】{by} 把 {who} 禁言了 {span}"
        if user_id and str(user_id) == self_id and sub_type != "lift_ban":
            result["muted_self"] = True
            result["mute_seconds"] = max(0, duration)
        return result

    if notice_type == "notify":
        if sub_type == "poke":
            poker = name(user_id)
            target = name(target_id) if target_id else "某人"
            result["text"] = f"【系统】{poker} 拍了拍 {target}"
            result["poke_me"] = bool(target_id and str(target_id) == self_id)
            return result
        if sub_type == "honor":
            honor_type = str(raw.get("honor_type") or "")
            title = HONOR_NAMES.get(honor_type, honor_type or "群头衔")
            result["text"] = f"【系统】{sender_name} 拿到了群头衔「{title}」"
            return result
        if sub_type == "lucky_king":
            target = name(target_id) if target_id else sender_name
            result["text"] = f"【系统】{target} 成为了运气王"
            return result
        result["text"] = f"【系统】{sender_name} 触发了 {sub_type or 'notify'} 事件"
        return result

    if notice_type == "group_admin":
        if sub_type == "set":
            result["text"] = f"【系统】{sender_name} 被设为了管理员"
        else:
            result["text"] = f"【系统】{sender_name} 的管理员被取消了"
        return result

    if notice_type == "group_increase":
        who = name(user_id)
        inviter = name(operator_id) if operator_id and str(operator_id) != str(user_id) else ""
        result["text"] = (
            f"【系统】{inviter} 邀请了 {who} 加入群聊"
            if inviter
            else f"【系统】{who} 加入了这个群"
        )
        return result

    if notice_type == "group_decrease":
        who = name(user_id)
        operator = name(operator_id) if operator_id and str(operator_id) != str(user_id) else ""
        if sub_type == "kick":
            result["text"] = f"【系统】{operator or '管理员'} 把 {who} 移出了群聊"
        elif sub_type == "kick_me":
            result["text"] = "【系统】机器人被移出了群聊"
        else:
            result["text"] = f"【系统】{who} 退出了群聊"
        return result

    if notice_type in ("group_recall", "friend_recall"):
        by = name(operator_id) if operator_id and str(operator_id) != str(user_id) else ""
        if by:
            result["text"] = f"【系统】{by} 撤回了 {sender_name} 的一条消息"
        else:
            result["text"] = f"【系统】{sender_name} 撤回了一条消息"
        return result

    if notice_type == "group_msg_emoji_like":
        likes = raw.get("likes") or []
        emoji = ""
        if isinstance(likes, list) and likes and isinstance(likes[0], dict):
            emoji = str(likes[0].get("emoji_id") or "")
        suffix = f"「{emoji}」" if emoji else ""
        result["text"] = f"【系统】{sender_name} 用表情{suffix}回应了一条消息"
        return result

    # 未覆盖的类型也保留一条，至少让模型知道"群里发生了点事"。
    label = _NOTICE_LABELS.get(notice_type, notice_type)
    result["text"] = f"【系统】{sender_name} 触发了{label}事件"
    return result
