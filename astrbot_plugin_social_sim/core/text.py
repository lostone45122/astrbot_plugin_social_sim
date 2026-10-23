"""文本处理工具。

移植自 qq-bridge 的同名职责（`src/md-to-plain.js` + 分句/静默判定逻辑），
按 AstrBot 场景重写：去 Markdown、清理控制字符、按行/空格分条、长度兜底、
“对方还没说完”的语感判定、以及“这句话是不是在找我”的判定。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "clean",
    "extract_json",
    "is_directed_at",
    "is_silent",
    "looks_like_unfinished",
    "split_bursts",
    "strip_markdown",
    "truncate",
    "wants_vision",
]

# 控制字符（保留 \n \t），以及零宽字符。群聊原文里经常混进这些。
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u200b-\u200f\u2028\u2029]")
# CQ 码：AstrBot 的 OneBot 适配器一般已经转成消息组件，但转发/引用文本里可能残留。
_CQ_RE = re.compile(r"\[CQ:([a-zA-Z_]+)(?:,[^\]]*)?\]")
_MD_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.S)
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_MD_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\(([^)\s]+)\)")
_MD_LINK_RE = re.compile(r"\[([^\]\n]{1,120})\]\(([^)\s]+)\)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_QUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.M)
_MD_LIST_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.M)
_MD_RULE_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.M)
_MD_BOLD_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
_MD_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)

# 对方话说到一半的常见收尾词（qq-bridge 的 looksLikeUnfinished 语音版）。
_UNFINISHED_TAILS = (
    "你知道",
    "你猜",
    "等下",
    "等等",
    "等我",
    "我跟你讲",
    "我跟你说",
    "但是",
    "但",
    "所以说",
    "然后",
    "那个",
    "就是",
    "对了",
    "还有",
    "而且",
    "因为",
    "所以",
    "不过",
    "至于",
    "或者",
    "比如",
    "如果",
    "要是",
    "说真的",
    "讲道理",
    "话说",
    "顺便",
    "还有一件事",
)
_UNFINISHED_TRAIL = ("…", "...", "，", ",", "、", "：", ":", "——", "-", "（", "(", "【", "[")

# 句读边界：长文本兜底切分用。
_SENTENCE_END = "。！？!?…\n；;"


def clean(text: object) -> str:
    """把任意输入变成一段干净的单行文本（去掉 CQ 码与控制字符）。"""
    s = "" if text is None else str(text)
    s = _CONTROL_RE.sub("", s)
    # 只把 CQ 码替换成可读占位，避免整段被吞掉。
    s = _CQ_RE.sub(lambda m: f"[{m.group(1)}]", s)
    return s.strip()


def truncate(text: object, limit: int) -> str:
    """按字符（码点）截断，绝不留下半个 emoji。"""
    s = "" if text is None else str(text)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return s
    if limit <= 0 or len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def strip_markdown(text: object) -> str:
    """Markdown → 纯文本。

    QQ / 微信等平台不渲染 Markdown，``**加粗**``、``# 标题``、代码块都会原样显示成
    噪声，所以发出去之前统一转成纯文本。链接保留成 ``文字 (url)`` 的形式。
    """
    s = "" if text is None else str(text)
    s = _MD_CODE_BLOCK_RE.sub(lambda m: m.group(1), s)
    s = _MD_INLINE_CODE_RE.sub(lambda m: m.group(1), s)
    s = _MD_IMAGE_RE.sub("", s)
    s = _MD_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", s)
    s = _MD_HEADING_RE.sub("", s)
    s = _MD_QUOTE_RE.sub("", s)
    s = _MD_LIST_RE.sub("", s)
    s = _MD_RULE_RE.sub("", s)
    s = _MD_STRIKE_RE.sub(lambda m: m.group(1), s)
    s = _MD_BOLD_RE.sub(lambda m: m.group(2), s)
    # 表格分隔行 |---|---| 直接丢掉，其余竖线换成空格。
    s = re.sub(r"^\s*\|?[\s:|-]{3,}\|?\s*$", "", s, flags=re.M)
    s = s.replace("|", " ")
    return s.strip()


def is_silent(text: object, token: str = "[SILENT]") -> bool:
    """判断模型这一轮是否选择“潜水/不说话”。

    qq-bridge 用 ``[SILENT]`` 表示沉默；这里同时容忍模型把标记包进引号、
    或不带方括号写 ``SILENT`` 的情况（只认整条输出就是该标记，避免误伤正文）。
    """
    if not text:
        return False
    s = str(text).strip()
    token = (token or "[SILENT]").strip()
    candidates = {token, token.strip("[]"), "SILENT", "[silent]", "（沉默）", "(沉默)"}
    if s in candidates:
        return True
    # 允许“标记 + 一点点说明”的写法，如 `[SILENT]（这轮不接）`。
    head = s.split("\n", 1)[0].strip()
    for c in candidates:
        if c and head == c:
            return True
        if c and head.startswith(c) and len(head) - len(c) <= 12:
            return True
    return False


def _wrap_long(part: str, max_chars: int) -> list[str]:
    """把超长的一段按句读切成多段；实在切不开就硬切。"""
    if max_chars <= 0 or len(part) <= max_chars:
        return [part]
    out: list[str] = []
    buf = ""
    for ch in part:
        buf += ch
        if ch in _SENTENCE_END and len(buf) >= max_chars * 0.6:
            out.append(buf.strip())
            buf = ""
        elif len(buf) >= max_chars:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [p for p in out if p] or [part[:max_chars]]


def split_bursts(
    text: object,
    max_chars: int = 500,
    split_mode: str = "newline",
    max_parts: int = 8,
) -> list[str]:
    """把模型输出拆成要逐条发送的 QQ 消息。

    与 qq-bridge 的差异（有意为之）：那一版用**空格**分条，对中文群聊很危险
    （``DeepSeek V3``、``iPhone 15`` 会被拦腰切断）。这里默认按**换行**分条，
    模型只要一行写一条即可；确实想沿用空格语义时把 ``split_mode`` 设为 ``space``。

    Args:
        text: 模型原始输出。
        max_chars: 单条消息长度上限，超出按句读兜底切分。
        split_mode: ``newline``（默认）或 ``space``。
        max_parts: 单轮最多发几条，防止模型一次刷屏。

    Returns:
        非空的消息文本列表（已去 Markdown、去空行）。
    """
    s = strip_markdown(clean(text))
    if not s.strip():
        return []
    if split_mode == "space":
        raw_parts = re.split(r"[ \t\u3000]+", s)
    else:
        raw_parts = re.split(r"\r?\n+", s)
    parts: list[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        parts.extend(_wrap_long(p, max_chars))
    # 去掉只有标点/装饰的碎片。
    parts = [p for p in parts if re.search(r"\w", p)]
    if max_parts > 0 and len(parts) > max_parts:
        parts = parts[:max_parts]
    return parts


def looks_like_unfinished(text: object) -> bool:
    """对方是不是话没说完（用于把“等待窗口”拉长一点，别抢话）。"""
    s = ("" if text is None else str(text)).strip()
    if not s:
        return False
    if s.endswith(_UNFINISHED_TRAIL):
        return True
    tail = s[-8:]
    return any(tail.endswith(t) for t in _UNFINISHED_TAILS)


def is_directed_at(
    text: object,
    names: Iterable[str] = (),
    keywords: Iterable[str] = (),
) -> bool:
    """这句话是不是在找“我”。

    判定顺序：@ 提及 → 名字出现 → 必回关键词。群聊里“有人直接找我”是最强的
    参与信号，qq-bridge 用它决定是否一定接话。
    """
    s = "" if text is None else str(text)
    if not s.strip():
        return False
    if re.search(r"\[at\s*[:：]", s, re.I) or "@" in s:
        # `[at:xxx]` / `[CQ:at]` 都算被点名；@ 后面跟名字的情况无法区分，保守算“有指向”。
        return True
    lowered = s.lower()
    for name in names:
        n = str(name or "").strip()
        if n and n.lower() in lowered:
            return True
    for kw in keywords:
        k = str(kw or "").strip()
        if k and k.lower() in lowered:
            return True
    return False


# ---------------------------------------------------------------- 识图触发

# "让你看图"的动词/名词线索。只有出现这些，才考虑加载图片。
_VISION_HINTS = (
    "看", "瞧", "瞅", "识别", "认一下", "认认", "这是什么", "这是啥", "图里",
    "画面", "描述", "像什么", "翻译一下", "表情包是",
)
# 明确的否定："别看了""不用看"——出现就一票否决。
_VISION_NEGATIVES = ("别看", "不用看", "不要看", "别管", "无视", "别识别")


def wants_vision(
    text: object,
    names: Iterable[str] = (),
    extra_words: Iterable[str] = (),
    require_name: bool = True,
) -> bool:
    """这句话是不是在"让你看图"。

    按需识图的核心开关：默认要求**既有点名（@/昵称）又有看图的意思**，
    这样群里两个人互相说"你看这个"不会把图喂给模型。

    Args:
        text: 消息文本。
        names: bot 的昵称/别名。
        extra_words: 额外触发词（配置）。
        require_name: 是否必须点名。关掉后只要出现看图动词就触发。

    Returns:
        是否请求识图。
    """
    body = "" if text is None else str(text)
    if not body.strip():
        return False
    lowered = body.lower()
    if any(neg in lowered for neg in _VISION_NEGATIVES):
        return False
    has_hint = any(hint in lowered for hint in _VISION_HINTS) or any(
        str(w).strip() and str(w).strip().lower() in lowered for w in extra_words
    )
    if not has_hint:
        return False
    if not require_name:
        return True
    if "@" in body or re.search(r"\[at[:：]", body, re.I):
        return True
    return any(str(n).strip() and str(n).strip().lower() in lowered for n in names)


# ---------------------------------------------------------------- JSON 抽取


def _balanced_candidates(raw: str, opener: str, closer: str, limit: int = 32) -> list[str]:
    """从左到右找出**括号配平**的候选片段。

    不能直接用正则 ``\\{[\\s\\S]*\\}``：那是贪婪匹配，从第一个 ``{`` 吃到最后一个
    ``}``，模型只要先写一句“根据分析如下：”就会跨越两个结构，``json.loads`` 必然失败，
    然后整轮解析结果被静默丢掉。这里的扫描还会跳过字符串字面量里的括号与转义，
    否则内容里出现 ``{`` 就会算错深度。（逻辑移植自 qq-bridge 的 ``extractJsonArray``。）
    """
    out: list[str] = []
    attempts = 0
    for i, ch in enumerate(raw):
        if ch != opener:
            continue
        if len(out) >= limit or attempts >= limit * 2:
            break
        attempts += 1
        depth = 0
        in_string = False
        escaped = False
        for j in range(i, len(raw)):
            c = raw[j]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    out.append(raw[i : j + 1])
                    break
    return out


def extract_json(text: object) -> Any:
    """从模型输出里抠出一段 JSON（对象或数组）。

    优先整体解析；失败则扫描配平候选，**优先返回含对象元素的数组/对象本身**，
    避免 ``根据 [1] 的分析：[{...}]`` 这种输出里 ``[1]`` 先到先得把真正结果挤掉。
    """
    raw = ("" if text is None else str(text)).strip()
    if not raw:
        return None
    # 去掉模型爱写的 ```json 围栏。
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        inner = fence.group(1).strip()
        try:
            return json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            raw = inner
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if len(raw) > 400_000:
        return None
    # 按**出现位置**排序后逐个尝试：模型先写什么结构就返回什么结构。
    # 这一点很重要——`根据 [1] 的分析：[{...}]` 里 `[1]` 本身也是合法 JSON 数组，
    # 若按“先对象再数组”的固定顺序找，就会把真正的结果挤掉。
    candidates: list[tuple[int, str]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        for candidate in _balanced_candidates(raw, opener, closer):
            candidates.append((raw.find(candidate), candidate))
    candidates.sort(key=lambda item: item[0])
    fallback: Any = None
    for _, candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        # 只认可“对象本身”或“装着对象的数组”：散文里的 `[1]` 这种下标引用
        # 也是合法 JSON 数组，先到先得会把真正的结果挤掉，所以它只作兜底。
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and any(isinstance(x, dict) for x in parsed):
            return parsed
        if fallback is None:
            fallback = parsed
    return fallback

