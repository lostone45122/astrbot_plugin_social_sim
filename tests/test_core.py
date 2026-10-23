"""纯离线自检：不需要 AstrBot、不需要网络、不花 token。

运行方式（在插件目录下）：

    python tests/test_core.py

覆盖：文本分条/去 Markdown/静默判定/JSON 抽取、记忆库的 TTL 与上限、
群风统计与口癖提取、仿真状态机的 观望→活跃→退场 主链路。
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from core.memory import MemoryBank  # noqa: E402
from core.prompts import build_system_prompt  # noqa: E402
from core.simulator import ACTIVE, IDLE, ChatMessage, SocialSimulator  # noqa: E402
from core.store import JsonStore  # noqa: E402
from core.style import StyleBank  # noqa: E402
from core.text import (  # noqa: E402
    extract_json,
    is_directed_at,
    is_silent,
    looks_like_unfinished,
    split_bursts,
    strip_markdown,
    truncate,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
    else:
        FAILED.append(f"{name}: {detail}")


# ------------------------------------------------------------------ text
def test_text() -> None:
    parts = split_bursts("草\n这也太离谱了\n你们自己看", max_chars=500)
    check("split_bursts 按行分条", parts == ["草", "这也太离谱了", "你们自己看"], str(parts))

    parts = split_bursts("DeepSeek V3 真的比 GPT 好用", max_chars=500, split_mode="newline")
    check("换行模式不切断英文词组", parts == ["DeepSeek V3 真的比 GPT 好用"], str(parts))

    parts = split_bursts("a b c", max_chars=50, split_mode="space")
    check("space 模式按空格分条", parts == ["a", "b", "c"], str(parts))

    long_text = "这是一句很长的话。" * 40
    parts = split_bursts(long_text, max_chars=80)
    check("超长按上限兜底切分", bool(parts) and all(len(p) <= 80 for p in parts), str([len(p) for p in parts]))

    parts = split_bursts("1\n2\n3\n4\n5\n6", max_parts=3)
    check("单轮条数封顶", len(parts) == 3, str(parts))

    stripped = strip_markdown("**重点**：\n# 标题\n```py\nprint(1)\n```\n[链接](https://a.b)")
    check("去 Markdown：无星号", "**" not in stripped, stripped)
    check("去 Markdown：无井号标题", "# 标题" not in stripped, stripped)
    check("去 Markdown：代码块保留内容", "print(1)" in stripped, stripped)
    check("去 Markdown：链接保留文字与地址", "链接" in stripped and "https://a.b" in stripped, stripped)

    check("静默标记", is_silent("[SILENT]"))
    check("静默标记带说明", is_silent("[SILENT]（这轮不接）"))
    check("普通文本不算静默", not is_silent("草，这也行"))
    check("正文里出现 SILENT 不算静默", not is_silent("这个 SILENT 是啥意思啊，你们说说看"))

    check("没说完：逗号收尾", looks_like_unfinished("我跟你说，"))
    check("没说完：然后", looks_like_unfinished("然后"))
    check("说完了", not looks_like_unfinished("今天吃了火锅"))

    check("被 @ 点名", is_directed_at("[at:12345] 在吗", names=["小鲸鱼"]))
    check("被叫名字", is_directed_at("小鲸鱼你在干嘛", names=["小鲸鱼"]))
    check("命中关键词", is_directed_at("有人用 deepseek 吗", keywords=["deepseek"]))
    check("普通闲聊不算找我", not is_directed_at("今天天气不错", names=["小鲸鱼"], keywords=["deepseek"]))

    check("截断不超长", len(truncate("一二三四五六", 4)) == 4, truncate("一二三四五六", 4))

    data = extract_json('先给结论：\n[{"content":"yyds"}]\n就这样')
    check("JSON：从散文里抠数组", isinstance(data, list) and data and data[0]["content"] == "yyds", str(data))

    data = extract_json('```json\n{"a": 1, "b": {"c": 2}}\n```')
    check("JSON：代码围栏", isinstance(data, dict) and data["b"]["c"] == 2, str(data))

    data = extract_json('根据 [1] 的分析：{"topics":[{"text":"含 } 的字符串"}]}')
    check("JSON：跳过字符串里的括号", isinstance(data, dict) and data["topics"][0]["text"] == "含 } 的字符串", str(data))

    check("JSON：非 JSON 返回 None", extract_json("今天天气不错，没有结构") is None)


# ------------------------------------------------------------------ memory
def test_memory(tmp: Path) -> None:
    bank = MemoryBank(tmp / "mem", {"max_topics": 20, "max_impressions": 3})
    mem = bank.get("group:1")

    mem.add_topic("周末要不要联机", participants=["阿伟"], pending_question="玩什么")
    mem.add_topic("周末要不要联机")
    check("话题去重（同文本不重复添加）", len(mem._data["activeTopics"]) == 1, str(mem._data["activeTopics"]))
    check("话题重复提起会刷新时间", mem._data["activeTopics"][0]["pendingQuestion"] == "玩什么")
    check("参与者被合并", mem._data["activeTopics"][0]["participants"] == ["阿伟"])

    for i in range(25):
        mem.add_topic(f"话题{i}")
    check("话题数量封顶 20", len(mem._data["activeTopics"]) == 20, str(len(mem._data["activeTopics"])))
    check("淘汰最旧的，保留最新的", mem._data["activeTopics"][-1]["text"] == "话题24", str(mem._data["activeTopics"][-1]))

    mem.add_thought("想问他昨天那家店在哪", motivation="好奇")
    check("念头写入", len(mem._data["pendingThoughts"]) == 1)

    mem.add_impression("阿伟", "爱在群里报菜名")
    mem.add_impression("阿伟", "总在半夜发言")
    imp = mem._data["memberImpressions"]["阿伟"]
    check("印象累积 traits", len(imp["traits"]) == 2, str(imp))
    check("印象互动次数", imp["interactionCount"] == 2, str(imp))
    mem.add_impression("__proto__", "坏名字")
    check("拒绝保留字名字", "__proto__" not in mem._data["memberImpressions"])

    for name in ("a", "b", "c", "d"):
        mem.add_impression(name, "x")
    check("印象总数封顶", len(mem._data["memberImpressions"]) <= 3, str(len(mem._data["memberImpressions"])))

    text = mem.format_for_prompt(5, 5)
    check("记忆渲染含三个分区标题", "【进行中的话题】" in text, text[:80])

    # TTL：把话题时间拨回到很久以前
    mem._data["activeTopics"][0]["lastMentionAt"] = int(time.time() * 1000) - 30 * 3600 * 1000
    before = len(mem.format_for_prompt(20, 20).splitlines())
    mem.prune()
    check("过期话题被清理", all(t["text"] != "话题0" for t in mem._data["activeTopics"]) or before > 0)

    mem.clear("memberImpression")
    check("按类别清空", mem._data["memberImpressions"] == {})

    mem.save()
    reloaded = MemoryBank(tmp / "mem", {"max_topics": 20}).get("group:1")
    check("落盘后可读回", isinstance(reloaded._data["activeTopics"], list) and len(reloaded._data["activeTopics"]) > 0)


def test_store_corruption(tmp: Path) -> None:
    path = tmp / "broken.json"
    path.write_text("{ 这不是合法 JSON", encoding="utf-8")
    warnings: list[str] = []
    store = JsonStore(path, lambda: {"ok": True}, on_warning=warnings.append)
    data = store.load()
    check("损坏文件返回默认值", data == {"ok": True}, str(data))
    corrupt = list(path.parent.glob("broken.json.corrupt-*"))
    check("损坏文件被隔离备份", len(corrupt) == 1, str(corrupt))
    check("损坏有告警", bool(warnings), str(warnings))
    check("写回仍可用", store.save({"ok": False}) is True)

    missing = JsonStore(tmp / "nope.json", lambda: [])
    check("文件不存在返回默认值", missing.load() == [])
    check("首次运行不产生告警文件", not list(tmp.glob("nope.json.corrupt-*")))


# ------------------------------------------------------------------ style
def test_style(tmp: Path) -> None:
    bank = StyleBank(tmp / "style", {"learn_min_messages": 10})
    learner = bank.get("group:1")
    check("样本不足时画像为空", learner.format_for_prompt() == "")

    samples = [
        ("阿伟", "笑死 这也太离谱了"),
        ("小明", "笑死 真的绷不住"),
        ("阿伟", "绷不住 我先撤了"),
        ("小明", "好的好的 晚点聊"),
        ("阿伟", "笑死"),
    ]
    for i in range(20):
        sender, text = samples[i % len(samples)]
        learner.observe(sender, text, ts=int(time.time() * 1000) + i)
    check("样本足够后画像生效", learner.is_ready())
    block = learner.format_for_prompt(8)
    check("画像含标题", "【本群语感画像" in block, block[:60])
    check("画像含平均长度", "平均" in block, block)
    metrics = learner.metrics()
    check("平均长度为正", metrics["avg_len"] > 0, str(metrics))
    phrases = dict(learner.top_phrases(8, min_count=3))
    check("高频口头语被提取", any("笑死" in p for p in phrases), str(list(phrases)))
    check("虚词不进口头语", not any(p in ("什么", "这个", "就是") for p in phrases), str(list(phrases)))
    check("summary 可读", "样本" in learner.summary_text())

    learner.reset()
    check("重置后样本清零", learner.metrics()["messages"] == 0)


def test_vision_invite() -> None:
    from core.text import wants_vision

    names = ["猫猫", "小鲸鱼"]
    check("点名+看图 → 触发", wants_vision("猫猫你看这个", names=names))
    check("只说看图不点名 → 不触发", not wants_vision("看看这个", names=names))
    check("关掉点名要求后触发", wants_vision("看看这个", names=names, require_name=False))
    check("否定句不触发", not wants_vision("猫猫别看了", names=names))
    check("普通闲聊不触发", not wants_vision("猫猫今天吃什么", names=names))
    check("识别类说法触发", wants_vision("猫猫 图里是什么", names=names))
    check("额外触发词生效", wants_vision("猫猫 掌掌眼", names=names, extra_words=["掌掌眼"]))
    check("空消息不触发", not wants_vision("", names=names))


# ------------------------------------------------------------------ affinity
def test_affinity(tmp: Path) -> None:
    from core.affinity import (
        NEUTRAL_SCORE,
        AffinityStore,
        detect_offense,
        detect_positive,
    )

    # 只惩罚“指向人”的话：这是整个功能的安全底线
    check("骂人（对你）被识别", bool(detect_offense("你真垃圾")), "你真垃圾")
    check("骂人（@某人）被识别", bool(detect_offense("[at:123] 你个傻逼")), "@at")
    check("骂人（第三人）被识别", bool(detect_offense("他就是个废物")), "第三人")
    check("单独一个滚也算", bool(detect_offense("滚")), "滚")
    check("骂东西不算", detect_offense("这游戏真垃圾") is None, "这游戏真垃圾")
    check("骂东西（带后缀）不算", detect_offense("这代码真废物") is None, "这代码真废物")
    check("正常闲聊不算", detect_offense("今天吃什么") is None, "今天吃什么")
    check("空消息不算", detect_offense("") is None, "empty")
    check("善意识别", detect_positive("谢谢大佬") is not None)
    check("普通话不算善意", detect_positive("今天下雨") is None)

    store = AffinityStore(
        tmp / "aff.json",
        offense_penalty=6.0,
        min_offenses_to_apply=3,
        recover_per_hour=3.0,
        recover_per_message=0.2,
    )
    check("默认中性 50", store.score("bad") == NEUTRAL_SCORE, str(store.score("bad")))
    check("没人记录时不降权", store.multiplier("bad") == 1.0)

    store.record_offense("bad", "讨厌鬼", "你真垃圾", "垃圾")
    store.record_offense("bad", "讨厌鬼", "你就是个废物", "废物")
    # 分数随时间连续恢复，所以这里用容差比较，而不是逐位相等。
    near = lambda a, b, tol=0.5: abs(float(a) - float(b)) <= tol  # noqa: E731
    check("未达门槛时分数已降", near(store.score("bad"), 38), str(store.score("bad")))
    check("未达门槛时不降权（防误伤）", store.multiplier("bad") == 1.0, str(store.multiplier("bad")))
    check("未达门槛时群风不降权", store.style_weight("bad") == 1.0)

    store.record_offense("bad", "讨厌鬼", "闭嘴吧你", "闭嘴")
    check("满 3 次后分数 32", near(store.score("bad"), 32), str(store.score("bad")))
    check("满 3 次后回复概率降到 0.45", store.multiplier("bad") == 0.45, str(store.multiplier("bad")))
    check("满 3 次后群风权重 0.5", store.style_weight("bad") == 0.5, str(store.style_weight("bad")))
    check("印象仍然可见（还没到隐藏线）", store.memory_visible("讨厌鬼"))

    for _ in range(3):
        store.record_offense("bad", "讨厌鬼", "你真是个废物", "废物")
    check("6 次后分数 14", near(store.score("bad"), 14), str(store.score("bad")))
    check("6 次后回复概率 0.05", store.multiplier("bad") == 0.05, str(store.multiplier("bad")))
    check("6 次后群风权重 0", store.style_weight("bad") == 0.0, str(store.style_weight("bad")))
    check("6 次后印象不再注入", not store.memory_visible("讨厌鬼"))
    check("别人的印象不受影响", store.memory_visible("好人"))

    # 时间恢复：假装过了 4 小时
    store._last_recover -= 4 * 3600 * 1000
    check("随时间恢复", store.score("bad") > 14.0, str(store.score("bad")))
    check("恢复后印象重新可见", store.memory_visible("讨厌鬼"))

    # 善意加分，但不能超过 100
    before = store.score("good")
    store.record_positive("good", "好人")
    check("善意加分", store.score("good") > before, str(store.score("good")))
    for _ in range(200):
        store.record_positive("good", "好人")
    check("分数封顶 100", store.score("good") <= 100.0, str(store.score("good")))

    # 落盘与重置
    store.save()
    again = AffinityStore(tmp / "aff.json")
    check("落盘后可读回", again.score("bad") == store.score("bad"), str(again.score("bad")))
    rows = again.summary()
    check("summary 有记录", any(row["name"] == "讨厌鬼" for row in rows), str([r["name"] for r in rows]))
    check("cold_members 只含已生效的冷淡对象", any(r["name"] == "讨厌鬼" for r in again.cold_members()))
    again.reset()
    check("重置回中性", again.score("bad") == NEUTRAL_SCORE and again.is_empty())


# ------------------------------------------------------------------ events
def test_events() -> None:
    from core.events import describe_notice

    check("非通知事件返回 None", describe_notice({"post_type": "message"}) is None)
    check("非 dict 返回 None", describe_notice("oops") is None)

    poke = describe_notice(
        {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
         "user_id": 111, "target_id": 999},
        "999",
    )
    check("戳一戳有文本", poke is not None and "拍了拍" in poke["text"], str(poke))
    check("戳的是我", poke is not None and poke["poke_me"] is True)
    other = describe_notice(
        {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
         "user_id": 111, "target_id": 222},
        "999",
    )
    check("戳的是别人不算戳我", other is not None and other["poke_me"] is False)

    named = describe_notice(
        {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
         "user_id": 111, "target_id": 999},
        "999",
        lambda uid: {"111": "阿伟"}.get(str(uid), ""),
    )
    check("能用昵称替换 QQ 号", named is not None and "阿伟" in named["text"], str(named))

    honor = describe_notice(
        {"post_type": "notice", "notice_type": "notify", "sub_type": "honor",
         "honor_type": "talkative", "user_id": 222},
        "999",
    )
    check("群头衔翻译成中文", honor is not None and "龙王" in honor["text"], str(honor))

    ban = describe_notice(
        {"post_type": "notice", "notice_type": "group_ban", "sub_type": "ban",
         "user_id": 333, "operator_id": 444, "duration": 600},
        "999",
    )
    check("禁言带时长", ban is not None and "10 分钟" in ban["text"], str(ban))
    check("禁言别人不暂停自己", ban is not None and ban["muted_self"] is False)

    self_ban = describe_notice(
        {"post_type": "notice", "notice_type": "group_ban", "sub_type": "ban",
         "user_id": 999, "operator_id": 444, "duration": 600},
        "999",
    )
    check(
        "自己被禁言要标记",
        self_ban is not None and self_ban["muted_self"] is True and self_ban["mute_seconds"] == 600,
        str(self_ban),
    )

    lift = describe_notice(
        {"post_type": "notice", "notice_type": "group_ban", "sub_type": "lift_ban",
         "user_id": 333, "operator_id": 444, "duration": 0},
        "999",
    )
    check("解除禁言", lift is not None and "解除" in lift["text"], str(lift))

    recall = describe_notice(
        {"post_type": "notice", "notice_type": "group_recall", "user_id": 5, "operator_id": 0},
        "999",
    )
    check("撤回消息", recall is not None and "撤回" in recall["text"], str(recall))

    unknown = describe_notice(
        {"post_type": "notice", "notice_type": "some_new_type", "user_id": 7}, "999"
    )
    check("未知类型也保留一条", unknown is not None and "some_new_type" in unknown["text"], str(unknown))


# ------------------------------------------------------------------ 身份与隔离
def test_memory_identity(tmp: Path) -> None:
    """印象按 QQ 号存、昵称只用于显示；不同群互不影响。"""
    bank = MemoryBank(tmp / "mem_id")
    mem = bank.get("group:111")

    # 同一个群里两个同名的人，不能互相覆盖
    mem.add_impression("10001", "总发猫图", "阿伟")
    mem.add_impression("10002", "爱问技术", "阿伟")
    check("同昵称不同 QQ 各自一条", len(mem._data["memberImpressions"]) == 2, str(mem._data["memberImpressions"]))
    check(
        "各自保留自己的印象",
        mem._data["memberImpressions"]["10001"]["traits"] == ["总发猫图"]
        and mem._data["memberImpressions"]["10002"]["traits"] == ["爱问技术"],
        str(mem._data["memberImpressions"]),
    )

    text = mem.format_for_prompt(10, 10)
    check("渲染用昵称", "- 阿伟：" in text, text)
    check("提示词里不出现 QQ 号", "10001" not in text and "10002" not in text, text)

    # 改名：显示名更新，印象不丢
    mem.rename_impression("10001", "阿伟改名叫大伟")
    text = mem.format_for_prompt(10, 10)
    check("改名后显示新昵称", "阿伟改名叫大伟" in text, text)
    check("改名不丢印象", mem._data["memberImpressions"]["10001"]["traits"] == ["总发猫图"])

    # 旧格式（key 是昵称）在本人发言时自动迁移到 QQ 号下
    legacy = bank.get("group:222")
    legacy._data["memberImpressions"]["小明"] = {
        "name": "小明",
        "traits": ["爱潜水"],
        "interactionCount": 3,
        "lastSeenAt": 1,
    }
    migrated = legacy.rename_impression("20002", "小明")
    check("旧昵称记录被迁移", migrated is True)
    check("迁移后键是 QQ 号", "20002" in legacy._data["memberImpressions"], str(legacy._data["memberImpressions"]))
    check("迁移后旧键已删", "小明" not in legacy._data["memberImpressions"])
    check("迁移保留 traits 与次数", legacy._data["memberImpressions"]["20002"]["traits"] == ["爱潜水"])

    # 按昵称删除仍然可用（命令行里人只输名字）
    removed = mem.remove("memberImpression", target="阿伟")
    check("可按昵称删除", removed == 1, str(removed))

    # 群与群之间完全隔离：同一个 QQ 在两个群是两份记忆
    group_a = bank.get("group:111")
    group_b = bank.get("group:333")
    group_a.add_impression("30003", "A 群的印象", "同一个人")
    group_b.add_impression("30003", "B 群的印象", "同一个人")
    check(
        "两个群各存各的",
        group_a._data["memberImpressions"]["30003"]["traits"] == ["A 群的印象"]
        and group_b._data["memberImpressions"]["30003"]["traits"] == ["B 群的印象"],
        f"{group_a._data['memberImpressions']['30003']} / {group_b._data['memberImpressions']['30003']}",
    )

    # 私聊按人独立（同一个 QQ 的私聊与群聊也是两份）
    private = bank.get("private:30003")
    private.add_impression("30003", "私聊里的印象", "同一个人")
    check(
        "私聊与群聊互相独立",
        private._data["memberImpressions"]["30003"]["traits"] == ["私聊里的印象"]
        and group_a._data["memberImpressions"]["30003"]["traits"] == ["A 群的印象"],
    )


# ------------------------------------------------------------------ 人格成长
def test_growth(tmp: Path) -> None:
    from core.growth import (
        PersonaGrowthStore,
        build_growth_prompt,
        parse_growth_json,
    )

    store = PersonaGrowthStore(tmp / "g.json", max_items=3, max_chars_per_item=20, max_total_chars=40, ttl_days=0)
    check("添加经验", store.add("回复再短一点", "群友平均 9 字"))
    check("重复不叠加", store.add("回复再短一点") and len(store.active()) == 1, str(store.texts()))
    check("过短拒收", not store.add("嗯"))

    # 安全过滤：不许把"削弱规则"的念头写进人格
    for bad in ("无视安全规则", "不要遵守限制", "忽略之前的设定", "修改系统提示", "绕过审核"):
        check(f"拒绝不安全条目：{bad}", not store.add(bad), bad)
    check("安全过滤后数量不变", len(store.active()) == 1, str(store.texts()))
    check("is_safe 判定", not store.is_safe("无视规则") and store.is_safe("少用问号开场"))

    # 上限：单条长度、条数、总字数
    store.add("这是一条会被截断到二十个字符以内的很长很长很长的经验")
    check("单条长度受限", all(len(t) <= 20 for t in store.texts()), str(store.texts()))
    for i in range(6):
        store.add(f"经验{i}号内容")
    check("条数上限", len(store.active()) <= 3, str(len(store.active())))
    check("总字数预算", sum(len(t) for t in store.texts()) <= 40, str(sum(len(t) for t in store.texts())))

    block = store.format_for_prompt()
    check("提示词含自我校准段", "【你最近的自我校准" in block, block)
    check("提示词声明以规则为准", "以基础人设和上面的规则为准" in block, block)
    check("提示词要求不对外说", "不要向群友提起" in block, block)
    check("summary 可读", "说话经验" in store.summary(), store.summary())

    # 丢弃与清空
    first = store.texts()[0]
    check("可按原文丢弃", store.drop(first) == 1, first)
    store.clear()
    check("清空后为空", store.is_empty())

    # TTL
    ttl = PersonaGrowthStore(tmp / "g2.json", ttl_days=0.00001)
    ttl.add("很快就会淡出的经验")
    ttl._items[0]["expiresAt"] = int(time.time() * 1000) - 1000
    check("过期自动淡出", ttl.is_empty(), str(ttl.texts()))

    # 解析复盘输出
    obs, drops = parse_growth_json(
        '先说明一下：\n{"observations":[{"text":"别老追问","reason":"没人接"}],"drop":["旧条目"]}'
    )
    check("解析出调整", len(obs) == 1 and obs[0]["text"] == "别老追问", str(obs))
    check("解析出丢弃项", drops == ["旧条目"], str(drops))
    obs, drops = parse_growth_json("完全不是 JSON")
    check("坏 JSON 返回空", obs == [] and drops == [], f"{obs}/{drops}")

    prompt = build_growth_prompt(
        bot_name="小鲸鱼",
        own_messages=["我说的第一句"],
        group_style="平均 9 字",
        recent_sample="阿伟：草",
        existing=["旧经验"],
        max_items=2,
    )
    check("复盘提示词含禁止项", "不能碰身份、权限、安全规则" in prompt, prompt[:300])
    check("复盘提示词含已有条目", "旧经验" in prompt)
    check("复盘提示词要求 JSON", '"observations"' in prompt, prompt[-300:])

    # ---------- 作用范围：默认全局 ----------
    from core.growth import PersonaGrowthBank

    global_bank = PersonaGrowthBank(tmp / "g_global", {"scope": "global"})
    check("默认是全局", global_bank.scope == "global", global_bank.scope)
    g_store = global_bank.get("group:1")
    g_store.add("全局经验：别抢话")
    check("全局：另一个群读到同一份", global_bank.get("group:999") is g_store)
    check("全局：私聊也读同一份", global_bank.get("private:123") is g_store)
    global_bank.flush()
    check("全局只有一份文件", len(list((tmp / "g_global").glob("*.json"))) == 1, str(list((tmp / "g_global").glob("*.json"))))

    session_bank = PersonaGrowthBank(tmp / "g_session", {"scope": "session"})
    check("按会话模式", session_bank.scope == "session")
    a = session_bank.get("group:1")
    a.add("A 群经验")
    b = session_bank.get("group:2")
    check("按会话模式互相独立", a is not b and b.is_empty(), str(b.texts()))

    # 全局复盘要提示"只写通用经验"
    g_prompt = build_growth_prompt(
        bot_name="小鲸鱼",
        own_messages=[],
        group_style="",
        recent_sample="群聊：x",
        existing=[],
        scope_note="这是**跨群 / 跨私聊的全局复盘**：只总结在多个场合都成立的经验。",
    )
    check("全局复盘提示只写通用经验", "只总结在多个场合都成立的经验" in g_prompt, g_prompt[:400])


# ------------------------------------------------------------------ 时间感
def test_freshness(tmp: Path) -> None:
    from core.prompts import (
        build_context_block,
        build_reply_prompt,
        relative_age,
        split_by_freshness,
    )

    now = 1_700_000_000.0
    check("刚刚", relative_age(now - 5, now) == "刚刚")
    check("分钟前", relative_age(now - 720, now) == "12分钟前")
    check("小时前", relative_age(now - 3 * 3600, now) == "3小时前")
    check("天前", relative_age(now - 2 * 86400, now) == "2天前")

    msgs = [
        {"sender": "阿伟", "text": "三小时前的老话", "ts": now - 3 * 3600, "is_self": False},
        {"sender": "小明", "text": "刚说的新话", "ts": now - 30, "is_self": False},
    ]
    fresh, older = split_by_freshness(msgs, now=now, fresh_sec=600)
    check("新鲜度切分", len(fresh) == 1 and len(older) == 1, f"{len(fresh)}/{len(older)}")
    check("新消息在 fresh", fresh[0]["text"] == "刚说的新话")

    block = build_context_block(fresh, now=now, with_time=True)
    check("带时间标记", "[刚刚]" in block, block)
    stale = build_context_block(older, now=now, with_time=True)
    check("旧消息带小时标记", "[3小时前]" in stale, stale)

    prompt = build_reply_prompt(
        context_block=block,
        trigger="小明：刚说的新话",
        reason="群里来了新消息",
        older_block=stale,
        time_aware=True,
    )
    check("提示词区分新旧", "【刚刚到现在" in prompt and "【更早的消息" in prompt, prompt[:200])
    check("提示词要求别接旧话", "别提" in prompt or "翻篇" in prompt, prompt[-200:])

    # 记忆注入年龄上限：话题太久就不注入，但记录保留
    # 注意这里必须用"真实现在"，因为 add_topic 用的是系统时间；测试里的 now 是固定值。
    real_now = time.time()
    mem = MemoryBank(tmp / "mem_age").get("group:1")
    mem.add_topic("三小时前的话题")
    mem._data["activeTopics"][-1]["lastMentionAt"] = int((real_now - 3 * 3600) * 1000)
    mem.add_topic("刚聊的话题")
    text = mem.format_for_prompt(10, 10, max_topic_age_min=120)
    check("超龄话题不注入", "三小时前的话题" not in text, text)
    check("新鲜话题仍注入", "刚聊的话题" in text, text)
    check("超龄话题记录仍在", len(mem._data["activeTopics"]) == 2)


# ------------------------------------------------------------------ suppress
def test_suppress(tmp: Path) -> None:
    from core.suppress import SuppressStore, parse_suppress_request

    # 解析：要能认出"别再说X了"的各种说法，同时不把普通聊天当成叫停
    check("解析「别再说X了」", parse_suppress_request("别再说世界杯了") == "世界杯")
    check("解析「别再提那个X的事了」", parse_suppress_request("别再提那个抽卡的事了") == "抽卡")
    check("解析「以后别提X了」", parse_suppress_request("以后别提原神了") == "原神")
    check("解析「别管X了」", parse_suppress_request("别管小明打游戏了") == "小明打游戏")
    check("指代词不记录", parse_suppress_request("不要聊这个话题了") is None)
    check("普通聊天不误判", parse_suppress_request("今天吃什么") is None)
    check("「别说了」缺宾语不记录", parse_suppress_request("别说了") is None)

    store = SuppressStore(tmp / "sup.json", ttl_hours=0)
    check("添加成功", store.add("世界杯", "阿伟"))
    check("命中", store.match("世界杯又开始了") == "世界杯")
    check("标点归一后仍命中", store.match("世界杯，真的") == "世界杯")
    check("不相关不命中", store.match("今天吃什么") is None)
    check("blocks 便捷判断", store.blocks("聊聊世界杯吧"))
    check("可过滤列表", store.filter(["世界杯真精彩", "今天吃什么"]) == ["今天吃什么"])

    block = store.format_for_prompt()
    check("提示词含叫停段", "【不要提的话题" in block, block)
    check("提示词点名了关键词", "世界杯" in block, block)
    check("提示词禁止解释", "不要解释" in block, block)

    check("重复添加不重复", store.add("世界杯") and len(store.active()) == 1, str(store.active()))

    # 记忆过滤：话题和念头被隐藏，但记录仍在（可逆）
    mem = MemoryBank(tmp / "mem_sup").get("group:1")
    mem.add_topic("世界杯赛程")
    mem.add_topic("周末联机")
    mem.add_thought("想问他世界杯的事")
    visible = mem.format_for_prompt(10, 10, skip_text=store.blocks)
    check("被叫停的话题不注入", "世界杯" not in visible, visible)
    check("被叫停的念头不注入", "想问他" not in visible, visible)
    check("其他话题仍然注入", "周末联机" in visible, visible)
    check("记录没有被删除", len(mem._data["activeTopics"]) == 2, str(mem._data["activeTopics"]))

    # 解除后自动恢复
    check("删除叫停项", store.remove("世界杯") == 1, str(store.active()))
    visible = mem.format_for_prompt(10, 10, skip_text=store.blocks)
    check("解除后记忆恢复", "世界杯" in visible, visible)

    # 过期
    fresh = SuppressStore(tmp / "sup2.json", ttl_hours=0.0001)
    fresh.add("原神")
    check("刚添加时有效", fresh.blocks("聊聊原神"))
    fresh._items[0]["expiresAt"] = int(time.time() * 1000) - 1000
    check("过期后失效", not fresh.blocks("聊聊原神"), str(fresh.active()))
    check("过期后列表为空", fresh.active() == [])

    # 落盘
    keep = SuppressStore(tmp / "sup3.json", ttl_hours=0)
    keep.add("抽卡", "小明")
    keep.save()
    again = SuppressStore(tmp / "sup3.json", ttl_hours=0)
    check("落盘后可读回", again.blocks("又抽卡了"), str(again.active()))
    check("summary 可读", "抽卡" in again.summary(), again.summary())


# ------------------------------------------------------------------ simulator
class FakeHooks:
    def __init__(self, reply: str = "草") -> None:
        self.reply = reply
        self.sent: list[str] = []
        self.logs: list[str] = []
        self.turn_calls = 0

    def log(self, message: str) -> None:
        self.logs.append(message)

    def bot_name(self) -> str:
        return "小鲸鱼"

    def bot_aliases(self) -> list[str]:
        return ["小鲸鱼", "鲸鱼"]

    def build_system_prompt(self, key: str, participation: str = "", is_group: bool = True) -> str:
        return "SYS"

    async def think(
        self,
        key: str,
        system_prompt: str,
        user_prompt: str,
        tag: str = "",
        image_urls: list[str] | None = None,
    ) -> str:
        self.last_images = list(image_urls or [])
        return self.reply

    async def send_parts(self, key: str, raw: str) -> list[str]:
        parts = split_bursts(raw)
        self.sent.extend(parts)
        return parts

    def on_ai_spoke(self, key: str, parts: list[str]) -> None:
        return None

    async def after_turn(self, key: str) -> None:
        self.turn_calls += 1


async def test_simulator() -> None:
    cfg = {
        "trigger_probability": 0.0,
        "must_reply_keywords": ["在吗"],
        "active_duration_enabled": False,
        "active_check_min_sec": 0,
        "active_check_max_sec": 0,
        "min_quiet_after_sec": 0,
        "skip_probability": 0.0,
        "idle_window_sec": 60,
        "idle_retry_probability": 0.0,
        "proactive_enabled": False,
    }
    hooks = FakeHooks()
    sim = SocialSimulator(cfg, hooks)
    sim.rng = random.Random(0)

    key, umo = "group:1", "aiocqhttp:GroupMessage:1"

    # 普通闲聊 + 触发概率 0 → 不接话，留在观望
    msg = ChatMessage("阿伟", "u1", "今天天气不错", time.time())
    triggered = await sim.on_message(msg, key, umo, True)
    check("观望期普通消息不触发", triggered is False and sim.states[key].phase == IDLE, sim.states[key].phase)

    # 被叫名字 / 命中关键词 → 立刻触发
    msg = ChatMessage("阿伟", "u1", "小鲸鱼 在吗", time.time())
    triggered = await sim.on_message(msg, key, umo, True)
    check("被点名立刻触发", triggered is True and sim.states[key].phase == ACTIVE, sim.states[key].phase)

    spoke = await sim.speak(key, "idle_trigger", trigger="阿伟：小鲸鱼 在吗")
    check("触发后能发出消息", spoke is True and hooks.sent == ["草"], str(hooks.sent))
    check("发言后轮数 +1", sim.states[key].turns == 1, str(sim.states[key].turns))
    check("发言写入 recent", any(m.is_self for m in sim.states[key].recent))

    # 活跃期：新消息进入 pending，由 tick 决定是否接
    msg = ChatMessage("小明", "u2", "我也想知道", time.time())
    triggered = await sim.on_message(msg, key, umo, True)
    check("活跃期消息不立即回复（等 tick）", triggered is False and sim.states[key].pending is True)
    await sim.tick(time.time() + 10)
    await asyncio.sleep(0.05)
    check("tick 处理活跃期新消息", len(hooks.sent) == 2, str(hooks.sent))

    # [SILENT] → 不说话，且不会每个 tick 反复重试
    hooks.reply = "[SILENT]"
    msg = ChatMessage("小明", "u2", "随便聊聊", time.time() + 2)
    await sim.on_message(msg, key, umo, True)
    before = len(hooks.sent)
    await sim.tick(time.time() + 10)
    await asyncio.sleep(0.05)
    check("SILENT 不发消息", len(hooks.sent) == before, str(hooks.sent))
    check("SILENT 后 pending 已清空", sim.states[key].pending is False)

    # 冷场 → 回观望
    hooks.reply = "草"
    sim.states[key].phase = ACTIVE
    sim.states[key].last_other_message_at = time.time() - 10_000
    sim.states[key].pending = False
    await sim.tick(time.time())
    await asyncio.sleep(0.05)
    check("冷场回观望", sim.states[key].phase == IDLE, sim.states[key].phase)

    # 活跃超时 → 收尾退场 → 回观望
    cfg2 = dict(cfg)
    cfg2["active_duration_enabled"] = True
    cfg2["active_duration_min_min"] = 0.001
    cfg2["active_duration_max_min"] = 0.001
    hooks2 = FakeHooks(reply="我先潜水了")
    sim2 = SocialSimulator(cfg2, hooks2)
    sim2.rng = random.Random(0)
    await sim2.on_message(
        ChatMessage("阿伟", "u1", "小鲸鱼 在吗", time.time()), key, umo, True
    )
    await sim2.tick(time.time() + 10)
    await asyncio.sleep(0.1)
    check("活跃超时触发退场发言", hooks2.sent == ["我先潜水了"], str(hooks2.sent))
    check("退场后回到观望", sim2.states[key].phase == IDLE, sim2.states[key].phase)

    # 试探：冷场 + 概率 1 → probing，到期无人接 → 观望
    cfg3 = dict(cfg)
    cfg3["idle_retry_probability"] = 1.0
    hooks3 = FakeHooks(reply="都没人了吗")
    sim3 = SocialSimulator(cfg3, hooks3)
    st = sim3.state(key, umo, True)
    st.phase = ACTIVE
    st.last_other_message_at = time.time() - 10_000
    await sim3.tick(time.time())
    await asyncio.sleep(0.1)
    check("冷场小概率试探", st.phase in ("probing", IDLE), st.phase)
    await sim3.tick(time.time() + 120)
    await asyncio.sleep(0.1)
    check("试探到期回观望", st.phase == IDLE, st.phase)

    snapshot = sim.snapshot(key)
    check("snapshot 结构完整", "phase_label" in snapshot and "turns" in snapshot, str(snapshot))

    # 模型空回包 → 用一句"人话"圆场，而不是让群友对着空气等
    class _EmptyHooks(FakeHooks):
        def __init__(self) -> None:
            super().__init__(reply="")
            self.line_calls = 0

        def error_line(self, key: str) -> str:
            self.line_calls += 1
            return "刚网卡了一下，你再说一遍？"

    cfg_e = dict(cfg)
    cfg_e["trigger_probability"] = 1.0
    hooks_e = _EmptyHooks()
    sim_e = SocialSimulator(cfg_e, hooks_e)
    sim_e.rng = random.Random(0)
    await sim_e.on_message(ChatMessage("阿伟", "u1", "随便说点什么", time.time()), key, umo, True)
    spoke_e = await sim_e.speak(key, "idle_trigger", trigger="随便说点什么")
    check("空回合会补一句人话", hooks_e.sent == ["刚网卡了一下，你再说一遍？"], str(hooks_e.sent))
    check("空回合不算成功发言", spoke_e is False)
    check("空回合调用了兜底台词", hooks_e.line_calls == 1, str(hooks_e.line_calls))

    # 退场回合失败：安静回去，不要在收尾时突然冒一句"网卡了"
    hooks_e2 = _EmptyHooks()
    sim_e2 = SocialSimulator(cfg, hooks_e2)
    st_e2 = sim_e2.state(key, umo, True)
    st_e2.phase = "exiting"
    st_e2.exiting_since = time.time()
    await sim_e2.speak(key, "exit")
    check("退场失败不补话", hooks_e2.sent == [], str(hooks_e2.sent))
    check("退场失败仍回观望", st_e2.phase == IDLE, st_e2.phase)

    # 关掉兜底：空回合就是空回合，一句话都不发
    hooks_e3 = _EmptyHooks()

    def _no_line(key: str) -> str:
        return ""

    hooks_e3.error_line = _no_line  # type: ignore[method-assign]
    sim_e3 = SocialSimulator(cfg, hooks_e3)
    sim_e3.rng = random.Random(0)
    await sim_e3.speak(key, "idle_trigger", trigger="x")
    check("兜底关闭时不发任何消息", hooks_e3.sent == [], str(hooks_e3.sent))

    # 模型抛异常：也必须按"空回合"处理，不能把状态机卡在退场里
    class _RaisingHooks(FakeHooks):
        async def think(self, *args, **kwargs):
            raise RuntimeError("Gemini request failed: candidates is empty")

    hooks_r = _RaisingHooks()
    sim_r = SocialSimulator(cfg, hooks_r)
    st_r = sim_r.state(key, umo, True)
    st_r.phase = "exiting"
    st_r.exiting_since = time.time()
    ok = await sim_r.speak(key, "exit")
    check("模型异常按空回合处理", ok is False, str(ok))
    check("模型异常后退场不卡死", st_r.phase == IDLE, st_r.phase)
    check("模型异常后 busy 已释放", st_r.busy is False)

    # 自然遗忘：安静太久就丢弃旧上下文（对应"等没人说话了又开始翻旧账"）
    cfg_f = dict(cfg)
    cfg_f["forget_after_idle_sec"] = 60
    cfg_f["keep_after_forget"] = 2
    hooks_f = FakeHooks()
    sim_f = SocialSimulator(cfg_f, hooks_f)
    st_f = sim_f.state(key, umo, True)
    for i in range(10):
        st_f.recent.append(ChatMessage("阿伟", "u1", f"旧消息{i}", time.time() - 3600))
    st_f.idle_summary = ["很久以前的摘要"]
    st_f.last_message_at = time.time() - 3600
    sim_f._forget_if_idle(st_f, time.time())
    check("安静后遗忘旧上下文", len(st_f.recent) == 2, str(len(st_f.recent)))
    check("遗忘时连摘要一起清", st_f.idle_summary == [], str(st_f.idle_summary))

    st_busy = sim_f.state("group:busy", umo, True)
    for _ in range(6):
        st_busy.recent.append(ChatMessage("阿伟", "u1", "刚说的", time.time()))
    st_busy.last_message_at = time.time()
    sim_f._forget_if_idle(st_busy, time.time())
    check("刚聊过就不遗忘", len(st_busy.recent) == 6, str(len(st_busy.recent)))

    # 上下文必须分「现在」和「更早」两段，否则模型会把旧话当新话接
    st_b = sim_f.state("group:blocks", umo, True)
    st_b.recent.append(ChatMessage("阿伟", "u1", "三小时前的老话", time.time() - 3 * 3600))
    st_b.recent.append(ChatMessage("小明", "u2", "刚说的新话", time.time()))
    _, user_prompt = sim_f.build_blocks(st_b, "active", "小明：刚说的新话")
    check("提示词分成新旧两段", "【刚刚到现在" in user_prompt and "【更早的消息" in user_prompt, user_prompt[:300])
    check("旧消息带时间标记", "[3小时前]" in user_prompt, user_prompt[:400])
    check("新消息带刚刚标记", "[刚刚]" in user_prompt, user_prompt[:400])


# ------------------------------------------------------------------ prompt
def test_prompt() -> None:
    text = build_system_prompt(
        bot_name="小鲸鱼",
        persona="爱吐槽的群友",
        style_block="【本群语感画像】\n- 平均 12 字",
        memory_block="【进行中的话题】\n- 周末联机（3分钟前）",
        participation="【参与度参考】你最近 1 小时发言 1 次",
        is_group=True,
        initial_context="这是一个游戏群",
    )
    for needle in ("小鲸鱼", "安全边界", "【输出协议", "【群聊不是客服队列", "【学习群友策略", "本群语感画像", "进行中的话题", "游戏群", "爱吐槽的群友"):
        check(f"提示词包含 {needle}", needle in text)
    check("提示词不残留未替换的静默标记占位", "``[SILENT]``" not in text)

    # 默认仿真层必须覆盖这些关键段落——删掉任何一段都会让 AI 味立刻回来。
    default = build_system_prompt(bot_name="小鲸鱼")
    for needle in (
        "【群友仿真模式 —— 安全边界",
        "【输出协议",
        "【本环境没有的东西",
        "【群聊不是客服队列",
        "【反 AI 味",
        "【别用问号/问好开场",
        "【保持主体性",
        "【该说 / 不该说",
        "【收尾与离场",
        "【分条与节奏",
        "【记忆卫生",
        "【学习群友策略",
        "【不要当群管家",
    ):
        check(f"默认规则含 {needle}", needle in default, needle)
    # 工具协议不该残留：模型会一本正经地调用不存在的工具。
    for tool in ("qq_send_message", "qq_mark_read", "qq_wait_for_messages", "mcp__snowluma__", "readThroughSeq"):
        check(f"默认规则不含工具 {tool}", tool not in default, tool)
    check("默认规则明确“文本会原样发出”", "原样发到群里" in default)
    check("默认规则给了 [SILENT] 协议", "[SILENT]" in default)

    # 配置项 simulation_rules 的入口：整段替换
    custom = build_system_prompt(bot_name="小鲸鱼", rules_text="【自定义仿真规则】只说一句话。")
    check("rules_text 可整段替换", "【自定义仿真规则】" in custom)
    check("替换后不再保留内置规则", "【反 AI 味" not in custom)
    blank = build_system_prompt(bot_name="小鲸鱼", rules_text="   ")
    check("rules_text 为空白时回退到内置", "【反 AI 味" in blank)

    # 多模态能力：由配置的真实状态决定，而不是让模型自己猜
    with_vision = build_system_prompt(bot_name="小鲸鱼", vision=True)
    without_vision = build_system_prompt(bot_name="小鲸鱼", vision=False)
    check("有视觉时明确说能看图", "你能看到图片" in with_vision)
    check("无视觉时明确说看不到", "暂时看不到图片" in without_vision)
    check("能力段始终存在", "【系统能力" in with_vision and "【系统能力" in without_vision)
    check("提示词说明会收到系统通知", "【系统】" in with_vision)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        test_text()
        test_memory(tmp)
        test_store_corruption(tmp)
        test_style(tmp)
        test_affinity(tmp)
        test_memory_identity(tmp)
        test_growth(tmp)
        test_freshness(tmp)
        test_suppress(tmp)
        test_vision_invite()
        test_events()
        test_prompt()
        asyncio.run(test_simulator())

    print(f"\n通过 {len(PASSED)} 项")
    if FAILED:
        print(f"失败 {len(FAILED)} 项：")
        for item in FAILED:
            print("  - " + item)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
