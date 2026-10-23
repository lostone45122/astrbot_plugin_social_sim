from __future__ import annotations

"""群友仿真插件的内部模块。

- ``text``      文本清洗 / 分条 / “话没说完”判定 / JSON 抽取
- ``store``     原子写 + 损坏隔离的 JSON 持久化
- ``memory``    三类记忆（话题 / 念头 / 群友印象）
- ``style``     群风统计（长度、连发率、口癖、活跃时段）
- ``prompts``   仿真提示词与上下文拼装
- ``simulator`` 观望 / 活跃 / 试探 / 退场 状态机
"""

__all__ = ["memory", "prompts", "simulator", "store", "style", "text"]
