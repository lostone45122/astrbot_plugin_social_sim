"""JSON 持久化底座。

qq-bridge 的一个关键教训（`src/slang-learner.js` 的注释里写得很明白）：
**“文件不存在”和“读失败”必须分开处理**。旧实现一律 ``catch { return [] }``，
于是一次杀软占用或一次半截写入就让整库看起来是空的，紧接着的保存把空内容写回去，
数据永久消失。

这里沿用同一套语义：
- ``FileNotFoundError`` → 返回默认值（首次运行）。
- 其它读取/解析失败 → 把原文另存为 ``.corrupt-<时间戳>`` 后返回默认值，
  并抛给调用方一个可记录的告警（不静默）。
- 写入走「临时文件 + os.replace」，崩溃不会留下半截 JSON。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["JsonStore", "deep_merge", "now_ms"]


def now_ms() -> int:
    """当前毫秒时间戳（qq-bridge 的状态文件统一用毫秒）。"""
    return int(time.time() * 1000)


def deep_merge(base: dict, patch: dict) -> dict:
    """把 patch 深合并进 base 的副本（只对 dict 递归）。"""
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class JsonStore:
    """一个线程/协程内安全使用的单文件 JSON 存储。

    AstrBot 的插件回调都在同一个事件循环里，读写之间不会真正并行，
    因此这里只用「临时文件 + 原子替换」保证崩溃安全，不额外加锁。
    """

    def __init__(
        self,
        path: Path,
        default_factory: Callable[[], Any],
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._default_factory = default_factory
        self._on_warning = on_warning
        self.read_only = False
        """读失败时置为 True：此时允许继续以空数据运行，但拒绝写回。"""

    # ---------------------------------------------------------------- 内部
    def _warn(self, message: str) -> None:
        if self._on_warning is not None:
            try:
                self._on_warning(message)
            except Exception:  # noqa: BLE001 - 告警失败绝不能影响主流程
                pass

    def _quarantine(self, text: str, reason: str) -> None:
        """把读不懂的原文另存一份，绝不覆盖丢弃。"""
        quarantine = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time() * 1000)}")
        try:
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            quarantine.write_text(text, encoding="utf-8")
            self._warn(f"{reason}，原文已另存为 {quarantine.name}")
        except OSError:
            self._warn(f"{reason}，且原文备份失败（文件可能被占用）")

    # ---------------------------------------------------------------- 读写
    def load(self) -> Any:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._default_factory()
        except OSError as exc:
            # 文件存在但读不出来：绝不能当成“空库”，也不能写回覆盖它。
            # 标记为只读，之后所有 save() 都是空操作，原文件保持原样等人工处理。
            self.read_only = True
            self._warn(
                f"读取 {self.path.name} 失败（{exc.__class__.__name__}）：{exc}；"
                "本次以空数据继续，并已禁止写回以免覆盖原文件"
            )
            return self._default_factory()

        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
        if not text.strip():
            return self._default_factory()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            self._quarantine(text, f"{self.path.name} JSON 解析失败：{exc}")
            return self._default_factory()
        return data

    def save(self, data: Any) -> bool:
        if self.read_only:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            with open(tmp, "w", encoding="utf-8", newline="\n") as fp:
                fp.write(payload)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp, self.path)
            return True
        except OSError as exc:
            self._warn(f"写入 {self.path.name} 失败：{exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def backup(self) -> Path | None:
        """把当前文件复制一份带时间戳的备份（手动清空前用）。"""
        if not self.path.exists():
            return None
        target = self.path.with_name(f"{self.path.name}.bak-{int(time.time() * 1000)}")
        try:
            shutil.copy2(self.path, target)
            return target
        except OSError:
            return None
