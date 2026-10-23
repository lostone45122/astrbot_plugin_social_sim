"""真环境加载测试：用**真正的** AstrBot 包加载本插件。

和 ``test_plugin_wiring.py`` 的区别：那个用桩件、不需要装 AstrBot；这个必须真的装上
AstrBot，因此能验证桩件验证不了的东西——

- 真实的 ``filter.command`` / ``filter.event_message_type`` / ``filter.on_llm_request``
  装饰器能否注册成功（包括 ``CommandFilter.init_handler_md`` 对注解的解析）；
- 真实的 ``Star`` 初始化（``StarTools.get_data_dir``、插件日志器解析）；
- 真实的 ``StarHandlerMetadata`` 是否出现在 ``star_handlers_registry`` 里。

没装 AstrBot 时会打印 SKIP 并以 0 退出，所以在任何机器上跑都不会“误报失败”。

运行：

    python tests/test_real_astrbot.py
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_NAME = "astrbot_plugin_social_sim"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name if cond else f"{name}: {detail}")


def main() -> int:
    try:
        import astrbot  # noqa: F401
        from astrbot.core.star.star_handler import star_handlers_registry
        from astrbot.core.star.star import star_map
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: 当前环境没有可用的 AstrBot（{exc.__class__.__name__}: {exc}）")
        return 0

    root = Path(tempfile.mkdtemp(prefix="ab_real_load_"))
    plugins_dir = root / "data" / "plugins"
    plugins_dir.mkdir(parents=True)
    target = plugins_dir / PLUGIN_NAME
    shutil.copytree(
        PLUGIN_DIR,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"),
    )
    sys.path.insert(0, str(root))

    module_name = f"data.plugins.{PLUGIN_NAME}.main"
    try:
        module = importlib.import_module(module_name)
        check("真实 AstrBot 下可导入 main.py", True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"\n失败：插件在真实 AstrBot 环境下无法导入")
        return 1

    plugin_cls = getattr(module, "SocialSimPlugin", None)
    check("导出 SocialSimPlugin", plugin_cls is not None)
    if plugin_cls is None:
        return 1

    from astrbot.core.star.base import Star

    check("插件类继承真实 Star", issubclass(plugin_cls, Star))

    handlers = [
        h
        for h in star_handlers_registry
        if (h.handler_module_path or "").startswith(module_name.rpartition(".")[0])
    ]
    names = {h.handler_name for h in handlers}
    for expected in (
        "on_all_message",
        "on_llm_request",
        "cmd_sim",
        "cmd_memory",
        "cmd_style",
        "cmd_persona",
        "cmd_env",
        "cmd_reset",
        "cmd_affinity",
        "cmd_suppress",
        "on_decorating_result",
    ):
        check(f"真实注册表里有 {expected}", expected in names, str(sorted(names)))

    check("至少注册了 11 个 handler", len(handlers) >= 11, str(len(handlers)))

    # 指令过滤器真实解析过注解；能拿到 handler_params 就说明这一步没崩。
    cmd_handler = next((h for h in handlers if h.handler_name == "cmd_sim"), None)
    check("cmd_sim 已注册", cmd_handler is not None)
    if cmd_handler is not None:
        cmd_filters = [f for f in cmd_handler.event_filters if type(f).__name__ == "CommandFilter"]
        check("cmd_sim 挂了 CommandFilter", bool(cmd_filters), str(cmd_handler.event_filters))
        if cmd_filters:
            check(
                "指令没有多余必填参数（否则用户得带参数才能触发）",
                not cmd_filters[0].handler_params,
                str(cmd_filters[0].handler_params),
            )
            check(
                "指令名解析正确",
                cmd_filters[0].get_complete_command_names() == ["仿真"],
                str(cmd_filters[0].get_complete_command_names()),
            )

    # 真实实例化：会走到 StarTools.get_data_dir() 与插件日志器解析。
    class _FakeContext:
        def get_config(self):
            return {"wake_prefix": ["/"]}

        async def get_using_provider_async(self, umo=None, **_):
            return None

        async def send_message(self, session, chain):
            return True

    try:
        plugin = plugin_cls(context=_FakeContext(), config={"enabled": True})
        check("真实 Star 下可实例化", True)
        check("拿到了数据目录", plugin._data_dir.exists(), str(plugin._data_dir))
        check("配置被读取", plugin.bot_name() == "群友", plugin.bot_name())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check("真实 Star 下可实例化", False, "见上方 traceback")

    check("star_map 里登记了模块", module_name in star_map, str(list(star_map)[:5]))

    shutil.rmtree(root, ignore_errors=True)

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
