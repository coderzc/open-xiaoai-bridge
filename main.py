import argparse
import os
import signal
import sys
import time

# Fix: Add onnxruntime library path for sherpa_onnx on macOS/Linux
# This ensures sherpa_onnx can find libonnxruntime at runtime
if sys.platform in ("darwin", "linux"):
    try:
        import onnxruntime as ort
        ort_lib_dir = os.path.join(os.path.dirname(ort.__file__), "capi")
        if os.path.exists(ort_lib_dir):
            if sys.platform == "darwin":
                os.environ.setdefault("DYLD_LIBRARY_PATH", "")
                if ort_lib_dir not in os.environ["DYLD_LIBRARY_PATH"]:
                    os.environ["DYLD_LIBRARY_PATH"] = ort_lib_dir + ":" + os.environ["DYLD_LIBRARY_PATH"]
            else:  # linux
                os.environ.setdefault("LD_LIBRARY_PATH", "")
                if ort_lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                    os.environ["LD_LIBRARY_PATH"] = ort_lib_dir + ":" + os.environ["LD_LIBRARY_PATH"]
    except ImportError:
        pass

from core.utils.config_loader import ensure_config_module_loaded

config_path = ensure_config_module_loaded()

from core.app import MainApp
from core.utils.config import ConfigManager
from core.utils.logger import logger


main_app_instance = None

# 启动配置（从环境变量读取）
enable_api_server = False  # 是否开启 API Server
enable_openclaw = False  # 是否连接 OpenClaw


def setup_config():
    """解析命令行参数和环境变量"""
    global enable_api_server, enable_openclaw

    parser = argparse.ArgumentParser(description="小爱音箱接入 Open XiaoAI")
    parser.parse_args()

    # 从环境变量读取配置
    enable_api_server = os.environ.get("API_SERVER_ENABLE", "").lower() in ("1", "true", "yes")
    # 兼容 OPENCLAW_ENABLE (新) 和 OPENCLAW_ENABLED (旧)
    openclaw_env = os.environ.get("OPENCLAW_ENABLE") or os.environ.get("OPENCLAW_ENABLED") or ""
    enable_openclaw = openclaw_env.lower() in ("1", "true", "yes")

    # Home Assistant 没有独立的环境变量开关，完全由 config.py 中的
    # homeassistant.enabled 控制（此处仅用于日志展示）。
    homeassistant_enabled = bool(
        ConfigManager.instance().get_app_config("homeassistant.enabled", False)
    )

    logger.info(
        f"[Main] ENV: API_SERVER_ENABLE={os.environ.get('API_SERVER_ENABLE') or 'not set (disabled)'}, "
        f"OPENCLAW_ENABLE={os.environ.get('OPENCLAW_ENABLE') or os.environ.get('OPENCLAW_ENABLED') or 'not set (disabled)'}"
    )
    logger.info(f"[Main] Using config file: {config_path}")

    # 打印模块启用情况
    logger.info("[Main] 模块启用情况:")
    logger.info("小爱指令拦截器启用", module="Main")
    logger.info(
        f"OpenClaw Bridge: {'启用' if enable_openclaw else '禁用'}",
        module="Main",
    )
    logger.info(
        f"Home Assistant Bridge (config.py 控制): {'启用' if homeassistant_enabled else '禁用'}",
        module="Main",
    )
    logger.info(
        f"API Server: {'启用' if enable_api_server else '禁用'}",
        module="Main",
    )


def run_services():
    """统一的服务启动入口"""
    global main_app_instance, enable_api_server, enable_openclaw

    # 统一使用 MainApp 管理所有服务
    main_app_instance = MainApp.instance(
        enable_openclaw=enable_openclaw,
    )
    main_app_instance.run(enable_api_server=enable_api_server)

    # 主线程保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    run_services()
    return 0


def setup_graceful_shutdown():
    def signal_handler(_sig, _frame):
        global main_app_instance

        # 关闭 MainApp（包含 API Server）
        if main_app_instance:
            main_app_instance.shutdown()

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_config()
    setup_graceful_shutdown()
    sys.exit(main())
