import threading
from pathlib import Path
from typing import Any, Callable

from core.utils.config_loader import (
    ensure_config_module_loaded,
    get_config_path,
    load_config_module,
)


class ConfigManager:
    """配置管理器 - 单例模式

    负责加载 config.py 中的 APP_CONFIG，并支持热重载与变更通知。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._state_lock = threading.RLock()
        self._reload_listeners: list[Callable[[dict[str, Any], dict[str, Any]], None]] = []
        self._config_path = get_config_path()
        self._config_module_name = "config"

        ensure_config_module_loaded()
        self._app_config = self._load_app_config()

    def _load_app_config(self) -> dict[str, Any]:
        """加载 config.py 中的 APP_CONFIG。"""
        module = ensure_config_module_loaded()
        app_config = getattr(module, "APP_CONFIG", None)
        if not isinstance(app_config, dict):
            raise ValueError("config.APP_CONFIG must be a dict")
        return app_config

    def get_config_path(self) -> Path:
        """返回配置文件路径。"""
        return self._config_path

    def get_app_config(self, path: str | None = None, default: Any = None) -> Any:
        """获取运行时 APP_CONFIG。"""
        with self._state_lock:
            if not path:
                return self._app_config

            value: Any = self._app_config
            for key in path.split("."):
                if not isinstance(value, dict):
                    return default
                value = value.get(key, default)
                if value is default:
                    return default
            return value

    def add_reload_listener(
        self, callback: Callable[[dict[str, Any], dict[str, Any]], None]
    ) -> None:
        """注册配置重载监听器。"""
        with self._state_lock:
            if callback not in self._reload_listeners:
                self._reload_listeners.append(callback)

    def reload_app_config(self) -> bool:
        """重新加载 config.py，并同步运行时配置。"""
        with self._state_lock:
            module = load_config_module(force_reload=True)
            next_config = getattr(module, "APP_CONFIG", None)
            if not isinstance(next_config, dict):
                raise ValueError("config.APP_CONFIG must be a dict")

            previous_config = self._app_config
            self._app_config = next_config

            listeners = list(self._reload_listeners)

        for listener in listeners:
            try:
                listener(previous_config, next_config)
            except Exception:
                continue

        return True

    @classmethod
    def instance(cls):
        """获取配置管理器实例（线程安全）"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
