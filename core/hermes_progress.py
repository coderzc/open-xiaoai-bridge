"""将 Hermes 工具生命周期事件安全转换为可播报进度。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization|secret|password)"
    r"\b\s*[:=]\s*\S+"
)


@dataclass(frozen=True)
class HermesToolProgress:
    """Hermes Chat Completions 发出的结构化工具生命周期事件。"""

    tool: str
    status: str
    tool_call_id: str
    label: str = ""
    emoji: str = ""

    @classmethod
    def from_json(cls, data: str) -> "HermesToolProgress | None":
        body = json.loads(data)
        if not isinstance(body, dict):
            return None
        tool = str(body.get("tool") or "").strip()
        status = str(body.get("status") or "").strip().lower()
        tool_call_id = str(body.get("toolCallId") or "").strip()
        if not tool or status not in {
            "running",
            "completed",
            "failed",
        }:
            return None
        return cls(
            tool=tool,
            status=status,
            tool_call_id=tool_call_id,
            label=str(body.get("label") or "").strip(),
            emoji=str(body.get("emoji") or "").strip(),
        )

    def technical_log(self) -> dict[str, str]:
        """返回已脱敏的生命周期字段，供技术日志使用。"""

        return {
            "tool": self.tool,
            "status": self.status,
            "tool_call_id": self.tool_call_id,
            "label": self._safe_log_label(self.label),
            "emoji": self.emoji,
        }

    @staticmethod
    def _safe_log_label(label: str) -> str:
        redacted = _URL_RE.sub("[URL]", str(label or ""))
        redacted = _SECRET_RE.sub(
            r"\1=[REDACTED]",
            redacted,
        )
        return redacted[:500]


# 与 Hermes agent/display.py 中 curated built-in _TOOL_VERBS 对应。
# 这里只提供适合 TTS 的中文短句；custom/plugin/MCP 一律使用通用提示，
# 并且绝不读取 argument preview 或 label，避免朗读机器参数和敏感内容。
HERMES_TOOL_VOICE_STATUS = {
    "web_search": "正在查找相关资料",
    "web_extract": "正在读取相关内容",
    "browser_navigate": "正在浏览相关页面",
    "browser_click": "正在处理网页操作",
    "browser_type": "正在处理网页内容",
    "read_file": "正在读取文件",
    "write_file": "正在处理文件",
    "patch": "正在修改内容",
    "search_files": "正在查找文件",
    "terminal": "正在执行相关操作",
    "execute_code": "正在运行程序",
    "image_generate": "正在生成图片",
    "video_generate": "正在生成视频",
    "text_to_speech": "正在生成语音",
    "vision_analyze": "正在查看图片",
    "session_search": "正在回顾之前的内容",
    "skill_view": "正在查看相关技能",
    "skills_list": "正在查找可用技能",
    "skill_manage": "正在更新相关技能",
    "delegate_task": "正在处理子任务",
    "cronjob": "正在安排任务",
    "clarify": "正在确认你的需求",
    "memory": "正在整理相关记忆",
    "todo": "正在处理任务列表",
}

_NO_TOOL_PROGRESS = "还在为你处理，请稍等"
_UNKNOWN_TOOL_PROGRESS = "正在处理，请稍等"
_COMPLETED_PROGRESS = "这一步已经完成，正在继续处理。"


class HermesProgressNarrator:
    """将真实 Hermes 工具事件转换为 voice-safe 进度提示。"""

    def __init__(self, user_text: str):
        # 保留现有调用签名，但用户问题不再参与 progress 语义选择。
        del user_text
        self._latest_status: tuple[str, str] | None = None
        self._received_tool_progress = False
        self._active_calls: dict[str, str] = {}
        self._announced: set[str] = set()
        self._completed_since_announcement = False
        self._organizing_announced = False

    def observe(self, event: HermesToolProgress):
        self._received_tool_progress = True
        tool = str(event.tool or "").strip().lower()
        message = HERMES_TOOL_VOICE_STATUS.get(tool)
        status_key = f"tool:{tool}" if message else "tool:unknown"
        call_key = event.tool_call_id or (
            f"{event.tool}:{event.status}"
        )
        if event.status == "running":
            self._latest_status = (
                message or _UNKNOWN_TOOL_PROGRESS,
                status_key,
            )
            self._active_calls[call_key] = status_key
        elif event.status == "completed":
            completed_key = self._active_calls.pop(call_key, None)
            if completed_key is not None:
                self._completed_since_announcement = True
                if (
                    self._latest_status
                    and self._latest_status[1] == completed_key
                ):
                    self._latest_status = None
        elif event.status == "failed":
            # 失败只结束对应调用，绝不能进入成功汇总路径。
            # Hermes 仍可继续执行其他工具并自行生成最终回答。
            failed_key = self._active_calls.pop(call_key, None)
            if (
                failed_key is not None
                and self._latest_status
                and self._latest_status[1] == failed_key
            ):
                self._latest_status = None

    def next_message(self) -> tuple[str, str] | None:
        """返回安全提示和用于语义去重的键。"""

        if (
            self._completed_since_announcement
            and self._announced
            and not self._organizing_announced
        ):
            self._completed_since_announcement = False
            self._organizing_announced = True
            return _COMPLETED_PROGRESS, "organizing"

        if self._latest_status:
            message, key = self._latest_status
            if key not in self._announced:
                self._announced.add(key)
                return message, key

        if not self._announced:
            key = (
                "tool:unknown"
                if self._received_tool_progress
                else "working"
            )
            self._announced.add(key)
            return (
                _UNKNOWN_TOOL_PROGRESS
                if self._received_tool_progress
                else _NO_TOOL_PROGRESS,
                key,
            )
        return None
