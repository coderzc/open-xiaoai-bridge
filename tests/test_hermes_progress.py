import unittest

from core.hermes_progress import (
    HERMES_TOOL_VOICE_STATUS,
    HermesProgressNarrator,
    HermesToolProgress,
)


EXPECTED_TOOL_VOICE_STATUS = {
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


class HermesProgressNarratorTest(unittest.TestCase):
    def test_all_curated_builtin_tools_have_exact_voice_mapping(self):
        self.assertEqual(
            EXPECTED_TOOL_VOICE_STATUS,
            HERMES_TOOL_VOICE_STATUS,
        )

    def test_web_search_uses_curated_status(self):
        narrator = HermesProgressNarrator("任意问题")
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="running",
                tool_call_id="call-1",
            )
        )
        self.assertEqual(
            ("正在查找相关资料", "tool:web_search"),
            narrator.next_message(),
        )

    def test_label_and_machine_arguments_are_never_spoken(self):
        dangerous_labels = {
            "execute_code": (
                "import pandas as pd\n"
                "file_path='/private/foo.xlsx'"
            ),
            "terminal": "rm -rf /something",
        }
        for tool, label in dangerous_labels.items():
            with self.subTest(tool=tool):
                narrator = HermesProgressNarrator("任意问题")
                narrator.observe(
                    HermesToolProgress(
                        tool=tool,
                        status="running",
                        tool_call_id=f"call-{tool}",
                        label=label,
                    )
                )
                message, _key = narrator.next_message()
                self.assertEqual(
                    HERMES_TOOL_VOICE_STATUS[tool],
                    message,
                )
                for unsafe in (
                    "pandas",
                    "/private/foo.xlsx",
                    "rm -rf",
                    "/something",
                ):
                    self.assertNotIn(unsafe, message)

    def test_unknown_plugin_tool_uses_safe_generic_status(self):
        narrator = HermesProgressNarrator("播放我的私人歌单")
        narrator.observe(
            HermesToolProgress(
                tool="spotify_private_mcp_search",
                status="running",
                tool_call_id="call-1",
                label="search user private playlist",
            )
        )
        message, key = narrator.next_message()
        self.assertEqual("正在处理，请稍等", message)
        self.assertEqual("tool:unknown", key)
        self.assertNotIn("spotify", message.lower())
        self.assertNotIn("playlist", message.lower())

    def test_no_tool_event_uses_long_wait_generic_status(self):
        narrator = HermesProgressNarrator("任意问题")
        self.assertEqual(
            ("还在为你处理，请稍等", "working"),
            narrator.next_message(),
        )
        self.assertIsNone(narrator.next_message())

    def test_user_query_business_terms_do_not_select_status(self):
        user_text = (
            "天气 飞书 Spotify 网易云 显卡价格 Home Assistant"
        )
        narrator = HermesProgressNarrator(user_text)
        self.assertEqual(
            ("还在为你处理，请稍等", "working"),
            narrator.next_message(),
        )

    def test_running_then_completed_uses_generic_completion(self):
        narrator = HermesProgressNarrator("任意问题")
        running = HermesToolProgress(
            tool="write_file",
            status="running",
            tool_call_id="call-1",
        )
        completed = HermesToolProgress(
            tool="write_file",
            status="completed",
            tool_call_id="call-1",
        )
        narrator.observe(running)
        self.assertEqual(
            ("正在处理文件", "tool:write_file"),
            narrator.next_message(),
        )
        narrator.observe(completed)
        self.assertEqual(
            ("这一步已经完成，正在继续处理。", "organizing"),
            narrator.next_message(),
        )
        self.assertIsNone(narrator.next_message())

    def test_running_then_failed_never_emits_success_summary(self):
        narrator = HermesProgressNarrator("任意问题")
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="running",
                tool_call_id="call-1",
            )
        )
        narrator.next_message()
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="failed",
                tool_call_id="call-1",
            )
        )
        self.assertIsNone(narrator.next_message())

    def test_completed_tool_can_summarize_when_another_tool_failed(self):
        narrator = HermesProgressNarrator("任意问题")
        for call_id in ("failed-call", "completed-call"):
            narrator.observe(
                HermesToolProgress(
                    tool="web_search",
                    status="running",
                    tool_call_id=call_id,
                )
            )
        narrator.next_message()
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="failed",
                tool_call_id="failed-call",
            )
        )
        self.assertIsNone(narrator.next_message())
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="completed",
                tool_call_id="completed-call",
            )
        )
        self.assertEqual(
            ("这一步已经完成，正在继续处理。", "organizing"),
            narrator.next_message(),
        )

    def test_repeated_same_tool_is_semantically_deduplicated(self):
        narrator = HermesProgressNarrator("任意问题")
        for call_id in ("call-1", "call-2"):
            narrator.observe(
                HermesToolProgress(
                    tool="web_search",
                    status="running",
                    tool_call_id=call_id,
                )
            )
        self.assertEqual(
            ("正在查找相关资料", "tool:web_search"),
            narrator.next_message(),
        )
        self.assertIsNone(narrator.next_message())

    def test_different_builtin_tools_can_emit_different_statuses(self):
        narrator = HermesProgressNarrator("任意问题")
        narrator.observe(
            HermesToolProgress(
                tool="web_search",
                status="running",
                tool_call_id="call-1",
            )
        )
        self.assertEqual(
            ("正在查找相关资料", "tool:web_search"),
            narrator.next_message(),
        )
        narrator.observe(
            HermesToolProgress(
                tool="web_extract",
                status="running",
                tool_call_id="call-2",
            )
        )
        self.assertEqual(
            ("正在读取相关内容", "tool:web_extract"),
            narrator.next_message(),
        )

    def test_raw_label_never_affects_unknown_tool_tts(self):
        label = (
            "https://private.example/path\n"
            "/private/user/file.json\n"
            '{"query":"中文私人搜索词"}'
        )
        narrator = HermesProgressNarrator("任意问题")
        narrator.observe(
            HermesToolProgress(
                tool="private_mcp_tool",
                status="running",
                tool_call_id="call-1",
                label=label,
            )
        )
        message, _key = narrator.next_message()
        self.assertEqual("正在处理，请稍等", message)
        for unsafe in (
            "private.example",
            "/private/user/file.json",
            "query",
            "中文私人搜索词",
            "private_mcp_tool",
        ):
            self.assertNotIn(unsafe, message)

    def test_technical_log_redacts_urls_and_secrets(self):
        event = HermesToolProgress(
            tool="web_search",
            status="running",
            tool_call_id="call-1",
            label=(
                "fetch https://private.example/path "
                "Authorization: Bearer-private"
            ),
        )
        label = event.technical_log()["label"]
        self.assertNotIn("private.example", label)
        self.assertNotIn("Bearer-private", label)
        self.assertIn("[URL]", label)
        self.assertIn("[REDACTED]", label)


if __name__ == "__main__":
    unittest.main()
