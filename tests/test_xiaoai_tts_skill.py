import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "xiaoai-tts"


def _load_script(name):
    scripts_dir = SKILL_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class XiaoAITTSSkillTest(unittest.TestCase):
    def test_skill_declares_hermes_path_and_required_bridge_url(self):
        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("${HERMES_SKILL_DIR}/tools/xiaoai-tts", content)
        self.assertIn("required_environment_variables:", content)
        self.assertIn("OPENXIAOAI_BASE_URL", content)
        self.assertIn("RESULT success=true completed=true", content)
        self.assertIn("--blocking", content)

    def test_native_text_api_failure_returns_nonzero(self):
        module = _load_script("play_text")
        stderr = io.StringIO()
        with (
            mock.patch.object(module, "api_request", return_value={"success": False}),
            mock.patch.object(sys, "argv", ["play_text.py", "测试"]),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            module.main()

        self.assertEqual(1, raised.exception.code)
        self.assertIn("RESULT success=false", stderr.getvalue())

    def test_doubao_api_failure_returns_nonzero(self):
        module = _load_script("tts_doubao")
        stderr = io.StringIO()
        with (
            mock.patch.object(module, "api_request", return_value={"success": False}),
            mock.patch.object(sys, "argv", ["tts_doubao.py", "测试"]),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            module.main()

        self.assertEqual(1, raised.exception.code)
        self.assertIn("RESULT success=false", stderr.getvalue())

    def test_blocking_native_text_reports_completed(self):
        module = _load_script("play_text")
        stdout = io.StringIO()
        with (
            mock.patch.object(module, "api_request", return_value={"success": True}),
            mock.patch.object(
                sys,
                "argv",
                ["play_text.py", "测试", "--blocking"],
            ),
            mock.patch("sys.stdout", stdout),
        ):
            module.main()

        self.assertIn(
            "RESULT success=true completed=true",
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
