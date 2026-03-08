from typing import Literal

from core.ref import get_xiaoai, set_speaker
from core.utils.base import json_decode, json_encode


class CommandResult:
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class SpeakerManager:
    status: Literal["playing", "paused", "idle"] = "idle"

    def __init__(self):
        set_speaker(self)

    async def get_playing(self, sync=False):
        """获取播放状态"""
        if sync:
            # 同步远端最新状态
            res = await self.run_shell("mphelper mute_stat")
            if "1" in res.stdout:
                self.status = "playing"
            elif "2" in res.stdout:
                self.status = "paused"
        return self.status

    async def set_playing(self, playing=True):
        """播放/暂停"""
        command = "mphelper play" if playing else "mphelper pause"
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def play(
        self,
        text=None,
        url=None,
        buffer=None,
        blocking=True,
        timeout=10 * 60 * 1000,
    ):
        """
        播放文字、音频链接、音频流

        参数:
            text: 文字内容
            url: 音频链接
            buffer: 音频流
            timeout: 超时时长（毫秒），默认10分钟
            blocking: 是否阻塞运行(仅对播放文字、音频链接有效)
        """
        if buffer is not None:
            return get_xiaoai().on_output_data(buffer)

        if blocking:
            command = (
                f"miplayer -f '{url}'"
                if url
                else f"/usr/sbin/tts_play.sh '{text.replace("'", "'\\''") or '你好'}'"
            )
            res = await self.run_shell(command, timeout=timeout)
            return res.exit_code == 0

        if url:
            data = json_encode({"url": url, "type": 1})
            command = f"ubus call mediaplayer player_play_url '{data}'"
        else:
            data = json_encode({"text": text or "你好", "save": 0})
            command = f"ubus call mibrain text_to_speech '{data}'"

        res = await self.run_shell(command, timeout=timeout)
        return '"code": 0' in res.stdout if res else False

    async def wake_up(self, awake=True, silent=True):
        """
        （取消）唤醒小爱

        参数:
            awake: 是否唤醒
            silent: 是否静默唤醒
        """

        if awake:
            if silent:
                command = 'ubus call pnshelper event_notify \'{"src":1,"event":0}\''
            else:
                command = 'ubus call pnshelper event_notify \'{"src":0,"event":0}\''
        else:
            command = """
                ubus call pnshelper event_notify '{"src":3, "event":7}'
                sleep 0.1
                ubus call pnshelper event_notify '{"src":3, "event":8}'
            """
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def ask_xiaoai(self, text: str, silent=False):
        """
        把文字指令交给原来的小爱执行

        参数:
            text: 文字指令
            silent: 是否静默执行
        """

        data = {"nlp": 1, "nlp_text": text}
        if not silent:
            data["tts"] = 1

        command = f"ubus call mibrain ai_service '{json_encode(data)}'"
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def abort_xiaoai(self):
        """
        中断原来小爱的运行

        注意：重启需要大约 1-2s 的时间，在此期间无法使用小爱音箱自带的 TTS 服务
        """
        res = await self.run_shell("/etc/init.d/mico_aivs_lab restart >/dev/null 2>&1")
        return res.exit_code == 0

    async def get_boot(self):
        """获取启动分区"""
        res = await self.run_shell("echo $(fw_env -g boot_part)")
        return res.stdout.strip()

    async def set_boot(self, boot_part: Literal["boot0", "boot1"]):
        """设置启动分区"""
        command = f"fw_env -s boot_part {boot_part} >/dev/null 2>&1 && echo $(fw_env -g boot_part)"
        res = await self.run_shell(command)
        return boot_part in res.stdout

    async def get_device(self):
        """获取设备型号、序列号信息"""
        res = await self.run_shell("echo $(micocfg_model) $(micocfg_sn)")
        info = res.stdout.strip().split(" ")
        return {
            "model": info[0] if len(info) > 0 else "unknown",
            "sn": info[1] if len(info) > 1 else "unknown",
        }

    async def get_mic(self):
        """获取麦克风状态"""
        res = await self.run_shell("[ ! -f /tmp/mipns/mute ] && echo on || echo off")
        status = "off"
        if "on" in res.stdout:
            status = "on"
        return status

    async def set_mic(self, on=True):
        """打开/关闭麦克风"""
        if on:
            command = (
                'ubus -t1 -S call pnshelper event_notify \'{"src":3, "event":7}\' 2>&1'
            )
        else:
            command = (
                'ubus -t1 -S call pnshelper event_notify \'{"src":3, "event":8}\' 2>&1'
            )
        res = await self.run_shell(command)
        return '"code":0' in res.stdout

    async def run_shell(self, script: str, timeout=10000):
        """
        执行脚本

        参数:
            script: 脚本内容
            timeout: 超时时间（毫秒）
        """
        res = "unknown"
        try:
            res = await get_xiaoai().run_shell(script, timeout=timeout)
            data = json_decode(res)
            if data:
                return CommandResult(
                    data.get("stdout", ""),
                    data.get("stderr", ""),
                    data.get("exit_code", 0),
                )
        except Exception:
            return CommandResult("error", res, -1)
