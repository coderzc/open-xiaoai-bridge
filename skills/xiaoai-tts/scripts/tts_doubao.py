#!/usr/bin/env python3
"""
火山 TTS 语音合成脚本
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import api_request


# 默认音色（2.0 高质量）
DEFAULT_SPEAKER = None


def tts_doubao(text, speaker=None, speed=1.0, emotion=None, 
               context_texts=None, app_id=None, access_key=None, 
               resource_id=None, blocking=False):
    """
    使用火山 TTS 播放文字
    
    Args:
        text: 要合成的文本
        speaker: 音色 ID（默认 zh_female_vv_uranus_bigtts）
        speed: 语速 0.8-2.0（默认 1.0）
        emotion: 情感参数（仅多情感音色支持）
        context_texts: 上下文指令（仅 2.0 音色支持）
        app_id: 火山 App ID（可选，默认从 config 读取）
        access_key: 火山 Access Key（可选，默认从 config 读取）
        resource_id: 资源 ID（可选，自动检测）
        blocking: 是否阻塞等待（默认 False，文档错误更正）
    """
    data = {
        "text": text,
        "speaker_id": speaker or DEFAULT_SPEAKER,
        "speed": speed,
        "blocking": blocking
    }
    
    # 可选参数
    if emotion:
        data["emotion"] = emotion
    if context_texts:
        data["context_texts"] = context_texts if isinstance(context_texts, list) else [context_texts]
    if app_id:
        data["app_id"] = app_id
    if access_key:
        data["access_key"] = access_key
    if resource_id:
        data["resource_id"] = resource_id
    
    result = api_request("/api/tts/doubao", method="POST", data=data)
    
    mode = "阻塞模式" if blocking else "非阻塞模式"
    speaker_info = f"[{speaker or DEFAULT_SPEAKER}]"
    emotion_info = f"[{emotion}]" if emotion else ""
    print(f"✅ 火山 TTS [{mode}]{speaker_info}{emotion_info}: {text[:50]}{'...' if len(text) > 50 else ''}")
    return result


def main():
    parser = argparse.ArgumentParser(description="火山 TTS 语音合成")
    parser.add_argument("text", help="要合成的文本")
    parser.add_argument("--speaker", "-s", default=DEFAULT_SPEAKER,
                        help=f"音色 ID（默认: {DEFAULT_SPEAKER}）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="语速 0.8-2.0（默认 1.0）")
    parser.add_argument("--emotion", "-e",
                        help="情感参数（如: happy, sad, angry, lovey-dovey）")
    parser.add_argument("--context", "-c",
                        help="上下文指令（仅 2.0 音色支持）")
    parser.add_argument("--blocking", action="store_true",
                        help="阻塞等待播放完成（默认非阻塞）")
    parser.add_argument("--app-id",
                        help="火山 App ID（可选）")
    parser.add_argument("--access-key",
                        help="火山 Access Key（可选）")
    
    args = parser.parse_args()
    
    # 验证语速范围
    if not 0.8 <= args.speed <= 2.0:
        print("❌ 错误: 语速必须在 0.8-2.0 之间")
        sys.exit(1)
    
    try:
        context = args.context
        if context:
            context = [context]
        
        result = tts_doubao(
            text=args.text,
            speaker=args.speaker,
            speed=args.speed,
            emotion=args.emotion,
            context_texts=context,
            app_id=args.app_id,
            access_key=args.access_key,
            blocking=args.blocking
        )
        
        if result.get("success"):
            if args.blocking:
                print("🎵 火山 TTS 播放完成")
                print("RESULT success=true completed=true")
            else:
                print("🎵 Bridge 已接受火山 TTS 后台播放请求")
                print("RESULT success=true accepted=true")
        else:
            print(f"❌ 播放失败: {result}", file=sys.stderr)
            print("RESULT success=false", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("RESULT success=false", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
