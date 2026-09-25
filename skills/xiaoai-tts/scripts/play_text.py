#!/usr/bin/env python3
"""
播放文字（小爱自带 TTS）
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import api_request


def play_text(text, blocking=False, timeout=60000):
    """
    使用小爱自带 TTS 播放文字
    
    Args:
        text: 要播放的文字
        blocking: 是否阻塞等待（默认 False，文档错误更正）
        timeout: 超时时间（毫秒，默认 60000）
    """
    data = {
        "text": text,
        "blocking": blocking,
        "timeout": timeout
    }
    
    result = api_request("/api/play/text", method="POST", data=data)
    
    mode = "阻塞模式" if blocking else "非阻塞模式"
    print(f"✅ 已发送播放请求 [{mode}]: {text[:50]}{'...' if len(text) > 50 else ''}")
    return result


def main():
    parser = argparse.ArgumentParser(description="小爱 TTS 播放文字")
    parser.add_argument("text", help="要播放的文字内容")
    parser.add_argument("--blocking", action="store_true", 
                        help="阻塞等待播放完成（默认非阻塞）")
    parser.add_argument("--no-blocking", action="store_true",
                        help="非阻塞模式（默认）")
    parser.add_argument("--timeout", type=int, default=60000,
                        help="超时时间（毫秒，默认 60000）")
    
    args = parser.parse_args()
    
    # 默认非阻塞（blocking=False）
    blocking = args.blocking if args.blocking else False
    
    try:
        result = play_text(args.text, blocking=blocking, timeout=args.timeout)
        if result.get("success"):
            if blocking:
                print("🎵 播放完成")
                print("RESULT success=true completed=true")
            else:
                print("🎵 Bridge 已接受后台播放请求")
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
