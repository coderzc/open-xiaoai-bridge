import logging
import sys
import os
from datetime import datetime
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化时间
        record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 添加颜色到日志级别
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)

class XiaozhiLogger:
    """小智日志记录器"""
    
    def __init__(self, name: str = "xiaozhi"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        # Get the logging level from the 'LOGLEVEL' environment variable,
        # defaulting to 'INFO' if not set.
        log_level = os.environ.get("LOGLEVEL", "INFO").upper()
        
        # 级别映射
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # 获取日志级别常量
        numeric_level = level_map.get(log_level, logging.INFO)

        self.logger.setLevel(numeric_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # 创建格式化器
        formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """调试日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """信息日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """严重错误日志"""
        self.logger.critical(message)
    
    def wakeup(self, keyword: str, speech_prob: Optional[float] = None):
        """唤醒日志"""
        if speech_prob:
            message = f"🔥 触发唤醒: {keyword} (speech_prob: {speech_prob:.2f})"
        else:
            message = f"🔥 触发唤醒: {keyword}"
        self.info(message)
    
    def user_speech(self, text: str):
        """用户语音日志"""
        self.info(f"💬 我说：{text}")
    
    def ai_response(self, text: str):
        """AI回复日志"""
        self.info(f"🤖 小智：{text}")
    
    def vad_event(self, event: str, details: str = ""):
        """VAD事件日志"""
        message = f"🎤 VAD: {event}"
        if details:
            message += f" ({details})"
        self.info(message)
    
    def kws_event(self, event: str, details: str = ""):
        """KWS事件日志"""
        message = f"🔍 KWS: {event}"
        if details:
            message += f" ({details})"
        self.info(message)
    
    def device_state(self, state: str):
        """设备状态日志"""
        self.info(f"📱 状态: {state}")

# 创建全局日志实例
logger = XiaozhiLogger()
