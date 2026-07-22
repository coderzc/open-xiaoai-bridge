"""QwenPaw continuous conversation controller."""

from core.external_conversation import ExternalConversationController
from core.qwenpaw import QwenPawManager


class QwenPawConversationController(ExternalConversationController):
    """Continuous conversation mode for QwenPaw."""

    CONFIG_PREFIX = "qwenpaw"
    BACKEND_NAME = "QwenPaw"
    LOG_MODULE = "QwenPaw Conv"
    WAKEUP_SOURCE = "qwenpaw"
    MANAGER = QwenPawManager
