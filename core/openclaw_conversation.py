"""OpenClaw continuous conversation controller."""

from core.external_conversation import ExternalConversationController
from core.openclaw import OpenClawManager


class OpenClawConversationController(ExternalConversationController):
    """Manages multi-turn OpenClaw conversation."""

    CONFIG_PREFIX = "openclaw"
    BACKEND_NAME = "OpenClaw"
    LOG_MODULE = "OpenClaw Conv"
    WAKEUP_SOURCE = "openclaw"
    MANAGER = OpenClawManager
