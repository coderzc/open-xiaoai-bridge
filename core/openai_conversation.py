"""OpenAI-compatible continuous conversation controller."""

from core.openai import OpenAIManager
from core.external_conversation import ExternalConversationController


class OpenAIConversationController(ExternalConversationController):
    """Manages multi-turn conversation for OpenAI-compatible services."""

    CONFIG_PREFIX = "openai"
    BACKEND_NAME = "OpenAI"
    LOG_MODULE = "OpenAI Conv"
    WAKEUP_SOURCE = "openai"
    MANAGER = OpenAIManager
