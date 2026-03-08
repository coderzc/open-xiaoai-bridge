from typing import Any

GLOBAL_STATES = {}


def set_app(app: Any):
    """Set the main application instance."""
    GLOBAL_STATES["app"] = app


def get_app() -> Any:
    """Get the main application instance."""
    return GLOBAL_STATES.get("app")


def set_xiaozhi(xiaozhi: Any):
    GLOBAL_STATES["xiaozhi"] = xiaozhi


def get_xiaozhi() -> Any:
    return GLOBAL_STATES.get("xiaozhi")


def set_xiaoai(xiaoai: Any):
    GLOBAL_STATES["xiaoai"] = xiaoai


def get_xiaoai() -> Any:
    return GLOBAL_STATES.get("xiaoai")


def set_vad(vad: Any):
    GLOBAL_STATES["vad"] = vad


def get_vad() -> Any:
    return GLOBAL_STATES.get("vad")


def set_audio_codec(opus_encoder: Any):
    GLOBAL_STATES["opus_encoder"] = opus_encoder


def get_audio_codec() -> Any:
    return GLOBAL_STATES.get("opus_encoder")


def get_speaker() -> Any:
    return GLOBAL_STATES.get("speaker")


def set_speaker(speaker: Any):
    GLOBAL_STATES["speaker"] = speaker


def set_kws(kws: Any):
    GLOBAL_STATES["kws"] = kws


def get_kws() -> Any:
    return GLOBAL_STATES.get("kws")


def set_speech_frames(speech_frames: Any):
    GLOBAL_STATES["speech_frames"] = speech_frames


def get_speech_frames() -> Any:
    return GLOBAL_STATES.get("speech_frames")
