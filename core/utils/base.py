import json
import os
import random


def get_env(key: str, default_value: str | None = None):
    return os.environ.get(key, default_value)


def to_set(data):
    if isinstance(data, list):
        return list(set(data))
    return data


def pick_one(data: list):
    if len(data) == 0:
        return None
    return data[random.randint(0, len(data) - 1)]


def json_encode(obj, pretty=False):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=4 if pretty else None)
    except Exception as _:
        return None


def json_decode(text):
    try:
        return json.loads(text)
    except Exception:
        return None
