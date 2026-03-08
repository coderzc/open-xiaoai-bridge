import os


def get_model_file_path(file_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../models", file_name))


def read_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
