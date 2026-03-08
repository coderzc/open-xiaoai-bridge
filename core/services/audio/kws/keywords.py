import re

from sherpa_onnx import text2token


def init_project_context():
    """动态导入父模块"""
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


init_project_context()

from config import APP_CONFIG
from core.utils.file import get_model_file_path


def get_args():
    tokens_type = "cjkchar+bpe"
    tokens = get_model_file_path("tokens.txt")
    bpe_model = get_model_file_path("bpe.model")
    output = get_model_file_path("keywords.txt")
    keywords = APP_CONFIG["wakeup"]["keywords"]
    texts = [f"{keyword.upper()}" for keyword in keywords]
    return locals()


def main():
    args = get_args()
    encoded_texts = text2token(
        args["texts"],
        tokens=args["tokens"],
        tokens_type=args["tokens_type"],
        bpe_model=args["bpe_model"],
    )
    with open(args["output"], "w", encoding="utf8") as f:
        for _, txt in enumerate(encoded_texts):
            line = "".join(txt)
            if re.match(r"^[▁A-Z\s]+$", line):
                f.write(" ".join(txt) + "\n")
            else:
                f.write(" ".join(txt) + f" @{line}" + "\n")


if __name__ == "__main__":
    main()
