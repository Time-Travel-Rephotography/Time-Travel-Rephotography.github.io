import os
from typing import Iterable


def optional_string(condition: bool, string: str):
    return string if condition else ""


def parent_dir(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def iterable_to_str(iterable: Iterable) -> str:
    return ','.join([str(x) for x in iterable])
