"""极简 .env 加载器，只依赖标准库。

从当前目录向上找 .env，把里面的 KEY=VALUE 写进 os.environ（不覆盖已有的环境变量）。
"""
from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def find_dotenv(start=None):
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        p = d / ".env"
        if p.is_file():
            return p
    return None


def load_dotenv(path=None, override=False):
    """加载 .env，返回实际写入的键名列表。重复调用只会真正读一次（除非指定 path）。"""
    global _loaded
    if _loaded and path is None and not override:
        return []
    p = Path(path) if path else find_dotenv()
    if path is None:
        _loaded = True
    if p is None or not p.is_file():
        return []
    written = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        written.append(key)
    return written
