"""通过 OpenRouter 的 Decisions API 调用 Jev，不需要 TypeSafe 的排队资格。

    POST https://openrouter.ai/api/alpha/decisions
    {"model": "~typesafe/jev-latest", "state": ..., "questions": {"move": {"type": "choice", ...}}}

只依赖标准库。用长连接复用 TLS，实时模式下能省掉每次握手的时间。
"""
from __future__ import annotations

import http.client
import json
import os
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit
from types import SimpleNamespace

HOST = "openrouter.ai"
PATH = "/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


@dataclass
class Choice:
    instructions: str | None = None
    criteria: dict | None = None

    def to_json(self):
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


class OpenRouterJevClient:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, timeout=10.0):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise SystemExit("缺少 OPENROUTER_API_KEY 环境变量")
        self.model = model
        self.timeout = timeout
        self._conn = None

    def _connection(self):
        if self._conn is None:
            # 自动沿用系统 / 环境变量里的 HTTPS 代理（如 Clash 的 127.0.0.1:7890）
            proxy = urllib.request.getproxies().get("https")
            if proxy:
                p = urlsplit(proxy if "://" in proxy else "http://" + proxy)
                self._conn = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=self.timeout)
                self._conn.set_tunnel(HOST, 443)
            else:
                self._conn = http.client.HTTPSConnection(HOST, timeout=self.timeout)
        return self._conn

    def system_one(self, state, questions):
        body = json.dumps({
            "model": self.model,
            "state": state,
            "questions": {k: q.to_json() for k, q in questions.items()},
        })
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "jev-pacman",
        }
        for attempt in range(2):  # 长连接被服务端关掉时重连一次
            try:
                conn = self._connection()
                conn.request("POST", PATH, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                break
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt:
                    raise
        if resp.status != 200:
            raise RuntimeError(f"OpenRouter HTTP {resp.status}: {data[:300].decode(errors='replace')}")
        payload = json.loads(data)
        answers = {k: SimpleNamespace(**v) for k, v in payload.get("answers", {}).items()}
        return SimpleNamespace(answers=answers, raw=payload)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
