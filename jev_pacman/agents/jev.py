"""Jev 智能体：每一步把局面作为 state、把“往哪走”作为一道 Choice 题交给 Jev。

真实接入（拿到资格后）：
    pip install typesafe-sdk
    设置环境变量 TYPESAFE_API_KEY
    agent = JevAgent.from_typesafe()

SDK 调用方式参照公开示例：
    TypeSafeClient().system_one(state=..., questions={...})
    response.answers[key].choice / .probabilities
如果正式版 SDK 的字段名不同，只需要改 _ask() 这一个方法。
"""
from __future__ import annotations

import time

INSTRUCTIONS = (
    "You are playing Pac-Man. Pick the direction Pac-Man should move next. "
    "Priority 1: survive. Never move toward a hunting ghost that is only a few steps away. "
    "Priority 2: while ghosts are frightened, chase a frightened ghost if you can reach it "
    "before frightened_ticks_left runs out. "
    "Priority 3: otherwise move toward the nearest pellets. Avoid pointless reversing."
)


def _fmt(v):
    return "none" if v is None else f"{v} steps"


def describe_move(d, f):
    parts = [
        f"Move {d}.",
        f"Nearest pellet: {_fmt(f['pellet_steps'])}.",
        f"Nearest hunting ghost: {_fmt(f['hunting_ghost_steps'])}.",
        f"Nearest frightened ghost: {_fmt(f['frightened_ghost_steps'])}.",
        f"Pellets within 6 steps: {f['pellets_within_6_steps']}.",
    ]
    if f["is_current_direction"]:
        parts.append("Keeps the current direction.")
    if f["reverses_direction"]:
        parts.append("Reverses direction.")
    return " ".join(parts)


class JevAgent:
    name = "jev"

    def __init__(self, client, choice_cls, include_map=True):
        self.client = client
        self.Choice = choice_cls
        self.include_map = include_map
        self.calls = 0
        self.errors = 0
        self.last_error = None
        self.last_probabilities = None
        self.last_confidence = None
        self.api_latencies = []

    @classmethod
    def from_typesafe(cls, **kw):
        try:
            from typesafe_sdk import Choice, TypeSafeClient
        except ImportError as e:
            raise SystemExit("未安装 typesafe-sdk，请先运行: pip install typesafe-sdk") from e
        return cls(TypeSafeClient(), Choice, **kw)

    @classmethod
    def from_openrouter(cls, model=None, **kw):
        from .openrouter_client import DEFAULT_MODEL, Choice, OpenRouterJevClient
        agent = cls(OpenRouterJevClient(model=model or DEFAULT_MODEL), Choice, **kw)
        agent.name = "jev-openrouter"
        return agent

    @classmethod
    def mock(cls, latency_ms=(70, 500), seed=None, **kw):
        from .mock_typesafe import Choice, MockTypeSafeClient
        agent = cls(MockTypeSafeClient(latency_ms=latency_ms, seed=seed), Choice, **kw)
        agent.name = "mockjev"
        return agent

    def act(self, state):
        moves = state["moves"]
        if len(moves) == 1:
            return next(iter(moves))
        payload = dict(state) if self.include_map else {k: v for k, v in state.items()
                                                         if k not in ("map", "map_legend")}
        criteria = {d: describe_move(d, f) for d, f in moves.items()}
        t0 = time.perf_counter()
        try:
            choice = self._ask(payload, criteria)
        except Exception as e:  # 网络错误、限流等：不中断游戏，退回到保持方向
            self.errors += 1
            self.last_error = repr(e)
            choice = None
        finally:
            self.api_latencies.append(time.perf_counter() - t0)
            self.calls += 1
        if choice not in moves:
            facing = state["pacman"]["facing"]
            choice = facing if facing in moves else next(iter(moves))
        return choice

    def _ask(self, payload, criteria):
        question = self.Choice(instructions=INSTRUCTIONS, criteria=criteria)
        resp = self.client.system_one(state=payload, questions={"move": question})
        ans = resp.answers["move"]
        self.last_probabilities = getattr(ans, "probabilities", None)
        self.last_confidence = getattr(ans, "confidence", None)
        choice = getattr(ans, "choice", None)
        choice = getattr(choice, "value", choice)  # 兼容枚举类型
        return None if choice is None else str(choice).upper()

    def close(self):
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
