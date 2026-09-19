"""离线模拟的 TypeSafe 客户端，接口形状仿照 typesafe_sdk，拿到 Jev 资格前用来跑通整条链路。

它只根据传进来的 state 和 questions 做决定（和真 Jev 看到的输入完全一样），
并随机休眠 70~500ms 模拟真实延迟，这样实时模式下的“决策跟不上画面”也能提前测。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

from ..features import heuristic_score, softmax


@dataclass
class Choice:
    instructions: str | None = None
    criteria: dict | None = None


@dataclass
class Noul:
    instructions: str | None = None


@dataclass
class Score:
    instructions: str | None = None
    criteria: list | None = None


class MockTypeSafeClient:
    def __init__(self, latency_ms=(70, 500), seed=None, temperature=10.0):
        self.latency_ms = latency_ms
        self.rng = random.Random(seed)
        self.temperature = temperature

    def system_one(self, state, questions):
        time.sleep(self.rng.uniform(*self.latency_ms) / 1000)
        answers = {}
        for key, q in questions.items():
            if isinstance(q, Choice):
                answers[key] = self._choice(state, q)
            elif isinstance(q, Noul):
                answers[key] = SimpleNamespace(noul=0.5, confidence=0.0)
            elif isinstance(q, Score):
                answers[key] = SimpleNamespace(score=q.criteria[0], probabilities=None, confidence=0.0)
        return SimpleNamespace(answers=answers)

    def _choice(self, state, q):
        options = list(q.criteria)
        moves = state.get("moves", {})
        if all(o in moves for o in options):
            left = state.get("frightened_ticks_left", 0)
            scores = {o: heuristic_score(moves[o], left) for o in options}
            probs = softmax(scores, self.temperature)
        else:
            probs = {o: 1 / len(options) for o in options}
        # 按概率采样，模拟模型偶尔“看走眼”
        pick = self.rng.choices(options, weights=[probs[o] for o in options])[0]
        return SimpleNamespace(choice=pick, probabilities=probs, confidence=probs[pick])

    def close(self):
        pass
