"""基线智能体，用来和 Jev 对比。所有智能体都实现 act(state) -> 动作名。"""
from __future__ import annotations

import random

from ..features import heuristic_score


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(list(state["moves"]))


class GreedyAgent:
    """规则基线：躲开近处的幽灵，追受惊幽灵，否则去最近的豆子。"""
    name = "greedy"

    def act(self, state):
        left = state["frightened_ticks_left"]
        return max(state["moves"], key=lambda d: heuristic_score(state["moves"][d], left))
